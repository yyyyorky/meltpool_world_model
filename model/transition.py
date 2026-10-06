"""Flow-matching DiT transition model."""
import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from einops import rearrange


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def timestep_embed(t, dim, max_period=10000):
    """Sinusoidal embedding of t in [0, 1], (B,) -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period)
                      * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0) * 1000.0
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class Attention(nn.Module):

    def __init__(self, dim, heads):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(B, N, C))


class DiTBlock(nn.Module):

    def __init__(self, dim, heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim))
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x, c):
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.adaLN(c).chunk(6, dim=-1)
        x = x + gate_a.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_a, scale_a))
        x = x + gate_m.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_m, scale_m))
        return x


class DiTTransition(nn.Module):
    """Velocity field over the latent residual, conditioned on the past window,
    the controls and the flow time.
    """

    def __init__(self, z_dim, past_horizon, ctrl_dim,
                 H=25, W=25, D=8, patch=(2, 2, 1),
                 dim=256, depth=8, heads=8, mlp_ratio=4):
        super().__init__()
        self.z_dim, self.H, self.W, self.D = z_dim, H, W, D
        self.patch = patch

        def _pad(n, p):
            total = (math.ceil(n / p) * p) - n
            return total // 2, total - total // 2

        self.pad_h, self.pad_w, self.pad_d = _pad(H, patch[0]), _pad(W, patch[1]), _pad(D, patch[2])
        Hp, Wp, Dp = H + sum(self.pad_h), W + sum(self.pad_w), D + sum(self.pad_d)
        self.gh, self.gw, self.gd = Hp // patch[0], Wp // patch[1], Dp // patch[2]
        self.N = self.gh * self.gw * self.gd
        self.dim = dim

        in_ch = z_dim * (past_horizon + 1)
        self.patch_embed = nn.Conv3d(in_ch, dim, kernel_size=patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, self.N, dim))

        self.t_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.ctrl_mlp = nn.Sequential(nn.Linear(ctrl_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

        self.blocks = nn.ModuleList([DiTBlock(dim, heads, mlp_ratio) for _ in range(depth)])

        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_out = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.head = nn.Linear(dim, z_dim * patch[0] * patch[1] * patch[2])

        self._init_weights()

    def _init_weights(self):
        def _basic(m):
            if isinstance(m, (nn.Linear, nn.Conv3d)):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(_basic)
        trunc_normal_(self.pos, std=0.02)
        for blk in self.blocks:
            nn.init.zeros_(blk.adaLN[-1].weight)
            nn.init.zeros_(blk.adaLN[-1].bias)
        nn.init.zeros_(self.adaLN_out[-1].weight)
        nn.init.zeros_(self.adaLN_out[-1].bias)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, ctrl, t):
        x = F.pad(x, (self.pad_d[0], self.pad_d[1], self.pad_w[0], self.pad_w[1],
                      self.pad_h[0], self.pad_h[1]), mode="replicate")
        h = self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos

        c = self.t_mlp(timestep_embed(t, self.dim)) + self.ctrl_mlp(ctrl)
        for blk in self.blocks:
            h = blk(h, c)

        shift, scale = self.adaLN_out(c).chunk(2, dim=-1)
        h = self.head(modulate(self.norm_out(h), shift, scale))

        p0, p1, p2 = self.patch
        v = rearrange(h, "b (gh gw gd) (z p0 p1 p2) -> b z (gh p0) (gw p1) (gd p2)",
                      gh=self.gh, gw=self.gw, gd=self.gd, z=self.z_dim,
                      p0=p0, p1=p1, p2=p2)
        return v[..., self.pad_h[0]:self.pad_h[0] + self.H,
                 self.pad_w[0]:self.pad_w[0] + self.W,
                 self.pad_d[0]:self.pad_d[0] + self.D]


def fm_loss(model, past_latents, last_frame, next_frame, ctrl,
            sample_weight=None, return_pred=False):
    """Rectified-flow loss on the residual, optionally with the residual estimate."""
    B = next_frame.shape[0]
    x1 = next_frame - last_frame
    x0 = torch.randn_like(x1)
    t = torch.rand(B, device=x1.device)
    tb = t.view(B, 1, 1, 1, 1)
    xt = (1 - tb) * x0 + tb * x1
    v_target = x1 - x0

    v = model(torch.cat([xt, past_latents], dim=1), ctrl, t)

    per_sample = F.mse_loss(v, v_target, reduction="none").mean(dim=[1, 2, 3, 4])
    loss = (per_sample * sample_weight).mean() if sample_weight is not None \
        else per_sample.mean()

    if return_pred:
        return loss, xt + (1 - tb) * v
    return loss


@torch.no_grad()
def fm_sample(model, past_latents, last_frame, ctrl, nfe=4, noise=None,
              requantize=None):
    """Euler-integrate from noise to the residual and return the next latent."""
    x = torch.randn_like(last_frame) if noise is None else noise
    dt = 1.0 / nfe
    for i in range(nfe):
        t = torch.full((x.shape[0],), i * dt, device=x.device)
        x = x + dt * model(torch.cat([x, past_latents], dim=1), ctrl, t)
    z = last_frame + x
    return requantize(z) if requantize is not None else z


def build_transition(model_cfg, past_horizon):
    """Build the DiT, or the 'vit' / 'transolver' baseline, from a config."""
    arch = model_cfg.get("arch", "dit")
    H, W, D = model_cfg["grid"]
    common = dict(z_dim=model_cfg["z_dim"], past_horizon=past_horizon,
                  ctrl_dim=model_cfg["cond_dim"], H=H, W=W, D=D,
                  dim=model_cfg["width"], depth=model_cfg["depth"],
                  heads=model_cfg["heads"], mlp_ratio=model_cfg["mlp_ratio"])
    if arch == "dit":
        return DiTTransition(patch=tuple(model_cfg["patch"]), **common)
    if arch == "vit":
        from model.vit import ViTTransition
        return ViTTransition(patch=tuple(model_cfg["patch"]), **common)
    if arch == "transolver":
        from model.transolver import TransolverTransition
        return TransolverTransition(slice_num=model_cfg["slice_num"], **common)
    raise ValueError(f"unknown transition arch {arch!r}")


def is_generative(model):
    return isinstance(model, DiTTransition)


def predict_next(model, past_latents, ctrl, noise=None, nfe=4, requantize=None):
    if is_generative(model):
        z_dim = model.z_dim
        return fm_sample(model, past_latents, past_latents[:, -z_dim:], ctrl,
                         nfe=nfe, noise=noise, requantize=requantize)
    z = model(past_latents, ctrl)
    return requantize(z) if requantize is not None else z


def load_transition(config_path, checkpoint, device):
    with open(config_path) as f:
        cfg = json.load(f)
    model = build_transition(cfg["model"], cfg["data"]["past_horizon"]).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model, cfg
