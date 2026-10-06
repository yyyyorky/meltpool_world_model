"""ViT transition baseline."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from einops import rearrange

from model.transition import Attention


class Block(nn.Module):

    def __init__(self, dim, heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        # nn.Identity keeps the state-dict index of the second linear layer.
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(),
                                 nn.Identity(), nn.Linear(dim * mlp_ratio, dim))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class ViTTransition(nn.Module):

    def __init__(self, z_dim, past_horizon, ctrl_dim, H=25, W=25, D=8,
                 patch=(2, 2, 1), dim=256, depth=8, heads=8, mlp_ratio=4):
        super().__init__()
        self.z_dim, self.H, self.W, self.D = z_dim, H, W, D
        self.patch = tuple(patch)

        def _pad(n, p):
            total = (math.ceil(n / p) * p) - n
            return total // 2, total - total // 2

        self.pad_h, self.pad_w, self.pad_d = _pad(H, patch[0]), _pad(W, patch[1]), _pad(D, patch[2])
        Hp, Wp, Dp = H + sum(self.pad_h), W + sum(self.pad_w), D + sum(self.pad_d)
        self.gh, self.gw, self.gd = Hp // patch[0], Wp // patch[1], Dp // patch[2]
        self.N = self.gh * self.gw * self.gd

        self.patch_embed = nn.Conv3d(z_dim * past_horizon, dim, kernel_size=self.patch,
                                     stride=self.patch)
        self.pos = nn.Parameter(torch.zeros(1, self.N, dim))
        self.cond_mlp = nn.Sequential(nn.Linear(ctrl_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([Block(dim, heads, mlp_ratio) for _ in range(depth)])
        self.norm_out = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, z_dim * patch[0] * patch[1] * patch[2])

        def _basic(m):
            if isinstance(m, (nn.Linear, nn.Conv3d)):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(_basic)
        trunc_normal_(self.pos, std=0.02)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, ctrl):
        last_frame = x[:, -self.z_dim:]
        x = F.pad(x, (self.pad_d[0], self.pad_d[1], self.pad_w[0], self.pad_w[1],
                      self.pad_h[0], self.pad_h[1]), mode="replicate")
        h = self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos
        h = h + self.cond_mlp(ctrl).unsqueeze(1)
        for blk in self.blocks:
            h = blk(h)
        h = self.head(self.norm_out(h))

        p0, p1, p2 = self.patch
        delta = rearrange(h, "b (gh gw gd) (z p0 p1 p2) -> b z (gh p0) (gw p1) (gd p2)",
                          gh=self.gh, gw=self.gw, gd=self.gd, z=self.z_dim,
                          p0=p0, p1=p1, p2=p2)
        delta = delta[..., self.pad_h[0]:self.pad_h[0] + self.H,
                      self.pad_w[0]:self.pad_w[0] + self.W,
                      self.pad_d[0]:self.pad_d[0] + self.D]
        return last_frame + delta
