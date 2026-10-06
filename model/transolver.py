"""Transolver transition baseline (Wu et al., 2024)."""
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from einops import rearrange


class MLP(nn.Module):

    def __init__(self, n_input, n_hidden, n_output):
        super().__init__()
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), nn.GELU())
        self.linear_post = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        return self.linear_post(self.linear_pre(x))


class PhysicsAttention3D(nn.Module):

    def __init__(self, dim, heads, dim_head, slice_num, H, W, D, kernel=3):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.H, self.W, self.D = H, W, D

        self.in_project_x = nn.Conv3d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_fx = nn.Conv3d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        torch.nn.init.orthogonal_(self.in_project_slice.weight)
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        # nn.Identity keeps the state-dict index of the output projection.
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Identity())

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, self.D, C).contiguous().permute(0, 4, 1, 2, 3).contiguous()

        fx_mid = self.in_project_fx(x).permute(0, 2, 3, 4, 1).contiguous() \
            .reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        x_mid = self.in_project_x(x).permute(0, 2, 3, 4, 1).contiguous() \
            .reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        slice_weights = self.softmax(
            self.in_project_slice(x_mid) / torch.clamp(self.temperature, min=0.1, max=5))
        slice_norm = slice_weights.sum(2)
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        q = self.to_q(slice_token)
        k = self.to_k(slice_token)
        v = self.to_v(slice_token)
        attn = self.softmax(torch.matmul(q, k.transpose(-1, -2)) * self.scale)
        out_slice_token = torch.matmul(attn, v)

        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        return self.to_out(rearrange(out_x, "b h n d -> b n (h d)"))


class TransolverBlock(nn.Module):

    def __init__(self, heads, dim, mlp_ratio, slice_num, H, W, D):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim)
        self.Attn = PhysicsAttention3D(dim, heads, dim // heads, slice_num, H, W, D)
        self.ln_2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim * mlp_ratio, dim)

    def forward(self, fx):
        fx = self.Attn(self.ln_1(fx)) + fx
        return self.mlp(self.ln_2(fx)) + fx


class TransolverTransition(nn.Module):

    def __init__(self, z_dim, past_horizon, ctrl_dim, H=25, W=25, D=8,
                 dim=256, depth=5, heads=8, slice_num=64, mlp_ratio=1):
        super().__init__()
        self.z_dim, self.H, self.W, self.D = z_dim, H, W, D
        self.dim = dim

        self.input_proj = nn.Sequential(nn.Conv3d(z_dim * past_horizon + 3, dim, kernel_size=1),
                                        nn.GELU())
        self.register_buffer("grid", self._create_grid())
        self.cond_fc = nn.Sequential(nn.Linear(ctrl_dim, dim), nn.GELU(),
                                     nn.Linear(dim, dim * 3))
        self.blocks = nn.ModuleList([TransolverBlock(heads, dim, mlp_ratio, slice_num, H, W, D)
                                     for _ in range(depth)])
        self.output_proj = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, z_dim))

        def _init(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        self.apply(_init)

    def _create_grid(self):
        """Normalized cell-centre coordinates of the latent grid, (3, H, W, D)."""
        def centres(lo, hi, n):
            v = torch.linspace(lo, hi, n + 1)
            return (v[:-1] + v[1:]) / 2
        xx, yy, zz = torch.meshgrid(centres(-5e-1, 5e-1, self.H), centres(-5e-1, 5e-1, self.W),
                                    centres(-2e-1, 1.2e-1, self.D), indexing="ij")
        return torch.stack([xx, yy, zz], dim=0)

    def forward(self, x, ctrl):
        B, _, H, W, D = x.shape
        last_frame = x[:, -self.z_dim:]
        x = torch.cat([x, self.grid.unsqueeze(0).expand(B, -1, -1, -1, -1)], dim=1)
        fx = self.input_proj(x).permute(0, 2, 3, 4, 1).contiguous().reshape(B, H * W * D, self.dim)

        gamma, beta, add_emb = self.cond_fc(ctrl).chunk(3, dim=-1)
        fx = (1 + gamma.unsqueeze(1)) * fx + beta.unsqueeze(1) + add_emb.unsqueeze(1)
        for block in self.blocks:
            fx = block(fx)

        fx = self.output_proj(fx).reshape(B, H, W, D, self.z_dim).permute(0, 4, 1, 2, 3).contiguous()
        return fx + last_frame
