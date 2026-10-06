"""Batched melt-pool metrics and CRPS for rollout evaluation.

Fields are (B, nx, ny, nz); the melt pool is alpha * muwafa > 0.5.
"""
import numpy as np
import torch

from model.autoencoder import T_AMBIENT, T_MAX

DX = DY = DZ = 1e-5
Z_MIN = -2e-4
Z_SUB = 0.0
MELT_THR = 0.5
T_CLAMP = 0.8149  # 3315 K
SCALAR_KEYS = ("depth_um", "depth_vox_um", "width_um", "length_um", "vol_mm3", "peakT_K")


def _contiguous_melt(f, sub_idx):
    """Melt mask restricted to the column connected to the substrate top."""
    ms = (f[..., :sub_idx] > MELT_THR).to(torch.int8)
    return torch.cummin(ms.flip(-1), dim=-1).values.flip(-1).bool()


def depth_subvoxel(alpha, muwafa):
    """Melt-pool depth with sub-voxel interpolation of the 0.5 iso-surface, (B,) m."""
    f = alpha * muwafa
    B, nx, ny, nz = f.shape
    sub_idx = min(int(round((Z_SUB - Z_MIN) / DZ)), nz)
    contig = _contiguous_melt(f, sub_idx)
    has = contig.any(-1)
    idx = torch.arange(sub_idx, device=f.device)
    k = torch.where(contig, idx, torch.full_like(idx, sub_idx)).amin(-1)
    k = k.clamp(max=sub_idx - 1)
    k_below = (k - 1).clamp(min=0)
    f_k = torch.gather(f, -1, k[..., None])[..., 0]
    f_b = torch.gather(f, -1, k_below[..., None])[..., 0]
    denom = f_k - f_b
    frac = torch.where((k > 0) & (denom.abs() > 1e-12), (f_k - MELT_THR) / denom,
                       torch.zeros_like(f_k)).clamp(0.0, 1.0)
    z_iso = Z_MIN + k.to(f.dtype) * DZ - frac * DZ
    z_iso = torch.where(has, z_iso, torch.full_like(z_iso, Z_SUB))
    return (Z_SUB - z_iso.reshape(B, -1).amin(1)).clamp(min=0.0)


def depth_voxel(alpha, muwafa):
    f = alpha * muwafa
    B, nx, ny, nz = f.shape
    sub_idx = min(int(round((Z_SUB - Z_MIN) / DZ)), nz)
    contig = _contiguous_melt(f, sub_idx)
    idx = torch.arange(sub_idx, device=f.device)
    k = torch.where(contig, idx, torch.full_like(idx, sub_idx)).reshape(B, -1).amin(1)
    depth = Z_SUB - (Z_MIN + k.to(f.dtype) * DZ)
    return torch.where(k < sub_idx, depth, torch.zeros_like(depth)).clamp(min=0.0)


def width_length(alpha, muwafa, dirs):
    """Melt-pool extent across and along the scan direction, in metres."""
    m2 = (alpha * muwafa > MELT_THR).any(-1)
    B, nx, ny = m2.shape
    xx = torch.arange(nx, device=m2.device, dtype=alpha.dtype)[:, None].expand(nx, ny) * DX
    yy = torch.arange(ny, device=m2.device, dtype=alpha.dtype)[None, :].expand(nx, ny) * DY
    d = dirs.to(alpha.dtype)
    d = d / (d.norm(dim=1, keepdim=True) + 1e-12)
    tan = xx[None] * d[:, 0, None, None] + yy[None] * d[:, 1, None, None]
    nor = -xx[None] * d[:, 1, None, None] + yy[None] * d[:, 0, None, None]

    def extent(p):
        big = torch.finfo(p.dtype).max
        lo = torch.where(m2, p, torch.full_like(p, big)).reshape(B, -1).amin(1)
        hi = torch.where(m2, p, torch.full_like(p, -big)).reshape(B, -1).amax(1)
        e = hi - lo
        return torch.where(m2.reshape(B, -1).any(1), e, torch.zeros_like(e))
    return extent(nor), extent(tan)


def melt_pool_scalars(T, alpha, muwafa, dirs):
    """Depth, width, length [um], volume [mm^3] and peak temperature [K]."""
    B = T.shape[0]
    width, length = width_length(alpha, muwafa, dirs)
    vol = (alpha * muwafa > MELT_THR).reshape(B, -1).sum(1).to(T.dtype) * (DX * DY * DZ) * 1e9
    peak = T.reshape(B, -1).amax(1) * (T_MAX - T_AMBIENT) + T_AMBIENT
    return dict(depth_um=(depth_subvoxel(alpha, muwafa) * 1e6).cpu().numpy(),
                depth_vox_um=(depth_voxel(alpha, muwafa) * 1e6).cpu().numpy(),
                width_um=(width * 1e6).cpu().numpy(), length_um=(length * 1e6).cpu().numpy(),
                vol_mm3=vol.cpu().numpy(), peakT_K=peak.cpu().numpy())


def relative_l2(pred, gt):
    B = pred.shape[0]
    return (pred - gt).reshape(B, -1).norm(dim=1) / (gt.reshape(B, -1).norm(dim=1) + 1e-8)


def iou(m_pred, m_gt):
    a, b = m_pred > MELT_THR, m_gt > MELT_THR
    B = a.shape[0]
    inter = (a & b).reshape(B, -1).sum(1).float()
    union = (a | b).reshape(B, -1).sum(1).float()
    return inter / union.clamp(min=1.0)


def crps_field(ens, gt):
    """Field-averaged CRPS of an ensemble (E, ...) against gt."""
    E = ens.shape[0]
    t1 = (ens - gt[None]).abs().mean(0).reshape(-1)
    if E == 1:
        return float(t1.mean())
    es = ens.reshape(E, -1).sort(0).values
    w = (2 * torch.arange(1, E + 1, device=ens.device) - E - 1).to(ens.dtype)[:, None]
    return float((t1 - (w * es).sum(0) / (E * E)).mean())


def crps_empirical(ens, obs):
    """Empirical CRPS along the member axis; ens (S, E), obs (S,)."""
    ens = np.asarray(ens, float)
    obs = np.asarray(obs, float)
    t1 = np.abs(ens - obs[:, None]).mean(1)
    E = ens.shape[1]
    if E == 1:
        return t1
    w = 2.0 * np.arange(1, E + 1) - E - 1
    return t1 - (np.sort(ens, axis=1) * w).sum(1) / (E * E)
