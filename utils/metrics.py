"""Melt-pool metrics. Fields are (..., nx, ny, nz) with z the depth axis."""
import numpy as np
import torch


def relative_l2_series(pred_series, target_series, eps=1e-8):
    T = pred_series.shape[0]
    diff = (pred_series - target_series).reshape(T, -1)
    ref = target_series.reshape(T, -1)
    return np.linalg.norm(diff, axis=1) / (np.linalg.norm(ref, axis=1) + eps)


def compute_snapshot_iou(pred_muwafa_series, target_muwafa_series, threshold=0.5):
    pred = pred_muwafa_series > threshold
    tgt = target_muwafa_series > threshold
    intersection = np.sum(pred & tgt, axis=(1, 2, 3))
    union = np.sum(pred | tgt, axis=(1, 2, 3))
    return intersection / np.maximum(union, 1)


def compute_meltpool_depth(alpha_series, muwafa_series, threshold=0.5,
                           dz=1e-5, z_min=-2e-4, z_substrate=0.0,
                           require_contiguous=True, subvoxel=False):
    """Melt-pool depth below the substrate, (T,) in metres."""
    T, nx, ny, nz = muwafa_series.shape
    f = (alpha_series * muwafa_series).astype(np.float64)
    mask = f > threshold

    z_coords = z_min + np.arange(nz) * dz
    depth_mask = mask
    if require_contiguous:
        sub_idx = min(int(round((z_substrate - z_min) / dz)), nz)
        depth_mask = np.zeros_like(mask)
        if sub_idx > 0:
            mask_sub = mask[:, :, :, :sub_idx]
            contig = np.minimum.accumulate(
                mask_sub[..., ::-1].astype(np.int8), axis=-1)[..., ::-1]
            depth_mask[:, :, :, :sub_idx] = contig.astype(bool)

    if not subvoxel:
        z_masked = np.where(depth_mask, z_coords[None, None, None, :], np.inf)
        z_min_vals = z_masked.reshape(T, -1).min(axis=1)
        depth = np.where(np.isinf(z_min_vals), 0.0, z_substrate - z_min_vals)
        return np.maximum(depth, 0.0)

    has_melt = depth_mask.any(axis=-1)
    k = np.argmax(depth_mask, axis=-1)
    k_below = np.maximum(k - 1, 0)

    f_k = np.take_along_axis(f, k[..., None], axis=-1)[..., 0]
    f_below = np.take_along_axis(f, k_below[..., None], axis=-1)[..., 0]
    z_k = z_min + k * dz

    denom = f_k - f_below
    frac = np.zeros_like(f_k)
    np.divide(f_k - threshold, denom, out=frac,
              where=(k > 0) & (np.abs(denom) > 1e-12))
    z_iso = z_k - np.clip(frac, 0.0, 1.0) * dz
    z_iso = np.where(has_melt, z_iso, z_substrate)
    depth = z_substrate - z_iso.reshape(T, -1).min(axis=1)
    return np.maximum(depth, 0.0)


def soft_threshold(x, threshold, sharpness=50.0):
    return torch.sigmoid(sharpness * (x - threshold))


def compute_meltpool_depth_series(muwafa_series, alpha_series=None,
                                  z_min=-2e-4, dz=1e-5, z_substrate=0.0,
                                  threshold=0.5, sharpness=100.0,
                                  smooth_max_beta=10.0,
                                  require_contiguous=True):
    """Differentiable melt-pool depth, (T,) in metres."""
    T = muwafa_series.shape[0]
    nz = muwafa_series.shape[-1]
    device, dtype = muwafa_series.device, muwafa_series.dtype

    z_coords = z_min + torch.arange(nz, device=device, dtype=dtype) * dz
    sub_idx = min(int(round((z_substrate - z_min) / dz)), nz)
    if sub_idx <= 0:
        return torch.zeros(T, device=device, dtype=dtype)

    melt = soft_threshold(muwafa_series[..., :sub_idx], threshold, sharpness)
    if require_contiguous:
        cum_contig, _ = torch.cummin(melt.flip(-1), dim=-1)
        melt = cum_contig.flip(-1)

    z_has_melt = melt.amax(dim=(1, 2))
    depth_per_level = z_substrate - z_coords[:sub_idx]
    effective_depth = depth_per_level.unsqueeze(0) * z_has_melt

    weights = torch.softmax(smooth_max_beta * effective_depth / dz, dim=-1)
    return (weights * effective_depth).sum(dim=-1)


def compute_meltpool_depth_map_series(muwafa_series, z_min=-2e-4, dz=1e-5,
                                      z_substrate=0.0, threshold=0.5,
                                      sharpness=20.0, smooth_max_beta=10.0,
                                      require_contiguous=True):
    """Differentiable melt-pool depth at every (x, y), (T, nx, ny)."""
    T, nx, ny, nz = muwafa_series.shape
    device, dtype = muwafa_series.device, muwafa_series.dtype

    z_coords = z_min + torch.arange(nz, device=device, dtype=dtype) * dz
    sub_idx = min(int(round((z_substrate - z_min) / dz)), nz)
    if sub_idx <= 0:
        return torch.zeros(T, nx, ny, device=device, dtype=dtype)

    melt = soft_threshold(muwafa_series[..., :sub_idx], threshold, sharpness)
    if require_contiguous:
        cum_contig, _ = torch.cummin(melt.flip(-1), dim=-1)
        melt = cum_contig.flip(-1)

    depth_per_level = z_substrate - z_coords[:sub_idx]
    effective_depth = depth_per_level.view(1, 1, 1, -1) * melt

    weights = torch.softmax(smooth_max_beta * effective_depth / dz, dim=-1)
    return (weights * effective_depth).sum(dim=-1)


def compute_support_thickness_map(alpha, z_min=-2e-4, dz=1e-5,
                                  z_substrate=0.0, threshold=0.5):
    """Solid thickness just below the substrate, (nx, ny) in metres."""
    sub_idx = min(int(round((z_substrate - z_min) / dz)), alpha.shape[-1])
    if sub_idx <= 0:
        return torch.zeros(alpha.shape[:2], device=alpha.device, dtype=alpha.dtype)

    below = alpha[..., :sub_idx].flip(-1)
    is_solid = (below > threshold).to(alpha.dtype)
    cum_solid = torch.cumprod(is_solid, dim=-1)
    return cum_solid.sum(dim=-1) * dz


def compute_max_relative_depth_series(depth_map, thickness_map,
                                      min_thickness=1e-5,
                                      smooth_max_beta=50.0,
                                      controllable_thickness=0.0):
    """Smooth max of melt depth over local support thickness, (T,)."""
    T = depth_map.shape[0]
    H_eff = torch.clamp(thickness_map, min=min_thickness)
    rel_map = depth_map / H_eff.unsqueeze(0)

    if controllable_thickness > 0.0:
        controllable = (thickness_map >= controllable_thickness).to(rel_map.dtype)
        rel_map = rel_map * controllable.unsqueeze(0)

    flat = rel_map.reshape(T, -1)
    weights = torch.softmax(smooth_max_beta * flat, dim=-1)
    return (weights * flat).sum(dim=-1)
