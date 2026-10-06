import json
import os

import numpy as np

from utils.vtkhdf_writer import VTKHDFImageDataWriter, cell_to_point_data


def export_cfd_control(out_path, points_m, powers_W, dt, T0, meta,
                       alpha0=None, speeds=None, final_point_m=None,
                       jump_power_off_dist=2.5e-4):
    """Write a CFD replay control file.

    Power is switched off on jumps longer than `jump_power_off_dist`.
    """
    pts = np.asarray(points_m, dtype=np.float64)
    if pts.shape[1] == 2:
        pts = np.concatenate([pts, np.zeros((pts.shape[0], 1))], axis=1)
    pw = np.asarray(powers_W, dtype=np.float64).reshape(-1)
    N = pw.shape[0]
    if pts.shape[0] != N or N == 0:
        raise ValueError(f"points_m ({pts.shape[0]}) and powers_W ({N}) mismatch")

    if final_point_m is None:
        final = pts[-1:].copy()
    else:
        final = np.asarray(final_point_m, dtype=np.float64).reshape(1, -1)
        if final.shape[1] == 2:
            final = np.concatenate([final, np.zeros((1, 1))], axis=1)
    pts_ext = np.concatenate([pts, final], axis=0)
    pw_ext = np.concatenate([pw, [0.0]])
    times = np.arange(N + 1, dtype=np.float64) * float(dt)

    hop = np.linalg.norm(np.diff(pts_ext[:, :2], axis=0), axis=1)
    n_off = 0
    if jump_power_off_dist is not None:
        mask = hop > float(jump_power_off_dist)
        n_off = int(np.count_nonzero(mask & (pw_ext[:-1] > 0)))
        pw_ext[:-1][mask] = 0.0

    T0 = np.asarray(T0, dtype=np.float32)
    if float(T0.max()) < 50.0:
        raise ValueError(
            "T0 looks normalized; pass physical Kelvin (denormalize first).")

    meta = dict(meta)
    meta.setdefault("dt", float(dt))
    meta.setdefault("n_intervals", int(N))
    meta["end_time"] = float(times[-1])
    meta["n_jump_power_off"] = n_off

    kwargs = dict(times=times, points_m=pts_ext, powers_W=pw_ext, T0=T0,
                  meta_json=json.dumps(meta))
    if alpha0 is not None:
        kwargs["alpha0"] = np.asarray(alpha0, dtype=np.float32)
    if speeds is not None:
        kwargs["speeds"] = np.asarray(speeds, dtype=np.float64).reshape(-1)
    np.savez_compressed(out_path, **kwargs)
    print(f"[cfd-export] {out_path}: {N} intervals x {dt:.3g}s"
          + (f", {n_off} interval(s) power-zeroed on jumps" if n_off else ""))
    return out_path


def save_fields_sequence_to_vtkhdf(fields, field_names, filename,
                                   x_range=(-5e-4, 5e-4), y_range=(-5e-4, 5e-4),
                                   z_range=(-2e-4, 1.2e-4), dt=5e-5):
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    n_steps, nx, ny, nz = fields[0].shape

    spacing = ((x_range[1] - x_range[0]) / nx,
               (y_range[1] - y_range[0]) / ny,
               (z_range[1] - z_range[0]) / nz)
    origin = (x_range[0], y_range[0], z_range[0])

    cell_specs = {name: np.float64 for name in field_names}
    cell_specs.update({"X": np.float64, "Y": np.float64, "Z": np.float64})
    point_specs = {name: np.float64 for name in field_names}

    x_c = np.linspace(x_range[0] + spacing[0] / 2, x_range[1] - spacing[0] / 2, nx)
    y_c = np.linspace(y_range[0] + spacing[1] / 2, y_range[1] - spacing[1] / 2, ny)
    z_c = np.linspace(z_range[0] + spacing[2] / 2, z_range[1] - spacing[2] / 2, nz)
    X, Y, Z = np.meshgrid(x_c, y_c, z_c, indexing="ij")

    with VTKHDFImageDataWriter(path=filename,
                               whole_extent=(0, nx, 0, ny, 0, nz),
                               origin=origin, spacing=spacing,
                               cell_data_specs=cell_specs,
                               point_data_specs=point_specs) as writer:
        for t in range(n_steps):
            cell_data = {name: f[t] for name, f in zip(field_names, fields)}
            cell_data.update({"X": X, "Y": Y, "Z": Z})
            point_data = {name: cell_to_point_data(f[t])
                          for name, f in zip(field_names, fields)}
            writer.add_timestep(time=t * dt, cell_data=cell_data,
                                point_data=point_data)
    print(f"Saved {n_steps} time steps to {filename}")
    return filename
