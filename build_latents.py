"""Encode CFD datasets into a latent archive with a frozen autoencoder."""
import argparse
import contextlib
import os

import h5py
import numpy as np
import torch
from tqdm import tqdm

from data.frames import normalize_temperature
from data.schedules import DT, laser_off_convention, parse_hatch_segments
from model.autoencoder import load_autoencoder
from utils.scan_path import interpolate_heat_source_positions_hatching

ENCODE_BATCH = 51


def control_params_from_laser(case, num_time_steps):
    """Per-interval controls [x, y, power]; control[k] drives frame k -> k+1."""
    points = np.asarray(case["laser/points"][:], dtype=np.float64)
    powers = np.asarray(case["laser/powers"][:], dtype=np.float64)
    n = num_time_steps
    if len(points) < n:
        raise ValueError(f"laser/points has {len(points)} entries < {n} frames")
    if len(powers) < n:
        powers = np.concatenate([powers, np.zeros(n - len(powers))])
    return np.concatenate([points[:n, :2], powers[:n, None]], axis=1)[:, None, :]


def control_params_from_waypoints(case, num_time_steps, dt, off_position="file"):
    """Per-interval controls of a constant-schedule case."""
    md = case["metadata"].attrs
    positions, powers = interpolate_heat_source_positions_hatching(
        case["waypoints"][:], md["scan_speed"], dt, num_time_steps,
        upsample_step=1, laser_power=md["laser_power"])
    if off_position == "interp":
        on = powers[:, 0] > 0
        positions = laser_off_convention(
            positions, on, np.where(on, float(md["scan_speed"]), 0.0),
            parse_hatch_segments(case["waypoints"][:]),
            int(round(float(md["laser_off_time"]) / DT)))
    return np.concatenate([positions, powers], axis=1)[:, None, :]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--autoencoder-config", required=True)
    parser.add_argument("--autoencoder", required=True)
    parser.add_argument("--sources", nargs="+", required=True,
                        help="CFD HDF5 archives, each as path or path:prefix")
    parser.add_argument("--out", required=True)
    parser.add_argument("--off-position", default="file", choices=["file", "interp"],
                        help="laser-off positions of waypoint cases: park (file) "
                             "or the laser/-archive convention (interp)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    sources = []
    for spec in args.sources:
        path, _, prefix = spec.partition(":")
        sources.append((path, prefix))

    model = load_autoencoder(args.autoencoder_config, args.autoencoder, args.device)
    print(f"Loaded autoencoder from {args.autoencoder}")

    if os.path.exists(args.out):
        raise SystemExit(f"{args.out} exists; refusing to overwrite")
    with h5py.File(args.out, "w") as f_out, contextlib.ExitStack() as stack:
        f_out.attrs["off_position"] = args.off_position
        in_files = [stack.enter_context(h5py.File(p, "r")) for p, _ in sources]

        work, seen = [], set()
        for (path, prefix), f_in in zip(sources, in_files):
            cases = sorted(k for k in f_in.keys() if k.startswith("case_"))
            mesh = f_in["global/mesh/mesh_resolution"][:]
            if "mesh_resolution" not in f_out:
                f_out.create_dataset("mesh_resolution", data=mesh)
            print(f"{path}: {len(cases)} cases | mesh {tuple(mesh)} | prefix '{prefix}'")
            for case_name in cases:
                out_name = prefix + case_name
                if out_name in seen:
                    raise ValueError(
                        f"Duplicate case '{out_name}' from {path}; give the "
                        f"source a unique prefix.")
                seen.add(out_name)
                work.append((f_in, case_name, out_name, *(int(v) for v in mesh)))

        for f_in, case_name, out_name, nx, ny, nz in tqdm(work, desc="Encoding"):
            case = f_in[case_name]
            n = case["fields/T"].shape[0]
            end_time = case["metadata"].attrs["end_time"]
            dt = end_time / (n - 1)

            if "laser" in case:
                ctrl = control_params_from_laser(case, n)
            else:
                ctrl = control_params_from_waypoints(case, n, dt, args.off_position)

            T = case["fields/T"][:].reshape((n, nz, nx, ny)).transpose(0, 3, 2, 1)
            alpha = case["fields/alpha.phase1"][:].reshape((n, nz, nx, ny)).transpose(0, 3, 2, 1)
            muwafa = case["fields/muwafa"][:].reshape((n, nz, nx, ny)).transpose(0, 3, 2, 1)

            inputs = torch.cat([
                torch.from_numpy(normalize_temperature(T)).unsqueeze(1).float(),
                torch.from_numpy(alpha).unsqueeze(1).float(),
            ], dim=1)

            with torch.no_grad():
                latent = torch.cat([
                    model.encode(inputs[i:i + ENCODE_BATCH].to(args.device)).cpu()
                    for i in range(0, n, ENCODE_BATCH)
                ]).numpy()

            f_out.create_dataset(f"latent/{out_name}", data=latent)
            f_out.create_dataset(f"muwafa/{out_name}", data=muwafa, compression="gzip")
            f_out.create_dataset(f"control_params/{out_name}", data=ctrl)
            for key, value in (("end_time", end_time),
                               ("scan_speed", case["metadata"].attrs["scan_speed"]),
                               ("laser_power", case["metadata"].attrs["laser_power"]),
                               ("num_time_steps", n)):
                f_out.create_dataset(f"metadata/{out_name}/{key}", data=value)

    print(f"Saved {len(work)} cases to {args.out}")


if __name__ == "__main__":
    main()
