"""Run MPC on a held-out case and compare it with the unoptimized schedule."""
import argparse
import json
import os
import random
import time

import h5py
import numpy as np
import torch

from control.mpc import MPCController
from data.frames import denormalize_temperature, sample_random_init_temp
from model.autoencoder import load_autoencoder
from model.transition import load_transition
from utils.export import export_cfd_control, save_fields_sequence_to_vtkhdf
from utils.metrics import (compute_max_relative_depth_series,
                           compute_meltpool_depth_map_series,
                           compute_meltpool_depth_series,
                           compute_support_thickness_map)
from utils.scan_path import (choose_perpendicular_hatch_angle,
                             generate_uniform_hatching_path)

X_RANGE = (-5e-4, 5e-4)
Y_RANGE = (-5e-4, 5e-4)
Z_RANGE = (-2e-4, 1.2e-4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--autoencoder-config", required=True)
    ap.add_argument("--autoencoder", required=True)
    ap.add_argument("--transition-config",
                    default="config/transition_finetune_n100.json")
    ap.add_argument("--transition",
                    default="checkpoints/M1e_n100/best_dit_stepper_20260923_130449.pth")
    ap.add_argument("--config", default="config/mpc.json")
    ap.add_argument("--data", required=True, help="held-out CFD HDF5 archive")
    ap.add_argument("--case", default="case_v1_p3_0")
    ap.add_argument("--out-dir", default="./opt_result")
    ap.add_argument("--hatch-spacing", type=float, default=1.0e-4)
    ap.add_argument("--laser-off-time", type=float, default=2.5e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--requantize", action=argparse.BooleanOptionalAction, default=True,
                    help="snap each predicted latent onto the FSQ code grid (default on; "
                         "--no-requantize keeps the continuous latent)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[mpc] device: {device}")

    with open(args.config) as f:
        cfg = json.load(f)

    ae = load_autoencoder(args.autoencoder_config, args.autoencoder, device)
    transition, t_cfg = load_transition(args.transition_config,
                                        args.transition, device)
    dt = cfg["dt"]

    with h5py.File(args.data, "r") as f:
        nx, ny, nz = (int(v) for v in f["global/mesh/mesh_resolution"][:])
        case = f[args.case]
        alpha0 = case["fields/alpha.phase1"][0, :].astype(np.float32) \
            .reshape(nz, ny, nx).transpose(2, 1, 0)
        muwafa0 = case["fields/muwafa"][0, :].astype(np.float32) \
            .reshape(nz, ny, nx).transpose(2, 1, 0)
    T0 = sample_random_init_temp(nx, ny, nz)
    print(f"[mpc] case {args.case}, mesh {nx}x{ny}x{nz}")

    z_center_idx = int(round((-Z_RANGE[0]) / ((Z_RANGE[1] - Z_RANGE[0]) / nz)))
    alpha_top = alpha0[:, :, max(0, z_center_idx - 1)]
    thickness_np = compute_support_thickness_map(
        torch.from_numpy(alpha0), z_min=Z_RANGE[0],
        dz=(Z_RANGE[1] - Z_RANGE[0]) / nz, z_substrate=0.0,
        threshold=0.5).numpy()
    hatch_angle = choose_perpendicular_hatch_angle(
        thickness_np, mask=(alpha_top > 0.5))
    waypoints = generate_uniform_hatching_path(
        alpha_top, x_range=X_RANGE, y_range=Y_RANGE,
        hatch_spacing=args.hatch_spacing, hatch_angle=hatch_angle,
        scan_speed=cfg["init_speed"], laser_off_time=args.laser_off_time)
    print(f"[mpc] hatching: {len(waypoints)} waypoint rows "
          f"(angle {np.rad2deg(hatch_angle):.0f} deg)")

    mpc = MPCController(
        ae=ae, transition=transition, waypoints=waypoints,
        initial_frame=np.stack([T0, alpha0, muwafa0]),
        objective=cfg["objective"],
        dt=dt, past_horizon=t_cfg["data"]["past_horizon"],
        init_speed=cfg["init_speed"], init_power=cfg["init_power"],
        speed_range=tuple(cfg["speed_range"]),
        power_range=tuple(cfg["power_range"]),
        idle_time=args.laser_off_time, t_norm_steps=cfg["t_norm_steps"],
        nfe=cfg["nfe"], horizon_steps=cfg["horizon_steps"],
        exec_stride=cfg["exec_stride"], inner_iters=cfg["inner_iters"],
        inner_lr=cfg["inner_lr"], grad_clip_norm=cfg["grad_clip_norm"],
        plan_detach_interval=cfg["plan_detach_interval"],
        requantize=args.requantize, device=device)

    os.makedirs(args.out_dir, exist_ok=True)
    prefix = f"{args.out_dir}/{args.case}"

    def masked_metrics(out):
        mask = mpc.active_mask()[:out["muwafa"].shape[0]]
        with torch.no_grad():
            depth = compute_meltpool_depth_series(
                out["muwafa"][mask], out["alpha"][mask], z_min=mpc.z_min,
                dz=mpc.dz, z_substrate=mpc.z_substrate)
            depth_map = compute_meltpool_depth_map_series(
                out["muwafa"][mask], z_min=mpc.z_min, dz=mpc.dz,
                z_substrate=mpc.z_substrate)
            rel = compute_max_relative_depth_series(depth_map, mpc.thickness_map)
        idx = torch.where(mask)[0].cpu().numpy()
        return depth.cpu().numpy() * 1e6, rel.cpu().numpy(), idx * dt * 1e3

    unopt, unopt_ctrl = mpc.open_loop_rollout()
    depth_u, rel_u, t_u = masked_metrics(unopt)
    _, speed_u, power_u = mpc.open_loop_speed_power_profile()
    save_fields_sequence_to_vtkhdf(
        [unopt[k].cpu().numpy() for k in ("temperature", "alpha", "muwafa")],
        ["Temperature", "Alpha", "Muwafa"], f"{prefix}_unoptimized.vtkhdf",
        dt=dt, x_range=X_RANGE, y_range=Y_RANGE, z_range=Z_RANGE)

    t0 = time.perf_counter()
    result = mpc.run()
    print(f"[mpc] total wall time {time.perf_counter() - t0:.2f}s, "
          f"{len(mpc.plan_times)} replans, "
          f"mean plan time {np.mean(mpc.plan_times):.3f}s")
    depth_m, rel_m, t_m = masked_metrics(result)
    _, speed_m, power_m = mpc.committed_speed_power_profile()
    save_fields_sequence_to_vtkhdf(
        [result[k].cpu().numpy() for k in ("temperature", "alpha", "muwafa")],
        ["Temperature", "Alpha", "Muwafa"], f"{prefix}_mpc.vtkhdf",
        dt=dt, x_range=X_RANGE, y_range=Y_RANGE, z_range=Z_RANGE)

    T0_phys = denormalize_temperature(T0)
    meta = {"case": args.case, "h5_file": args.data, "seed": args.seed,
            "autoencoder": args.autoencoder, "transition": args.transition}
    unopt_np = unopt_ctrl.cpu().numpy()
    export_cfd_control(f"{prefix}_unopt_cfd_control.npz",
                       points_m=unopt_np[:, :2] * 1e-3,
                       powers_W=unopt_np[:, 2] * 1e3, dt=dt, T0=T0_phys,
                       alpha0=alpha0, speeds=speed_u,
                       meta={**meta, "controller": "unoptimized"})
    mpc_np = mpc.committed_control_array
    export_cfd_control(f"{prefix}_mpc_cfd_control.npz",
                       points_m=mpc_np[:, :2] * 1e-3,
                       powers_W=mpc_np[:, 2] * 1e3, dt=dt, T0=T0_phys,
                       alpha0=alpha0, speeds=speed_m,
                       meta={**meta, "controller": "mpc"})

    np.savez(f"{prefix}_results_mpc.npz",
             time_active_ms_unopt=t_u, depth_um_unopt=depth_u,
             max_rel_depth_unopt=rel_u, unopt_speed=speed_u, unopt_power=power_u,
             time_active_ms_opt=t_m, depth_um_opt=depth_m,
             max_rel_depth_opt=rel_m, opt_speed=speed_m, opt_power=power_m,
             plan_times=np.array(mpc.plan_times),
             loss_history=np.array(mpc.loss_history),
             thickness_map_um=thickness_np * 1e6)
    print(f"[mpc] done — outputs in {args.out_dir}/")


if __name__ == "__main__":
    main()
