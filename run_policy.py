"""Deploy the policy on a held-out case and compare it with a constant schedule."""
import argparse
import math
import os
import random
import time

import h5py
import numpy as np
import torch

from control.plant import THICKNESS_OBS_SCALE, load_plant_config
from control.networks import Actor
from data.frames import denormalize_temperature, sample_random_init_temp
from model.autoencoder import load_autoencoder
from model.transition import fm_sample, load_transition
from utils.export import export_cfd_control, save_fields_sequence_to_vtkhdf
from utils.metrics import (compute_max_relative_depth_series,
                           compute_meltpool_depth_map_series,
                           compute_meltpool_depth_series,
                           compute_support_thickness_map)
from utils.scan_path import (choose_perpendicular_hatch_angle,
                             generate_uniform_hatching_path, parse_scan_tracks)

T_CLAMP = 0.8149


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actor", required=True)
    ap.add_argument("--autoencoder-config", required=True)
    ap.add_argument("--autoencoder", required=True)
    ap.add_argument("--transition-config",
                    default="config/transition_finetune_n100.json")
    ap.add_argument("--transition",
                    default="checkpoints/M1e_n100/best_dit_stepper_20260923_130449.pth")
    ap.add_argument("--env-config", default="config/policy_env.json")
    ap.add_argument("--data", required=True, help="held-out CFD HDF5 archive")
    ap.add_argument("--case", default="case_v4_p4_0")
    ap.add_argument("--out-dir", default="./opt_result")
    ap.add_argument("--hatch-spacing", type=float, default=1.0e-4)
    ap.add_argument("--baseline-speed", type=float, default=1.8)
    ap.add_argument("--baseline-power", type=float, default=250.0)
    ap.add_argument("--laser-off-time", type=float, default=2.5e-4)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--requantize", action=argparse.BooleanOptionalAction, default=True,
                    help="snap each predicted latent onto the FSQ code grid (default on; "
                         "--no-requantize keeps the continuous latent)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[deploy] device: {device}")

    ae = load_autoencoder(args.autoencoder_config, args.autoencoder, device)
    transition, t_cfg = load_transition(args.transition_config,
                                        args.transition, device)
    past_horizon = t_cfg["data"]["past_horizon"]
    cfg = load_plant_config(args.env_config)
    z_dim = t_cfg["model"]["z_dim"]
    dt = cfg.dt
    V_lo, V_hi = cfg.speed_range
    P_lo, P_hi = cfg.power_range

    with h5py.File(args.data, "r") as f:
        nx, ny, nz = (int(v) for v in f["global/mesh/mesh_resolution"][:])
        case = f[args.case]
        alpha0 = case["fields/alpha.phase1"][0, :].astype(np.float32) \
            .reshape(nz, ny, nx).transpose(2, 1, 0)
    T0 = sample_random_init_temp(nx, ny, nz)
    print(f"[deploy] case {args.case}, mesh {nx}x{ny}x{nz}")

    z_center_idx = int(round((-cfg.z_min) / ((1.2e-4 - cfg.z_min) / nz)))
    substrate_top_idx = max(0, z_center_idx - 1)
    alpha_top = alpha0[:, :, substrate_top_idx]
    x_range = (cfg.x_min, cfg.x_min + cfg.nx * cfg.dx)
    y_range = (cfg.y_min, cfg.y_min + cfg.ny * cfg.dx)

    thickness = compute_support_thickness_map(
        torch.from_numpy(alpha0).to(device), z_min=cfg.z_min, dz=cfg.dz,
        z_substrate=cfg.z_substrate, threshold=0.5).detach()
    hatch_angle = choose_perpendicular_hatch_angle(
        thickness.cpu().numpy(), mask=(alpha_top > 0.5))
    waypoints = generate_uniform_hatching_path(
        alpha_top, x_range=x_range, y_range=y_range,
        hatch_spacing=args.hatch_spacing, hatch_angle=hatch_angle,
        scan_speed=args.baseline_speed, laser_off_time=args.laser_off_time)
    segs = parse_scan_tracks(waypoints)
    if len(segs) == 0:
        raise RuntimeError("no scan segments produced from hatching")
    print(f"[deploy] {len(segs)} hatching segments "
          f"(angle {np.rad2deg(hatch_angle):.0f} deg)")

    polyline = np.asarray([p for s in segs for p in (s[0], s[1])],
                          dtype=np.float32)
    cum_dist = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(polyline[1:] - polyline[:-1], axis=1))]
    ).astype(np.float32)
    seg_starts = [np.asarray(s[0], dtype=np.float32) for s in segs]
    seg_ends = [np.asarray(s[1], dtype=np.float32) for s in segs]
    seg_lens = [float(np.linalg.norm(e - s)) for s, e in zip(seg_starts, seg_ends)]
    seg_dirs = [(e - s) / max(L, 1e-12)
                for s, e, L in zip(seg_starts, seg_ends, seg_lens)]
    seg_start_arc = [float(cum_dist[2 * i]) for i in range(len(segs))]
    scan_len = float(sum(seg_lens))
    idle_steps = int(args.laser_off_time // dt)

    def path_pos(s):
        s = float(np.clip(s, 0.0, cum_dist[-1]))
        k = max(0, min(int(np.searchsorted(cum_dist, s, side="right")) - 1,
                       len(cum_dist) - 2))
        seg_len = float(cum_dist[k + 1] - cum_dist[k])
        a = (s - float(cum_dist[k])) / seg_len if seg_len > 0 else 0.0
        return polyline[k] + a * (polyline[k + 1] - polyline[k])

    def thickness_at(xy):
        ix = int(np.clip(round((xy[0] - cfg.x_min) / cfg.dx), 0, cfg.nx - 1))
        iy = int(np.clip(round((xy[1] - cfg.y_min) / cfg.dx), 0, cfg.ny - 1))
        return float(thickness[ix, iy].item())

    def thickness_patch(xy):
        p = cfg.thickness_patch_size
        ix = int(round((float(xy[0]) - cfg.x_min) / cfg.dx))
        iy = int(round((float(xy[1]) - cfg.y_min) / cfg.dx))
        half = p // 2
        x0, y0 = ix - half, iy - half
        sx0, sx1 = max(x0, 0), min(x0 + p, cfg.nx)
        sy0, sy1 = max(y0, 0), min(y0 + p, cfg.ny)
        out = np.zeros((p, p), dtype=np.float32)
        if sx0 < sx1 and sy0 < sy1:
            out[sx0 - x0: sx1 - x0, sy0 - y0: sy1 - y0] = \
                thickness[sx0:sx1, sy0:sy1].cpu().numpy()
        return out / THICKNESS_OBS_SCALE

    def build_observation(latent_stack, laser_xy, scan_dir, prev_action,
                          steps, max_steps, arc_s):
        step_d = cfg.future_lookahead_speed * dt
        future = np.array([thickness_at(path_pos(arc_s + (i + 1) * step_d))
                           for i in range(cfg.future_lookahead_steps)],
                          dtype=np.float32) / THICKNESS_OBS_SCALE
        scalars = np.concatenate([
            np.asarray(laser_xy, dtype=np.float32) * 1e3,
            np.asarray(scan_dir, dtype=np.float32),
            np.asarray(prev_action, dtype=np.float32),
            np.array([steps / max(max_steps, 1)], dtype=np.float32),
            future,
        ]).astype(np.float32)
        return {
            "latent": latent_stack.detach(),
            "scalars": torch.from_numpy(scalars).to(device).unsqueeze(0),
            "thickness_patch": torch.from_numpy(
                thickness_patch(laser_xy)).to(device).unsqueeze(0),
        }

    latent_shape = (past_horizon * z_dim, 25, 25, 8)
    scalar_dim = 7 + cfg.future_lookahead_steps
    actor = Actor(latent_shape, scalar_dim, cfg.thickness_patch_size).to(device)
    actor.load_state_dict(torch.load(args.actor, map_location=device))
    actor.eval()
    print(f"[deploy] loaded actor {args.actor}")

    def policy_picker(obs):
        with torch.no_grad():
            raw = actor(obs)
            return np.array([2.0 * torch.sigmoid(raw[0]).item() - 1.0,
                             2.0 * torch.sigmoid(raw[1]).item() - 1.0],
                            dtype=np.float32)

    a_v = float(np.clip(2.0 * (args.baseline_speed - V_lo) / (V_hi - V_lo) - 1.0,
                        -1.0, 1.0))
    a_p = float(np.clip(2.0 * (args.baseline_power - P_lo) / (P_hi - P_lo) - 1.0,
                        -1.0, 1.0))
    baseline_picker = lambda obs: np.array([a_v, a_p], dtype=np.float32)

    frame = torch.from_numpy(np.stack([T0, alpha0])).float().unsqueeze(0).to(device)
    with torch.no_grad():
        init_latent = ae.encode(frame)

    def rollout(action_picker, name, max_steps):
        latent_stack = init_latent.repeat(1, past_horizon, 1, 1, 1).contiguous()
        latent_traj = [latent_stack[:, -z_dim:].detach().clone()]
        seg_idx, seg_s, idle_remaining = 0, 0.0, 0
        done = False
        laser_xy = seg_starts[0].copy()
        scan_dir = seg_dirs[0].copy()
        arc_s = seg_start_arc[0]
        prev_action = np.array([0.0, 0.0], dtype=np.float32)
        speeds, powers, xys, active = [], [], [], []
        t_policy, t_transition = [], []
        t_total = time.perf_counter()
        steps = 0
        while steps < max_steps and not done:
            if idle_remaining > 0:
                v, P = 0.0, 0.0
                active.append(False)
            else:
                obs = build_observation(latent_stack, laser_xy, scan_dir,
                                        prev_action, steps, max_steps, arc_s)
                t0 = time.perf_counter()
                action = action_picker(obs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_policy.append(time.perf_counter() - t0)
                action = np.clip(action, -1.0, 1.0)
                v = V_lo + 0.5 * (action[0] + 1.0) * (V_hi - V_lo)
                P = P_lo + 0.5 * (action[1] + 1.0) * (P_hi - P_lo)
                prev_action = action
                active.append(True)

            cond = torch.tensor([[laser_xy[0] * 1e3, laser_xy[1] * 1e3, P * 1e-3,
                                  min((steps + 1) / cfg.t_norm_steps, 1.0)]],
                                dtype=torch.float32, device=device)
            t0 = time.perf_counter()
            with torch.no_grad():
                g = torch.Generator(device=device)
                g.manual_seed((args.seed * 1_000_003 + steps) % (2 ** 63 - 1))
                last = latent_stack[:, -z_dim:]
                x0 = torch.randn(last.shape, generator=g, device=device)
                pred = fm_sample(transition, latent_stack, last, cond,
                                 nfe=cfg.nfe, noise=x0,
                                 requantize=ae.requantize if args.requantize else None)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_transition.append(time.perf_counter() - t0)

            latent_stack = torch.cat([latent_stack, pred], dim=1)[:, z_dim:]
            latent_traj.append(pred.detach().clone())
            speeds.append(v)
            powers.append(P)
            xys.append(laser_xy.copy())
            steps += 1

            if idle_remaining > 0:
                idle_remaining -= 1
                if idle_remaining == 0:
                    seg_idx += 1
                    seg_s = 0.0
                    laser_xy = seg_starts[seg_idx].copy()
                    scan_dir = seg_dirs[seg_idx].copy()
                    arc_s = seg_start_arc[seg_idx]
            else:
                seg_s = min(seg_s + v * dt, seg_lens[seg_idx])
                laser_xy = (seg_starts[seg_idx] + seg_s * seg_dirs[seg_idx]).astype(np.float32)
                arc_s = seg_start_arc[seg_idx] + seg_s
                if seg_s >= seg_lens[seg_idx] - 1e-12:
                    if seg_idx < len(segs) - 1:
                        idle_remaining = idle_steps
                        if idle_remaining == 0:
                            seg_idx += 1
                            seg_s = 0.0
                            laser_xy = seg_starts[seg_idx].copy()
                            scan_dir = seg_dirs[seg_idx].copy()
                            arc_s = seg_start_arc[seg_idx]
                    else:
                        done = True
        wall = time.perf_counter() - t_total
        active = np.array(active, dtype=bool)
        print(f"[deploy] {name}: {steps} steps ({int(active.sum())} active) "
              f"in {wall:.2f}s "
              f"[policy {1e3 * np.mean(t_policy):.2f} ms, "
              f"transition {1e3 * np.mean(t_transition):.2f} ms per step]")
        return {"latent_trajectory": torch.cat(latent_traj, dim=0),
                "speeds": np.array(speeds, dtype=np.float32),
                "powers": np.array(powers, dtype=np.float32),
                "xys": np.array(xys, dtype=np.float32),
                "active": active, "final_xy": laser_xy.copy(),
                "wall_s": wall,
                "policy_ms": float(np.mean(t_policy) * 1e3),
                "transition_ms": float(np.mean(t_transition) * 1e3)}

    max_steps = args.max_steps or int(math.ceil(scan_len / (V_lo * dt))
                                      + (len(segs) - 1) * idle_steps + 1)
    baseline = rollout(baseline_picker,
                       f"baseline v={args.baseline_speed} P={args.baseline_power}",
                       max_steps)
    policy = rollout(policy_picker, "policy", max_steps)

    def decode_traj(latent_traj, batch=16):
        fields = [[], [], []]
        with torch.no_grad():
            for i in range(0, latent_traj.shape[0], batch):
                rec = ae.decode(latent_traj[i:i + batch])
                rec = torch.cat([rec[:, 0:1].clamp(max=T_CLAMP), rec[:, 1:]], dim=1)
                for c in range(3):
                    fields[c].append(rec[:, c].cpu())
        return [torch.cat(f, 0) for f in fields]

    T_bl, A_bl, M_bl = decode_traj(baseline["latent_trajectory"])
    T_po, A_po, M_po = decode_traj(policy["latent_trajectory"])

    def metrics_of(M, A, active):
        with torch.no_grad():
            depth = compute_meltpool_depth_series(
                M[1:].to(device), A[1:].to(device), z_min=cfg.z_min, dz=cfg.dz,
                z_substrate=cfg.z_substrate, threshold=0.5, sharpness=20.0)
            depth_map = compute_meltpool_depth_map_series(
                M[1:].to(device), z_min=cfg.z_min, dz=cfg.dz,
                z_substrate=cfg.z_substrate, threshold=0.5, sharpness=20.0)
            rel = compute_max_relative_depth_series(depth_map, thickness)
        idx = np.where(active)[0]
        return (depth.cpu().numpy()[idx] * 1e6, rel.cpu().numpy()[idx],
                (idx + 1) * dt * 1e3)

    depth_bl, rel_bl, t_bl = metrics_of(M_bl, A_bl, baseline["active"])
    depth_po, rel_po, t_po = metrics_of(M_po, A_po, policy["active"])

    os.makedirs(args.out_dir, exist_ok=True)
    prefix = f"{args.out_dir}/{args.case}"

    for (T_, A_, M_), tag in (((T_bl, A_bl, M_bl), "baseline"),
                              ((T_po, A_po, M_po), "policy")):
        save_fields_sequence_to_vtkhdf(
            [T_.numpy(), A_.numpy(), M_.numpy()],
            ["Temperature", "Alpha", "Muwafa"],
            f"{prefix}_{tag}.vtkhdf", dt=dt,
            x_range=x_range, y_range=y_range, z_range=(cfg.z_min, 1.2e-4))

    T0_phys = denormalize_temperature(T0)
    meta = {"case": args.case, "h5_file": args.data, "seed": args.seed,
            "requantize": args.requantize,
            "actor": args.actor, "autoencoder": args.autoencoder,
            "transition": args.transition}
    for res, tag in ((baseline, "baseline"), (policy, "policy")):
        export_cfd_control(
            f"{prefix}_{tag}_cfd_control.npz",
            points_m=res["xys"], powers_W=res["powers"], dt=dt,
            T0=T0_phys, alpha0=alpha0, speeds=res["speeds"],
            final_point_m=res["final_xy"],
            meta={**meta, "controller": tag})

    np.savez(f"{prefix}_results_policy.npz",
             t_ms_baseline=t_bl, depth_baseline_um=depth_bl, rel_baseline=rel_bl,
             speed_baseline=baseline["speeds"], power_baseline=baseline["powers"],
             active_baseline=baseline["active"],
             t_ms_policy=t_po, depth_policy_um=depth_po, rel_policy=rel_po,
             speed_policy=policy["speeds"], power_policy=policy["powers"],
             active_policy=policy["active"],
             thickness_map_um=thickness.cpu().numpy() * 1e6,
             policy_ms_step=policy["policy_ms"],
             transition_ms_step=policy["transition_ms"])
    print(f"[deploy] done — outputs in {args.out_dir}/")


if __name__ == "__main__":
    main()
