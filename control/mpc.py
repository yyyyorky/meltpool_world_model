"""Gradient-based receding-horizon control through the frozen world model."""
import math
import time

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from utils.metrics import (compute_meltpool_depth_series,
                           compute_support_thickness_map)
from utils.scan_path import parse_scan_tracks

T_CLAMP = 0.8149


class MPCController:

    def __init__(self, ae, transition, waypoints, initial_frame, objective,
                 dt=5e-5, past_horizon=5, init_speed=1.8, init_power=250.0,
                 speed_range=(0.8, 2.4), power_range=(80.0, 300.0),
                 idle_time=2.5e-4, t_norm_steps=81, nfe=2,
                 horizon_steps=10, exec_stride=2, inner_iters=10,
                 inner_lr=0.5, grad_clip_norm=1.0,
                 plan_detach_interval=0, noise_seed=12345,
                 requantize=False, device=None, verbose=True):
        self.device = device or next(ae.parameters()).device
        self.ae = ae
        self.transition = transition
        for p in ae.parameters():
            p.requires_grad_(False)
        for p in transition.parameters():
            p.requires_grad_(False)

        self.dt = dt
        self.past_horizon = past_horizon
        self.speed_min, self.speed_max = speed_range
        self.power_min, self.power_max = power_range
        self.idle_step = idle_time // dt
        self.t_norm_steps = float(t_norm_steps)
        self.nfe = int(nfe)
        self.requantize = bool(requantize)
        self._noise_seed = int(noise_seed)
        self._noise_cache = {}

        self.horizon_steps = int(horizon_steps)
        self.exec_stride = int(exec_stride)
        assert 0 < self.exec_stride <= self.horizon_steps
        self.inner_iters = int(inner_iters)
        self.inner_lr = float(inner_lr)
        self.grad_clip_norm = grad_clip_norm
        self.plan_detach_interval = int(plan_detach_interval)
        self.verbose = verbose

        self.d_fuse = objective["d_fuse"]
        self.d_bond = objective["d_bond"]
        self.safe_rel = objective["safe_rel"]
        self.rel_hard = objective["rel_hard"]
        self.sigma_lo = objective["sigma_lo"]
        self.sigma_hi = objective["sigma_hi"]
        self.sigma_blend = objective["sigma_blend"]
        self.w_band = objective["w_band"]
        self.w_depth_var = objective["w_depth_var"]
        self.target_depth = objective["target_depth"]
        self.anticipation_aggregator = objective["anticipation_aggregator"]
        self.future_lookahead_steps = int(objective["future_lookahead_steps"])
        self.future_lookahead_speed = objective["future_lookahead_speed"]

        self.dx = 1e-5
        self.z_min = -2e-4
        self.dz = 1e-5
        self.z_substrate = 0.0
        self.x_min = -5e-4
        self.y_min = -5e-4

        self.raw_segments = parse_scan_tracks(waypoints)
        self.num_segments = self.raw_segments.shape[0]
        self.seg_lengths = [
            float(np.linalg.norm(self.raw_segments[i, 1] - self.raw_segments[i, 0]))
            for i in range(self.num_segments)]

        inv_sig = lambda v, lo, hi: math.log((v - lo) / (hi - v + 1e-12) + 1e-12)
        self._init_raw_speed = inv_sig(
            float(np.clip(init_speed, self.speed_min + 1e-6, self.speed_max - 1e-6)),
            self.speed_min, self.speed_max)
        self._init_raw_power = inv_sig(
            float(np.clip(init_power, self.power_min + 1e-6, self.power_max - 1e-6)),
            self.power_min, self.power_max)

        frame_t = torch.from_numpy(initial_frame).float().to(self.device)
        with torch.no_grad():
            self.init_latent = ae.encode(frame_t[:2].unsqueeze(0))
            self.thickness_map = compute_support_thickness_map(
                frame_t[1], z_min=self.z_min, dz=self.dz,
                z_substrate=self.z_substrate, threshold=0.5)
        self.latent_C = self.init_latent.shape[1]
        self._tk_nx, self._tk_ny = self.thickness_map.shape

        self._build_global_path()
        self._reset_trajectory()

    def _reset_trajectory(self):
        self.raw_speed = []
        self.raw_power = []
        self._step_meta = []
        self._seg_active_steps = [[] for _ in range(self.num_segments)]
        self._cur_seg = 0
        self._cur_seg_cum = 0.0
        self._idle_remaining = 0
        self._done = (self.num_segments == 0)
        if not self._done:
            self._append_step(0, "active", 0,
                              self._init_raw_speed, self._init_raw_power)

    def _append_step(self, seg_idx, kind, intra_offset, raw_spd, raw_pwr):
        trainable = (kind == "active")
        self.raw_speed.append(torch.tensor(raw_spd, dtype=torch.float32,
                                           device=self.device,
                                           requires_grad=trainable))
        self.raw_power.append(torch.tensor(raw_pwr, dtype=torch.float32,
                                           device=self.device,
                                           requires_grad=trainable))
        t = len(self.raw_speed) - 1
        self._step_meta.append({"seg_idx": seg_idx, "kind": kind,
                                "intra_offset": intra_offset})
        if kind == "active":
            self._seg_active_steps[seg_idx].append(t)
            spd = (self.speed_min + (self.speed_max - self.speed_min)
                   / (1.0 + math.exp(-raw_spd)))
            remaining = self.seg_lengths[seg_idx] - self._cur_seg_cum
            if remaining <= spd * self.dt + 1e-12:
                self._cur_seg_cum = self.seg_lengths[seg_idx]
                if seg_idx < self.num_segments - 1:
                    self._idle_remaining = int(self.idle_step)
                else:
                    self._done = True
            else:
                self._cur_seg_cum += spd * self.dt
        else:
            self._idle_remaining -= 1
            if self._idle_remaining == 0:
                self._cur_seg += 1
                if self._cur_seg >= self.num_segments:
                    self._done = True
                else:
                    self._cur_seg_cum = 0.0

    def _extend_one(self):
        if self._done:
            return False
        spd = float(self.raw_speed[-1].detach().item())
        pwr = float(self.raw_power[-1].detach().item())
        if self._idle_remaining > 0:
            intra = int(self.idle_step) - self._idle_remaining
            self._append_step(self._cur_seg, "idle", intra, spd, pwr)
        else:
            intra = len(self._seg_active_steps[self._cur_seg])
            self._append_step(self._cur_seg, "active", intra, spd, pwr)
        return True

    def _extend_to_length(self, target_len):
        while len(self.raw_speed) < target_len and not self._done:
            self._extend_one()

    def _extend_full(self):
        while not self._done:
            self._extend_one()

    @property
    def total_steps(self):
        return len(self.raw_speed)

    def get_speed(self, seg_idx):
        """Bounded per-step speeds of one segment."""
        idxs = self._seg_active_steps[seg_idx]
        if not idxs:
            return torch.zeros(0, device=self.device)
        raws = torch.stack([self.raw_speed[t] for t in idxs])
        return self.speed_min + (self.speed_max - self.speed_min) * torch.sigmoid(raws)

    def _build_global_path(self):
        """Scan polyline with cumulative arc length."""
        pts, seg_start_idx = [], []
        for i in range(self.num_segments):
            seg_start_idx.append(len(pts))
            pts.append(np.asarray(self.raw_segments[i, 0], dtype=np.float64))
            pts.append(np.asarray(self.raw_segments[i, 1], dtype=np.float64))
        if not pts:
            self._path_pts = np.zeros((0, 2))
            self._path_cum = np.zeros((0,))
            self._seg_start_arc = []
            return
        pts = np.stack(pts, axis=0)
        d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        self._path_pts = pts
        self._path_cum = np.concatenate([[0.0], np.cumsum(d)])
        self._seg_start_arc = [float(self._path_cum[i]) for i in seg_start_idx]

    def _path_pos(self, s):
        cum = self._path_cum
        s = float(np.clip(s, 0.0, cum[-1]))
        j = max(0, min(int(np.searchsorted(cum, s, side="right")) - 1,
                       len(cum) - 2))
        seg_len = cum[j + 1] - cum[j]
        r = 0.0 if seg_len <= 1e-12 else (s - cum[j]) / seg_len
        return self._path_pts[j] + r * (self._path_pts[j + 1] - self._path_pts[j])

    def _thickness_at_xy(self, x_m, y_m):
        ix = int(np.clip(round((x_m - self.x_min) / self.dx), 0, self._tk_nx - 1))
        iy = int(np.clip(round((y_m - self.y_min) / self.dx), 0, self._tk_ny - 1))
        return float(self.thickness_map[ix, iy].item())

    def _global_arc_s_of_step(self, t):
        meta = self._step_meta[t]
        seg_idx = meta["seg_idx"]
        base = self._seg_start_arc[seg_idx]
        L = self.seg_lengths[seg_idx]
        if meta["kind"] != "active":
            return base + L
        spd = self.get_speed(seg_idx).detach()
        if spd.numel() == 0:
            return base
        cum = torch.cat([torch.zeros(1, device=spd.device, dtype=spd.dtype),
                         torch.cumsum(spd[:-1] * self.dt, dim=0)])
        j = min(meta["intra_offset"], cum.shape[0] - 1)
        return base + min(float(cum[j].item()), L)

    def _future_thickness(self, arc_s):
        step = self.future_lookahead_speed * self.dt
        return np.array([
            self._thickness_at_xy(*self._path_pos(arc_s + (i + 1) * step))
            for i in range(self.future_lookahead_steps)], dtype=np.float32)

    def _step_noise(self, idx):
        """Fixed per-step noise, so rollouts are reproducible."""
        idx = int(idx)
        if idx not in self._noise_cache:
            g = torch.Generator(device=self.device)
            g.manual_seed(self._noise_seed + idx)
            shape = (1, self.latent_C, *self.init_latent.shape[2:])
            self._noise_cache[idx] = torch.randn(shape, generator=g,
                                                 device=self.device)
        return self._noise_cache[idx]

    def _predict_latent(self, latent_stack, cond, x0):
        last = latent_stack[:, -self.latent_C:]
        x = x0
        dt = 1.0 / self.nfe
        for k in range(self.nfe):
            t = torch.full((x.shape[0],), k * dt, device=x.device)
            x = x + dt * self.transition(torch.cat([x, latent_stack], dim=1),
                                         cond, t)
        z = last + x
        return self.ae.requantize(z) if self.requantize else z

    def _rollout_from(self, latent_stack, control_slice, use_checkpoint,
                      detach_interval, t_start):
        """Returns (decoded fields, final latent stack)."""
        def _step_fn(ls_in, ctrl_i, x0_i):
            pred_latent = self._predict_latent(ls_in, ctrl_i.unsqueeze(0), x0_i)
            recon = self.ae.decode(pred_latent)
            recon = torch.cat([recon[:, 0:1].clamp(max=T_CLAMP), recon[:, 1:]],
                              dim=1)
            return pred_latent, recon

        fields = {"temperature": [], "alpha": [], "muwafa": []}
        cur = latent_stack
        for i in range(control_slice.shape[0]):
            if detach_interval > 0 and i > 0 and i % detach_interval == 0:
                cur = cur.detach()
            x0 = self._step_noise(t_start + i)
            if use_checkpoint and cur.requires_grad:
                pred_latent, recon = checkpoint(_step_fn, cur, control_slice[i],
                                                x0, use_reentrant=False)
            else:
                pred_latent, recon = _step_fn(cur, control_slice[i], x0)
            fields["temperature"].append(recon[0, 0])
            fields["alpha"].append(recon[0, 1])
            fields["muwafa"].append(recon[0, 2])
            cur = torch.cat([cur, pred_latent], dim=1)[:, self.latent_C:]
        return {k: torch.stack(v, dim=0) for k, v in fields.items()}, cur

    def open_loop_rollout(self):
        """Rollout at the initial speed and power."""
        self._extend_full()
        ctrl = self._build_window_controls(0, self.total_steps)
        stack = self.init_latent.repeat(1, self.past_horizon, 1, 1, 1)
        with torch.no_grad():
            out, _ = self._rollout_from(stack, ctrl.detach(),
                                        use_checkpoint=False,
                                        detach_interval=0, t_start=0)
        return out, ctrl.detach()

    def _window_active_mask(self, t_start, n):
        end = min(t_start + n, self.total_steps)
        mask = torch.zeros(end - t_start, dtype=torch.bool, device=self.device)
        for k in range(end - t_start):
            if self._step_meta[t_start + k]["kind"] == "active":
                mask[k] = True
        return mask

    def _build_window_controls(self, t_start, n):
        """Rows [x_mm, y_mm, power_kW, t_norm], differentiable in speed and power."""
        end = min(t_start + n, self.total_steps)
        rows, seg_cache = [], {}
        for t in range(t_start, end):
            meta = self._step_meta[t]
            seg_idx = meta["seg_idx"]
            if seg_idx not in seg_cache:
                seg_start = torch.tensor(self.raw_segments[seg_idx, 0],
                                         dtype=torch.float32, device=self.device)
                direction = torch.tensor(
                    self.raw_segments[seg_idx, 1] - self.raw_segments[seg_idx, 0],
                    dtype=torch.float32, device=self.device)
                L = torch.tensor(self.seg_lengths[seg_idx],
                                 dtype=torch.float32, device=self.device)
                spd = self.get_speed(seg_idx)
                cum = torch.cat([torch.zeros(1, device=self.device, dtype=spd.dtype),
                                 torch.cumsum(spd[:-1] * self.dt, dim=0)])
                ratio_end = ((cum[-1] + spd[-1] * self.dt) / L).clamp(0.0, 1.0)
                end_pos = (seg_start + ratio_end * direction).detach()
                seg_cache[seg_idx] = (seg_start, direction, L, cum, end_pos)
            seg_start, direction, L, cum, end_pos = seg_cache[seg_idx]

            t_norm = torch.tensor(min(t / self.t_norm_steps, 1.0),
                                  device=self.device, dtype=torch.float32)
            if meta["kind"] == "active":
                ratio = (cum[meta["intra_offset"]] / L).clamp(0.0, 1.0)
                pos = seg_start + ratio * direction
                pwr_kW = ((self.power_min + (self.power_max - self.power_min)
                           * torch.sigmoid(self.raw_power[t])) * 1e-3)
                rows.append(torch.stack([pos[0] * 1e3, pos[1] * 1e3, pwr_kW, t_norm]))
            else:
                rows.append(torch.stack([
                    end_pos[0] * 1e3, end_pos[1] * 1e3,
                    torch.tensor(0.0, device=self.device), t_norm]))
        if not rows:
            return torch.zeros(0, 4, device=self.device, dtype=torch.float32)
        return torch.stack(rows, dim=0)

    def _window_objective(self, rollout_out, t_start, ctrl_slice):
        muwafa, alpha = rollout_out["muwafa"], rollout_out["alpha"]
        wmask = self._window_active_mask(t_start, muwafa.shape[0])
        muwafa_a, alpha_a = muwafa[wmask], alpha[wmask]
        if muwafa_a.shape[0] < 1:
            return torch.tensor(0.0, device=self.device), {}

        depth = compute_meltpool_depth_series(
            muwafa_a, alpha_a, z_min=self.z_min, dz=self.dz,
            z_substrate=self.z_substrate, threshold=0.5, sharpness=20.0)

        local_thk, fut_agg = [], []
        for k in range(muwafa.shape[0]):
            t = t_start + k
            if t >= self.total_steps or self._step_meta[t]["kind"] != "active":
                continue
            local_thk.append(self._thickness_at_xy(
                float(ctrl_slice[k, 0].item()) / 1e3,
                float(ctrl_slice[k, 1].item()) / 1e3))
            ft = self._future_thickness(self._global_arc_s_of_step(t))
            if ft.size == 0:
                fut_agg.append(0.0)
            elif self.anticipation_aggregator == "mean":
                fut_agg.append(float(ft.mean()))
            else:
                fut_agg.append(float(ft.min()))
        local_thk = torch.tensor(local_thk, device=self.device, dtype=depth.dtype)
        fut_agg = torch.tensor(fut_agg, device=self.device, dtype=depth.dtype)

        h_eff = torch.minimum(local_thk, fut_agg)
        d_tgt = torch.minimum(
            torch.full_like(depth, self.d_fuse),
            torch.minimum(self.rel_hard * local_thk,
                          torch.clamp(self.safe_rel * h_eff, min=self.d_bond)))
        s_eff = self.sigma_lo + (self.sigma_hi - self.sigma_lo) * torch.sigmoid(
            (depth - d_tgt) / max(self.sigma_blend, 1e-12))
        band = self.w_band * ((depth - d_tgt) / s_eff) ** 2

        if self._prev_committed_depth is None:
            first = depth[:1].detach()
        else:
            first = torch.tensor([self._prev_committed_depth],
                                 device=self.device, dtype=depth.dtype)
        depth_prev = torch.cat([first, depth[:-1]], dim=0)
        depth_var = self.w_depth_var * ((depth - depth_prev) / self.target_depth) ** 2

        loss = (band + depth_var).sum()
        return loss, {"band": float(band.mean().item()),
                      "depth_var": float(depth_var.mean().item())}

    def _live_parameters(self):
        params = []
        for t in range(self._t_global, len(self.raw_speed)):
            if self.raw_speed[t].requires_grad:
                params.append(self.raw_speed[t])
            if self.raw_power[t].requires_grad:
                params.append(self.raw_power[t])
        return params

    def _rebuild_uncommitted_schedule(self):
        """Rebuild the uncommitted schedule, warm-started from the last window."""
        if self.num_segments == 0:
            return
        tg = self._t_global

        ws_speed, ws_power = [], []
        for t in range(tg, len(self.raw_speed)):
            if self._step_meta[t]["kind"] == "active":
                ws_speed.append(float(self.raw_speed[t].detach().item()))
                ws_power.append(float(self.raw_power[t].detach().item()))
        ws_idx = 0

        if tg == 0:
            self._cur_seg, self._cur_seg_cum = 0, 0.0
            self._idle_remaining, self._done = 0, False
        else:
            last = self._step_meta[tg - 1]
            seg = last["seg_idx"]
            if last["kind"] == "idle":
                done_idle = 0
                for t in range(tg - 1, -1, -1):
                    m = self._step_meta[t]
                    if m["kind"] == "idle" and m["seg_idx"] == seg:
                        done_idle += 1
                    else:
                        break
                rem = int(self.idle_step) - done_idle
                if rem > 0:
                    self._cur_seg, self._cur_seg_cum = seg, self.seg_lengths[seg]
                    self._idle_remaining, self._done = rem, False
                else:
                    self._idle_remaining = 0
                    self._cur_seg, self._cur_seg_cum = seg + 1, 0.0
                    self._done = (self._cur_seg >= self.num_segments)
            else:
                covered = 0.0
                for t in range(tg):
                    m = self._step_meta[t]
                    if m["kind"] == "active" and m["seg_idx"] == seg:
                        covered += self._committed_speeds[t] * self.dt
                if self.seg_lengths[seg] - covered <= 1e-12:
                    self._cur_seg, self._cur_seg_cum = seg, self.seg_lengths[seg]
                    if seg < self.num_segments - 1:
                        self._idle_remaining, self._done = int(self.idle_step), False
                    else:
                        self._idle_remaining, self._done = 0, True
                else:
                    self._cur_seg, self._cur_seg_cum = seg, covered
                    self._idle_remaining, self._done = 0, False

        self.raw_speed = self.raw_speed[:tg]
        self.raw_power = self.raw_power[:tg]
        self._step_meta = self._step_meta[:tg]
        self._seg_active_steps = [[] for _ in range(self.num_segments)]
        for t in range(tg):
            m = self._step_meta[t]
            if m["kind"] == "active":
                self._seg_active_steps[m["seg_idx"]].append(t)

        target = tg + self.horizon_steps
        while len(self.raw_speed) < target and not self._done:
            if len(self.raw_speed) == 0:
                spd_carry, pwr_carry = self._init_raw_speed, self._init_raw_power
            else:
                spd_carry = float(self.raw_speed[-1].detach().item())
                pwr_carry = float(self.raw_power[-1].detach().item())
            if self._idle_remaining > 0:
                intra = int(self.idle_step) - self._idle_remaining
                self._append_step(self._cur_seg, "idle", intra, spd_carry, pwr_carry)
            else:
                intra = len(self._seg_active_steps[self._cur_seg])
                if ws_idx < len(ws_speed):
                    spd_raw, pwr_raw = ws_speed[ws_idx], ws_power[ws_idx]
                    ws_idx += 1
                else:
                    spd_raw, pwr_raw = spd_carry, pwr_carry
                self._append_step(self._cur_seg, "active", intra, spd_raw, pwr_raw)

    def _plan_window(self):
        self._rebuild_uncommitted_schedule()
        self._extend_to_length(self._t_global + self.horizon_steps)
        live = self._live_parameters()
        losses, last_loss_dict = [], {}
        if not live:
            return losses, last_loss_dict
        opt = torch.optim.Adam(live, lr=self.inner_lr)
        for _ in range(self.inner_iters):
            opt.zero_grad()
            ctrl = self._build_window_controls(self._t_global, self.horizon_steps)
            if ctrl.shape[0] == 0:
                break
            out, _ = self._rollout_from(self._latent_stack, ctrl,
                                        use_checkpoint=True,
                                        detach_interval=self.plan_detach_interval,
                                        t_start=self._t_global)
            loss, last_loss_dict = self._window_objective(out, self._t_global, ctrl)
            if loss.requires_grad:
                loss.backward()
                if self.grad_clip_norm:
                    torch.nn.utils.clip_grad_norm_(live, self.grad_clip_norm)
                opt.step()
            losses.append(loss.item())
        return losses, last_loss_dict

    def _execute(self, n_commit):
        self._extend_to_length(self._t_global + n_commit)
        ctrl = self._build_window_controls(self._t_global, n_commit)
        n = ctrl.shape[0]
        if n == 0:
            return 0
        with torch.no_grad():
            out, new_stack = self._rollout_from(
                self._latent_stack, ctrl.detach(), use_checkpoint=False,
                detach_interval=0, t_start=self._t_global)
            amask = self._window_active_mask(self._t_global, n)
            if amask.any():
                d = compute_meltpool_depth_series(
                    out["muwafa"][amask], out["alpha"][amask],
                    z_min=self.z_min, dz=self.dz, z_substrate=self.z_substrate,
                    threshold=0.5, sharpness=20.0)
                self._prev_committed_depth = float(d[-1].item())

        for k in range(n):
            t = self._t_global + k
            if self._step_meta[t]["kind"] == "active":
                spd = (self.speed_min + (self.speed_max - self.speed_min)
                       * torch.sigmoid(self.raw_speed[t])).detach().item()
            else:
                spd = 0.0
            self._committed_speeds.append(float(spd))
            self.raw_speed[t].requires_grad_(False)
            self.raw_power[t].requires_grad_(False)
            self._committed_controls.append(ctrl[k].detach().cpu().numpy())

        for key in self._committed_frames:
            self._committed_frames[key].append(out[key])
        self._latent_stack = new_stack.detach()
        self._t_global += n
        return n

    def run(self):
        self._reset_trajectory()
        self._latent_stack = self.init_latent.repeat(1, self.past_horizon, 1, 1, 1)
        self._t_global = 0
        self._prev_committed_depth = None
        self._committed_frames = {"temperature": [], "alpha": [], "muwafa": []}
        self._committed_controls = []
        self._committed_speeds = []
        self.plan_times = []
        self.loss_history = []

        iteration = 0
        while not (self._done and self._t_global >= self.total_steps):
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            t0 = time.perf_counter()
            losses, loss_dict = self._plan_window()
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self.plan_times.append(time.perf_counter() - t0)
            if losses:
                self.loss_history.append(losses[-1])

            n_commit = min(self.exec_stride, self.total_steps - self._t_global)
            if n_commit == 0:
                break
            actual = self._execute(n_commit)
            iteration += 1
            if self.verbose:
                done_mm = self.committed_path_length() * 1e3
                total_mm = sum(self.seg_lengths) * 1e3
                last = f"{losses[-1]:.4f}" if losses else "n/a"
                print(f"[iter {iteration:3d}] path={done_mm:6.3f}/{total_mm:6.3f} mm "
                      f"plan_time={self.plan_times[-1]:.2f}s "
                      f"last_loss={last} committed={actual} {loss_dict}")
            if actual == 0:
                break
        return self.result_fields

    def committed_path_length(self):
        seg_cum = [0.0] * self.num_segments
        for t, spd in enumerate(self._committed_speeds):
            meta = self._step_meta[t]
            if meta["kind"] == "active":
                seg_cum[meta["seg_idx"]] += spd * self.dt
        return sum(min(c, self.seg_lengths[i]) for i, c in enumerate(seg_cum))

    @property
    def result_fields(self):
        if not self._committed_frames["temperature"]:
            empty = torch.zeros(0, device=self.device)
            return {"temperature": empty, "alpha": empty, "muwafa": empty}
        return {k: torch.cat(v, dim=0) for k, v in self._committed_frames.items()}

    @property
    def committed_control_array(self):
        if not self._committed_controls:
            return np.zeros((0, 4), dtype=np.float32)
        return np.stack(self._committed_controls, axis=0)

    def committed_speed_power_profile(self):
        """Applied speed [m/s] and power [W] per step."""
        ctrl = self.committed_control_array
        speeds = np.array(self._committed_speeds, dtype=np.float32)
        return np.arange(ctrl.shape[0]), speeds, (ctrl[:, 2] * 1e3).astype(np.float32)

    def active_mask(self):
        mask = torch.zeros(self.total_steps, dtype=torch.bool, device=self.device)
        for i in range(self.num_segments):
            for t in self._seg_active_steps[i]:
                mask[t] = True
        return mask

    def open_loop_speed_power_profile(self):
        self._extend_full()
        T = self.total_steps
        speeds = np.zeros(T, dtype=np.float32)
        powers = np.zeros(T, dtype=np.float32)
        for t in range(T):
            if self._step_meta[t]["kind"] == "active":
                speeds[t] = (self.speed_min + (self.speed_max - self.speed_min)
                             * torch.sigmoid(self.raw_speed[t])).detach().item()
                powers[t] = (self.power_min + (self.power_max - self.power_min)
                             * torch.sigmoid(self.raw_power[t])).detach().item()
        return np.arange(T), speeds, powers
