"""Scan-path generation and parsing. Waypoint arrays separate segments by NaN rows."""
import numpy as np

OFF_POSITION = np.array([1e-3, 1e-3])


def interpolate_heat_source_positions_hatching(waypoints, scan_speed, dt,
                                               num_time_steps, upsample_step=1,
                                               laser_power=200.0):
    """Walk the path at constant speed; returns positions (N, 2) and powers (N, 1)."""
    waypoints_2d = waypoints[:, :2] if waypoints.ndim > 1 and waypoints.shape[1] > 2 else waypoints

    raw_segments, current = [], []
    for point in waypoints_2d:
        if np.any(np.isnan(point)):
            if current:
                raw_segments.append(np.array(current))
                current = []
        else:
            current.append(point)
    if current:
        raw_segments.append(np.array(current))

    n_total = num_time_steps * upsample_step
    if not raw_segments:
        return np.full((n_total, 2), np.nan), np.zeros((n_total, 1))

    segment_info = []
    for seg in raw_segments:
        is_pseudo = np.isclose(seg[0, 0], 1.0, atol=0.01)
        length = np.sum(np.linalg.norm(np.diff(seg, axis=0), axis=1)) if len(seg) > 1 else 0
        info = {
            "duration": length / scan_speed if scan_speed > 0 else 0,
            "points": seg,
            "laser_on": not is_pseudo,
        }
        if info["laser_on"] and len(seg) > 1:
            lengths = np.linalg.norm(np.diff(seg, axis=0), axis=1)
            info["sub_lengths"] = lengths
            info["cum_dist"] = np.concatenate([[0], np.cumsum(lengths)])
        segment_info.append(info)

    cumulative_times = np.zeros(len(segment_info) + 1)
    for i, info in enumerate(segment_info):
        cumulative_times[i + 1] = cumulative_times[i] + info["duration"]
    total_duration = cumulative_times[-1]

    sub_dt = dt / upsample_step
    sub_offsets = (np.arange(upsample_step) - upsample_step + 1) * sub_dt
    all_times = (np.arange(num_time_steps)[:, None] * dt + sub_offsets[None, :]).ravel()

    neg_mask = all_times < 0
    all_times_clamped = np.clip(all_times, 0, total_duration)
    n_segments = len(segment_info)
    seg_indices = np.clip(
        np.searchsorted(cumulative_times[1:], all_times_clamped, side="right"),
        0, n_segments - 1)

    positions = np.zeros((n_total, 2))
    powers = np.zeros((n_total, 1))

    for s in range(n_segments):
        mask = (seg_indices == s) & ~neg_mask
        if not np.any(mask):
            continue
        info = segment_info[s]
        if not info["laser_on"]:
            positions[mask] = OFF_POSITION
            continue

        dists = (all_times_clamped[mask] - cumulative_times[s]) * scan_speed
        points = info["points"]
        if len(points) == 1:
            positions[mask] = points[0]
        else:
            cum_dist, sub_lengths = info["cum_dist"], info["sub_lengths"]
            dists = np.clip(dists, 0, cum_dist[-1])
            sub_idx = np.clip(np.searchsorted(cum_dist, dists, side="right") - 1,
                              0, len(sub_lengths) - 1)
            ratios = np.where(sub_lengths[sub_idx] > 0,
                              (dists - cum_dist[sub_idx]) / sub_lengths[sub_idx], 0.0)
            positions[mask] = points[sub_idx] + ratios[:, None] * (points[sub_idx + 1] - points[sub_idx])

        powers[mask, 0] = np.where(all_times[mask] > total_duration, 0.0, laser_power)

    positions[neg_mask] = OFF_POSITION
    return positions, powers


def parse_scan_tracks(waypoints):
    """Waypoints -> (n_segments, 2, 2) start/end pairs."""
    wp = waypoints[:, :2] if waypoints.ndim > 1 and waypoints.shape[1] > 2 else waypoints

    segments, current = [], []
    for pt in wp:
        if np.any(np.isnan(pt)):
            if len(current) >= 2:
                arr = np.array(current)
                if not np.isclose(arr[0, 0], 1.0, atol=0.01):
                    segments.append(np.stack([arr[:-1], arr[1:]], axis=1))
            current = []
        else:
            current.append(pt)
    if len(current) >= 2:
        arr = np.array(current)
        if not np.isclose(arr[0, 0], 1.0, atol=0.01):
            segments.append(np.stack([arr[:-1], arr[1:]], axis=1))
    return np.concatenate(segments, axis=0) if segments else np.empty((0, 2, 2))


def _line_box_intersections(point, direction, x_bounds, y_bounds):
    px, py = point
    dx, dy = direction
    pts = []
    if abs(dx) > 1e-12:
        for xe in x_bounds:
            t = (xe - px) / dx
            y_int = py + t * dy
            if y_bounds[0] <= y_int <= y_bounds[1]:
                pts.append((xe, y_int))
    if abs(dy) > 1e-12:
        for ye in y_bounds:
            t = (ye - py) / dy
            x_int = px + t * dx
            if x_bounds[0] <= x_int <= x_bounds[1]:
                pts.append((x_int, ye))
    out = []
    for p in pts:
        if not any(np.hypot(p[0] - q[0], p[1] - q[1]) < 1e-12 for q in out):
            out.append(p)
    return out


def _clip_to_substrate(start, end, alpha_top, x_range, y_range, threshold,
                       sample_dx):
    nx, ny = alpha_top.shape
    cell_dx = (x_range[1] - x_range[0]) / nx
    cell_dy = (y_range[1] - y_range[0]) / ny

    seg = np.array(end) - np.array(start)
    length = float(np.linalg.norm(seg))
    if length < 1e-15:
        return []
    ts = np.linspace(0.0, 1.0, max(2, int(np.ceil(length / sample_dx)) + 1))
    xs = start[0] + ts * seg[0]
    ys = start[1] + ts * seg[1]
    i = np.clip(np.floor((xs - x_range[0]) / cell_dx).astype(int), 0, nx - 1)
    j = np.clip(np.floor((ys - y_range[0]) / cell_dy).astype(int), 0, ny - 1)
    inside = alpha_top[i, j] > threshold

    runs, in_run, t_start = [], False, 0.0
    for k, flag in enumerate(inside):
        if flag and not in_run:
            in_run, t_start = True, ts[k]
        elif not flag and in_run:
            in_run = False
            if ts[k - 1] > t_start:
                runs.append((t_start, ts[k - 1]))
    if in_run:
        runs.append((t_start, ts[-1]))
    return [((start[0] + t0 * seg[0], start[1] + t0 * seg[1]),
             (start[0] + t1 * seg[0], start[1] + t1 * seg[1]))
            for t0, t1 in runs]


def choose_perpendicular_hatch_angle(overhang_field, mask=None,
                                     return_info=False):
    """Axis-aligned scan angle crossing the overhang edge."""
    f = np.asarray(overhang_field, dtype=float)
    gx = np.abs(np.gradient(f, axis=0))
    gy = np.abs(np.gradient(f, axis=1))
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        Gx, Gy = float(gx[m].sum()), float(gy[m].sum())
    else:
        Gx, Gy = float(gx.sum()), float(gy.sum())
    angle = 0.0 if Gx >= Gy else (np.pi / 2.0)
    return (angle, Gx, Gy) if return_info else angle


def generate_uniform_hatching_path(alpha_top, x_range=(-5e-4, 5e-4),
                                   y_range=(-5e-4, 5e-4), hatch_spacing=1.0e-4,
                                   hatch_angle=0.0, scan_speed=1.0,
                                   laser_off_time=2.5e-4,
                                   substrate_threshold=0.5,
                                   min_track_length=None, sample_dx=None):
    """Serpentine hatching clipped to the substrate mask, (N, 3)."""
    alpha_top = np.asarray(alpha_top)
    nx, ny = alpha_top.shape
    cell_dx = (x_range[1] - x_range[0]) / nx
    cell_dy = (y_range[1] - y_range[0]) / ny
    if sample_dx is None:
        sample_dx = 0.5 * min(cell_dx, cell_dy)
    if min_track_length is None:
        min_track_length = 2.0 * hatch_spacing

    scan_dir = np.array([np.cos(hatch_angle), np.sin(hatch_angle)])
    hatch_dir = np.array([-np.sin(hatch_angle), np.cos(hatch_angle)])

    corners = np.array([[x_range[0], y_range[0]], [x_range[1], y_range[0]],
                        [x_range[1], y_range[1]], [x_range[0], y_range[1]]])
    proj = corners @ hatch_dir
    top = float(proj.max()) - 0.5 * hatch_spacing
    bot = float(proj.min()) + 0.5 * hatch_spacing
    if top < bot:
        offsets = np.array([], dtype=float)
    else:
        n_lines = int(np.floor((top - bot) / hatch_spacing)) + 1
        offsets = top - np.arange(n_lines) * hatch_spacing

    lines = []
    for offset in offsets:
        pts = _line_box_intersections(offset * hatch_dir, scan_dir, x_range, y_range)
        if len(pts) < 2:
            continue
        pts_sorted = sorted(pts, key=lambda p: p[0] * scan_dir[0] + p[1] * scan_dir[1])
        subs = _clip_to_substrate(pts_sorted[0], pts_sorted[-1], alpha_top,
                                  x_range, y_range, substrate_threshold, sample_dx)
        kept = [(np.array(s), np.array(e)) for s, e in subs
                if np.hypot(e[0] - s[0], e[1] - s[1]) >= min_track_length]
        if kept:
            lines.append(kept)
    if not lines:
        return np.empty((0, 3), dtype=float)

    waypoints, reverse = [], False
    for subs in lines:
        for s, e in (reversed(subs) if reverse else subs):
            if reverse:
                s, e = e, s
            waypoints.append([s[0], s[1], 0.0])
            waypoints.append([e[0], e[1], 0.0])
            waypoints.append([np.nan, np.nan, np.nan])
            waypoints.append([1.0, 0.0, 1.0])
            waypoints.append([1.0, scan_speed * laser_off_time, 1.0])
            waypoints.append([np.nan, np.nan, np.nan])
        reverse = not reverse
    return np.array(waypoints, dtype=float)
