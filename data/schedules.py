"""Fields and laser schedules of the CFD test archives.

Both archive layouts are supported: constant-parameter archives with
`waypoints`, and schedule archives with a `laser/` group. Control row k =
[x mm, y mm, P kW, (k+1)/n_t] drives frame k -> k+1.
"""
import json

import numpy as np

from data.frames import normalize_temperature
from utils.scan_path import interpolate_heat_source_positions_hatching

DT = 5e-5  # s
OFF_PARK = np.array([1e-3, 1e-3])  # laser-off park position [m]


def list_cases(f):
    """Case names in natural (v, p) order.

    The index seeds each case's noise, so the order must not change.
    """
    return sorted([k for k in f.keys() if k.startswith("case_")],
                  key=lambda s: [int(t) if t.isdigit() else t for t in
                                 s.replace("case_v", "").replace("_p", " ").replace("_", " ").split()])


def mesh_resolution(f):
    return tuple(int(v) for v in f["global/mesh/mesh_resolution"][:])


def read_fields(case, nx, ny, nz, n_frames=None):
    """(T_norm, alpha, muwafa), each (n, nx, ny, nz)."""
    n_t = case["fields/T"].shape[0]
    n = n_t if n_frames is None else min(n_frames, n_t)

    def field(name):
        d = case[f"fields/{name}"][:n].astype(np.float32)
        return d.reshape(n, nz, ny, nx).transpose(0, 3, 2, 1)

    return normalize_temperature(field("T")), field("alpha.phase1"), field("muwafa")


def _jsonable(v):
    if isinstance(v, (bytes, np.bytes_)):
        v = v.decode()
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, np.ndarray):
        v = v.tolist()
    if isinstance(v, str):
        s = v.strip()
        if s[:1] in "[{":
            try:
                return json.loads(s)
            except Exception:
                return v
    return v


def case_metadata(case):
    """metadata/ and laser/ attributes as one dict; laser/ keys are prefixed."""
    md = {k: _jsonable(v) for k, v in case["metadata"].attrs.items()}
    if "laser" in case:
        for k, v in case["laser"].attrs.items():
            md[f"laser_{k}"] = _jsonable(v)
        prof = md.get("laser_profile")
        if isinstance(prof, dict):
            for pk, pv in prof.items():
                md.setdefault(f"laser_profile_{pk}", _jsonable(pv))
    md["schedule_type"] = str(md.get("laser_schedule_type", md.get("laser_mode", "constant")))
    md["schedule_tag"] = str(md.get("laser_tag", md["schedule_type"]))
    md["nominal_power"] = float(md.get("laser_nominal_power", md.get("laser_power", np.nan)))
    md["nominal_speed"] = float(md.get("laser_nominal_speed", md.get("scan_speed", np.nan)))
    return md


def parse_hatch_segments(waypoints):
    """Laser-on segments of a waypoint array.

    NaN rows separate segments; segments starting at x ~ 1.0 are laser-off
    pseudo paths.
    """
    wp = waypoints[:, :2] if waypoints.ndim > 1 and waypoints.shape[1] > 2 else waypoints
    segs, cur = [], []
    for p in wp:
        if np.any(np.isnan(p)):
            if cur:
                segs.append(np.array(cur, np.float64))
                cur = []
        else:
            cur.append(p)
    if cur:
        segs.append(np.array(cur, np.float64))
    return [sg for sg in segs if not np.isclose(sg[0, 0], 1.0, atol=0.01)]


def on_off_runs(on):
    runs, k = [], 0
    while k < len(on):
        a = k
        while k < len(on) and on[k] == on[a]:
            k += 1
        runs.append((a, k - 1, bool(on[a])))
    return runs


def laser_off_convention(pos, on, speed, hatch_segments, m_full=5, dt=DT):
    """Move laser-off positions to the convention of the laser/ archives.

    The first off step continues along the previous segment (clipped at its
    end); the remaining steps move linearly to the start of the next segment.
    """
    pos = np.array(pos, np.float64, copy=True)
    n = len(pos)
    seg_i = -1
    for a, b, is_on in on_off_runs(on):
        if is_on:
            seg_i += 1
            continue
        m_here = b + 1 - a
        if seg_i < 0:
            pos[a:b + 1] = hatch_segments[0][0] if hatch_segments else pos[b + 1]
            continue
        seg = hatch_segments[min(seg_i, len(hatch_segments) - 1)]
        d = seg[-1] - seg[0]
        dir_i = d / (np.linalg.norm(d) + 1e-30)
        p_last = pos[a - 1]
        remaining = max(float(np.dot(seg[-1] - p_last, dir_i)), 0.0)
        pa = p_last + min(speed[a - 1] * dt, remaining) * dir_i
        has_next = seg_i + 1 < len(hatch_segments)
        m = m_here if (b + 1 < n and has_next) else m_full
        S = hatch_segments[seg_i + 1][0] if has_next else pa
        for j in range(m_here):
            pos[a + j] = pa + (S - pa) * (j / m) if j < m else S
    return pos


def read_schedule(case, n_t, off_position="file"):
    """Laser schedule: pos (n_t, 2) m, power (n_t,) W, speed (n_t,) m/s, on (n_t,).

    off_position: 'file' keeps the stored laser-off positions, 'interp' applies
    laser_off_convention, 'park' uses the (1, 1) mm park position.
    """
    md = case["metadata"].attrs
    if "laser" in case:
        pts = np.asarray(case["laser/points"][:], np.float64)
        pw = np.asarray(case["laser/powers"][:], np.float64)
        if len(pts) < n_t:
            raise ValueError(f"laser/points has {len(pts)} < {n_t} entries")
        if len(pw) < n_t:
            pw = np.concatenate([pw, np.zeros(n_t - len(pw))])
        if "laser/speeds" in case:
            sp = np.asarray(case["laser/speeds"][:], np.float64)
            if len(sp) < n_t:
                sp = np.concatenate([sp, np.zeros(n_t - len(sp))])
        else:
            sp = np.concatenate([np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1) / DT, [0.0]])
        pos, power, speed = pts[:n_t, :2], pw[:n_t], sp[:n_t]
        legacy = False
    else:
        v0, P0 = float(md["scan_speed"]), float(md["laser_power"])
        pos, pw = interpolate_heat_source_positions_hatching(
            case["waypoints"][:], v0, DT, n_t, upsample_step=1, laser_power=P0)
        pos, power = pos[:n_t], pw[:n_t, 0].copy()
        power[-1] = 0.0
        speed = np.where(power > 0, v0, 0.0)
        legacy = True
    on = power > 0
    speed = np.where(on, speed, 0.0)
    pos = np.array(pos, np.float64, copy=True)
    if off_position == "park":
        pos[~on] = OFF_PARK
    elif off_position == "interp" and legacy:
        on_raw = np.asarray(pw)[:n_t].reshape(-1) > 0
        m_full = int(round(float(md["laser_off_time"]) / DT)) if "laser_off_time" in md else 5
        pos = laser_off_convention(pos, on_raw, np.where(on_raw, float(md["scan_speed"]), 0.0),
                                   parse_hatch_segments(case["waypoints"][:]), m_full)
    return dict(pos=pos, power=power, speed=speed, on=on)


def control_rows(sched, n_t, n_steps):
    """(n_steps, 4) rows [x mm, y mm, P kW, (k+1)/n_t]."""
    rows = np.zeros((n_steps, 4), np.float32)
    rows[:, 0:2] = sched["pos"][:n_steps] * 1e3
    rows[:, 2] = sched["power"][:n_steps] * 1e-3
    rows[:, 3] = np.minimum((np.arange(n_steps) + 1) / n_t, 1.0)
    return rows


def scan_directions(sched, n_steps):
    """Unit scan direction per interval; off intervals keep the last direction."""
    pos, on = sched["pos"], sched["on"]
    dirs = np.zeros((n_steps, 2))
    last = np.array([-1.0, 0.0])
    for k in range(n_steps):
        if on[k] and k + 1 < len(pos):
            d = pos[k + 1] - pos[k]
            nrm = np.linalg.norm(d)
            if 1e-12 < nrm < 5e-4:
                last = d / nrm
        dirs[k] = last
    return dirs
