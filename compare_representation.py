"""Reconstruction error of several autoencoders on the same CFD frames.

The first model is the reference. With --n-boot > 0, every other model is
compared with it by a paired bootstrap over cases.
"""
import argparse
import json
import os
import time

import h5py
import numpy as np
import torch

from data.schedules import mesh_resolution, read_fields
from evaluate_rollout import select_cases
from model.autoencoder import T_AMBIENT, T_MAX
from model.autoencoder_ablation import load_ablation_autoencoder
from utils import rollout_metrics as M

CHANNELS = ("T", "alpha", "mu")
T_SCALE = T_MAX - T_AMBIENT


def parse_models(specs):
    out = []
    for s in specs:
        name, _, rest = s.partition("=")
        cfg, _, ckpt = rest.partition(":")
        if not (name and cfg and ckpt):
            raise SystemExit(f"--models entry {s!r} is not name=config:checkpoint")
        out.append((name, cfg, ckpt))
    return out


@torch.no_grad()
def score_case(models, fields, S, batch, device):
    """Per-frame relative L2 (S, 3) of each model, and the CFD melt flags."""
    T, a, m = fields
    rel = {k: np.zeros((S, 3)) for k in models}
    melt = np.zeros(S, bool)
    for s in range(1, S + 1, batch):
        e = min(s + batch, S + 1)
        Tg, ag, mg = (torch.from_numpy(x[s:e]).to(device) for x in (T, a, m))
        gt = (Tg * T_SCALE + T_AMBIENT, ag, mg)
        melt[s - 1:e - 1] = (mg > M.MELT_THR).reshape(e - s, -1).any(1).cpu().numpy()
        for k, ae in models.items():
            rec = ae.decode(ae.encode(torch.stack([Tg, ag], dim=1)))
            rec[:, 0].clamp_(max=M.T_CLAMP)
            pr = (rec[:, 0] * T_SCALE + T_AMBIENT, rec[:, 1], rec[:, 2])
            for c in range(3):
                rel[k][s - 1:e - 1, c] = M.relative_l2(pr[c], gt[c]).double().cpu().numpy()
    return rel, melt


def summ(v):
    v = np.asarray(v, float)
    q25, q50, q75 = np.percentile(v, [25, 50, 75])
    return {"n": int(v.size), "median": float(q50), "q25": float(q25), "q75": float(q75),
            "mean": float(v.mean())}


def paired_bootstrap(r_b, r_a, mask, n_boot, seed=0):
    """Paired case bootstrap of the mean difference b - a."""
    rows = [i for i in range(mask.shape[0]) if mask[i].any()]
    ca = np.array([r_a[i][mask[i]].mean() for i in rows])
    cb = np.array([r_b[i][mask[i]].mean() for i in rows])
    d = cb - ca
    boot = d[np.random.default_rng(seed).integers(0, d.size, size=(n_boot, d.size))].mean(1)
    return {"n_cases": int(d.size), "mean_diff_model_minus_ref": float(d.mean()),
            "n_cases_ref_better": int((d > 0).sum()),
            "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "p_two_sided": float(min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()))),
            "p_floor": 2.0 / n_boot}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="CFD test archive (HDF5)")
    ap.add_argument("--models", nargs="+", required=True,
                    help="name=config:checkpoint ...; the first is the reference")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", default="", help="comma list of case names, or a .json/.txt list")
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--batch", type=int, default=8, help="frames per GPU batch")
    ap.add_argument("--n-boot", type=int, default=20000, help="paired bootstrap resamples (0: off)")
    ap.add_argument("--per-frame", action="store_true", help="also store the per-frame errors")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_path = os.path.join(args.out, "repr_metrics.json")
    if os.path.exists(out_path) and not args.overwrite:
        raise SystemExit(f"{out_path} exists; pass --overwrite to replace it")
    os.makedirs(args.out, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)

    specs = parse_models(args.models)
    ref = specs[0][0]
    models, info = {}, {}
    for name, cfg, ckpt in specs:
        models[name] = load_ablation_autoencoder(cfg, ckpt, device)
        info[name] = dict(config=cfg, checkpoint=ckpt, cls=type(models[name]).__name__)
        print(f"{name:12s} {info[name]['cls']:22s} {ckpt}", flush=True)

    S = args.steps
    with h5py.File(args.data, "r") as f:
        nx, ny, nz = mesh_resolution(f)
        names = select_cases(f, args.cases)
        print(f"{args.data}: {len(names)} cases x steps 1..{S}", flush=True)
        rel = {k: np.zeros((len(names), S, 3)) for k in models}
        melt = np.zeros((len(names), S), bool)
        t0 = time.time()
        for i, n in enumerate(names):
            fields = read_fields(f[n], nx, ny, nz, n_frames=S + 1)
            if fields[0].shape[0] < S + 1:
                raise SystemExit(f"{n} has {fields[0].shape[0]} < {S + 1} frames")
            r, melt[i] = score_case(models, fields, S, args.batch, device)
            for k in models:
                rel[k][i] = r[k]
            print(f"  [{i + 1}/{len(names)}] {n} | {time.time() - t0:5.0f} s", flush=True)

    subset = {"T": np.ones_like(melt), "alpha": np.ones_like(melt), "mu": melt}
    metrics = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data": {"path": os.path.abspath(args.data), "n_cases": len(names), "steps": S,
                 "n_melt_frames": int(melt.sum()), "cases": names},
        "models": info, "reference": ref,
        "definitions": {
            "relL2": "||x_hat - x|| / (||x|| + 1e-8) per frame over the volume; T in K (CFD T "
                     "clipped to [300, 4000] K, decoded T clamped at 3315 K), alpha, mu",
            "frames": f"steps 1..{S}; mu only on frames with CFD mu > {M.MELT_THR}"},
        "relL2": {k: {c: summ(rel[k][:, :, j][subset[c]]) for j, c in enumerate(CHANNELS)}
                  for k in models},
    }
    if args.n_boot > 0:
        metrics["paired_bootstrap"] = {
            f"{k}_vs_{ref}": {c: paired_bootstrap(rel[k][:, :, j], rel[ref][:, :, j], subset[c], args.n_boot)
                              for j, c in enumerate(CHANNELS)}
            for k in list(models)[1:]}
    if args.per_frame:
        metrics["per_frame"] = {"steps": list(range(1, S + 1)), "melt": melt.tolist(),
                                **{k: rel[k].round(7).tolist() for k in models}}
    with open(out_path, "w") as fh:
        json.dump(metrics, fh, indent=1)

    print(f"\nmedian (IQR) relative L2 over steps 1..{S} (mu: melt frames)")
    for k in models:
        r = metrics["relL2"][k]
        print(f"  {k:12s} " + "  ".join(
            f"{c} {100 * r[c]['median']:.2f}% ({100 * r[c]['q25']:.2f}-{100 * r[c]['q75']:.2f})"
            for c in CHANNELS))
    fmt = lambda p: f"<{2.0 / args.n_boot:.0e}" if p == 0 else f"{p:.2g}"
    for key, pr in metrics.get("paired_bootstrap", {}).items():
        print(f"  {key}: P " + " ".join(f"{c} {fmt(pr[c]['p_two_sided'])}" for c in CHANNELS))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
