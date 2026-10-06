"""Roll out held-out CFD cases and score every step against CFD.

Member e of case c is seeded seed + 1000003 e + c, so member 0 does not depend
on the batch, shard or ensemble size. Writes one <name>.npz per model.
"""
import argparse
import glob
import json
import os
import time

import h5py
import numpy as np
import torch

from data.schedules import (DT, case_metadata, control_rows, list_cases, mesh_resolution,
                            read_fields, read_schedule, scan_directions)
from model.autoencoder import T_AMBIENT, T_MAX, load_autoencoder
from model.transition import is_generative, load_transition, predict_next
from utils import rollout_metrics as M

T_SCALE = T_MAX - T_AMBIENT


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="CFD test archive (HDF5)")
    p.add_argument("--autoencoder-config", required=True)
    p.add_argument("--autoencoder", required=True)
    p.add_argument("--transition-config", required=True,
                   help="config of the transition models (architecture block)")
    p.add_argument("--transition", required=True,
                   help="comma list of name=checkpoint (same architecture)")
    p.add_argument("--out", required=True, help="output directory, one <name>.npz per model")
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--nfe", type=int, default=None, help="Euler steps (default: config)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--requantize", action=argparse.BooleanOptionalAction, default=True,
                   help="snap each predicted latent onto the FSQ code grid (default on; "
                        "--no-requantize keeps the continuous latent)")
    p.add_argument("--n-ens", type=int, default=8, help="noise members per case (DiT only)")
    p.add_argument("--batch-size", type=int, default=4, help="cases per GPU batch")
    p.add_argument("--decode-chunk", type=int, default=16, help="decoder batch (cases x members)")
    p.add_argument("--cases", default="", help="comma list of case names, or a .json/.txt list")
    p.add_argument("--off-position", default="interp", choices=["file", "park", "interp"],
                   help="laser-off position convention (see data.schedules.read_schedule)")
    p.add_argument("--shard", default="0/1", help="i/n: roll cases[i::n] into <out>/<name>/shard_i.npz")
    p.add_argument("--merge", action="store_true", help="merge <out>/<name>/shard_*.npz")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def parse_models(spec):
    out = []
    for item in spec.split(","):
        item = item.strip()
        if item:
            name, _, path = item.rpartition("=")
            out.append((name or os.path.splitext(os.path.basename(path))[0], path))
    return out


def select_cases(f, spec):
    names = list_cases(f)
    if not spec:
        return names
    if os.path.isfile(spec):
        txt = open(spec).read()
        wanted = json.loads(txt) if spec.endswith(".json") else txt.split()
    else:
        wanted = [s.strip() for s in spec.split(",") if s.strip()]
    missing = [w for w in wanted if w not in names]
    if missing:
        raise SystemExit(f"cases not in {f.filename}: {missing[:5]}")
    return [n for n in names if n in set(wanted)]


def save_result(path, arrays, meta, overwrite):
    if os.path.exists(path) and not overwrite:
        raise SystemExit(f"{path} exists; pass --overwrite to replace it")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, meta=json.dumps(meta), **arrays)


def load_result(path):
    z = np.load(path, allow_pickle=False)
    out = {k: z[k] for k in z.files if k != "meta"}
    out["meta"] = json.loads(str(z["meta"]))
    return out


def load_case(case, name, case_id, nx, ny, nz, S, off_position):
    n_t = case["fields/T"].shape[0]
    steps = min(S, n_t - 1)
    sched = read_schedule(case, n_t, off_position)
    return dict(name=name, id=case_id, steps=steps, sched=sched,
                fields=read_fields(case, nx, ny, nz, n_frames=steps + 1),
                ctrl=control_rows(sched, n_t, steps), dirs=scan_directions(sched, steps),
                meta=case_metadata(case))


def gt_scalars(c, device, chunk=16):
    T, a, m = c["fields"]
    out = {k: [] for k in M.SCALAR_KEYS}
    with torch.no_grad():
        for s in range(1, c["steps"] + 1, chunk):
            e = min(s + chunk, c["steps"] + 1)
            sc = M.melt_pool_scalars(torch.from_numpy(T[s:e]).to(device),
                                     torch.from_numpy(a[s:e]).to(device),
                                     torch.from_numpy(m[s:e]).to(device),
                                     torch.from_numpy(c["dirs"][s - 1:e - 1]).to(device))
            for k in out:
                out[k].append(sc[k])
    return {k: np.concatenate(v) for k, v in out.items()}


@torch.no_grad()
def rollout_batch(ae, model, cases, past_horizon, nfe, seed, E, decode_chunk, device,
                  requantize=False):
    """Roll B cases x E members together."""
    BE = len(cases) * E
    S = max(c["steps"] for c in cases)
    init = np.stack([np.stack([c["fields"][0][0], c["fields"][1][0]], 0) for c in cases], 0)
    z0 = ae.encode(torch.from_numpy(init).to(device))
    z_dim = z0.shape[1]
    z0 = z0.repeat_interleave(E, dim=0)
    stack = z0.repeat(1, past_horizon, 1, 1, 1)
    gens = []
    for c in cases:
        for e in range(E):
            g = torch.Generator(device=device)
            g.manual_seed(int(seed) + 1_000_003 * e + int(c["id"]))
            gens.append(g)

    keys = ("wm_relT", "wm_relA", "wm_iou") + tuple("wm_" + s for s in M.SCALAR_KEYS)
    out = [{k: np.full((E, c["steps"]), np.nan, np.float32) for k in keys} for c in cases]
    for o, c in zip(out, cases):
        for k in ("crps_T", "crps_alpha", "crps_muwafa"):
            o[k] = np.full(c["steps"], np.nan, np.float32)

    for i in range(S):
        rows = np.stack([c["ctrl"][min(i, c["steps"] - 1)] for c in cases], 0)
        ctrl = torch.from_numpy(rows).to(device).repeat_interleave(E, dim=0)
        noise = torch.stack([torch.randn(z0.shape[1:], generator=g, device=device) for g in gens], 0)
        pred = predict_next(model, stack, ctrl, noise=noise, nfe=nfe,
                            requantize=ae.requantize if requantize else None)
        stack = torch.cat([stack, pred], dim=1)[:, z_dim:]
        for c0 in range(0, BE, decode_chunk):
            rec = ae.decode(pred[c0:c0 + decode_chunk])
            rec[:, 0].clamp_(max=M.T_CLAMP)
            for j in range(rec.shape[0]):
                b, e = divmod(c0 + j, E)
                c, o = cases[b], out[b]
                if i >= c["steps"]:
                    continue
                Tg, ag, mg = (torch.from_numpy(fld[i + 1]).to(device) for fld in c["fields"])
                Tp, ap, mp = rec[j, 0], rec[j, 1], rec[j, 2]
                o["wm_relT"][e, i] = M.relative_l2((Tp * T_SCALE + T_AMBIENT)[None],
                                                   (Tg * T_SCALE + T_AMBIENT)[None])[0].item()
                o["wm_relA"][e, i] = M.relative_l2(ap[None], ag[None])[0].item()
                o["wm_iou"][e, i] = M.iou(mp[None], mg[None])[0].item()
                d = torch.from_numpy(c["dirs"][i:i + 1]).to(device)
                sc = M.melt_pool_scalars(Tp[None], ap[None], mp[None], d)
                for k in M.SCALAR_KEYS:
                    o["wm_" + k][e, i] = sc[k][0]
                if e == 0:
                    c["_ens"] = ([], [], [])
                for lst, x in zip(c["_ens"], (Tp * T_SCALE + T_AMBIENT, ap, mp)):
                    lst.append(x)
                if e == E - 1:
                    ens_T, ens_a, ens_m = (torch.stack(lst) for lst in c["_ens"])
                    o["crps_T"][i] = M.crps_field(ens_T, Tg * T_SCALE + T_AMBIENT)
                    o["crps_alpha"][i] = M.crps_field(ens_a, ag)
                    o["crps_muwafa"][i] = M.crps_field(ens_m, mg)
                    c["_ens"] = None
    return out


def pack(common_rows, model_rows, S, E):
    N = len(common_rows)
    arr = {}
    for k in model_rows[0]:
        arr[k] = np.full((N, S) if k.startswith("crps_") else (N, E, S), np.nan, np.float32)
    for k in M.SCALAR_KEYS:
        arr["gt_" + k] = np.full((N, S), np.nan, np.float32)
    for k in ("P_W", "v_mps", "x_mm", "y_mm", "dir_x", "dir_y"):
        arr[k] = np.full((N, S), np.nan, np.float32)
    arr["on"] = np.zeros((N, S), bool)
    arr["valid"] = np.zeros((N, S), bool)
    names, ids, metas, n_steps = [], [], [], []
    for n, (row, mr) in enumerate(zip(common_rows, model_rows)):
        c, g = row["case"], row["gt"]
        s, sch = c["steps"], c["sched"]
        for k, v in mr.items():
            arr[k][n, ..., :s] = v
        for k in M.SCALAR_KEYS:
            arr["gt_" + k][n, :s] = g[k]
        arr["P_W"][n, :s] = sch["power"][:s]
        arr["v_mps"][n, :s] = sch["speed"][:s]
        arr["on"][n, :s] = sch["on"][:s]
        arr["x_mm"][n, :s] = sch["pos"][:s, 0] * 1e3
        arr["y_mm"][n, :s] = sch["pos"][:s, 1] * 1e3
        arr["dir_x"][n, :s] = c["dirs"][:s, 0]
        arr["dir_y"][n, :s] = c["dirs"][:s, 1]
        arr["valid"][n, :s] = True
        names.append(c["name"])
        ids.append(c["id"])
        n_steps.append(s)
        metas.append(json.dumps(c["meta"], default=str))
    arr["case_names"] = np.array(names)
    arr["case_id"] = np.array(ids, np.int64)
    arr["n_steps"] = np.array(n_steps, np.int64)
    arr["case_meta_json"] = np.array(metas)
    arr["step"] = np.arange(1, S + 1)
    arr["t_ms"] = arr["step"] * DT * 1e3
    return arr


def merge(args, models):
    for name, _ in models:
        shards = sorted(glob.glob(os.path.join(args.out, name, "shard_*.npz")))
        if not shards:
            raise SystemExit(f"[{name}] no shards in {os.path.join(args.out, name)}")
        parts = [load_result(s) for s in shards]
        keys = [k for k in parts[0] if k not in ("meta", "step", "t_ms")]
        arrays = {k: np.concatenate([p[k] for p in parts], axis=0) for k in keys}
        order = np.argsort(arrays["case_id"])
        arrays = {k: v[order] for k, v in arrays.items()}
        arrays["step"], arrays["t_ms"] = parts[0]["step"], parts[0]["t_ms"]
        meta = parts[0]["meta"]
        meta.update(n_cases=int(len(arrays["case_names"])), merged_from=shards)
        out = os.path.join(args.out, f"{name}.npz")
        save_result(out, arrays, meta, args.overwrite)
        print(f"[{name}] merged {len(shards)} shards -> {out} ({meta['n_cases']} cases)")


def main():
    args = parse_args()
    models = parse_models(args.transition)
    if args.merge:
        merge(args, models)
        return

    device = torch.device(args.device)
    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    ae = load_autoencoder(args.autoencoder_config, args.autoencoder, device)
    nets = {}
    for name, path in models:
        nets[name], t_cfg = load_transition(args.transition_config, path, device)
    past_horizon = t_cfg["data"]["past_horizon"]
    nfe = args.nfe if args.nfe is not None else t_cfg["model"].get("nfe", 4)
    generative = all(is_generative(m) for m in nets.values())
    E = args.n_ens if generative else 1
    S = args.steps

    with h5py.File(args.data, "r") as f:
        nx, ny, nz = mesh_resolution(f)
        ids = {n: i for i, n in enumerate(list_cases(f))}
        mine = select_cases(f, args.cases)[shard_i::shard_n]
        print(f"{args.data}: shard {shard_i}/{shard_n}, {len(mine)} cases | E={E} S={S} "
              f"NFE={nfe} batch={args.batch_size} off-position={args.off_position}"
              + (" requantize" if args.requantize else ""), flush=True)
        per_model = {n: [] for n in nets}
        common_rows = []
        t0 = time.time()
        for b0 in range(0, len(mine), args.batch_size):
            names = mine[b0:b0 + args.batch_size]
            cases = [load_case(f[n], n, ids[n], nx, ny, nz, S, args.off_position) for n in names]
            gts = [gt_scalars(c, device) for c in cases]
            common_rows += [dict(case=c, gt=g) for c, g in zip(cases, gts)]
            for name, net in nets.items():
                res = rollout_batch(ae, net, cases, past_horizon, nfe, args.seed, E,
                                    args.decode_chunk, device, requantize=args.requantize)
                per_model[name] += res
                mae = np.mean([np.abs(r["wm_depth_um"][0] - g["depth_um"]).mean()
                               for r, g in zip(res, gts)])
                print(f"  [{b0 + len(names)}/{len(mine)}] {name}: member-0 depth MAE "
                      f"{mae:6.2f} um | {time.time() - t0:5.0f} s", flush=True)
            for c in cases:
                c["fields"] = None

    for name, path in models:
        meta = dict(ckpt=path, ckpt_name=name, ae_ckpt=args.autoencoder,
                    transition_config=args.transition_config, data=args.data, steps=S, nfe=nfe,
                    seed=args.seed, n_ens=E, past_horizon=past_horizon,
                    off_position=args.off_position, batch_size=args.batch_size,
                    requantize=args.requantize,
                    decode_chunk=args.decode_chunk, n_cases=len(common_rows), shard=args.shard)
        out = (os.path.join(args.out, f"{name}.npz") if shard_n == 1
               else os.path.join(args.out, name, f"shard_{shard_i}.npz"))
        save_result(out, pack(common_rows, per_model[name], S, E), meta, args.overwrite)
        print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
