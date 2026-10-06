"""Train the flow-matching DiT transition model."""
import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler, random_split
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from data.latents import LatentWindowDataset
from model.autoencoder import load_autoencoder
from model.transition import build_transition, fm_loss, fm_sample

START_BONUS = 15.0
EARLY_WEIGHT = 40.0
EARLY_DECAY_TAU = 0.6


def balanced_sampler(dataset):
    """Over-sample the first two prediction steps of every case."""
    if isinstance(dataset, Subset):
        parent, indices = dataset.dataset, dataset.indices
    else:
        parent, indices = dataset, range(len(dataset))
    weights = []
    for idx in indices:
        _, time_idx = parent.index_map[idx]
        weights.append(START_BONUS if time_idx == 0
                       else START_BONUS / 2 if time_idx == 1 else 1.0)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def early_step_weight(time_idx):
    return 1.0 + (EARLY_WEIGHT - 1.0) * torch.exp(-time_idx.float() / EARLY_DECAY_TAU)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True, help="latent HDF5 archive")
    parser.add_argument("--out-dir", default="checkpoints")
    parser.add_argument("--init", default=None, help="warm-start checkpoint (overrides the config)")
    parser.add_argument("--train-cases", default=None, help="override data.train_cases")
    parser.add_argument("--val-cases", default=None, help="override data.val_cases")
    parser.add_argument("--epochs", type=int, default=None, help="override training.epochs")
    parser.add_argument("--max-steps", type=int, default=None, help="override training.max_steps")
    parser.add_argument("--wandb-mode", default=None, choices=["online", "offline", "disabled"])
    parser.add_argument("--resume", action="store_true",
                        help="save the training state to <out-dir>/train_state_<config> at every "
                             "validation and continue from it if present")
    parser.add_argument("--max-hours", type=float, default=0.0,
                        help="with --resume: save and exit after the first epoch past this wall time")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    if cfg["model"].get("arch", "dit") != "dit":
        raise SystemExit("train_transition.py trains the flow-matching DiT only; the ViT and "
                         "Transolver baselines are provided for loading only.")
    data_cfg, train_cfg, model_cfg = cfg["data"], cfg["training"], cfg["model"]
    for key, value in (("train_cases", args.train_cases), ("val_cases", args.val_cases)):
        if value:
            data_cfg[key] = value
    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.max_steps is not None:
        train_cfg["max_steps"] = args.max_steps
    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    t_start = time.time()

    set_seed(data_cfg["seed"])
    accelerator = Accelerator(log_with="wandb")

    state_dir = os.path.join(args.out_dir, "train_state_" + os.path.splitext(os.path.basename(args.config))[0])
    resume_meta = None
    if args.resume and os.path.isfile(os.path.join(state_dir, "meta.json")):
        with open(os.path.join(state_dir, "meta.json")) as f:
            resume_meta = json.load(f)
        if resume_meta["done"]:
            if accelerator.is_main_process:
                print(f"{state_dir} is marked done; nothing to do.")
            return
        start_time = resume_meta["start_time"]

    muwafa_cfg = cfg.get("muwafa")
    use_muwafa = muwafa_cfg is not None and muwafa_cfg["weight"] > 0
    rollout_max = train_cfg["rollout_max"]
    rollout_warmup_epochs = train_cfg["rollout_warmup_epochs"]
    val_interval = train_cfg.get("val_interval", 1)
    rollout_val_steps = min(train_cfg.get("rollout_val_steps", 0), rollout_max)
    rollout_val_batches = train_cfg.get("rollout_val_batches", 24)
    patience = train_cfg.get("early_stop_patience", 0)
    min_delta = train_cfg.get("early_stop_min_delta", 0.0)
    guard_mult = train_cfg.get("divergence_guard_mult", 0.0)
    guard_patience = train_cfg.get("divergence_guard_patience", 3)
    nfe = model_cfg.get("nfe", 4)
    z_dim = model_cfg["z_dim"]
    past_horizon = data_cfg["past_horizon"]

    def make_dataset(cases=None):
        return LatentWindowDataset(args.data, past_horizon=past_horizon,
                                   max_rollout_step=rollout_max, load_muwafa=use_muwafa,
                                   cases=cases)

    if data_cfg.get("train_cases") and data_cfg.get("val_cases"):
        with open(data_cfg["train_cases"]) as f:
            train_cases = json.load(f)
        with open(data_cfg["val_cases"]) as f:
            val_cases = json.load(f)
        overlap = set(train_cases) & set(val_cases)
        assert not overlap, f"train/val case lists overlap: {sorted(overlap)[:5]}"
        train_dataset, val_dataset = make_dataset(train_cases), make_dataset(val_cases)
        n_time_steps = train_dataset.n_time_steps
    else:
        dataset = make_dataset()
        n_time_steps = dataset.n_time_steps
        val_size = int(data_cfg["val_ratio"] * len(dataset))
        train_dataset, val_dataset = random_split(
            dataset, [len(dataset) - val_size, val_size],
            generator=torch.Generator().manual_seed(data_cfg["seed"]))
    if accelerator.is_main_process:
        print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=train_cfg["batch_size"],
        sampler=balanced_sampler(train_dataset),
        num_workers=train_cfg["num_workers"], pin_memory=True)
    val_loader = DataLoader(
        val_dataset, batch_size=train_cfg["val_batch_size"], shuffle=False,
        num_workers=train_cfg["num_workers"], pin_memory=True)

    model = build_transition(model_cfg, past_horizon)
    init_ckpt = args.init or cfg.get("init_checkpoint")
    if init_ckpt:
        model.load_state_dict(torch.load(init_ckpt, map_location="cpu"))
    if accelerator.is_main_process:
        print(f"DiT: {sum(p.numel() for p in model.parameters()):,} parameters"
              + (f", initialized from {init_ckpt}" if init_ckpt else ""))

    ae = None
    if use_muwafa:
        ae = load_autoencoder(muwafa_cfg["autoencoder_config"],
                              muwafa_cfg["autoencoder_checkpoint"], accelerator.device)
        for p in ae.parameters():
            p.requires_grad_(False)

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if "bias" in name or "norm" in name or name == "pos"
         else decay).append(p)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg["optimizer"]["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg["optimizer"]["lr"])

    num_epochs = train_cfg["epochs"]
    steps_per_epoch = int(np.ceil(len(train_loader) / accelerator.num_processes))
    if train_cfg.get("max_steps"):
        num_epochs = int(np.ceil(train_cfg["max_steps"] / steps_per_epoch))
    if accelerator.is_main_process:
        print(f"{num_epochs} epochs x {steps_per_epoch} optimizer steps")
    # The prepared scheduler steps once per process, hence the unsharded length.
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, cfg["optimizer"]["warmup_steps"], num_epochs * len(train_loader))

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler)
    grad_clip = cfg["optimizer"]["grad_clip"]

    wandb_init = {"name": f"transition_{start_time}",
                  "mode": args.wandb_mode or cfg["wandb"].get("mode", "online")}
    if resume_meta and resume_meta.get("wandb_id"):
        wandb_init.update(id=resume_meta["wandb_id"], resume="allow")
    accelerator.init_trackers(project_name=cfg["wandb"]["project"], config=cfg,
                              init_kwargs={"wandb": wandb_init})
    wandb_id = None
    if accelerator.is_main_process:
        try:
            wandb_id = accelerator.get_tracker("wandb", unwrap=True).id
        except Exception:
            pass

    def build_cond(control_params, time_idx, r):
        t_norm = ((time_idx.float() + r) / n_time_steps).clamp(max=1.0)
        return torch.cat([control_params[:, r, :], t_norm.unsqueeze(-1)], dim=-1)

    os.makedirs(args.out_dir, exist_ok=True)
    best_path = os.path.join(args.out_dir, f"transition_{start_time}.pth")
    final_path = os.path.join(args.out_dir, f"transition_{start_time}_final.pth")
    st = dict(best_val=float("inf"), best_epoch=0, best_step=0, no_improve=0, divergence=0,
              global_step=0, epoch=0)
    if resume_meta:
        accelerator.load_state(state_dir)
        st.update({k: resume_meta[k] for k in st})

    def save_state(done, stop_reason=None):
        accelerator.wait_for_everyone()
        accelerator.save_state(state_dir)
        if accelerator.is_main_process:
            with open(os.path.join(state_dir, "meta.json"), "w") as f:
                json.dump(dict(st, start_time=start_time, num_epochs=num_epochs, done=done,
                               stop_reason=stop_reason, wandb_id=wandb_id), f, indent=2)
        accelerator.wait_for_everyone()

    def save_weights(path):
        if accelerator.is_main_process:
            accelerator.save(accelerator.unwrap_model(model).state_dict(), path)

    for epoch in range(st["epoch"], num_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}",
                    disable=not accelerator.is_main_process)
        for batch in pbar:
            if use_muwafa:
                past_latents, control_params, target_latents, time_idx, target_muwafa = batch
            else:
                past_latents, control_params, target_latents, time_idx = batch
            B = past_latents.shape[0]

            R = (torch.randint(1, rollout_max + 1, (1,)).item()
                 if rollout_max > 1 and epoch >= rollout_warmup_epochs else 1)

            loss_mu = None
            past = past_latents.reshape(B, -1, *past_latents.shape[3:])
            w = early_step_weight(time_idx)
            if R > 1:
                net = accelerator.unwrap_model(model)
                with torch.no_grad():
                    for r in range(R - 1):
                        pred = fm_sample(net, past, past[:, -z_dim:],
                                         build_cond(control_params, time_idx, r), nfe=nfe)
                        past = torch.cat([past, pred], dim=1)[:, z_dim:]
            cond = build_cond(control_params, time_idx, R - 1)
            last_frame = past[:, -z_dim:]
            next_frame = target_latents[:, R - 1]
            optimizer.zero_grad()
            if use_muwafa:
                loss_fm, pred_residual = fm_loss(model, past, last_frame, next_frame, cond,
                                                 sample_weight=w, return_pred=True)
                muwafa_pred = ae.decode(last_frame + pred_residual)[:, 2]
                per_sample = F.l1_loss(muwafa_pred, target_muwafa[:, R - 1],
                                       reduction="none").mean(dim=[1, 2, 3])
                loss_mu = (per_sample * w).mean()
                loss = loss_fm + muwafa_cfg["weight"] * loss_mu
            else:
                loss = fm_loss(model, past, last_frame, next_frame, cond, sample_weight=w)

            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            log = {"train/loss": loss.item(), "train/lr": scheduler.get_last_lr()[0],
                   "train/rollout_step": R}
            if loss_mu is not None:
                log["train/loss_muwafa"] = loss_mu.item()
            accelerator.log(log, step=st["global_step"])
            pbar.set_postfix(loss=f"{loss.item():.4f}", R=R)
            st["global_step"] += 1
        st["epoch"] = epoch + 1

        stop_reason = None
        validated = (epoch + 1) % val_interval == 0
        if validated:
            model.eval()
            net = accelerator.unwrap_model(model)
            tot = dict(rel_l2=0.0, mu_l1=0.0, roll=0.0, n=0, n_roll=0)
            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    if use_muwafa:
                        past_latents, control_params, target_latents, time_idx, target_muwafa = batch
                    else:
                        past_latents, control_params, target_latents, time_idx = batch
                    B = past_latents.shape[0]
                    past = past_latents.reshape(B, -1, *past_latents.shape[3:])
                    cond = build_cond(control_params, time_idx, 0)
                    pred = fm_sample(net, past, past[:, -z_dim:], cond, nfe=nfe)
                    all_pred = accelerator.gather(pred)
                    all_target = accelerator.gather(target_latents[:, 0])
                    if accelerator.is_main_process:
                        tot["rel_l2"] += (torch.norm(all_pred - all_target)
                                          / (torch.norm(all_target) + 1e-8)).item()
                        tot["n"] += 1
                    if use_muwafa:
                        mu_pred = accelerator.gather(ae.decode(pred)[:, 2])
                        mu_target = accelerator.gather(target_muwafa[:, 0])
                        if accelerator.is_main_process:
                            tot["mu_l1"] += (torch.norm(mu_pred - mu_target, p=1)
                                             / (torch.norm(mu_target, p=1) + 1e-8)).item()
                    # K-step rollout error, logged only
                    if rollout_val_steps >= 2 and i < rollout_val_batches:
                        past_r, pred_r = past, pred
                        for r in range(1, rollout_val_steps):
                            past_r = torch.cat([past_r, pred_r], dim=1)[:, z_dim:]
                            pred_r = fm_sample(net, past_r, past_r[:, -z_dim:],
                                               build_cond(control_params, time_idx, r), nfe=nfe)
                        all_r = accelerator.gather(pred_r)
                        all_tr = accelerator.gather(target_latents[:, rollout_val_steps - 1])
                        if accelerator.is_main_process:
                            tot["roll"] += (torch.norm(all_r - all_tr)
                                            / (torch.norm(all_tr) + 1e-8)).item()
                            tot["n_roll"] += 1

            should_save = False
            if accelerator.is_main_process:
                rel_l2 = tot["rel_l2"] / max(tot["n"], 1)
                log = {"val/rel_l2": rel_l2, "epoch": epoch + 1}
                if use_muwafa:
                    log["val/muwafa_rel_l1"] = tot["mu_l1"] / max(tot["n"], 1)
                if tot["n_roll"]:
                    log[f"val/rollout{rollout_val_steps}_rel_l2"] = tot["roll"] / tot["n_roll"]
                prev_best = st["best_val"]
                if rel_l2 < st["best_val"]:
                    st.update(best_val=rel_l2, best_epoch=epoch + 1, best_step=st["global_step"])
                    should_save = True
                st["no_improve"] = 0 if rel_l2 < prev_best * (1.0 - min_delta) else st["no_improve"] + 1
                st["divergence"] = (st["divergence"] + 1 if guard_mult > 0
                                    and np.isfinite(st["best_val"])
                                    and rel_l2 > guard_mult * st["best_val"] else 0)
                if patience and st["no_improve"] >= patience:
                    stop_reason = (f"early_stop: no improvement for {st['no_improve']} validations; "
                                   f"best {st['best_val']:.6f} at epoch {st['best_epoch']}")
                if guard_mult and st["divergence"] >= guard_patience:
                    stop_reason = (f"divergence_guard: {rel_l2:.6f} > {guard_mult:g} x best "
                                   f"{st['best_val']:.6f} for {st['divergence']} validations")
                log.update({"val/no_improve_count": st["no_improve"],
                            "val/divergence_count": st["divergence"]})
                accelerator.log(log, step=st["global_step"])
                print(" | ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}"
                                 for k, v in log.items()))
            accelerator.wait_for_everyone()
            if should_save:
                save_weights(best_path)
                if accelerator.is_main_process:
                    print(f"  -> new best checkpoint {best_path}")

        # broadcast rank 0's decision so all ranks stop together
        flags = torch.tensor([int(stop_reason is not None),
                              int(args.max_hours > 0 and time.time() - t_start > 3600 * args.max_hours)],
                             device=accelerator.device)
        flags = accelerator.gather(flags).reshape(-1, 2).max(0).values
        if accelerator.is_main_process:
            best_t = torch.tensor([st["best_val"]], device=accelerator.device)
        else:
            best_t = torch.tensor([float("inf")], device=accelerator.device)
        st["best_val"] = float(accelerator.gather(best_t).min().item())
        last = (epoch + 1) >= num_epochs or bool(flags[0])
        if last:
            accelerator.wait_for_everyone()
            save_weights(final_path)
            if accelerator.is_main_process:
                print(f"Stopped after epoch {epoch + 1} / step {st['global_step']}: "
                      f"{stop_reason or 'budget_complete'}; best {st['best_val']:.4f} at "
                      f"epoch {st['best_epoch']} -> {best_path}, final -> {final_path}")
        if args.resume and (validated or last or flags[1]):
            save_state(done=last, stop_reason=(stop_reason or "budget_complete") if last else None)
        if last:
            break
        if flags[1]:
            if accelerator.is_main_process:
                print(f"Wall-time budget reached after epoch {epoch + 1}; resume with --resume")
            break

    accelerator.end_training()


if __name__ == "__main__":
    main()
