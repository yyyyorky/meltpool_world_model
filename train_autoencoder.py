"""Train the FSQ autoencoder, or with --variant continuous the ablation model."""
import argparse
import json
import os
import time
from datetime import datetime

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler, random_split
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from data.frames import FrameDataset, denormalize_temperature
from model.autoencoder_ablation import build_ablation_autoencoder

MASS_WEIGHT = 1.0
IC_WEIGHT = 1e-6
IC_WEIGHT_FIRST_EPOCH = 1.0
START_BONUS = 15.0


def balanced_sampler(dataset):
    """Over-sample t=0 and t=1 frames."""
    parent, indices = dataset.dataset, dataset.indices
    weights = []
    for idx in indices:
        _, time_idx = parent.index_map[idx]
        weights.append(START_BONUS if time_idx == 0
                       else START_BONUS / 2 if time_idx == 1 else 1.0)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def gradient_loss_3d(pred, target):
    """MSE between finite-difference gradients of two fields."""
    kx = torch.tensor([[[[-1, 0, 1]]]], dtype=torch.float32, device=pred.device)
    ky = torch.tensor([[[[-1], [0], [1]]]], dtype=torch.float32, device=pred.device)
    kz = torch.tensor([[[[-1]], [[0]], [[1]]]], dtype=torch.float32, device=pred.device)
    loss = 0.0
    for k, pad in ((kx.expand(1, 1, 1, 1, 3), (0, 0, 1)),
                   (ky.expand(1, 1, 1, 3, 1), (0, 1, 0)),
                   (kz.expand(1, 1, 3, 1, 1), (1, 0, 0))):
        loss = loss + F.mse_loss(F.conv3d(pred, k, padding=pad),
                                 F.conv3d(target, k, padding=pad))
    return loss


def compute_loss(pred, target, is_t0_mask, ic_weight):
    per_channel = F.mse_loss(pred, target, reduction="none").mean(dim=(0, 2, 3, 4))
    recon_loss = per_channel.sum()

    alpha_pred = pred[:, 1].sum(dim=(1, 2, 3))
    alpha_target = target[:, 1].sum(dim=(1, 2, 3))
    mass_loss = (((alpha_pred - alpha_target) / (alpha_target.abs() + 1e-8)) ** 2).mean()

    pred_T = pred[:, 0:1].contiguous()
    target_T = target[:, 0:1].contiguous()
    grad_loss = gradient_loss_3d(pred_T, target_T)

    mask = is_t0_mask.squeeze(-1).bool()
    if mask.any():
        ic_penalty = (F.mse_loss(pred_T[mask], target_T[mask])
                      + gradient_loss_3d(pred_T[mask], target_T[mask]))
    else:
        ic_penalty = pred.new_tensor(0.0)

    total = recon_loss + MASS_WEIGHT * mass_loss + grad_loss + ic_weight * ic_penalty
    return total, recon_loss, mass_loss, grad_loss, ic_penalty


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True, help="CFD frames HDF5 archive")
    parser.add_argument("--variant", default=None, choices=["fsq", "continuous"],
                        help="bottleneck (default: config model.bottleneck, else fsq)")
    parser.add_argument("--out-dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=None, help="override training.epochs")
    parser.add_argument("--save-epochs", default=None,
                        help="comma list overriding training.save_epochs")
    parser.add_argument("--wandb-mode", default=None, choices=["online", "offline", "disabled"])
    parser.add_argument("--resume", action="store_true",
                        help="save the training state to <out-dir>/train_state_<config> after every "
                             "epoch and continue from it if present")
    parser.add_argument("--max-hours", type=float, default=0.0,
                        help="with --resume: save and exit after the first epoch past this wall time")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    variant = args.variant or cfg["model"].get("bottleneck", "fsq")
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.save_epochs is not None:
        cfg["training"]["save_epochs"] = [int(e) for e in args.save_epochs.split(",") if e.strip()]
    save_epochs = set(cfg["training"].get("save_epochs", []))
    loss_cfg = cfg.get("loss", {})
    ic_first = loss_cfg.get("ic_weight_first_epoch", IC_WEIGHT_FIRST_EPOCH)
    ic_later = loss_cfg.get("ic_weight", IC_WEIGHT)
    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    t_start = time.time()

    set_seed(cfg["data"]["seed"])
    accelerator = Accelerator(log_with="wandb")

    state_dir = os.path.join(args.out_dir, "train_state_" + os.path.splitext(os.path.basename(args.config))[0])
    resume_meta = None
    if args.resume and os.path.isfile(os.path.join(state_dir, "meta.json")):
        with open(os.path.join(state_dir, "meta.json")) as f:
            resume_meta = json.load(f)
        if resume_meta["variant"] != variant:
            raise SystemExit(f"{state_dir} holds a '{resume_meta['variant']}' run, not '{variant}'")
        if resume_meta["done"]:
            if accelerator.is_main_process:
                print(f"{state_dir} is marked done; nothing to do.")
            return
        start_time = resume_meta["start_time"]

    dataset = FrameDataset(args.data)
    val_size = int(cfg["data"]["val_ratio"] * len(dataset))
    train_dataset, val_dataset = random_split(
        dataset, [len(dataset) - val_size, val_size],
        generator=torch.Generator().manual_seed(cfg["data"]["seed"]))
    if accelerator.is_main_process:
        print(f"Variant: {variant}, Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=cfg["training"]["batch_size"],
        sampler=balanced_sampler(train_dataset),
        num_workers=cfg["training"]["num_workers"], pin_memory=True)
    val_loader = DataLoader(
        val_dataset, batch_size=cfg["training"]["val_batch_size"], shuffle=False,
        num_workers=cfg["training"]["num_workers"], pin_memory=True)

    model = build_ablation_autoencoder(cfg["model"], variant)

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if "bias" in name else decay).append(p)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg["optimizer"]["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg["optimizer"]["lr"])

    num_epochs = cfg["training"]["epochs"]
    total_steps = num_epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(0.1 * total_steps), total_steps)

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler)

    wandb_init = {"name": f"autoencoder_{variant}_{start_time}" if variant != "fsq"
                  else f"autoencoder_{start_time}",
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

    os.makedirs(args.out_dir, exist_ok=True)
    stem = "autoencoder" if variant == "fsq" else f"autoencoder_{variant}"
    ckpt_path = os.path.join(args.out_dir, f"{stem}_{start_time}.pth")
    st = dict(best_val=float("inf"), best_epoch=0, epoch=0)
    if resume_meta:
        accelerator.load_state(state_dir)
        st.update({k: resume_meta[k] for k in st})
        if accelerator.is_main_process:
            print(f"Resumed from {state_dir} after epoch {st['epoch']} "
                  f"(best {st['best_val']:.4f} at epoch {st['best_epoch']})")

    def save_state(done):
        accelerator.wait_for_everyone()
        accelerator.save_state(state_dir)
        if accelerator.is_main_process:
            with open(os.path.join(state_dir, "meta.json"), "w") as f:
                json.dump(dict(st, start_time=start_time, num_epochs=num_epochs, variant=variant,
                               done=done, wandb_id=wandb_id), f, indent=2)
        accelerator.wait_for_everyone()

    for epoch in range(st["epoch"], num_epochs):
        model.train()
        ic_weight = ic_first if epoch == 0 else ic_later
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}",
                    disable=not accelerator.is_main_process)
        for input_data, target_data, is_t0_mask in pbar:
            optimizer.zero_grad()
            out = model(input_data)
            loss, recon, mass, grad, ic = compute_loss(
                out["reconstruction"], target_data, is_t0_mask, ic_weight)
            loss = loss + out["emb_loss"]
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            accelerator.log({
                "train/loss": loss.item(), "train/recon": recon.item(),
                "train/mass": mass.item(), "train/grad": grad.item(),
                "train/ic": ic.item(), "train/emb": out["emb_loss"].item(),
            })
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        model.eval()
        rel_l2_total, n_batches = 0.0, 0
        with torch.no_grad():
            for input_data, target_data, _ in val_loader:
                pred = model(input_data)["reconstruction"]
                pred = accelerator.gather(pred)
                target = accelerator.gather(target_data)
                if accelerator.is_main_process:
                    pred = torch.cat([denormalize_temperature(pred[:, :1]),
                                      pred[:, 1:]], dim=1)
                    target = torch.cat([denormalize_temperature(target[:, :1]),
                                        target[:, 1:]], dim=1)
                    rel_l2_total += (torch.norm(pred - target)
                                     / (torch.norm(target) + 1e-8)).item()
                    n_batches += 1

        should_save = False
        if accelerator.is_main_process:
            rel_l2 = rel_l2_total / max(n_batches, 1)
            print(f"Epoch {epoch + 1}, Val Relative L2: {rel_l2:.4f}, "
                  f"LR: {scheduler.get_last_lr()[0]:.2e}")
            accelerator.log({"val/rel_l2": rel_l2})
            if rel_l2 < st["best_val"]:
                st["best_val"], st["best_epoch"] = rel_l2, epoch + 1
                should_save = True

        accelerator.wait_for_everyone()
        if should_save:
            accelerator.save(accelerator.unwrap_model(model).state_dict(), ckpt_path)
        if epoch + 1 in save_epochs:
            path = os.path.join(args.out_dir, f"{stem}_{start_time}_epoch{epoch + 1}.pth")
            accelerator.save(accelerator.unwrap_model(model).state_dict(), path)
            if accelerator.is_main_process:
                print(f"  -> snapshot {path}")

        st["epoch"] = epoch + 1
        over_time = torch.tensor(
            [int(args.max_hours > 0 and time.time() - t_start > 3600 * args.max_hours)],
            device=accelerator.device)
        over_time = bool(accelerator.gather(over_time).max().item())
        last = epoch + 1 >= num_epochs
        if args.resume:
            save_state(done=last)
        if over_time and not last:
            if accelerator.is_main_process:
                print(f"Wall-time budget reached after epoch {epoch + 1}; resume with --resume")
            break

    accelerator.end_training()
    if accelerator.is_main_process:
        print(f"Best validation relative L2: {st['best_val']:.4f} "
              f"(epoch {st['best_epoch']}) -> {ckpt_path}")


if __name__ == "__main__":
    main()
