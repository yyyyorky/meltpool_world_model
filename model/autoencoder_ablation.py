"""Continuous-bottleneck autoencoder for the representation ablation.

Same network as FSQAutoencoder with the rounding removed, so both share
parameter names and load with strict=True.
"""
import json
import re

import torch

from model.autoencoder import FSQAutoencoder, build_autoencoder

VARIANTS = ("fsq", "continuous")


class ContinuousAutoencoder(FSQAutoencoder):

    def _quantize(self, h):
        m = self.morph_dim
        half_l = (self.fsq._levels.float() - 1) / 2
        shape = [1] * h.ndim
        shape[1] = m
        s = self.fsq.scale
        z_morph = torch.tanh(h[:, :m] * s) * half_l.view(*shape) / s
        z_T = torch.tanh(h[:, m:m + 1])
        z_alpha = torch.tanh(h[:, m + 1:])
        return torch.cat([z_morph, z_T, z_alpha], dim=1), h.new_zeros((), dtype=torch.float32)

    def requantize(self, z):
        return z


def build_ablation_autoencoder(model_cfg, variant=None):
    """FSQ or continuous autoencoder; `variant` defaults to model_cfg['bottleneck']."""
    variant = variant or model_cfg.get("bottleneck", "fsq")
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    if variant == "fsq":
        return build_autoencoder(model_cfg)
    return ContinuousAutoencoder(
        in_channels=model_cfg["in_channels"],
        latent_dim=model_cfg["latent_dim"],
        base_channels=model_cfg["base_channels"],
        ch_mult=tuple(model_cfg["ch_mult"]),
        num_res_blocks=model_cfg["num_res_blocks"],
        fsq_levels=tuple(model_cfg["fsq_levels"]),
        fsq_scale=model_cfg["fsq_scale"],
        amplitude_levels=model_cfg["amplitude_levels"],
        decoder_base_channels=model_cfg["decoder_base_channels"],
        decoder_ch_mult=tuple(model_cfg["decoder_ch_mult"]),
        decoder_num_res_blocks=model_cfg["decoder_num_res_blocks"],
    )


def clean_state_dict(sd):
    """Unwrap a checkpoint dict and strip 'module.' / '_orig_mod.' prefixes."""
    for k in ("model_state_dict", "state_dict", "model"):
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
            sd = sd[k]
    return {re.sub(r"^(module\.|_orig_mod\.)+", "", k): v for k, v in sd.items()}


def load_ablation_autoencoder(config_path, checkpoint, device, variant=None):
    with open(config_path) as f:
        cfg = json.load(f)
    model = build_ablation_autoencoder(cfg["model"], variant).to(device)
    sd = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(clean_state_dict(sd), strict=True)
    model.eval()
    return model
