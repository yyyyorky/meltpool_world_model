"""Branched FSQ autoencoder for the melt-pool state."""
import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

T_AMBIENT = 300.0
T_MAX = 4000.0
T_SOLIDUS = 1877.0
T_LIQUIDUS = 1923.0


def compute_muwafa(T_normalized, alpha):
    """Liquid-metal fraction from normalized temperature and phase indicator."""
    T = T_normalized * (T_MAX - T_AMBIENT) + T_AMBIENT
    T_mid = 0.5 * (T_SOLIDUS + T_LIQUIDUS)
    dT = T_LIQUIDUS - T_SOLIDUS
    return alpha * 0.5 * (1 + torch.tanh(4 * (T - T_mid) / dT))


class FSQ(nn.Module):
    """Finite scalar quantization (Mentzer et al., 2023).

    Unlike the original, even L has no half-step offset, so L = 4 gives three
    values and L = 2 only zero. The trained weights depend on this rounding.
    """

    def __init__(self, levels, scale=10.0):
        super().__init__()
        self.register_buffer("_levels", torch.tensor(levels, dtype=torch.long))
        self.scale = scale
        self.dim = len(levels)
        self.codebook_size = math.prod(levels)

    def quantize(self, z):
        half_l = (self._levels.float() - 1) / 2
        shape = [1] * z.ndim
        shape[1] = self.dim
        z = torch.tanh(z * self.scale) * half_l.view(*shape)
        z = z + (z.round() - z).detach()
        return z / self.scale

    def forward(self, z):
        z_q = self.quantize(z)
        commitment = 0.25 * F.mse_loss(z, z_q.detach())
        z_q = z + (z_q - z).detach()
        return z_q, commitment

    def snap(self, z_q):
        """Snap a quantized latent back onto the code grid (straight-through)."""
        half_l = (self._levels.float() - 1) / 2
        shape = [1] * z_q.ndim
        shape[1] = self.dim
        bound = half_l.round().view(*shape)
        k = torch.minimum(torch.maximum((z_q * self.scale).round(), -bound), bound)
        return z_q + (k / self.scale - z_q).detach()


class ContinuousTokenizerLayer(nn.Module):
    """Uniform amplitude quantizer over tanh-bounded values."""

    def __init__(self, num_levels=1024):
        super().__init__()
        self.num_levels = num_levels
        self.register_buffer("half_l", torch.tensor((num_levels - 1) / 2.0))

    def forward(self, z):
        z_bounded = torch.tanh(z)
        z_scaled = z_bounded * self.half_l
        z_q = (z_scaled + (z_scaled.round() - z_scaled).detach()) / self.half_l
        commitment = F.mse_loss(z_bounded, z_q.detach())
        return z_q, commitment

    def snap(self, z_q):
        """Snap an amplitude back onto the nearest level (straight-through)."""
        bound = self.half_l.round()
        k = torch.clamp((z_q * self.half_l).round(), -bound, bound)
        return z_q + (k / self.half_l - z_q).detach()


class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, num_groups=32):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(num_groups, in_channels), in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(num_groups, out_channels), out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.shortcut(x)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, 3, stride=2, padding=0)

    def forward(self, x):
        return self.conv(F.pad(x, (0, 1, 0, 1, 0, 1)))


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="trilinear",
                                       align_corners=False))


class Encoder(nn.Module):

    def __init__(self, in_channels, base_channels, ch_mult, num_res_blocks,
                 latent_dim):
        super().__init__()
        self.conv_in = nn.Conv3d(in_channels, base_channels, 3, padding=1)

        in_ch = base_channels
        self.down = nn.ModuleList()
        for i, mult in enumerate(ch_mult):
            out_ch = base_channels * mult
            level = nn.Module()
            level.block = nn.ModuleList()
            for _ in range(num_res_blocks):
                level.block.append(ResBlock(in_ch, out_ch))
                in_ch = out_ch
            level.downsample = Downsample(in_ch) if i < len(ch_mult) - 1 else None
            self.down.append(level)

        self.mid_block1 = ResBlock(in_ch, in_ch)
        self.mid_block2 = ResBlock(in_ch, in_ch)
        self.norm_out = nn.GroupNorm(min(32, in_ch), in_ch)
        self.conv_out = nn.Conv3d(in_ch, latent_dim, 3, padding=1)

    def forward(self, x):
        h = self.conv_in(x)
        for level in self.down:
            for block in level.block:
                h = block(h)
            if level.downsample is not None:
                h = level.downsample(h)
        h = self.mid_block2(self.mid_block1(h))
        return self.conv_out(F.silu(self.norm_out(h)))


class Decoder(nn.Module):

    def __init__(self, in_channels, base_channels, ch_mult, num_res_blocks,
                 num_groups=32):
        super().__init__()
        top_ch = base_channels * ch_mult[-1]
        self.conv_in = nn.Conv3d(in_channels, top_ch, 3, padding=1)
        self.mid_block1 = ResBlock(top_ch, top_ch, num_groups)
        self.mid_block2 = ResBlock(top_ch, top_ch, num_groups)

        in_ch = top_ch
        self.up = nn.ModuleList()
        for i in reversed(range(len(ch_mult))):
            out_ch = base_channels * ch_mult[i]
            level = nn.Module()
            level.block = nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                level.block.append(ResBlock(in_ch, out_ch, num_groups))
                in_ch = out_ch
            level.upsample = Upsample(in_ch) if i > 0 else None
            self.up.append(level)

        self.norm_out = nn.GroupNorm(min(num_groups, in_ch), in_ch)
        self.conv_out = nn.Conv3d(in_ch, 1, 3, padding=1)

    def forward(self, z):
        h = self.mid_block2(self.mid_block1(self.conv_in(z)))
        for level in self.up:
            for block in level.block:
                h = block(h)
            if level.upsample is not None:
                h = level.upsample(h)
        return self.conv_out(F.silu(self.norm_out(h)))


class FSQAutoencoder(nn.Module):
    """Latent channels: [morphology, amplitude_T, amplitude_alpha].

    decode returns [T, alpha, muwafa].
    """

    def __init__(self, in_channels=2, latent_dim=10, base_channels=64,
                 ch_mult=(1, 2, 4), num_res_blocks=2,
                 fsq_levels=(5, 4, 4, 3, 3, 3, 2, 2), fsq_scale=10.0,
                 amplitude_levels=1024, decoder_base_channels=32,
                 decoder_ch_mult=(1, 2, 4), decoder_num_res_blocks=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.morph_dim = len(fsq_levels)
        assert latent_dim - self.morph_dim == 2, \
            "latent_dim must equal len(fsq_levels) + 2 amplitude channels"

        self.encoder = Encoder(in_channels, base_channels, ch_mult,
                               num_res_blocks, latent_dim)

        decoder_in = self.morph_dim + 1
        self.decoder_T = Decoder(decoder_in, decoder_base_channels,
                                 decoder_ch_mult, decoder_num_res_blocks)
        self.decoder_alpha = Decoder(decoder_in, decoder_base_channels,
                                     decoder_ch_mult, decoder_num_res_blocks)

        self.quant_conv = nn.Conv3d(latent_dim, latent_dim, 1)
        self.post_quant_conv_T = nn.Conv3d(decoder_in, decoder_in, 1)
        self.post_quant_conv_alpha = nn.Conv3d(decoder_in, decoder_in, 1)

        self.fsq = FSQ(list(fsq_levels), fsq_scale)
        self.continuous_tokenizer_T = ContinuousTokenizerLayer(amplitude_levels)
        self.continuous_tokenizer_alpha = ContinuousTokenizerLayer(amplitude_levels)

        self.temperature_activation = nn.Softplus()

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _quantize(self, h):
        z_morph, commit_m = self.fsq(h[:, :self.morph_dim])
        z_T, commit_T = self.continuous_tokenizer_T(
            h[:, self.morph_dim:self.morph_dim + 1])
        z_alpha, commit_a = self.continuous_tokenizer_alpha(
            h[:, self.morph_dim + 1:])
        return torch.cat([z_morph, z_T, z_alpha], dim=1), commit_m + commit_T + commit_a

    def encode(self, x):
        z, _ = self._quantize(self.quant_conv(self.encoder(x)))
        return z

    def requantize(self, z):
        """Snap a predicted latent onto the code grid of `encode` (straight-through)."""
        m = self.morph_dim
        return torch.cat([self.fsq.snap(z[:, :m]),
                          self.continuous_tokenizer_T.snap(z[:, m:m + 1]),
                          self.continuous_tokenizer_alpha.snap(z[:, m + 1:])], dim=1)

    def decode(self, z):
        z_morph = z[:, :self.morph_dim]
        h_T = self.post_quant_conv_T(
            torch.cat([z_morph, z[:, self.morph_dim:self.morph_dim + 1]], dim=1))
        h_a = self.post_quant_conv_alpha(
            torch.cat([z_morph, z[:, self.morph_dim + 1:]], dim=1))

        T = self.temperature_activation(self.decoder_T(h_T))
        alpha = torch.sigmoid(self.decoder_alpha(h_a))
        muwafa = compute_muwafa(T, alpha)
        return torch.cat([T, alpha, muwafa], dim=1)

    def forward(self, x):
        z, emb_loss = self._quantize(self.quant_conv(self.encoder(x)))
        return {"reconstruction": self.decode(z), "z": z, "emb_loss": emb_loss}


def build_autoencoder(model_cfg):
    return FSQAutoencoder(
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


def load_autoencoder(config_path, checkpoint, device):
    with open(config_path) as f:
        cfg = json.load(f)
    model = build_autoencoder(cfg["model"]).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model
