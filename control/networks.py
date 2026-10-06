"""Actor network of the amortized policy."""
import torch
import torch.nn as nn


class LatentVolumeCNN(nn.Module):

    def __init__(self, in_shape, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_shape[0], 64, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv3d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            flat_dim = int(self.net(torch.zeros(1, *in_shape)).shape[1])
        self.head = nn.Sequential(nn.Linear(flat_dim, out_dim), nn.GELU())

    def forward(self, x):
        return self.head(self.net(x))


class ThicknessPatchCNN(nn.Module):

    def __init__(self, patch_shape, out_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            flat_dim = int(self.net(torch.zeros(1, 1, *patch_shape)).shape[1])
        self.head = nn.Sequential(nn.Linear(flat_dim, out_dim), nn.GELU())

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return self.head(self.net(x))


class ObservationEncoder(nn.Module):

    def __init__(self, latent_shape, scalar_dim, patch_size,
                 cnn_out_dim=256, scalar_out_dim=32, thickness_out_dim=32):
        super().__init__()
        self.features_dim = cnn_out_dim + scalar_out_dim + thickness_out_dim
        self.cnn = LatentVolumeCNN(latent_shape, cnn_out_dim)
        self.scalar_mlp = nn.Sequential(
            nn.Linear(scalar_dim, scalar_out_dim), nn.GELU(),
            nn.Linear(scalar_out_dim, scalar_out_dim), nn.GELU(),
        )
        self.thickness_cnn = ThicknessPatchCNN((patch_size, patch_size),
                                               thickness_out_dim)

    def forward(self, obs):
        return torch.cat([self.cnn(obs["latent"]),
                          self.scalar_mlp(obs["scalars"]),
                          self.thickness_cnn(obs["thickness_patch"])], dim=-1)


class Actor(nn.Module):
    """Maps an observation to raw (speed, power) logits."""

    def __init__(self, latent_shape, scalar_dim, patch_size, n_actions=2):
        super().__init__()
        self.ext = ObservationEncoder(latent_shape, scalar_dim, patch_size)
        fd = self.ext.features_dim
        self.head = nn.Sequential(nn.Linear(fd, 256), nn.GELU(),
                                  nn.Linear(256, 256), nn.GELU(),
                                  nn.Linear(256, n_actions))

    def forward(self, obs):
        return self.head(self.ext(obs))[0]
