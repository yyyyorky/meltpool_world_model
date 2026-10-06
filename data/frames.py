"""Frame dataset for autoencoder training."""
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from model.autoencoder import T_AMBIENT, T_MAX


def normalize_temperature(T):
    """Kelvin -> [0, 1] over [T_AMBIENT, T_MAX], clipped."""
    return np.clip((T - T_AMBIENT) / (T_MAX - T_AMBIENT), 0.0, 1.0)


def denormalize_temperature(T_normalized):
    return T_normalized * (T_MAX - T_AMBIENT) + T_AMBIENT


def random_init_temp(shape=(100, 100, 32), n_gaussians=30, z_center=19,
                     sigma_broad=(10, 20), sigma_sharp=(10, 15),
                     sigma_z_below=15, sigma_z_above=7.5,
                     sharp_ratio=0, peak_value=0.25, seed=None):
    """Random residual-heat field in normalized temperature."""
    rng = np.random.default_rng(seed)
    nx, ny, nz = shape

    x = np.arange(nx, dtype=np.float64)
    y = np.arange(ny, dtype=np.float64)
    z = np.arange(nz, dtype=np.float64)

    dz = z - z_center
    z_profile = np.where(dz <= 0,
                         np.exp(-0.5 * (dz / sigma_z_below) ** 2),
                         np.exp(-0.5 * (dz / sigma_z_above) ** 2))

    bg = np.zeros(shape, dtype=np.float64)
    for _ in range(4):
        bcx = rng.uniform(0.2 * nx, 0.8 * nx)
        bcy = rng.uniform(0.2 * ny, 0.8 * ny)
        bsig = rng.uniform(max(nx, ny) * 0.6, max(nx, ny) * 1.2)
        bg += (np.exp(-0.5 * ((x - bcx) / bsig) ** 2)[:, None, None]
               * np.exp(-0.5 * ((y - bcy) / bsig) ** 2)[None, :, None]
               * z_profile[None, None, :])

    field = np.zeros(shape, dtype=np.float64)
    n_sharp = max(round(n_gaussians * sharp_ratio), 1)
    n_broad = max(n_gaussians - n_sharp, 1)
    for i in range(n_broad + n_sharp):
        cx = rng.uniform(0, nx)
        cy = rng.uniform(0, ny)
        if i < n_broad:
            sigma_xy = rng.uniform(*sigma_broad)
            amp = rng.uniform(0.1, 0.3)
        else:
            sigma_xy = rng.uniform(*sigma_sharp)
            amp = rng.uniform(0.7, 1.0)
        field += (amp * np.exp(-0.5 * ((x - cx) / sigma_xy) ** 2)[:, None, None]
                  * np.exp(-0.5 * ((y - cy) / sigma_xy) ** 2)[None, :, None]
                  * z_profile[None, None, :])

    field = 0.2 * bg / bg.max() + 0.8 * field / field.max()
    fmax = field.max()
    if fmax > 0:
        field = field / fmax * peak_value
    return field


def sample_random_init_temp(nx, ny, nz, zlim=(-2e-4, 1.2e-4)):
    z_center = int(round((-zlim[0]) / ((zlim[1] - zlim[0]) / nz)))
    return random_init_temp(
        shape=(nx, ny, nz),
        z_center=z_center,
        peak_value=float(np.random.uniform(0, 0.25)),
        seed=int(np.random.randint(0, 2 ** 31)),
        sigma_z_below=float(np.random.uniform(10, 20)),
        sigma_z_above=float(np.random.uniform(5, 10)),
        sharp_ratio=0,
    )


class FrameDataset(Dataset):
    """Yields (input [T, alpha], target [T, alpha, muwafa], is_t0).

    At t=0 the temperature is replaced by a random residual-heat field.
    """

    def __init__(self, h5_file):
        self.h5_path = h5_file
        self._h5_file = None

        with h5py.File(h5_file, "r") as f:
            self.nx, self.ny, self.nz = f["global/mesh/mesh_resolution"][:]
            self.cases = [k for k in f.keys() if k.startswith("case_")]
            self.n_time_steps = f[self.cases[0]]["fields/T"].shape[0]

        self.index_map = [(c, t) for c in range(len(self.cases))
                          for t in range(self.n_time_steps)]

    def __len__(self):
        return len(self.index_map)

    def _file(self):
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, "r")
        return self._h5_file

    def _reshape(self, flat):
        """Fortran-ordered flat field -> (nx, ny, nz)."""
        return flat.reshape((self.nx, self.ny, self.nz), order="F")

    def __getitem__(self, idx):
        case_idx, time_idx = self.index_map[idx]
        case = self._file()[self.cases[case_idx]]

        alpha = self._reshape(case["fields/alpha.phase1"][time_idx, :])
        muwafa = self._reshape(case["fields/muwafa"][time_idx, :])
        if time_idx == 0:
            T = sample_random_init_temp(self.nx, self.ny, self.nz)
        else:
            T = normalize_temperature(self._reshape(case["fields/T"][time_idx, :]))

        input_data = np.stack([T, alpha], axis=0)
        target_data = np.stack([T, alpha, muwafa], axis=0)
        return (torch.from_numpy(input_data).float(),
                torch.from_numpy(target_data).float(),
                torch.tensor([float(time_idx == 0)]))
