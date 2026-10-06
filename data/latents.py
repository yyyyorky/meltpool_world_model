"""Latent window dataset for transition-model training."""
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

PAST_NOISE_STD = 1e-4


class LatentWindowDataset(Dataset):
    """Yields (past window, controls, target latents, step[, target muwafa])."""

    def __init__(self, h5_file, past_horizon, max_rollout_step,
                 load_muwafa=False, cases=None):
        self.h5_path = h5_file
        self._h5_file = None
        self.past_horizon = past_horizon
        self.max_rollout_step = max_rollout_step
        self.load_muwafa = load_muwafa

        with h5py.File(h5_file, "r") as f:
            if cases is None:
                self.cases = sorted(f["latent"].keys())
            else:
                self.cases = list(cases)
                missing = [c for c in self.cases if c not in f["latent"]]
                if missing:
                    raise KeyError(f"{len(missing)} case(s) not in {h5_file}: {missing[:5]}")
                if len(set(self.cases)) != len(self.cases):
                    raise ValueError("duplicate case names")
            first = self.cases[0]
            self.latent_shape = f[f"latent/{first}"].shape[1:]
            self.n_time_steps = f[f"latent/{first}"].shape[0]
            self.control_dim = int(np.prod(f[f"control_params/{first}"].shape[1:]))
            if load_muwafa and f"muwafa/{first}" not in f:
                raise KeyError(
                    "load_muwafa=True but the latent archive has no muwafa "
                    "group; rebuild it with build_latents.py from a dataset "
                    "that contains fields/muwafa.")

        self.index_map = [
            (c, t) for c in range(len(self.cases))
            for t in range(1, self.n_time_steps - self.max_rollout_step + 1)
        ]
        print(f"LatentWindowDataset: {len(self.cases)} cases, "
              f"{len(self.index_map)} samples, latent {self.latent_shape}")

    def __len__(self):
        return len(self.index_map)

    def _file(self):
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, "r")
        return self._h5_file

    def __getitem__(self, idx):
        case_idx, t = self.index_map[idx]
        case = self.cases[case_idx]
        f = self._file()
        latent = f[f"latent/{case}"]
        R = self.max_rollout_step

        past_indices = np.maximum(0, t + np.arange(-self.past_horizon, 0))
        past = np.stack([latent[i] for i in past_indices], axis=0)

        ctrl = np.stack([f[f"control_params/{case}"][i]
                         for i in range(t - 1, t - 1 + R)], axis=0).astype(np.float64)
        ctrl[:, :, :2] *= 1e3
        ctrl[:, :, 2] *= 1e-3
        ctrl = ctrl.reshape(R, -1)

        targets = np.stack([latent[i] for i in range(t, t + R)], axis=0)

        past = torch.from_numpy(past).float()
        past = past + torch.randn_like(past) * PAST_NOISE_STD

        out = (past,
               torch.from_numpy(ctrl).float(),
               torch.from_numpy(targets).float(),
               torch.tensor(t).long())
        if not self.load_muwafa:
            return out

        muwafa = np.stack([f[f"muwafa/{case}"][i] for i in range(t, t + R)], axis=0)
        return out + (torch.from_numpy(muwafa).float(),)
