"""Plant and observation settings of the policy."""
import json
from dataclasses import dataclass

THICKNESS_OBS_SCALE = 2e-4


@dataclass
class PlantConfig:
    dt: float = 5e-5
    speed_range: tuple = (0.8, 2.4)
    power_range: tuple = (80.0, 300.0)
    t_norm_steps: int = 81
    nfe: int = 4
    z_min: float = -2e-4
    dz: float = 1e-5
    z_substrate: float = 0.0
    x_min: float = -5e-4
    y_min: float = -5e-4
    dx: float = 1e-5
    nx: int = 100
    ny: int = 100
    thickness_patch_size: int = 64
    future_lookahead_steps: int = 5
    future_lookahead_speed: float = 3.0


def load_plant_config(path):
    with open(path) as f:
        cfg = json.load(f)
    for key in ("speed_range", "power_range"):
        cfg[key] = tuple(cfg[key])
    return PlantConfig(**cfg)
