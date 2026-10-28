<h1 align="center">Generative World Models Enable Predictive Control of Laser Melt Pool Dynamics</h1>
<h4 align="center">
<a href="https://arxiv.org/abs/2610.06250"><img alt="Static Badge" src="https://img.shields.io/badge/arXiv-2610.06250-b31b1b"></a>
<a href="https://libdrive.ethz.ch/index.php/s/WibqAP2gBTfDdW3"><img alt="Static Badge" src="https://img.shields.io/badge/Data-Archive-green"></a>
</h4>
<div align="center">
  <span class="author-block">
    <a>Yiyang Yan</a><sup>1</sup>,</span>
  <span class="author-block">
    <a>Markus Bambach</a><sup>1</sup>, and</span>
    <span class="author-block">
    <a>Mohamadreza Afrasiabi</a><sup>1</sup></span>
</div>
<div align="center">
  <span class="author-block"><sup>1</sup>Advanced Manufacturing Lab, ETH Zurich</span>
</div>

## Overview
World models, which learn how environments respond to actions, are emerging as a powerful paradigm for planning through imagined futures, transforming decision-making across games, robotics, and autonomous driving. Bringing this capability to manufacturing could enable process decisions on timescales inaccessible to high-fidelity simulation. Here we introduce a generative world model for localized, highly dynamic laser melt pools that predicts evolution from histories of temperature and phase morphology under candidate actions. Its generative latent dynamics capture the effects of unresolved melt flow, enabling more accurate recursive rollouts than deterministic regressors under transient laser inputs. Because the learned dynamics are differentiable, the model can serve directly as a predictive control plant. Gradients through imagined futures optimize laser schedules that regulate melt-pool depth over previously unseen geometry, path, and initialization. We further distill this optimization into an amortized policy that produces control actions in a single forward pass, providing a proof of concept for real deployment on machines.

This repository contains a discrete generative world model of laser powder bed fusion melt pools, together with two controllers that plan laser power and scan speed through it. An FSQ autoencoder compresses the 3D temperature and phase fields into a discrete latent code, and a flow-matching diffusion transformer advances the latent state under the laser controls.

## Repository

| Path | Contents |
|---|---|
| `train_autoencoder.py`, `build_latents.py`, `train_transition.py` | Autoencoder, latent encoding and transition training |
| `evaluate_rollout.py` | Ensemble rollouts of held-out cases scored against CFD |
| `run_mpc.py`, `run_policy.py` | Receding-horizon MPC and SHAC policy deployment |
| `model/`, `data/`, `control/`, `utils/` | Networks, datasets, controllers, metrics and export |
| `config/` | Settings and case splits of the paper's runs |
| `checkpoints/` | Trained weights (not in git) |

## Installation

Tested with Python 3.13, PyTorch 2.10 and CUDA 12.6:

```bash
pip install -r requirements.txt
```

Training used 4 GPUs via `accelerate` (`config/accelerate_4gpu.yaml`); a single GPU also works. Both controllers run on one GPU.

## Data

Download the HDF5 datasets from the [data archive](https://libdrive.ethz.ch/index.php/s/WibqAP2gBTfDdW3) and place them one directory above the repository root. The large training archives are split into `.part0` / `.part1` and are joined with `cat`. The archive is password-protected during peer review; the password is available from the authors on request.

## Usage

```bash
# 1. Autoencoder
accelerate launch --config_file config/accelerate_4gpu.yaml train_autoencoder.py \
    --config config/autoencoder.json --data ../lpbf_dataset_overhang_20260428_train.h5

# 2. Latent archives: constant schedules (base model) and time-varying schedules (fine-tuning)
python build_latents.py --autoencoder-config config/autoencoder.json \
    --autoencoder checkpoints/best_model_ae_20260614_234059.pth --off-position interp \
    --sources ../lpbf_dataset_overhang_20260428_train.h5:const_ \
    --out ../latent_overhang_20260428_ae0614_interp.h5
python build_latents.py --autoencoder-config config/autoencoder.json \
    --autoencoder checkpoints/best_model_ae_20260614_234059.pth \
    --sources ../lpbf_dataset_control_overhang_20260614_train.h5:ctrl_ \
              ../lpbf_dataset_band_overhang_20260617_train.h5:band_ \
    --out ../latent_combined_20260616_train.h5

# 3a. Base transition model on constant schedules
accelerate launch --config_file config/accelerate_4gpu.yaml train_transition.py \
    --config config/transition_base.json --data ../latent_overhang_20260428_ae0614_interp.h5 \
    --out-dir checkpoints/M0_const

# 3b. Fine-tune the base model on time-varying schedules
accelerate launch --config_file config/accelerate_4gpu.yaml train_transition.py \
    --config config/transition_finetune_n100.json --data ../latent_combined_20260616_train.h5 \
    --init checkpoints/M0_const/transition_<start>.pth --out-dir checkpoints/M1e_n100

# 4. Rollout evaluation
python evaluate_rollout.py --data ../lpbf_dataset_overhang_20260421_test.h5 --cases case_v1_p3_0 \
    --autoencoder-config config/autoencoder.json \
    --autoencoder checkpoints/best_model_ae_20260614_234059.pth \
    --transition-config config/transition_finetune_n100.json \
    --transition M1e_n100=checkpoints/M1e_n100/best_dit_stepper_20260923_130449.pth \
    --n-ens 8 --out results/example

# 5. Control (MPC; run_policy.py takes the same arguments plus --actor)
python run_mpc.py --autoencoder-config config/autoencoder.json \
    --autoencoder checkpoints/best_model_ae_20260614_234059.pth \
    --transition-config config/transition_finetune_n100.json \
    --transition checkpoints/M1e_n100/best_dit_stepper_20260923_130449.pth \
    --data ../lpbf_dataset_overhang_20260421_test.h5 --case case_v1_p3_0
```

Step 3b starts from the base model. `--init` takes the best checkpoint of step 3a, `transition_<start>.pth`. Without `--init`, fine-tuning starts from the released base model, `checkpoints/M0_const/best_dit_stepper_20260914_135158.pth`, which is the `init_checkpoint` of every `config/transition_finetune_n*.json`. Each script lists its options with `--help`.

## Models

All models share the frozen autoencoder `checkpoints/best_model_ae_20260614_234059.pth`.

| Model | Config |
|---|---|
| Base DiT (constant schedules) | `config/transition_base.json` |
| Fine-tuned DiT, n = 10–100 (n = 100 used for control) | `config/transition_finetune_n*.json` |
| ViT / Transolver baselines (load only) | `config/baseline_{vit,transolver}.json` |
| Pure Transient reference (load only) | `config/transition_pure_transient.json` |
| SHAC policy actor | `checkpoints/shac_actor_mpcobjc.pt` |

## Citation

```bibtex
@article{yan2026generative,
  title   = {Generative World Models Enable Predictive Control of Laser Melt Pool Dynamics},
  author  = {Yan, Yiyang and Bambach, Markus and Afrasiabi, Mohamadreza},
  journal = {arXiv preprint arXiv:2610.06250},
  year    = {2026}
}
```

## Contact

For questions or support, please contact:
- **Author**: Yiyang Yan
- **Corresponding author**: Mohamadreza Afrasiabi (afrasiabi@ethz.ch)
- **Institution**: Advanced Manufacturing Lab, ETH Zurich
