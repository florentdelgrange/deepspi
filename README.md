# Deep SPI: Safe Policy Improvement via World Models

[![arXiv](https://img.shields.io/badge/arXiv-2510.12312-b31b1b.svg)](https://arxiv.org/abs/2510.12312)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)

This is the official implementation of **Deep SPI** from the paper:
> **Deep SPI: Safe Policy Improvement via World Models** (ICLR 2026)  
> Florent Delgrange, Raphaël Avalos, Willem Röpke  
> [Paper](https://arxiv.org/abs/2510.12312) | [Blog Post](https://florentdelgrange.com/post/deep_spi/) | [Project Page](https://florentdelgrange.com/project/deep_spi/)

## 🎯 The Problem

When you train a deep RL policy with auxiliary losses to improve the representation (the observation encoder before your value/critic heads), you face a critical timing problem:

The representation optimized under your behavioral policy may not be reliable for the next policy. Add auxiliary losses to regularize the latent space, improve the policy, and suddenly the encoder has shifted, invalidating the very representation you were relying on.

## 💡 The Solution

Deep SPI couples world-model learning with controlled policy updates. The key insight: improve the policy **step by step, in a neighborhood that keeps it close to regions where the world model is well-calibrated**.

This way, updates that look good in the model actually translate to improvements in the real environment.

## 🔑 Key Features

- **Monotonic improvement bounds**: Formal guarantees that each policy update improves the true environment
- **Representation stability**: Encoder doesn't degrade as the policy improves
- **Practical efficiency**: Works in online settings without pre-collected datasets
- **Competitive performance**: Matches or exceeds PPO and DeepMDPs on ALE-57
- **Fully reproducible**: JAX-based with locked dependencies and comprehensive benchmarks

For a detailed algorithm explanation, see the [paper](https://arxiv.org/abs/2510.12312), [blog post](https://florentdelgrange.com/post/deep_spi/), or [project page](https://florentdelgrange.com/project/deep_spi/).

## ⚡ Quick Start

### Requirements
- Python 3.10
- `uv` package manager (recommended for exact reproducibility)

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/florentdelgrange/deepspi.git
cd deepspi
```

2. **Set up Python 3.10 environment**:
```bash
uv python install 3.10
uv venv .venv --python 3.10
source .venv/bin/activate
```

3. **Install dependencies** (with exact reproducibility via lock file):
```bash
export UV_FIND_LINKS=https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
uv sync --active --frozen --python 3.10
```

4. **Export CUDA/cuDNN libraries for JAX**:
```bash
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:$PWD/.venv/lib/python3.10/site-packages/nvidia/cublas/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset JAX_PLATFORMS JAX_PLATFORM_NAME
```

5. **Verify GPU detection**:
```bash
python - <<'PY'
import jax
print("devices:", jax.devices())
print("backend:", jax.default_backend())
PY
```

### Training

Run Deep SPI training on Atari (e.g., Breakout):

```bash
python deep_spi.py --env-id=Breakout-v5 --total-timesteps=10000000
```

Key hyperparameters:
- `--num-envs`: Number of vectorized environments (default: 128)
- `--num-steps`: Steps per environment before update (default: 8)
- `--learning-rate`: Policy learning rate (default: 2.5e-4)

## 📊 Results

Deep SPI is competitive with strong baselines on ALE-57 (Atari 57) while maintaining formal improvement guarantees.

For full results and benchmark analysis, see the [paper](https://arxiv.org/abs/2510.12312).

## 📚 Code Structure

```
deepspi/
├── deep_spi.py                          # Main Deep SPI training script
├── deep_mdp_ppo.py                      # DeepMDPs baseline implementation
├── dream_spi.py                         # Dream environment variant
├── ppo_atari_envpool_xla_jax_scan.py    # PPO baseline for comparison
├── networks/                            # Network architectures
│   ├── architectures.py                 # Core network modules
│   └── lipschitz.py                     # Lipschitz-constrained layers
├── envs/                                # Environment utilities
│   └── confounding_gridworld.py         # Toy environment
├── utils/                               # Utility functions
│   ├── loss.py                          # Loss functions
│   ├── distributions.py                 # Probability distributions
│   ├── cleanrl_atari_wrappers.py        # Atari environment wrappers
│   ├── activations.py                   # Custom activation functions
│   └── logs.py                          # Logging utilities
└── pyproject.toml                       # Project configuration
```

## 🛠️ Troubleshooting

**JAX falls back to CPU** (`No GPU/TPU found`):
- Check that `LD_LIBRARY_PATH` includes:
  - `.venv/.../nvidia/cudnn/lib`
  - `.venv/.../nvidia/cublas/lib`
  - `/usr/local/cuda/lib64`
- Verify `JAX_PLATFORMS` is not set to `cpu`

**Atari environment unavailable**:
```bash
python - <<'PY'
import envpool
envs = envpool.list_all_envs()
print(f"Breakout-v5 available: {'Breakout-v5' in envs}")
PY
```

**Lock file validation**:
```bash
uv lock --check  # Check if uv.lock matches pyproject.toml
uv lock --python 3.10  # Regenerate lock if needed
```

## 📖 Citation

If you use Deep SPI in your research, please cite:

```bibtex
@inproceedings{delgrange2026deepspi,
  title={Deep SPI: Safe Policy Improvement via World Models},
  author={Delgrange, Florent and Avalos, Rapha\"{e}l and R\"{o}pke, Willem},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Florent Delgrange** — [Homepage](https://florentdelgrange.com) | [GitHub](https://github.com/florentdelgrange)
- **Raphaël Avalos** — [Homepage](https://avalos.fr)
- **Willem Röpke** — [Homepage](https://wilrop.github.io)

## 💬 Questions?

Please open an [issue](https://github.com/florentdelgrange/deepspi/issues) or contact the authors.
