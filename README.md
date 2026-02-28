## Installation

### Requirements
- Python `3.10`
- `uv`

Tested on Linux x86_64, as required for `envpool` official wheels

## Reproducible GPU Setup With `uv` (Recommended)

This repository includes a checked-in `uv.lock`. The commands below use that lockfile to recreate the exact dependency set.

1) Create and activate a Python 3.10 environment:
```bash
uv python install 3.10
uv venv .venv --python 3.10
source .venv/bin/activate
```

2) Install from lock/spec into the active `.venv`:
```bash
export UV_FIND_LINKS=https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
uv sync --active --frozen --python 3.10
```

3) Export CUDA/cuDNN runtime libraries for JAX:
```bash
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:$PWD/.venv/lib/python3.10/site-packages/nvidia/cublas/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset JAX_PLATFORMS
unset JAX_PLATFORM_NAME
```

4) Verify GPU is detected by JAX:
```bash
python - <<'PY'
import jax
print("devices:", jax.devices())
print("backend:", jax.default_backend())
PY
```

## Validate Or Refresh Lock File

Validate that `uv.lock` matches `pyproject.toml`:
```bash
uv lock --check
```

If you need to regenerate the lock:
```bash
export UV_FIND_LINKS=https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
uv lock --python 3.10
```
```

## Notes / Troubleshooting

- If JAX falls back to CPU (`No GPU/TPU found`), check:
  - `LD_LIBRARY_PATH` includes `.venv/.../nvidia/cudnn/lib`, `.venv/.../nvidia/cublas/lib`, and `/usr/local/cuda/lib64`
  - `JAX_PLATFORMS` is not set to `cpu`
- If `Breakout-v5` is unavailable, default Atari runs fail at environment creation. Check:
```bash
python - <<'PY'
import envpool
envs = envpool.list_all_envs()
print("num envs:", len(envs))
print("Breakout-v5 available:", "Breakout-v5" in envs)
PY
```
