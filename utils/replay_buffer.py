from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from deep_spi import Storage

class ReplayBuffer:
    """
    CPU 2-D ring buffer over time with fixed width (number of streams / envs).
    Shapes: (T_cap, W_cap, ...). Total capacity = T_cap * W_cap.
    """

    def __init__(self, T_cap: int, example: Storage, *, use_mirror: bool = False):
        """
        Args:
            T_cap: time capacity (integer B_T_cap).
            example: a Storage whose arrays have shape (1, W_cap, ...).
                     We infer dtypes and trailing shapes from it; only W_cap is used.
        """
        assert T_cap > 1
        # Infer W_cap from example
        leaves = jtu.tree_leaves(example)
        assert len(leaves) > 0, "Example Storage must have at least one array."
        W_cap = int(np.array(leaves[0]).shape[1])
        self.T_cap = int(T_cap)
        self.W_cap = int(W_cap)
        self.use_mirror = bool(use_mirror)

        # Allocate CPU arrays for each stored field
        def alloc(x):
            x = np.asarray(x)
            trailing = x.shape[2:]  # drop (1, W_cap)
            if self.use_mirror:  # <-- [B2]
                return np.empty((2 * self.T_cap, self.W_cap) + trailing, dtype=x.dtype)
            else:
                return np.empty((self.T_cap, self.W_cap) + trailing, dtype=x.dtype)

        self._cpu = jax.tree_map(alloc, example)
        if hasattr(example, "returns") and example.returns is not None:
            self._cpu = self._cpu.replace(returns=None)
        if hasattr(example, "advantages") and example.advantages is not None:
            self._cpu = self._cpu.replace(advantages=None)

        # Ring pointers and size (over time dimension only)
        self.t_write = 0          # next write time index in [0, T_cap)
        self.t_size = 0           # how many valid timesteps currently stored (<= T_cap)

    @property
    def size(self) -> int:
        """Total number of elements currently stored."""
        return self.t_size * self.W_cap

    # --------- Append ---------

    def add(self, batch: Storage):
        """
        Append a batch with shape (T_new, W_cap, ...).
        Width must match W_cap; time can be any T_new >= 1.
        """
        # Basic validations
        bw = jtu.tree_leaves(batch)
        T_new = int(np.asarray(bw[0]).shape[0])
        assert T_new > 0
        assert int(np.asarray(bw[0]).shape[1]) == self.W_cap, "Width mismatch."

        start = self.t_write
        end = (start + T_new) % self.T_cap

        if not self.use_mirror:

            if start + T_new <= self.T_cap:
                # Single segment write
                self._cpu = jax.tree_map(
                    lambda dst, src: (dst.__setitem__(slice(start, start + T_new), np.asarray(src)), dst)[1],
                    self._cpu, batch
                )
            else:
                # Wrap-around write
                first = self.T_cap - start
                second = T_new - first
                self._cpu = jax.tree_map(
                    lambda dst, src: (
                        dst.__setitem__(slice(start, self.T_cap), np.asarray(src)[:first]),
                        dst.__setitem__(slice(0, second), np.asarray(src)[first:]),
                        dst
                    )[2],
                    self._cpu, batch
                )

        else:
            if start + T_new <= self.T_cap:
                def write2(dst, src):
                    src = np.asarray(src)
                    dst[start:start + T_new] = src
                    dst[start + self.T_cap: start + self.T_cap + T_new] = src
                    return dst

                self._cpu = jax.tree_map(write2, self._cpu, batch)
            else:
                first = self.T_cap - start
                second = T_new - first

                def write_wrap2(dst, src):
                    src = np.asarray(src)
                    # tail segment
                    dst[start:self.T_cap] = src[:first]
                    dst[start + self.T_cap: self.T_cap + self.T_cap] = src[:first]
                    # head segment
                    dst[0:second] = src[first:]
                    dst[self.T_cap: self.T_cap + second] = src[first:]
                    return dst

                self._cpu = jax.tree_map(write_wrap2, self._cpu, batch)

        self.t_write = end
        self.t_size = min(self.T_cap, self.t_size + T_new)

    # --------- Sampling ---------
    def sample(
            self,
            rng: jax.random.PRNGKey,
            length: int,
            width: int,
            *,
            avoid_cross_terminal: bool = False,
            compute_next_from_obs: bool = True,
            device: Optional[jax.Device] = None,
    ) -> Storage:
        assert length >= 1 and width >= 1

        rng_length, rng_width = jax.random.split(rng, 2)

        # If we compute next_*, we need one extra time step after the block
        min_needed = length + (1 if compute_next_from_obs else 0)

        # Must have enough valid time steps currently in the buffer
        assert self.t_size >= min_needed

        # S = how many valid rows in time; 'oldest' is the absolute ring index of the oldest row
        S = self.t_size
        oldest = (self.t_write - S) % self.T_cap

        # Choose the device once (GPU if available)
        if device is None:
            gpus = [d for d in jax.local_devices() if d.platform == "gpu"]
            device = gpus[0] if gpus else jax.local_devices()[0]

        # Draw a start time RELATIVE to 'oldest' so we can compute absolute ring indices
        max_start = S - min_needed
        t_start_rel = int(jax.random.randint(rng_length, (), 0, max_start + 1))

        # -------- Mirror path --------
        if self.use_mirror:
            base = oldest + t_start_rel
            t_slice_abs = slice(base, base + length)

            # columns
            if width == self.W_cap:
                cols = np.arange(self.W_cap, dtype=np.int32)
            elif width < self.W_cap:
                cols = np.asarray(jax.random.choice(rng_width, self.W_cap, (width,), replace=False))
                cols = np.sort(cols).astype(np.int32, copy=True)
            else:
                cols = np.asarray(jax.random.randint(rng_width, (width,), 0, self.W_cap))
                cols = np.sort(cols).astype(np.int32, copy=True)

            if avoid_cross_terminal:
                dones = np.asarray(self._cpu.dones)
                db = dones[t_slice_abs, :][:, cols]
                if db.ndim > 2:
                    db = np.any(db.reshape(length, width, -1), axis=2)
                if np.any(db[:-1]):
                    rng, r1 = jax.random.split(rng)
                    if width == self.W_cap:
                        cols = np.arange(self.W_cap, dtype=np.int32)
                    elif width < self.W_cap:
                        cols = np.asarray(jax.random.choice(r1, self.W_cap, (width,), replace=False))
                        cols = np.sort(cols).astype(np.int32, copy=True)
                    else:
                        cols = np.asarray(jax.random.randint(r1, (width,), 0, self.W_cap))
                        cols = np.sort(cols).astype(np.int32, copy=True)

            def take_block_mirror(x):
                x = np.asarray(x)  # shape (2*T_cap, W, …)
                return x[t_slice_abs, cols]

            block_cpu = jax.tree_map(take_block_mirror, self._cpu)

            next_obs = next_dones = None
            if compute_next_from_obs:
                obs_all = np.asarray(self._cpu.obs)
                dones_all = np.asarray(self._cpu.dones)
                next_obs = obs_all[slice(base + 1, base + 1 + length), cols]
                next_dones = dones_all[slice(base + 1, base + 1 + length), cols]

            to_dev = lambda a: jax.device_put(a, device=device)
            block_gpu = jax.tree_map(to_dev, block_cpu)

            out = block_gpu
            if hasattr(self._cpu, "returns"):
                out = out.replace(returns=None)
            if hasattr(self._cpu, "advantages"):
                out = out.replace(advantages=None)
            if compute_next_from_obs:
                out = out.replace(
                    next_obs=jax.device_put(next_obs, device=device),
                    next_dones=jax.device_put(next_dones, device=device),
                )
            return out

        # Build the absolute ring indices for the block using modular arithmetic (no concat)
        t_rel = np.arange(length, dtype=np.int32)  # (L,)
        t_idx_rel = t_start_rel + t_rel  # (L,)
        t_idx_abs = (oldest + t_idx_rel) % self.T_cap  # (L,)

        # If next_* is requested, the "next time" indices are just +1 modulo T_cap
        if compute_next_from_obs:
            t_idx_abs_next = (t_idx_abs + 1) % self.T_cap  # (L,)

        # Pick the width columns efficiently
        if width == self.W_cap:
            cols = np.arange(self.W_cap, dtype=np.int32)  # fast path: take all columns
        elif width < self.W_cap:
            cols = np.asarray(jax.random.choice(rng_width, self.W_cap, (width,), replace=False))
            cols = np.sort(cols).astype(np.int32, copy=True)  # make writable
        else:
            cols = np.asarray(jax.random.randint(rng_width, (width,), 0, self.W_cap))
            cols = np.sort(cols).astype(np.int32, copy=True)

        # forbid terminals inside the first length-1 rows (no copy; direct modular gather)
        if avoid_cross_terminal:
            dones = np.asarray(self._cpu.dones)  # (T_cap, W, …)
            db = dones[t_idx_abs[:, None], cols[None, ...]]  # (L, width, …)
            if db.ndim > 2:
                db = np.any(db.reshape(length, width, -1), axis=2)
            if np.any(db[:-1]):
                # simple one-resample of columns if invalid
                rng, r1 = jax.random.split(rng)
                if width == self.W_cap:
                    cols = np.arange(self.W_cap, dtype=np.int32)
                elif width < self.W_cap:
                    cols = np.asarray(jax.random.choice(r1, self.W_cap, (width,), replace=False))
                    cols = np.sort(cols).astype(np.int32, copy=True)
                else:
                    cols = np.asarray(jax.random.randint(r1, (width,), 0, self.W_cap))
                    cols = np.sort(cols).astype(np.int32, copy=True)

        # Gather the block directly from the ring by absolute time/column indices (no linear view)
        def take_block(x):
            x = np.asarray(x)  # (T_cap, W, …)
            x_t = x[t_idx_abs]  # (L, W, …)
            return x_t[:, cols]  # (L, width, …)

        block_cpu = jax.tree_map(take_block, self._cpu)

        # Derive next_* using the +1 absolute indices (again no copies of big buffers)
        next_obs = next_dones = None
        if compute_next_from_obs:
            obs_all = np.asarray(self._cpu.obs)
            dones_all = np.asarray(self._cpu.dones)
            next_obs = obs_all[t_idx_abs_next][:, cols]  # (L, width, …)
            next_dones = dones_all[t_idx_abs_next][:, cols]  # (L, width, …)

        # Move the sampled block to the device (usually GPU)
        to_dev = lambda a: jax.device_put(a, device=device)
        block_gpu = jax.tree_map(to_dev, block_cpu)

        # Optionally synthesize returns/advantages only if your Storage schema has them
        def zeros_like_if_present(x_spec):
            if x_spec is None:
                return None
            # Choose dtype from spec array if you have one; default float32
            dtype = np.asarray(x_spec).dtype if hasattr(x_spec, "dtype") else np.float32
            return jax.device_put(jnp.zeros((length, width) + np.asarray(x_spec).shape[2:], dtype=dtype),
                                  device=device)

        out = block_gpu
        if hasattr(self._cpu, "returns"):
            out = out.replace(returns=zeros_like_if_present(getattr(self._cpu, "returns")))
        if hasattr(self._cpu, "advantages"):
            out = out.replace(advantages=zeros_like_if_present(getattr(self._cpu, "advantages")))

        # Attach computed next_* if your Storage dataclass defines them
        if compute_next_from_obs:
            out = out.replace(
                next_obs=jax.device_put(next_obs, device=device),
                next_dones=jax.device_put(next_dones, device=device),
            )

        return out