# gpu_replay_buffer.py
# A JAX-first, GPU-resident replay buffer optimized for memory efficiency and JIT performance.
#
# Updates in this version:
# - Auto-narrow actions to uint8 when action_n_classes <= 256, unless you explicitly set action_dtype.
# - Store and return logprobs on device (no placeholders). Toggle via RBConfig.store_logprobs.
#
import inspect
from functools import partial
from typing import Optional, Tuple, Dict, Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

# Import Storage to maintain compatibility with deep_spi/dream_spi training code
from deep_spi import Storage
import utils.flashbax as fbx


@struct.dataclass
class RBConfig:
    T_cap: int                        # time capacity (ring over time)
    W_cap: int                        # width capacity (number of env streams)
    store_next_obs: bool = False      # store next_obs explicitly (uses more memory)
    store_logprobs: bool = True       # store logprobs explicitly (user requires this)
    # Dtypes (None => choose sensible defaults below)
    obs_dtype: Any = jnp.uint8
    reward_dtype: Any = jnp.float32
    action_dtype: Optional[Any] = None   # if None and action_n_classes<=256 -> uint8 else int32
    done_dtype: Any = jnp.bool_
    logprob_dtype: Any = jnp.float32
    # Action space size hint for auto-narrowing
    action_n_classes: Optional[int] = None
    original_action_dtype: Optional[Any] = None  # for reference only, not used
    # Device placement
    device: Optional[jax.Device] = None


@struct.dataclass
class DeviceReplayBuffer:
    # Main storage (GPU-resident)
    obs: jnp.ndarray                  # (T_cap, W_cap, *obs_shape)
    actions: jnp.ndarray              # (T_cap, W_cap, *action_shape or scalar)
    rewards: jnp.ndarray              # (T_cap, W_cap, ...)
    dones: jnp.ndarray                # (T_cap, W_cap, ...)
    logprobs: Optional[jnp.ndarray]   # (T_cap, W_cap, ...) if stored

    # Static config
    config: RBConfig = struct.field(pytree_node=False)

    # Optional extras
    values: Optional[jnp.ndarray] = None
    returns: Optional[jnp.ndarray] = None
    advantages: Optional[jnp.ndarray] = None
    next_obs: Optional[jnp.ndarray] = None
    next_dones: Optional[jnp.ndarray] = None

    # Ring pointers & sizes (over time dimension)
    ptr_t: jnp.int32 = jnp.int32(0)   # next write position along T
    size_t: jnp.int32 = jnp.int32(0)  # how many valid time steps currently filled (<= T_cap)

    # Cached shapes (static metadata, non-pytree)
    obs_shape: Tuple[int, ...] = struct.field(pytree_node=False, default=())
    action_shape: Tuple[int, ...] = struct.field(pytree_node=False, default=())
    reward_shape: Tuple[int, ...] = struct.field(pytree_node=False, default=())
    done_shape: Tuple[int, ...] = struct.field(pytree_node=False, default=())
    logprob_shape: Tuple[int, ...] = struct.field(pytree_node=False, default=())

    @property
    def size(self) -> int:
        """Total number of elements currently stored."""
        return int(self.size_t) * int(self.config.W_cap)


def _maybe_shape(x) -> Optional[Tuple[int, ...]]:
    if x is None:
        return None
    s = tuple(x.shape)
    assert len(s) >= 2, "Storage example arrays must be at least 2D (T, W, ...)."
    return s[2:]


def _infer_shapes_from_example(example: Storage) -> Dict[str, Optional[Tuple[int, ...]]]:
    # example fields are shaped (1, W, ...). We keep the trailing shape after (T, W)
    return dict(
        obs_shape=_maybe_shape(example.obs),
        action_shape=_maybe_shape(example.actions),
        reward_shape=_maybe_shape(example.rewards),
        done_shape=_maybe_shape(example.dones),
        logprob_shape=_maybe_shape(example.logprobs),
    )


def init_replay_buffer(config: RBConfig, example: Storage, device: Optional[jax.Device] = None) -> DeviceReplayBuffer:
    """Initialize a GPU-resident replay buffer based on an example Storage batch.
    The example must have leading dims (T=1, W=config.W_cap, ...)."""
    if device is None:
        devs = [d for d in jax.local_devices() if d.platform == "gpu"]
        device = devs[0] if devs else jax.local_devices()[0]

    shapes = _infer_shapes_from_example(example)

    T, W = int(config.T_cap), int(config.W_cap)
    assert int(example.obs.shape[1]) == W, f"Example width {example.obs.shape[1]} != config.W_cap {W}"
    assert T > 1 and W > 0

    # Choose action dtype if not provided
    action_dtype = config.action_dtype
    if action_dtype is None:
        if (config.action_n_classes is not None) and (config.action_n_classes <= 256):
            action_dtype = jnp.uint8
        else:
            action_dtype = jnp.int32

    if example.actions is not None:
        original_action_dtype = example.actions.dtype
    else:
        original_action_dtype = None

    config = config.replace(action_dtype=action_dtype, original_action_dtype=original_action_dtype, device=device)

    def zeros(shape, dtype):
        with jax.default_device(device):
            return jnp.zeros((T, W) + tuple(shape), dtype=dtype)


    obs = zeros(shapes["obs_shape"], config.obs_dtype)
    actions = zeros(shapes["action_shape"], action_dtype)
    rewards = zeros(shapes["reward_shape"], config.reward_dtype)
    dones = zeros(shapes["done_shape"], config.done_dtype)

    next_obs = zeros(shapes["obs_shape"], config.obs_dtype) if config.store_next_obs else None
    next_dones = zeros(shapes["done_shape"], config.done_dtype) if config.store_next_obs else None

    logprobs = zeros(shapes["logprob_shape"], config.logprob_dtype) if config.store_logprobs and (shapes["logprob_shape"] is not None) else None

    rb = DeviceReplayBuffer(
        obs=obs, actions=actions, rewards=rewards, dones=dones, logprobs=logprobs,
        values=None, returns=None, advantages=None,
        next_obs=next_obs, next_dones=next_dones,
        ptr_t=jnp.int32(0), size_t=jnp.int32(0),
        config=config,
        obs_shape=shapes["obs_shape"] or (),
        action_shape=shapes["action_shape"] or (),
        reward_shape=shapes["reward_shape"] or (),
        done_shape=shapes["done_shape"] or (),
        logprob_shape=shapes["logprob_shape"] or (1,),
    )
    return rb


def _two_segment_slices(ptr_t: jnp.int32, T_cap: int, T_new: int):
    """Compute (len1, start1) and (len2, start2) for writing a contiguous T_new block
    into a ring buffer of size T_cap starting at ptr_t along the time axis."""
    len1 = jnp.minimum(T_new, T_cap - ptr_t)
    len2 = T_new - len1
    return (len1, ptr_t), (len2, jnp.int32(0))

def _update_ring_time(x: jnp.ndarray, values: jnp.ndarray, ptr_t: jnp.int32) -> jnp.ndarray:
    """Write `values` with shape (T_new, W, ...) into `x` at time indices
    [ptr_t, ptr_t+1, ..., ptr_t+T_new-1] modulo T_cap, using a single scatter.
    Works with dynamic ptr_t under jit/pjit."""
    T_new, T_cap = values.shape[0], x.shape[0]
    t_idx = (ptr_t + jnp.arange(T_new, dtype=jnp.int32)) % jnp.int32(T_cap)
    return x.at[t_idx, ...].set(values)


@partial(jax.jit, donate_argnums=(0,))   # donate the 0th positional arg → rb
def add_batch(
    rb: DeviceReplayBuffer,
    batch: Storage,                        # (T_new, W_cap, ...)
):
    """Append a batch of size (T_new, W_cap, ...). Width must match W_cap.
    Returns (new_rb, info) with info including new ptr/size for logging.
    This is a pure JAX function suitable for jit/scan."""
    T_new = batch.obs.shape[0]
    W = rb.config.W_cap
    assert batch.obs.shape[1] == W, "Width mismatch."

    # Coerce dtypes for storage to match configured saving dtypes (cast on write to save memory)
    obs_in = batch.obs.astype(rb.obs.dtype)
    act_in = batch.actions.astype(rb.actions.dtype)
    rew_in = batch.rewards.astype(rb.rewards.dtype)
    don_in = batch.dones.astype(rb.dones.dtype)

    # Write each field with at most two dynamic_update_slice ops
    new_obs = _update_ring_time(rb.obs, obs_in, rb.ptr_t)
    new_actions = _update_ring_time(rb.actions, act_in, rb.ptr_t)
    new_rewards = _update_ring_time(rb.rewards, rew_in, rb.ptr_t)
    new_dones = _update_ring_time(rb.dones, don_in, rb.ptr_t)

    # Optional fields
    if rb.config.store_next_obs and (batch.next_obs is not None):
        nobs_in = batch.next_obs.astype(rb.obs.dtype)
        new_next_obs = _update_ring_time(rb.next_obs, nobs_in, rb.ptr_t)
    else:
        new_next_obs = rb.next_obs

    if rb.config.store_next_obs and (batch.next_dones is not None):
        ndon_in = batch.next_dones.astype(rb.dones.dtype)
        new_next_dones = _update_ring_time(rb.next_dones, ndon_in, rb.ptr_t)
    else:
        new_next_dones = rb.next_dones

    if rb.config.store_logprobs and (batch.logprobs is not None) and (rb.logprobs is not None):
        lp_in = batch.logprobs.astype(rb.logprobs.dtype)
        new_logprobs = _update_ring_time(rb.logprobs, lp_in, rb.ptr_t)
    else:
        new_logprobs = rb.logprobs

    # Advance pointers/sizes
    T_cap = rb.config.T_cap
    new_ptr = (rb.ptr_t + jnp.int32(T_new)) % jnp.int32(T_cap)
    new_size = jnp.minimum(rb.size_t + jnp.int32(T_new), jnp.int32(T_cap))

    new_rb = rb.replace(
        obs=new_obs, actions=new_actions, rewards=new_rewards, dones=new_dones,
        logprobs=new_logprobs,
        next_obs=new_next_obs, next_dones=new_next_dones,
        ptr_t=new_ptr, size_t=new_size
    )
    info = dict(ptr_t=new_ptr, size_t=new_size)
    return new_rb, info


def _valid_window_bounds(ptr_t: jnp.int32, size_t: jnp.int32, T_cap: int) -> Tuple[jnp.int32, jnp.int32]:
    """Compute [oldest, newest] (inclusive) absolute time indices currently valid in the ring."""
    oldest = (ptr_t - size_t) % T_cap
    newest = (ptr_t - 1) % T_cap
    return jnp.int32(oldest), jnp.int32(newest)


def _rel_to_abs(oldest: jnp.int32, rel: jnp.ndarray, T_cap: int) -> jnp.ndarray:
    return (oldest + rel) % jnp.int32(T_cap)


@partial(jax.jit, static_argnames=("batch_size", "compute_next_from_obs", "avoid_cross_terminal"))
def sample_batch(
    rb: DeviceReplayBuffer,
    rng: jax.Array,
    batch_size: int,
    *,
    compute_next_from_obs: bool = True,
    avoid_cross_terminal: bool = False,
):
    """Uniformly sample (batch_size,) independent transitions (1-step). Returns updated rb (same ptr/size)
    and a Storage on the same device.
      - If compute_next_from_obs=True and rb.config.store_next_obs=False, next_obs/next_dones are computed on the fly.
      - If avoid_cross_terminal=True, transitions where dones[t,w] is True are avoided via one-shot re-sample of columns.
    """
    B = int(batch_size)

    T_cap = rb.config.T_cap
    W_cap = rb.config.W_cap

    oldest, newest = _valid_window_bounds(rb.ptr_t, rb.size_t, T_cap)

    # If we compute next_*, we must avoid selecting the very last valid time index (no next exists there).
    need_next = jnp.array(compute_next_from_obs and (not rb.config.store_next_obs))
    max_rel = rb.size_t - jnp.where(need_next, 1, 0)  # number of valid relative time positions
    max_rel = jnp.maximum(max_rel, 1)

    rng_t, rng_w = jax.random.split(rng, 2)

    # Sample relative time offsets in [0, max_rel-1], then map to absolute ring indices.
    rel_t = jax.random.randint(rng_t, (B,), 0, max_rel)
    t_idx = _rel_to_abs(oldest, rel_t, T_cap)  # (B,)

    # Sample width indices
    w_idx = jax.random.randint(rng_w, (B,), 0, W_cap)

    # Optionally avoid terminals (done=True). We do a one-shot resample of columns where needed.
    if avoid_cross_terminal:
        dones_at = rb.dones[t_idx, w_idx]
        rng_w2 = jax.random.fold_in(rng_w, 1)
        w_alt = jax.random.randint(rng_w2, (B,), 0, W_cap)
        w_idx = jnp.where(dones_at, w_alt, w_idx)

    # Gather fields
    obs = rb.obs[t_idx, w_idx]
    actions = rb.actions[t_idx, w_idx]
    if rb.config.original_action_dtype is not None and (rb.actions.dtype != rb.config.original_action_dtype):
        actions = actions.astype(rb.config.original_action_dtype)

    rewards = rb.rewards[t_idx, w_idx]
    dones = rb.dones[t_idx, w_idx]
    logprobs = rb.logprobs[t_idx, w_idx] if (rb.logprobs is not None) else None

    # Prepare next_* (either from storage or computed on the fly)
    if rb.config.store_next_obs and (rb.next_obs is not None):
        next_obs = rb.next_obs[t_idx, w_idx]
        next_dones = rb.next_dones[t_idx, w_idx] if (rb.next_dones is not None) else dones
    elif compute_next_from_obs:
        t_next = (t_idx + 1) % T_cap
        next_obs = rb.obs[t_next, w_idx]
        next_dones = dones
    else:
        next_obs = obs
        next_dones = dones

    def add_leading_batch(x):  # ensure shape is (B, 1, ...)
        if x is None:
            return None
        s = x.shape
        return x.reshape((B, 1) + s[1:]) if (x.ndim >= 1) else x

    out = Storage(
        obs=add_leading_batch(obs.astype(jnp.float32) if rb.obs.dtype == jnp.uint8 else obs),
        actions=add_leading_batch(actions),
        logprobs=add_leading_batch(logprobs) if logprobs is not None else None,
        dones=add_leading_batch(dones),
        values=None,
        advantages=None,
        returns=None,
        rewards=add_leading_batch(rewards.astype(jnp.float32) if rb.rewards.dtype != jnp.float32 else rewards),
        next_obs=add_leading_batch(next_obs.astype(jnp.float32) if rb.obs.dtype == jnp.uint8 else next_obs),
        next_dones=add_leading_batch(next_dones),
        hist_logprobs=None,
    )
    return out

# decorator
@partial(jax.jit, static_argnames=("batch_width","batch_length","avoid_cross_terminal"))
def sample_from_rb(replay_buffer, rng, batch_width, batch_length, *, avoid_cross_terminal=False):
    rb = replay_buffer
    W = int(batch_width); L = int(batch_length)
    T_cap = rb.config.T_cap; W_cap = rb.config.W_cap

    oldest, newest = _valid_window_bounds(rb.ptr_t, rb.size_t, T_cap)

    need_next = jnp.array((not rb.config.store_next_obs))
    contig_needed = L + jnp.where(need_next, 1, 0)
    max_rel = jnp.maximum(rb.size_t - contig_needed + 1, 1)

    rng, rng_w = jax.random.split(rng); rng, rng_t = jax.random.split(rng)
    w_idx  = jax.random.randint(rng_w, (W,), 0, W_cap)
    rel_t0 = jax.random.randint(rng_t, (W,), 0, max_rel)
    t0     = _rel_to_abs(oldest, rel_t0, T_cap)                         # (W,)

    # build indices directly in (L, W) layout
    offs = jnp.arange(L, dtype=jnp.int32)[:, None]                      # (L, 1)
    t_mat = (t0[None, :] + offs) % jnp.int32(T_cap)                     # (L, W)

    if avoid_cross_terminal:
        early = t_mat[:-1, :]                                           # (L-1, W)
        dones_early = rb.dones[early, w_idx[None, :]]
        if dones_early.ndim > 2:
            dones_early = dones_early.reshape(dones_early.shape[0], dones_early.shape[1], -1)
            dones_early = jnp.any(dones_early, axis=2)
        bad = jnp.any(dones_early, axis=0)                              # (W,)
        rng, rng_t2 = jax.random.split(rng)
        rel_alt = jax.random.randint(rng_t2, (W,), 0, max_rel)
        t0_alt = _rel_to_abs(oldest, rel_alt, T_cap)
        t0 = jnp.where(bad, t0_alt, t0)
        t_mat = (t0[None, :] + offs) % jnp.int32(T_cap)

    def g(a): return None if a is None else a[t_mat, w_idx[None, :]]

    obs      = g(rb.obs)
    actions  = g(rb.actions)
    rewards  = g(rb.rewards)
    dones    = g(rb.dones)
    logprobs = g(rb.logprobs) if (rb.logprobs is not None) else None

    if rb.config.store_next_obs and (rb.next_obs is not None):
        next_obs   = g(rb.next_obs)
        next_dones = g(rb.next_dones) if (rb.next_dones is not None) else dones
    else:
        t_mat_next = (t_mat + 1) % jnp.int32(T_cap)
        next_obs   = rb.obs[t_mat_next, w_idx[None, :]]
        next_dones = dones

    if rb.obs.dtype == jnp.uint8:
        obs = obs.astype(jnp.float32); next_obs = next_obs.astype(jnp.float32)
    if rb.rewards.dtype != jnp.float32:
        rewards = rewards.astype(jnp.float32)

    return Storage(
        obs=obs, actions=actions, logprobs=logprobs, dones=dones,
        values=None, advantages=None, returns=None,
        rewards=rewards, next_obs=next_obs, next_dones=next_dones,
        hist_logprobs=None,
    )


# -------- Convenience utilities --------

def estimate_bytes(config: RBConfig, example: Storage) -> int:
    """Rough VRAM estimate for the configured buffer, in bytes."""
    shapes = _infer_shapes_from_example(example)
    T, W = config.T_cap, config.W_cap

    # choose action dtype as in init
    action_dtype = config.action_dtype
    if action_dtype is None:
        if (config.action_n_classes is not None) and (config.action_n_classes <= 256):
            action_dtype = jnp.uint8
        else:
            action_dtype = jnp.int32

    def nbytes(shape, dtype):
        return T * W * (int(jnp.ones(shape, dtype=dtype).nbytes))

    total = 0
    total += nbytes(shapes["obs_shape"], config.obs_dtype)
    total += nbytes(shapes["action_shape"], action_dtype)
    total += nbytes(shapes["reward_shape"], config.reward_dtype)
    total += nbytes(shapes["done_shape"], config.done_dtype)
    if config.store_next_obs:
        total += nbytes(shapes["obs_shape"], config.obs_dtype)
        total += nbytes(shapes["done_shape"], config.done_dtype)
    if config.store_logprobs and (shapes["logprob_shape"] is not None):
        total += nbytes(shapes["logprob_shape"], config.logprob_dtype)
    return int(total)

# -------- Flashbax --------

def _make_flashbax_rb(args, example_storage, *, n_envs: int, action_n_classes: int, prioritize_ends: float = 0.):
    """
    Returns:
      buffer:   Flashbax TrajectoryBuffer (callables)
      state:    BufferState
      add_roll: jitted function(state, storage) -> state    # adds a full rollout (T,W,...)
      sample_f: jitted function(state, rng) -> batch_dict
      can_samp: jitted function(state) -> bool scalar
    """
    W = int(n_envs)                   # add batch size
    B = int(args.rb_batch_size)       # sample batch size
    L = int(args.rb_sequence_length)  # sample sequence length
    rb_prefill = np.ceil(args.rb_prefill // n_envs).astype(int)
    period = 1

    # start sampling after at least one full sequence per env
    min_len_time_axis = max(L, rb_prefill)
    max_size = int(args.replay_buffer_capacity)  # total timesteps (time * env)

    buffer = fbx.make_trajectory_buffer(
        add_batch_size=W,
        sample_batch_size=B,
        sample_sequence_length=L+1,  # +1 for next_obs, next_done
        period=period,
        min_length_time_axis=min_len_time_axis,
        max_size=max_size,  # Flashbax sets max_length_time_axis = max_size // W internally
        prioritize_ends=prioritize_ends,
    )

    # Decide action packing
    orig_action_dtype = example_storage.actions.dtype
    use_u8_actions = (action_n_classes is not None) and (action_n_classes <= 256)

    example_ts = {
        "obs":       example_storage.obs[0, 0],                         # (C,H,W) or whatever trailing dims
        "reward":    example_storage.rewards[0, 0].astype(jnp.float32), # scalar or trailing dims per env-step
        "done":      example_storage.dones[0, 0].astype(jnp.bool_),     # bool
        "logprobs":  example_storage.logprobs[0, 0],                    # trailing dims per env-step
    }
    if use_u8_actions:
        example_ts["action_u8"] = example_storage.actions[0, 0].astype(jnp.uint8)
    else:
        example_ts["action"]    = example_storage.actions[0, 0].astype(orig_action_dtype)

    init_f   = jax.jit(buffer.init)
    add_one  = jax.jit(buffer.add, donate_argnums=(0,))
    sample_f = jax.jit(buffer.sample)
    can_samp = jax.jit(buffer.can_sample)

    @partial(jax.jit, donate_argnums=(0,))
    def add_rollout(state, storage):
        """Append a whole rollout (T,W,...) in one compiled scan."""
        T = storage.obs.shape[0]

        def _one(carry, t):
            # Each field must be (W, 1, ...)  ← time axis on axis=1
            ts = {
                "obs": jnp.expand_dims(storage.obs[t], axis=1),
                "reward": jnp.expand_dims(storage.rewards[t].astype(jnp.float32), axis=1),
                "done": jnp.expand_dims(storage.dones[t].astype(jnp.bool_), axis=1),
                "logprobs": jnp.expand_dims(storage.logprobs[t], axis=1),
            }
            if use_u8_actions:
                ts["action_u8"] = jnp.expand_dims(storage.actions[t].astype(jnp.uint8), axis=1)
            else:
                ts["action"] = jnp.expand_dims(storage.actions[t], axis=1)
            return add_one(carry, ts), None

        state, _ = jax.lax.scan(_one, state, jnp.arange(T))
        return state

    state = init_f(example_ts)
    meta = {"use_u8_actions": use_u8_actions, "orig_action_dtype": orig_action_dtype}
    return buffer, state, add_rollout, sample_f, can_samp, meta


def sample_from_fbx_rb(replay_buffer, rng, batch_width, batch_length, prioritize_ends: bool = False):
    """Convenience wrapper to match sample_from_rb interface."""
    fbx_buffer, fbx_state, _add_rollout, fbx_sample, _can_sample, meta = replay_buffer
    use_u8_actions = bool(meta["use_u8_actions"])
    orig_action_dtype = meta["orig_action_dtype"]

    batch = fbx_sample(fbx_state, rng)
    exp = getattr(batch, "experience", batch)

    def _TB(x):
        return jnp.swapaxes(x, 0, 1)  # (B, L, …) to (L, B, …)

    obs_all     = _TB(exp["obs"])
    rewards_all = _TB(exp["reward"])
    dones_all   = _TB(exp["done"])
    if use_u8_actions:
        actions_all = _TB(exp["action_u8"]).astype(orig_action_dtype)
    else:
        actions_all = _TB(exp["action"])
    logprobs_all = exp.get("logprobs", None)
    logprobs_all = None if logprobs_all is None else _TB(logprobs_all)

    L = int(batch_length)

    obs        = obs_all[:L]
    actions    = actions_all[:L]
    rewards    = rewards_all[:L]
    dones      = dones_all[:L]
    next_obs   = obs_all[1:]
    next_dones = dones_all[1:]
    logprobs   = None if (logprobs_all is None) else logprobs_all[:L]

    return Storage(
        obs=obs, actions=actions, rewards=rewards, dones=dones,
        next_obs=next_obs, next_dones=next_dones,
        logprobs=logprobs, values=None, advantages=None, returns=None,
        hist_logprobs=None,
    )