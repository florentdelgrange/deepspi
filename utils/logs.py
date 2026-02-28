import time
from typing import Optional, Dict

import flax
from flax.core.frozen_dict import unfreeze
import jax
import numpy as np
import jax.numpy as jnp
from flax.training.train_state import TrainState
from torch.utils.tensorboard import SummaryWriter

from utils.scores import atari_human_normalized_scores, human_normalized_score, rel_improvement


from flax.serialization import to_state_dict

def flatten_dict(params, parent_key: str = '', sep: str = '/'):
    """Flattens nested pytrees (Flax structs, FrozenDicts, dicts, tuples/lists) into {path: leaf}.

    Examples of keys: 'agent/actor_network_params/0/conv/kernel'.
    """
    # Convert non-dict containers (e.g., Flax struct dataclasses) to plain Python containers
    if not isinstance(params, (dict, flax.core.FrozenDict)):
        try:
            params = to_state_dict(params)
        except Exception:
            # Fallback: try __dict__ (best-effort)
            params = getattr(params, '__dict__', params)

    items = []
    if isinstance(params, (dict, flax.core.FrozenDict)):
        iterable = params.items()
    elif isinstance(params, (tuple, list)):
        iterable = enumerate(params)
    else:
        # Leaf value
        return {parent_key: params} if parent_key else {'': params}

    for k, v in iterable:
        k_str = str(k)
        new_key = f"{parent_key}{sep}{k_str}" if parent_key else k_str
        if isinstance(v, (dict, flax.core.FrozenDict, tuple, list)) or \
           (not isinstance(v, (np.ndarray, jnp.ndarray)) and hasattr(v, '__dict__')):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _last_scalar(x):
    """Return a Python float from possibly batched arrays (epochs[, minibatches])."""
    x = jnp.asarray(x)
    return float(jnp.ravel(x)[-1])

# Helper to select final grads from a grads pytree with possible leading scan dims
def _select_final_grads_tree(grads_tree):
    """Return a grads pytree where leading scan dims are stripped (final epoch/minibatch).
    Works if leaves are shaped like [..., ... , *param_shape] or already param-shaped.
    """
    def _sel(x):
        x = jnp.asarray(x)
        if x.ndim >= 2:
            return x[-1, -1]
        elif x.ndim == 1:
            return x[-1]
        else:
            return x
    return jax.tree_map(_sel, grads_tree)


# Helper to log tree (params or grads)
def _log_tree(writer: SummaryWriter, prefix: str, tree, step: int):
    """Log histograms and basic stats for a (possibly FrozenDict) pytree."""
    if tree is None:
        return

    # Convert to flat dict of leaves
    if isinstance(tree, flax.core.FrozenDict):
        flat = flatten_dict(unfreeze(tree))
    elif isinstance(tree, dict):
        flat = flatten_dict(tree)
    else:
        data = getattr(tree, 'params', tree)
        if isinstance(data, flax.core.FrozenDict):
            flat = flatten_dict(unfreeze(data))
        elif isinstance(data, dict):
            flat = flatten_dict(data)
        else:
            return

    # Keep only array leaves
    leaves = {k: np.asarray(jax.device_get(v)) for k, v in flat.items() if isinstance(v, (np.ndarray, jnp.ndarray))}
    if not leaves:
        raise RuntimeError(f"No array leaves found in tree for prefix '{prefix}' to log.")

    # Global stats across all leaves
    total_sq = 0.0
    absmax = 0.0
    for v in leaves.values():
        if v.size == 0:
            continue
        total_sq += float(np.sum(v.astype(np.float64) ** 2))
        absmax = max(absmax, float(np.max(np.abs(v))))
    writer.add_scalar(f"{prefix}/global_l2", float(np.sqrt(total_sq)), step)
    writer.add_scalar(f"{prefix}/global_absmax", absmax, step)

    # Aggregate histogram across all leaves (downsampled to keep size reasonable)
    all_vals = np.concatenate([v.ravel() for v in leaves.values() if v.size > 0])
    if all_vals.size == 0:
        raise RuntimeError(f"No numeric content found in leaves for prefix '{prefix}'.")
    max_points_tb = 25000
    if all_vals.size > max_points_tb:
        sel = np.random.default_rng(0).choice(all_vals.size, size=max_points_tb, replace=False)
        all_vals = all_vals[sel]
    writer.add_histogram(f"{prefix}/ALL", all_vals, step)
    wrote_count = 0
    # Per-leaf histograms + stats
    for k, v in leaves.items():
        if v.size == 0:
            continue
        name = f"{prefix}/{k}"
        try:
            writer.add_histogram(name, v, step)
        except Exception as e:
            raise RuntimeError(f"TensorBoard histogram logging failed for '{name}' (shape={v.shape}) at step {step}: {e}") from e
        writer.add_scalar(f"{name}:mean", float(np.mean(v)), step)
        writer.add_scalar(f"{name}:std", float(np.std(v)), step)
        writer.add_scalar(f"{name}:absmax", float(np.max(np.abs(v))), step)
        wrote_count += 1
    print(f"[TB] wrote {wrote_count} histograms under '{prefix}' at step {step}")



# Helper to log ALL ndarray leaves (params or grads) to W&B: per-leaf L2 norm + histogram, chunked for large trees
def _wandb_log_all(prefix: str, tree, step: int, *, bins: int = 64, max_points: int = 25000, chunk_size: int = 200):
    """Log ALL ndarray leaves (per-leaf L2 + histogram) to W&B under `prefix/...`.
    - `tree` can be a FrozenDict, dict, or Flax struct; we flatten to path->leaf.
    - Histograms are downsampled to at most `max_points` per leaf.
    - Payload is chunked to avoid too-large single wandb.log calls.
    Raises informative errors if nothing is found or if W&B is unavailable.
    """
    try:
        import wandb
    except Exception as e:
        raise ImportError("wandb is required for param/grad logging but is not available.") from e

    # Convert to flat dict of leaves
    if isinstance(tree, flax.core.FrozenDict):
        flat = flatten_dict(unfreeze(tree))
    elif isinstance(tree, dict):
        flat = flatten_dict(tree)
    else:
        data = getattr(tree, 'params', tree)
        if isinstance(data, flax.core.FrozenDict):
            flat = flatten_dict(unfreeze(data))
        elif isinstance(data, dict):
            flat = flatten_dict(data)
        else:
            # As a last resort, try to_state_dict via flatten_dict internals
            flat = flatten_dict(data)

    leaves = {k: np.asarray(jax.device_get(v)) for k, v in flat.items() if isinstance(v, (np.ndarray, jnp.ndarray))}
    if not leaves:
        raise RuntimeError(f"No ndarray leaves found to log for prefix '{prefix}'.")

    # Helper: chunk dict into batches for wandb.log
    def _chunks(items, n):
        it = iter(items)
        while True:
            batch = list()
            try:
                for _ in range(n):
                    batch.append(next(it))
            except StopIteration:
                pass
            if not batch:
                break
            yield batch

    # Build (key, value) generator to keep memory reasonable
    kv_iter = []
    for name, arr in leaves.items():
        # L2 norm
        kv_iter.append((f"{prefix}/{name}", float(np.linalg.norm(arr))))
        # Histogram (downsample if huge)
        flat_v = arr.ravel()
        if flat_v.size > max_points:
            idx = np.random.default_rng(0).choice(flat_v.size, size=max_points, replace=False)
            flat_v = flat_v[idx]
        kv_iter.append((f"{prefix}/{name}:hist", wandb.Histogram(flat_v, num_bins=bins)))

    # Flush in chunks
    total_logged = 0
    for batch in _chunks(kv_iter, chunk_size):
        payload = {k: v for k, v in batch}
        try:
            # Do not pass step when syncing with TensorBoard; include it as a metric instead.
            payload_with_step = dict(payload)
            payload_with_step["global_step"] = step
            wandb.log(payload_with_step)
        except Exception as e:
            raise RuntimeError(
                f"W&B logging failed for prefix '{prefix}' at step {step} while logging keys "
                f"{[k for k,_ in batch][:5]}...: {e}"
            ) from e
        total_logged += len(batch)

    # print(f"[W&B] logged {total_logged//2} leaves (norm+hist) under '{prefix}' at step {step}")


def _safe_lr_from_state(ts: TrainState):
    """
    Best-effort extraction of the current learning rate from a TrainState built with
    optax.multi_transform + chain + inject_hyperparams(adam/adamw).

    Handles these nestings:
      - multi_transform: opt_state.inner_states is a dict of label -> chain state
      - chain: opt_state is a tuple of transform states
      - inject_hyperparams: state has a dict attribute .hyperparams with 'learning_rate'
      - wrappers: states may expose .inner_state (single) or .inner_states (dict)

    Returns float or None if not found.
    """
    def _as_float(x):
        try:
            return float(x())  # schedule callable
        except TypeError:
            try:
                return float(x.item())  # jnp scalar
            except Exception:
                try:
                    return float(x)
                except Exception:
                    return None

    def _extract(state):
        # 1) Direct hyperparams dict
        hp = getattr(state, "hyperparams", None)
        if isinstance(hp, dict) and "learning_rate" in hp:
            return _as_float(hp["learning_rate"])  # may be callable/array/float

        # 2) Chain: tuple of states (e.g., (clip_state, inject_state, ...))
        if isinstance(state, tuple):
            for s in state:
                lr = _extract(s)
                if lr is not None:
                    return lr

        # 3) Single inner_state wrapper
        inner = getattr(state, "inner_state", None)
        if inner is not None:
            lr = _extract(inner)
            if lr is not None:
                return lr

        # 4) Multi inner_states (multi_transform)
        inner_dict = getattr(state, "inner_states", None)
        if isinstance(inner_dict, dict):
            # Prefer common labels first, then fallback to any
            for label in ("agent", "wm"):
                if label in inner_dict:
                    lr = _extract(inner_dict[label])
                    if lr is not None:
                        return lr
            for s in inner_dict.values():
                lr = _extract(s)
                if lr is not None:
                    return lr

        return None

    try:
        return _extract(ts.opt_state)
    except Exception:
        return None


def subtree_norms(grads, prefix):
    flat = flatten_dict(unfreeze(grads))
    for k, v in flat.items():
        if isinstance(v, (np.ndarray, jnp.ndarray)):
            n = float(jnp.linalg.norm(v))
            if n > 0:
                print(f"{prefix}/{k}: {n:.4e}")

def log(
        global_step: int,
        avg_episodic_return: Optional[float],
        max_avg_return: Optional[float],
        agent_state: TrainState,
        v_loss: jnp.ndarray,
        pg_loss: jnp.ndarray,
        entropy_loss: jnp.ndarray,
        approx_kl: jnp.ndarray,
        drift_penalty_mean: jnp.ndarray,
        transition_loss: Optional[jnp.ndarray],
        reward_loss: Optional[jnp.ndarray],
        scores_mean: Dict[str, float],
        gradient_penalty: Optional[jnp.ndarray],
        start_time: float,
        iteration_time_start: float,
        episode_stats: Optional[flax.core.FrozenDict],
        loss: jnp.ndarray,
        args: flax.core.FrozenDict,
        writer: SummaryWriter,
        grads: Optional[flax.core.FrozenDict] = None,
        prefix='losses',
        multiple_lr: bool = False,
        intermediate_step: Optional[int] = None,
        print_logs: bool = True,
        debug_grads: bool = False,
        maximizer_loss: Optional[jnp.ndarray] = None,
        alpha: Optional[jnp.ndarray] = None,
):
    if print_logs:
        print(f"global_step={global_step}, avg_episodic_return={avg_episodic_return}")

    if args.track:
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if avg_episodic_return is not None:
            writer.add_scalar("charts/avg_episodic_return", avg_episodic_return, global_step)
            if args.env_id in atari_human_normalized_scores:
                writer.add_scalar(
                    "charts/human_normalized_score", human_normalized_score(args.env_id, avg_episodic_return), global_step)
            if args.env_id in scores_mean and max_avg_return is not None:
                score = max_avg_return if args.compare_scores_max else avg_episodic_return
                rel_imp = rel_improvement(score, args.env_id, scores_mean)
                writer.add_scalar("charts/relative_improvement", rel_imp, global_step)
                if args.env_id in atari_human_normalized_scores:
                    rel_imp_hns = rel_improvement(
                        human_normalized_score(args.env_id, score), args.env_id,
                        {args.env_id: human_normalized_score(args.env_id, scores_mean[args.env_id])})
                    writer.add_scalar("charts/relative_improvement_hns", rel_imp_hns, global_step)

        if episode_stats is not None:
            writer.add_scalar(
                "charts/avg_episodic_length", np.mean(jax.device_get(episode_stats.returned_episode_lengths)),
                global_step)

        lr_prefix = prefix if multiple_lr else "charts"
        lr_value = _safe_lr_from_state(agent_state)
        if lr_value is not None:
            writer.add_scalar(f"{lr_prefix}/learning_rate", lr_value, global_step)

        writer.add_scalar(f"{prefix}/value_loss", _last_scalar(v_loss), global_step)
        writer.add_scalar(f"{prefix}/policy_loss", _last_scalar(pg_loss), global_step)
        writer.add_scalar(f"{prefix}/entropy", _last_scalar(entropy_loss), global_step)
        writer.add_scalar(f"{prefix}/approx_kl", _last_scalar(approx_kl), global_step)
        writer.add_scalar(f"{prefix}/loss", _last_scalar(loss), global_step)
        if alpha is not None:
            writer.add_scalar(f"{prefix}/alpha", np.array(alpha), global_step)
        if transition_loss is not None:
            writer.add_scalar(f"{prefix}/transition_loss", _last_scalar(transition_loss), global_step)
        if reward_loss is not None:
            writer.add_scalar(f"{prefix}/reward_loss", _last_scalar(reward_loss), global_step)
        if gradient_penalty is not None:
            writer.add_scalar(f"{prefix}/gradient_penalty", _last_scalar(gradient_penalty), global_step)
        if args.drift_formulation:
            writer.add_scalar(f"{prefix}/drift", _last_scalar(drift_penalty_mean), global_step)
        if maximizer_loss is not None:
            writer.add_scalar(f"{prefix}/maximizer_loss", _last_scalar(maximizer_loss), global_step)

        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        if intermediate_step:
            writer.add_scalar(
                f"{prefix}/SPS_update", int(intermediate_step / (time.time() - iteration_time_start)), global_step
            )
        else:
            writer.add_scalar(
                "charts/SPS_update", int(args.num_envs * args.num_steps / (time.time() - iteration_time_start)), global_step
            )

    # Log ALL params and/or grads (entire tree), per-leaf L2 + histogram to W&B
    if args.track_params:
        _wandb_log_all("params", agent_state.params, global_step)

    if args.track_grads and grads is not None:
        final_grads = _select_final_grads_tree(grads)
        _wandb_log_all("grads", final_grads, global_step)
        if debug_grads:
            subtree_norms(grads, f'{prefix}_grads')

    if print_logs:
        print(
            "SPS:", int(global_step / (time.time() - start_time)),
            f'transition_loss: {_last_scalar(transition_loss):.6g}' if transition_loss is not None else '',
            f'reward_loss: {_last_scalar(reward_loss):.6g}' if reward_loss is not None else '',
            f'gradient_penalty: {_last_scalar(gradient_penalty):.6g}' if gradient_penalty is not None else '',
            f'alpha: {alpha:.4g}' if alpha is not None else '',
        )
