import os
import random
import time
import warnings
from dataclasses import dataclass, replace
from functools import partial
from typing import Callable, Tuple, Dict, Optional, Union

import gym
import jax
import jax.tree_util as jtu
import numpy as np
import optax
import tyro
from flax import struct
from flax.core import unfreeze
from flax.training.train_state import TrainState
from torch.utils.tensorboard import SummaryWriter
from jax import numpy as jnp

from deep_spi import Args, DeepSPIAgent, check_and_process_args, make_env, EpisodeStatistics, FullParams, Storage, \
    linear_schedule, step_env_wrapped, WorldModelParams, AgentParams
from networks.architectures import NetworkConv, DiscreteActionTransitionNetwork, \
    DiscreteActionRewardNetwork, DonePredictor, Critic, NetworkFCOutput, Actor
from utils.logs import flatten_dict, log
from utils.replay_buffer import ReplayBuffer
from utils.gpu_replay_buffer import sample_from_rb as rb_gpu_sample_batch, init_replay_buffer, RBConfig, \
    _make_flashbax_rb, sample_from_fbx_rb
from utils.gpu_replay_buffer import add_batch as rb_gpu_add_to_buffer
from utils.gpu_replay_buffer import DeviceReplayBuffer
from utils.scores import load_env_mean_std, atari_human_normalized_scores, human_normalized_score, rel_improvement
# jax.config.update("jax_disable_jit", True)
# jax.config.update("jax_debug_nans", True)

# Fix weird OOM https://github.com/google/jax/discussions/6332#discussioncomment-1279991
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6"
# Fix CUDNN non-determinisim; https://github.com/google/jax/issues/4823#issuecomment-952835771
os.environ["TF_XLA_FLAGS"] = "--xla_gpu_autotune_level=2 --xla_gpu_deterministic_reductions"
os.environ["TF_CUDNN DETERMINISTIC"] = "1"


def flatten_first_two(x):
    # Works for both real and imagined storages: (T, B, ...) -> (N, ...)
    return x.reshape((-1,) + x.shape[2:])

def build_minibatches(subkey, flat_batch, N, num_minibatches: int):
    """Shuffle with one shared permutation, then reshape into [M, S, ...]."""
    perm = jax.random.permutation(subkey, N)

    def take_and_reshape(x):
        x = x[perm]
        new_shape = (num_minibatches, -1) + x.shape[1:]
        return x.reshape(new_shape)

    return jax.tree_map(take_and_reshape, flat_batch)


# taken from: https://github.com/google/brax/blob/main/brax/training/agents/ppo/train.py
def convert_data(x: jnp.ndarray, num_minibatches: int, key: jax.random.PRNGKey) -> jnp.ndarray:
    x = jax.random.permutation(key, x)
    x = jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])
    return x


@dataclass
class DreamSPIArgs(Args):
    """Arguments for the Deep SPI Dreamer algorithm."""
    debug: bool = False
    """Debug mode: if True, will print additional information and produce sanity checks."""
    imagination_horizon: int = 8
    """Number of steps to unroll the world model during latent imagination."""
    encoder_learning_rate: float = 2e-4
    """Learning rate for the encoder (representation learning)."""
    encoder_anneal_lr: bool = True
    """Whether to anneal the encoder learning rate."""
    actor_learning_rate: float = 2.75e-5
    """Learning rate for the actor."""
    actor_anneal_lr: bool = True
    """Whether to anneal the actor learning rate."""
    critic_learning_rate: float = 2.75e-5
    """Learning rate for the critic."""
    critic_anneal_lr: bool = True
    """Whether to anneal the critic learning rate."""
    world_model_learning_rate: float = 2e-4
    """Learning rate for the world model."""
    world_model_anneal_lr: bool = True
    """Whether to anneal the world model learning rate."""
    global_anneal_lr: bool = True
    """Whether to anneal all learning rates (encoder, actor, critic, world model)."""
    n_imagination_rollouts: int = 1
    """Number of full imagination rollouts to perform from the same batch of experiences."""
    n_max_updates: int = 5
    """Number of updates for the discriminator (via a maximizer) per world model update."""
    maximizer_learning_rate: float = 1e-4
    """Learning rate for the maximizer (to update the discriminator)."""
    wm_clip_coef: float = 0.1
    """Clipping coefficient for the ratio in the world model loss."""
    imagination_epochs: int = 1
    """Number of epochs to train the world model on imagined data."""
    imagination_num_minibatches: Optional[int] = None
    """Number of minibatches to split the imagined data into (for world model updates). If not provided, infer from batch size."""
    weight_decay: float = 0.
    """Apply weight decay. Zero means no weight decay."""
    disable_wd_ac: bool = False
    """Whether to disable weight decay on actor and critic parameters."""
    dyna_style: bool = False
    """Whether to use Dyna-style actor-critic updates (on real data) in addition to imagined data."""
    progressive: bool = False
    """Whether to use progressive updates along training steps (first on real data, then on imagined data)."""
    progression_scheme: str = "linear"
    """Progression schedule: 'linear' or 'exp'."""
    use_replay_buffer: bool = False
    """Whether to use a replay buffer."""
    replay_buffer_capacity: int = 500_000
    """Capacity of the replay buffer (if used)."""
    env_steps_per_wm_update: int = 4
    """Number of (true) env steps per world model update. (!!) Only used when a replay buffer is enabled."""
    rb_batch_size: int = 64
    """Batch size (=width) for sampling from the replay buffer (if used)."""
    rb_sequence_length: int = 32
    """Sequence length for sampling from the replay buffer (if used)."""
    rb_prefill: int = 12_500
    """Number of random steps to prefill the replay buffer (if used)."""
    rb_type: str = 'flashbax'
    """Type of replay buffer to use (if any): 'flashbax', 'cpu', or 'gpu'."""
    rb_print_stats: bool = False
    """Whether to print replay buffer statistics (if used)."""
    prioritize_new_samples: float = 0.
    """Degree of prioritization of the new samples when sampling from the replay buffer (0 to 1)."""
    soft_done: bool = False
    """Whether to use soft-dones (Bernoulli probabilities) instead of hard dones (0/1)."""
    vtrace_log_ratio_clip: float = 3.
    """Clipping value for V-trace log ratios (only used when both use_replay_buffer and use_dyna_style are True)."""
    use_v_trace: bool = False
    """Whether to use V-trace corrections when using a replay buffer."""

    # to be filled in runtime
    wm_num_minibatches: int = 0
    """Number of minibatches for world model updates (computed in runtime)."""


@struct.dataclass
class TrainStatesConfig:
    """Parameter paths for the labels used in multi_transform optimizers. The """
    repr: Dict[str, Tuple[Tuple[Union[str, int]], ...]]
    actor: Dict[str, Tuple[Tuple[Union[str, int]], ...]]
    critic: Dict[str, Tuple[Tuple[Union[str, int]], ...]]
    world_model: Dict[str, Tuple[Tuple[Union[str, int]], ...]]
    maximizer: Optional[Dict[str, Tuple[Tuple[Union[str, int]], ...]]]

@struct.dataclass
class SeparateTrainStatesConfig(TrainStatesConfig):
    repr: Dict[str, Tuple[Tuple[Union[str, int]], ...]] = struct.field(default_factory=lambda: {
        'repr': (
            ('agent', 'actor_network_params', 0),
        ),
        'frozen': (
            ('world_model', ),
            ('agent', 'actor_network_params', 1),
            ('agent', 'actor_params', ),
            ('agent', 'critic_network_params'),
            ('agent', 'critic_params', ),
            ('discriminator',),
        )
    })

    actor: Dict[str, Tuple[Tuple[Union[str, int]], ...]] = struct.field(default_factory=lambda: {
        'actor': (
            ('agent', 'actor_network_params', 1),
            ('agent', 'actor_params', ),
        ),
        'frozen': (
            ('world_model', ),
            ('agent', 'actor_network_params', 0),
            ('agent', 'critic_network_params'),
            ('agent', 'critic_params', ),
        )
    })

    critic: Dict[str, Tuple[Tuple[Union[str, int]], ...]] = struct.field(default_factory=lambda: {
        'critic': (
            ('agent', 'critic_network_params', 1),  # we need to freeze this when decoupled_repr=False
            ('agent', 'critic_params', ),
        ),
        'frozen': (
            ('world_model', ),
            ('agent', 'critic_network_params', 0),  # even with a decoupled representation, the obs encoder is shared
            ('agent', 'actor_network_params'),
            ('agent', 'actor_params', ),
            ('discriminator',)
        )
    })

    world_model: Dict[str, Tuple[Tuple[Union[str, int]], ...]] = struct.field(default_factory=lambda: {
        'world_model':(
            ('world_model', ),
        ),
        'frozen': (
            ('agent', ),
            ('discriminator',),
        )
    })

    maximizer: Optional[Dict[str, Tuple[Tuple[Union[str, int]], ...]]] = struct.field(default_factory=lambda: {
        'maximizer': (
            ('discriminator',),
        ),
        'frozen': (
            ('world_model', ),
            ('agent', ),
        )
    })


@struct.dataclass
class DreamSPIAgent(DeepSPIAgent):
    repr_train_state: TrainState  # TrainState for the representation (encoder)
    world_model_train_state: TrainState  # TrainState for the world model
    actor_train_state: TrainState  # TrainState for the actor
    critic_train_state: TrainState  # TrainState for the critic
    maximizer_train_state: Optional[TrainState]

    # compiled functions
    get_wm_rollout_losses: Callable = struct.field(pytree_node=False)
    get_action_and_value_wm: Callable = struct.field(pytree_node=False)
    compute_gae_for_wm: Callable = struct.field(pytree_node=False)
    compute_aux_losses_wm: Callable = struct.field(pytree_node=False)
    world_model_loss: Callable = struct.field(pytree_node=False)
    world_model_loss_grad_fn: Callable = struct.field(pytree_node=False)
    update_world_model: Callable = struct.field(pytree_node=False)
    imagine_once: Callable = struct.field(pytree_node=False)
    imagine: Callable = struct.field(pytree_node=False)
    get_logprobs_and_value: Callable = struct.field(pytree_node=False)
    maximizer_grad_fn: Optional[Callable] = struct.field(pytree_node=False, default=None)
    update_maximizer: Optional[Callable] = struct.field(pytree_node=False, default=None)
    compute_v_trace: Callable = struct.field(pytree_node=False, default=None)

    @classmethod
    def create(
            cls,
            args: DreamSPIArgs,
            envs: gym.vector.VectorEnv,
            raw_step_env: Callable,
            key: jax.random.PRNGKey,
            use_done_predictor: bool = True,  # this input is ignored, as DeepSPIDreamer always uses a done predictor
            latent_obs: bool = True,  # this input is ignored, as DeepSPIDreamer always uses latent observations
    ) -> "DreamSPIAgent":
        # transition and reward losses are handled in the world model
        parent_args = replace(args, transition_loss_coef=0., reward_loss_coef=0.)
        parent = DeepSPIAgent.create(
            parent_args, envs, raw_step_env, key, use_done_predictor=True, latent_obs=True)
        fields = dict()

        dreamer_clip_coef = args.clip_coef
        wm_clip_coef = args.wm_clip_coef
        args = replace(args, clip_coef=wm_clip_coef)

        compute_aux_losses_wm = partial(
            cls._compute_auxiliary_losses, args=args, transition_network=parent.transition_network,
            reward_network=parent.reward_network, actor_fc=parent.actor_fc, envs=envs,
            pi_sample_fn=parent.pi_sample, done_predictor=parent.done_predictor,
            discriminator=parent.discriminator)
        fields["compute_aux_losses_wm"] = jax.jit(compute_aux_losses_wm)

        get_wm_rollout_losses = partial(
            cls._get_action_and_value2, args=args, actor=parent.actor, actor_conv=parent.actor_conv,
            actor_fc=parent.actor_fc, critic=parent.critic, critic_conv=parent.critic_conv,
            critic_fc=parent.critic_fc, compute_auxiliary_losses=fields["compute_aux_losses_wm"],
            latent_obs=False)
        fields["get_wm_rollout_losses"] = jax.jit(get_wm_rollout_losses)

        compute_gae_for_wm = partial(
            cls._compute_gae,  args=args,
            actor_conv=parent.actor_conv, actor_fc=parent.actor_fc,
            critic=parent.critic, critic_conv=parent.critic_conv, critic_fc=parent.critic_fc,
            compute_gae_once=parent.compute_gae_once, latent_obs=False)
        fields["compute_gae_for_wm"] = jax.jit(compute_gae_for_wm)

        alpha_ppo_loss_wm = partial(
            cls._alpha_ppo_loss,
            ppo_loss_fn=cls._off_policy_ppo_loss
                if args.use_replay_buffer and not args.drift_formulation
                # the off_policy correction is handled inside the drift formulation of the ppo loss
                else cls._ppo_loss)
        fields["world_model_loss"] = partial(
            alpha_ppo_loss_wm, args=args,
            get_action_and_value2=fields["get_wm_rollout_losses"],
            # V-trace already handles off-policy correction
            off_policy_correction=args.use_replay_buffer and not args.use_v_trace,
            use_v_trace=args.use_v_trace)
        fields["world_model_loss_grad_fn"] = jax.value_and_grad(fields["world_model_loss"], has_aux=True)

        update_world_model = partial(
            cls._update_ppo, args=replace(args, num_minibatches=args.wm_num_minibatches),
            ppo_loss_grad_fn=fields["world_model_loss_grad_fn"], num_minibatches=args.wm_num_minibatches,
            off_policy_correction=args.use_replay_buffer)
        fields["update_world_model"] = jax.jit(update_world_model)

        get_action_and_value_wm = partial(
            cls._get_action_and_value, args=args, critic=parent.critic, actor_conv=parent.actor_conv,
            actor_fc=parent.actor_fc, critic_conv=parent.critic_conv, critic_fc=parent.critic_fc,
            pi_sample_fn=parent.pi_sample, latent_obs=False)
        fields["get_action_and_value_wm"] = jax.jit(get_action_and_value_wm)

        _step_env_wrapped = partial(step_env_wrapped, step_env_fn=raw_step_env)
        fields["step_once"] = partial(
            cls._step_once, env_step_fn=_step_env_wrapped, get_action_and_value_fn=fields["get_action_and_value_wm"])
        fields["rollout"] = partial(cls._rollout, step_once_fn=fields["step_once"], max_steps=args.num_steps)

        args = replace(args, clip_coef=dreamer_clip_coef, drift_coef=1.)  # a drift coef of 1 allows recovering the original PPO loss

        alpha_ppo_loss = partial(cls._alpha_ppo_loss, ppo_loss_fn=cls._ppo_loss)
        ppo_loss_fn = partial(
            alpha_ppo_loss, args=args,
            get_action_and_value2=parent.get_action_and_value2,
            off_policy_correction=False,  # deep-spi on latent imagination is on-policy
            use_v_trace=False,  # GAE is used in the latent space
        )
        ppo_loss_grad_fn = jax.value_and_grad(ppo_loss_fn, has_aux=True)
        update_ppo = partial(
            cls._alpha_update_ppo, args=replace(args, update_epochs=args.imagination_epochs),
            ppo_loss_grad_fn=ppo_loss_grad_fn,  num_minibatches=args.num_minibatches, off_policy_correction=False)
        fields["update_ppo"] = jax.jit(update_ppo)

        imagine_once = partial(
            cls._imagine_step_once, step_fn=cls._imagine_step_fn, get_action_and_value_fn=parent.get_action_and_value,
            transition_network=parent.transition_network, reward_network=parent.reward_network,
            done_predictor=parent.done_predictor, soft_done=args.soft_done) #update_episode_statistics_fn=cls.update_episode_statistics)
        fields["imagine_once"] = jax.jit(imagine_once)
        fields["imagine"] = partial(
            cls._imagine_rollout, imagine_once_fn=fields["imagine_once"], max_steps=args.imagination_horizon,
            soft_done=args.soft_done)

        if args.wasserstein_discriminator:
            # Maximizer train state is initialized in the parent class
            mean_discrepancy = partial(cls._mean_discrepancy, discriminator=parent.discriminator, )
            fields["maximizer_grad_fn"] = jax.value_and_grad(mean_discrepancy, has_aux=False)
            update_maximizer = partial(
                cls._update_maximizer, args=args, mean_discrepancy_grad_fn=fields["maximizer_grad_fn"],
                actor_conv=parent.actor_conv, transition_network=parent.transition_network,
                num_minibatches=args.wm_num_minibatches)
            fields["update_maximizer"] = jax.jit(update_maximizer)

        if args.use_replay_buffer:
            assert args.rb_type in ('flashbax', 'cpu', 'gpu'), f"Invalid replay buffer type: {args.rb_type}"
            compute_v_trace = partial(
                cls._compute_v_trace, args=args, actor_conv=parent.actor_conv, actor_fc=parent.actor_fc,
                critic=parent.critic, critic_conv=parent.critic_conv, critic_fc=parent.critic_fc,
                compute_vtrace_once=cls._compute_vtrace_once,
            )
            fields["compute_v_trace"] = jax.jit(compute_v_trace)

        get_logprobs_and_value = partial(
            cls._get_logprobs_and_value, args=args,
            actor=parent.actor, critic=parent.critic,actor_conv=parent.actor_conv, actor_fc=parent.actor_fc,
            critic_conv=parent.critic_conv, critic_fc=parent.critic_fc)
        fields["get_logprobs_and_value"] = jax.jit(get_logprobs_and_value)

        # train_state for both agent and world model; use masking to control which parts are updated (safeguard)
        # Build masks once from initial params
        initial_params = parent.train_state.params  # FullParams
        train_states = cls.initialize_train_states(initial_params, args)
        fields = {**fields, **train_states}

        fields = {**fields, **{key: value for key, value in parent.__dict__.items() if key not in fields}}
        return cls(**fields)

    @classmethod
    def initialize_train_states(cls, initial_params, args: DreamSPIArgs) -> Dict[str, TrainState]:
        """Initialize the train states for the agent and world model using multi_transform.
        We build two optimizers:
          - agent optimizer: updates agent (except actor encoder index 0), freezes world_model
          - wm optimizer: updates world_model + actor encoder index 0, freezes the rest
        """
        linear_schedule_repr = partial(
            linear_schedule,
            args=replace(args, learning_rate=args.encoder_learning_rate,
                         num_minibatches=args.wm_num_minibatches),)
        linear_schedule_wm = partial(
            linear_schedule,
            args=replace(args, learning_rate=args.world_model_learning_rate,
                         num_minibatches=args.wm_num_minibatches),)
        actor_critic_num_it = args.num_iterations *  args.n_imagination_rollouts
        linear_schedule_actor = partial(
            linear_schedule,
            args=replace(
                args,
                learning_rate=args.actor_learning_rate,
                update_epochs=args.imagination_epochs,
                num_iterations=actor_critic_num_it))
        linear_schedule_critic = partial(
            linear_schedule,
            args=replace(
                args,
                update_epochs=args.imagination_epochs,
                learning_rate=args.critic_learning_rate,
                num_iterations=actor_critic_num_it))
        config = SeparateTrainStatesConfig()
        if not args.decoupled_repr:
            config.critic['critic'] = (('agent', 'critic_params', ), )
            config.critic['frozen'] += (('agent', 'critic_network_params', 1),)
        fields: Dict[str, TrainState] = {}

        component_names = ('repr', 'actor', 'critic', 'world_model') \
                          + (('maximizer',) if args.wasserstein_discriminator else ())

        # helpers to label leaves
        def _normalize_path(path):
            """Convert JAX path entries to plain strings/ints for reliable prefix checks."""
            out = []
            for e in path:
                name = getattr(e, "name", None)  # Attribute access (dataclass fields)
                key = getattr(e, "key", None)  # Dict keys
                idx = getattr(e, "idx", None)  # Sequence indices
                if name is not None:
                    out.append(name)
                elif key is not None:
                    out.append(key)
                elif idx is not None:
                    out.append(idx)
                else:
                    out.append(e)
            return tuple(out)

        def path_startswith(path, prefix):
            p = _normalize_path(path)
            if len(p) < len(prefix):
                return False
            for a, b in zip(p, prefix):
                if a != b:
                    return False
            return True

        def build_labels(name, p):
            def label_fn(path, _):
                local_config = getattr(config, name)
                for key in local_config.keys():
                    for prefix in local_config[key]:
                        if path_startswith(path, prefix):
                            return key
                return "frozen"
            return jtu.tree_map_with_path(label_fn, p)

        repr_labels = build_labels('repr', initial_params)
        actor_labels = build_labels('actor', initial_params)
        critic_labels = build_labels('critic', initial_params)
        world_model_labels = build_labels('world_model', initial_params)
        if args.wasserstein_discriminator:
            maximizer_labels = build_labels('maximizer', initial_params)
        else:
            maximizer_labels = None

        # --- Sanity checks for labels -----------------------------------
        from flax.serialization import to_state_dict as _to_state

        def _assert_all_labels(who, labels_tree, path_tuple, expected):
            """
            Assert that *all leaves* under `path_tuple` equal `expected`.
            Works with Flax structs by converting to a plain state dict.
            """
            ld = _to_state(labels_tree)

            # Robust navigation: support dicts with stringified indices and lists/tuples
            node = ld
            for k in path_tuple:
                if isinstance(node, dict):
                    if k in node:
                        node = node[k]
                    elif isinstance(k, int) and str(k) in node:
                        node = node[str(k)]
                    else:
                        raise KeyError(f"Path not found in labels for {who}: {'/'.join(map(str, path_tuple))} (missing key: {k})")
                elif isinstance(node, (list, tuple)):
                    if isinstance(k, int) and 0 <= k < len(node):
                        node = node[k]
                    else:
                        raise KeyError(f"Sequence index not valid in labels for {who}: {'/'.join(map(str, path_tuple))} (bad index: {k})")
                else:
                    raise KeyError(f"Cannot descend into non-container at {who}:{'/'.join(map(str, path_tuple))}; stuck at {k} with type {type(node)}")

            bad = []

            def _walk(n, rel=()):
                if isinstance(n, dict):
                    for kk, vv in n.items():
                        _walk(vv, rel + (kk,))
                elif isinstance(n, (list, tuple)):
                    for i, vv in enumerate(n):
                        _walk(vv, rel + (i,))
                else:
                    # Leaf: a label string or None
                    # If it's None, there are no parameters here; skip.
                    if n is None:
                        return
                    if n != expected:
                        bad.append('/'.join(map(str, path_tuple + rel)))

            _walk(node)

            # Build a preview with actual label values for the first few mismatches
            def _get_rel(n, rel):
                cur = n
                for kk in rel:
                    if isinstance(cur, dict):
                        if kk in cur:
                            cur = cur[kk]
                        elif isinstance(kk, int) and str(kk) in cur:
                            cur = cur[str(kk)]
                        else:
                            return None
                    elif isinstance(cur, (list, tuple)):
                        if isinstance(kk, int) and 0 <= kk < len(cur):
                            cur = cur[kk]
                        else:
                            return None
                    else:
                        return cur
                return cur

            preview_items = []
            for p in bad[:5]:
                # p is like 'world_model/discriminator_params'
                rel = tuple(p.split('/'))
                if rel[:len(path_tuple)] == tuple(map(str, path_tuple)):
                    rel = rel[len(path_tuple):]
                val = _get_rel(node, rel)  # get actual label at that path
                preview_items.append(f"{p} -> {val}")
            preview = '[' + ', '.join(preview_items) + (', ...' if len(bad) > 5 else '') + ']'

            assert not bad, (
                f"[{who}] label mismatch under {'/'.join(map(str, path_tuple))}: "
                f"expected all leaves == '{expected}', got mismatches at {preview}"
                + ("..." if len(bad) > 5 else "")
            )

        # --- sanity check: all leaves under each prefix have the expected label ---
        for name in component_names:
            local_config = getattr(config, name)
            for key in local_config.keys():
                for prefix in local_config[key]:
                    _assert_all_labels(name, locals()[f"{name}_labels"], prefix, key)
            print(f"[optimizer] all sanity checks passed for {name} labels")

        # --- Build optimizers ---
        # define per-label transforms
        repr_active_tx = optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.inject_hyperparams(optax.adamw)(
                learning_rate=linear_schedule_repr if args.encoder_anneal_lr else args.encoder_learning_rate,
                eps=1e-5,
                weight_decay=args.weight_decay,
            ),
        )
        world_model_active_tx = optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.inject_hyperparams(optax.adamw)(
                learning_rate=linear_schedule_wm if args.world_model_anneal_lr else args.world_model_learning_rate,
                eps=1e-5,
                weight_decay=args.weight_decay,
            ),
        )
        actor_active_tx = optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.inject_hyperparams(optax.adamw)(
                learning_rate=linear_schedule_actor if args.actor_anneal_lr else args.actor_learning_rate,
                eps=1e-5,
                weight_decay=args.weight_decay and not args.disable_wd_ac,
            ),
        )
        critic_active_tx = optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.inject_hyperparams(optax.adamw)(
                learning_rate=linear_schedule_critic if args.critic_anneal_lr else args.critic_learning_rate,
                eps=1e-5,
                weight_decay=args.weight_decay if not args.disable_wd_ac else 5e-7,
            ),
        )
        if args.wasserstein_discriminator:
            # Maximizer optimizer: update discriminator only
            maximizer_active_tx = optax.chain(
                optax.clip_by_global_norm(args.max_grad_norm),
                optax.inject_hyperparams(optax.adamw)(
                    learning_rate=args.maximizer_learning_rate,
                    b1=0.,
                    b2=0.9,
                    weight_decay=1e-3,
                ),
            )
        frozen_tx = optax.set_to_zero()

        # ----- build multi_transform optimizers ---------------------------------------
        for name in component_names:
            transforms = {key: frozen_tx for key in component_names if key != name}
            transforms[name] = locals()[f"{name}_active_tx"]
            transforms["frozen"] = frozen_tx
            opt_tx = optax.multi_transform(
                transforms,
                locals()[f"{name}_labels"],
            )

            fields[f"{name}_train_state"] = TrainState.create(params=initial_params, tx=opt_tx, apply_fn=None)
        
        if 'maximizer' not in component_names:
            fields['maximizer_train_state'] = None

        return fields

    @staticmethod
    def update_episode_statistics(
            stats: EpisodeStatistics,
            reward: jnp.ndarray,  # [B] reward at current step (include terminal reward)
            done: jnp.ndarray,  # [B] bools
    ) -> EpisodeStatistics:
        done = done.astype(bool)

        new_returns = stats.episode_returns + reward
        new_lengths = stats.episode_lengths + 1

        # When an episode ends *this* step, we “publish” the return/length
        returned_episode_returns = jnp.where(done, new_returns, stats.returned_episode_returns)
        returned_episode_lengths = jnp.where(done, new_lengths, stats.returned_episode_lengths)

        # Reset accumulators for envs that ended; keep accumulating otherwise
        episode_returns = jnp.where(done, jnp.zeros_like(new_returns), new_returns)
        episode_lengths = jnp.where(done, jnp.zeros_like(new_lengths), new_lengths)

        return stats.replace(
            episode_returns=episode_returns,
            episode_lengths=episode_lengths,
            returned_episode_returns=returned_episode_returns,
            returned_episode_lengths=returned_episode_lengths,
        )

    @staticmethod
    def _get_logprobs_and_value(
            agent_state: TrainState,
            obs: np.ndarray,
            action: np.ndarray,
            args: Args,
            actor: Actor,
            critic: Critic,
            actor_conv: NetworkConv,
            actor_fc: NetworkFCOutput,
            critic_conv: NetworkConv,
            critic_fc: NetworkFCOutput,
    ):
        """calculate value, logprob, and update storage for the given action"""
        actor_conv_params, actor_fc_params = agent_state.params.agent.actor_network_params
        critic_conv_params, critic_fc_params = agent_state.params.agent.critic_network_params
        actor_params = agent_state.params.agent.actor_params

        z = actor_conv.apply(actor_conv_params, obs)
        actor_hidden = actor_fc.apply(actor_fc_params, z)
        logits = actor.apply(actor_params, actor_hidden)
        logprob = jax.nn.log_softmax(logits)[jnp.arange(action.shape[0]), action]

        if not args.decoupled_repr:
            critic_hidden = actor_hidden
        else:
            critic_hidden = critic_conv.apply(critic_conv_params, obs)
            critic_hidden = critic_fc.apply(critic_fc_params, critic_hidden)

        value = critic.apply(agent_state.params.agent.critic_params, critic_hidden).squeeze(1)
        return logprob, value

    @staticmethod
    def _imagine_step_fn(
            world_model_params: WorldModelParams,
            z: np.ndarray,
            action: np.ndarray,
            key: jax.random.PRNGKey,
            # episode_stats: EpisodeStatistics,
            transition_network: DiscreteActionTransitionNetwork,
            reward_network: DiscreteActionRewardNetwork,
            done_predictor: DonePredictor,
            soft_done: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, jax.random.PRNGKey]:
        """Perform latent imagination using the world model (step once in the world model)."""
        transition_params = world_model_params.transition_network_params
        reward_params = world_model_params.reward_network_params
        done_predictor_params = world_model_params.done_predictor_params
        dist = transition_network.apply(transition_params, (z, action))
        key, subkey = jax.random.split(key)
        z_prime = dist.sample(seed=subkey)
        reward = reward_network.apply(reward_params, (z, action))
        done_dist = done_predictor.apply(done_predictor_params, z_prime)
        key, subkey = jax.random.split(key)
        if soft_done:
            done = done_dist.probs_parameter().astype(jnp.float32)
        else:
            done = done_dist.sample(seed=subkey).astype(jnp.int32)
            expand = (None,) * (z.ndim - 1)  # expand done to match z except batch dim
            mask = done[(...,) + expand]     # broadcast along all other dims
            z_prime = jnp.where(mask, z, z_prime)

        # episode_stats = update_episode_statistics_fn(episode_stats, reward, done)
        return jax.lax.stop_gradient(z_prime), jax.lax.stop_gradient(reward), jax.lax.stop_gradient(done), key

    @staticmethod
    def _imagine_step_once(
            carry, _, step_fn,
            # to be fixed
            get_action_and_value_fn, transition_network, reward_network, done_predictor, # update_episode_statistics_fn
            soft_done: bool = False,
        ):
        agent_state, z, done, key = carry
        world_model_params = agent_state.params.world_model
        action, logprob, value, key = get_action_and_value_fn(agent_state, z, key)  # latent_obs must be True
        z_prime, reward, next_done, key = step_fn(
            world_model_params, z, action, key,
            transition_network, reward_network, done_predictor,
            soft_done=soft_done,)
        # update_episode_statistics_fn)
        storage = Storage(
            obs=z,
            actions=action,
            logprobs=logprob,
            dones=done,
            values=value,
            rewards=reward,
            returns=jnp.zeros_like(reward),
            advantages=jnp.zeros_like(reward),
            next_obs=z_prime,
            next_dones=next_done,
        )
        return ((agent_state, z_prime, next_done, key), storage)

    @staticmethod
    def _imagine_rollout(agent_state, z_prime, next_done, key, imagine_once_fn, max_steps, soft_done: bool = False):
        next_done = next_done.astype(jnp.float32) if soft_done else next_done.astype(jnp.int32)
        (agent_state, z_prime, next_done, key), storage = jax.lax.scan(
            imagine_once_fn, (agent_state, z_prime, next_done, key), (), max_steps
        )
        return agent_state, z_prime, next_done, storage, key

    @classmethod
    def _alpha_ppo_loss(
            cls,
            params, obs, a, logp, mb_advantages, mb_returns, reward, done, next_obs, next_done, key, hist_logprobs,
            alpha: float,
            args: Args, get_action_and_value2: Callable,
            off_policy_correction: bool = False,
            use_v_trace: bool = False,
            ppo_loss_fn: Callable = None,  # to be set in partial
    ):
        """Compute the PPO loss with a scaling factor `alpha`."""
        loss, aux = ppo_loss_fn(
            params, obs, a, logp, mb_advantages, mb_returns, reward, done, next_obs, next_done, key, hist_logprobs,
            args=args,  # pass through as keyword
            get_action_and_value2=get_action_and_value2,
            off_policy_correction=off_policy_correction,
            use_v_trace=use_v_trace,
        )
        alpha = jax.lax.stop_gradient(alpha)
        return alpha * loss, aux

    @staticmethod
    def _alpha_update_ppo(
            alpha: float,
            train_states: Tuple[TrainState, ...],
            storage: Storage,  # Storage with shapes (T, B, ...), where T is the number of steps and B is the batch size
            key: jax.random.PRNGKey,
            args: DreamSPIArgs,
            ppo_loss_grad_fn: Callable,
            num_minibatches: int = 1,  # Number of minibatches to use for PPO update
            off_policy_correction: bool = False,  # whether to use off-policy correction
    ):
        """
        Memory-friendly PPO update:
          - Flattens (T,B,...) -> (N,...)
          - Uses one permutation index for all fields (alignment)
          - Aggregates metrics inside scans (no per-minibatch stacking)
          - Returns per-epoch averages (1D arrays of length update_epochs) + last grads
        `alpha` is a scaling factor for the PPO loss; set it to one for standard PPO/Deep SPI.
        `alpha` will be provided as the second argument of `ppo_loss_grad_fn`.
        Set alpha to one for standard PPO/Deep SPI.
        """
        # ---- build flat dataset ----
        flat = jax.tree_map(flatten_first_two, storage)
        N = flat.obs.shape[0]
        _build_minibatches = partial(build_minibatches, N=N, num_minibatches=num_minibatches)

        # -------- inner scan: over minibatches (aggregate stats; return None to avoid stacking) --------
        def zeros_stats():
            z = jnp.array(0., jnp.float32)
            return dict(loss=z, pg=z, v=z, ent=z, kl=z, drift=z, t=z, r=z, gp=z)

        def add_stats(acc, loss, pg, v, ent, kl, drift, tloss, rloss, gp, grads):
            return {
                "loss": acc["loss"] + loss,
                "pg": acc["pg"] + pg,
                "v": acc["v"] + v,
                "ent": acc["ent"] + ent,
                "kl": acc["kl"] + kl,
                "drift": acc["drift"] + drift,
                "t": acc["t"] + tloss,
                "r": acc["r"] + rloss,
                "gp": acc["gp"] + gp,
            }

        def update_minibatch(carry, mb):
            train_states, key, acc, last_grads = carry
            (loss, (pg_loss, v_loss, ent_loss, approx_kl, drift_penalty_mean,
                    transition_loss, reward_loss, gradient_penalty, key)), grads = ppo_loss_grad_fn(
                train_states[0].params,
                mb.obs,
                mb.actions,
                mb.logprobs,
                mb.advantages,
                mb.returns,
                mb.rewards,
                mb.dones,
                mb.next_obs,
                mb.next_dones,
                key,
                mb.hist_logprobs if off_policy_correction else jnp.ones_like(mb.logprobs),
                alpha,
            )
            updated_train_states = []
            for train_state in train_states:
                new_train_state = train_state.apply_gradients(grads=grads)
                updated_train_states.append(new_train_state)
            acc = add_stats(acc, loss, pg_loss, v_loss, ent_loss, approx_kl,
                            drift_penalty_mean, transition_loss, reward_loss, gradient_penalty, grads)
            # Return None as ys to avoid stacking per-minibatch tensors
            return (tuple(updated_train_states), key, acc, grads), None

        # outer scan: over epochs (returns per-epoch averages)
        def update_epoch(carry, _):
            train_states, key, last_grads = carry
            key, subkey = jax.random.split(key)
            shuffled = _build_minibatches(subkey, flat)

            init_acc = zeros_stats()
            init_last_grads = jax.tree_map(lambda x: jnp.zeros_like(x), train_states[0].params)
            (train_states, key, acc, last_grads), _ = jax.lax.scan(
                update_minibatch, (train_states, key, init_acc, init_last_grads), shuffled
            )

            # Average over minibatches (scalar stats)
            denom = jnp.asarray(num_minibatches, dtype=jnp.float32)
            avg = jax.tree_map(lambda x: x / denom, acc)  # dict of scalars

            # Pack to arrays so scan can stack per-epoch metrics (shape [epochs])
            epoch_stats = (
                avg["loss"], avg["pg"], avg["v"], avg["ent"], avg["kl"], avg["drift"],
                avg["t"], avg["r"], avg["gp"]
            )
            # Keep last_grads in the carry so we do not stack the full grads tree across epochs
            return (train_states, key, last_grads), epoch_stats

        # Initialize a zero-like grads tree to seed the carry
        init_last_grads = jax.tree_map(lambda x: jnp.zeros_like(x), train_states[0].params)

        (train_states, key, last_grads), (loss, pg_loss, v_loss, entropy_loss,
                                         approx_kl, drift_penalty_mean,
                                         transition_loss, reward_loss, gradient_penalty) = jax.lax.scan(
            update_epoch, (train_states, key, init_last_grads), xs=None, length=args.update_epochs
        )

        return (
            train_states,
            loss, pg_loss, v_loss, entropy_loss, approx_kl, drift_penalty_mean,
            transition_loss, reward_loss, gradient_penalty,
            last_grads,
            key,
        )

    @classmethod
    def _update_ppo(
            cls,
            train_states: Tuple[TrainState, ...],
            storage: Storage,  # Storage with shapes (T, B, ...), where T is the number of steps and B is the batch size
            key: jax.random.PRNGKey,
            args: DreamSPIArgs,
            ppo_loss_grad_fn: Callable,
            num_minibatches: int = 1,  # Number of minibatches to use for PPO update
            off_policy_correction: bool = False,  # whether to use off-policy correction
    ):
        return cls._alpha_update_ppo(
            1., train_states,  storage, key, args, ppo_loss_grad_fn, num_minibatches, off_policy_correction)

    @staticmethod
    def _off_policy_ppo_loss(
            params,
            obs,
            a,
            logp,
            mb_advantages,
            mb_returns,
            reward,
            done,
            next_obs,
            next_done,
            key,
            hist_logprobs,
            args: DreamSPIArgs,
            get_action_and_value2: Callable,
            off_policy_correction: bool = True,  # dummy arg, not used here
            use_v_trace: bool = True,  # dummy arg, not used here
    ):
        """
        Off-policy PPO loss with auxiliary losses for the world model.
        Implements the loss from: https://ojs.aaai.org/index.php/AAAI/article/view/26099
        Only use when dyna_style=False and use_replay_buffer=True.
        """
        newlogprob, entropy, newvalue, transition_loss, reward_loss, gradient_penalty, key = get_action_and_value2(
            params, obs, a, reward, done, next_obs, next_done, key)
        if not args.piecewise_auxiliary_ratio:
            reward_loss = reward_loss.mean()
            transition_loss = transition_loss.mean()
        auxiliary_loss = args.reward_loss_coef * reward_loss + args.transition_loss_coef * transition_loss

        clip_low = 1.0 - args.clip_coef
        clip_high = 1.0 + args.clip_coef

        logratio = newlogprob - hist_logprobs
        ratio = jnp.exp(logratio)
        clip_low = jnp.exp(logp - hist_logprobs) * clip_low
        clip_high = jnp.exp(logp - hist_logprobs) * clip_high
        clipped_ratio = jnp.clip(ratio, clip_low, clip_high)

        if args.norm_adv:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

        log_ratio_on_policy= newlogprob - logp
        ratio_on_policy = jnp.exp(log_ratio_on_policy)
        approx_kl = ((ratio_on_policy - 1) - log_ratio_on_policy).mean()

        pg_loss1 = -(mb_advantages - auxiliary_loss) * ratio
        pg_loss2 = -(mb_advantages - auxiliary_loss) * clipped_ratio
        pg_loss = jnp.maximum(pg_loss1, pg_loss2)

        pg_loss = pg_loss.mean()

        # Value loss
        if args.use_huber:
            v_loss = jnp.mean(optax.huber_loss(newvalue - mb_returns, delta=1.0))
        else:
            v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

        drift_penalty_mean = jnp.array(0.0)

        entropy_loss = entropy.mean()
        reward_loss = reward_loss.mean()
        transition_loss = transition_loss.mean()
        loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + args.lambda_gp * gradient_penalty
        return loss, (
            pg_loss, v_loss, entropy_loss, jax.lax.stop_gradient(approx_kl),
            drift_penalty_mean, transition_loss, reward_loss, gradient_penalty, key)

    @staticmethod
    def _compute_vtrace_once(
            carry,
            inp,
            gamma,
            rho_bar=1.0,
            c_bar=1.0,
            log_ratio_clip=7.0,
            gae_lambda: float = 1.0,
            weighted_advantage: bool = False
    ):
        """
        Reverse-time one-step of V-trace:
          - carry: (v_tp1_hat, g_tp1) both (B,)
          - inp:   (next_done, v_tp1, v_t, r_t, logp_t, histlogp_t) each (B,)
        Returns:
          - new carry: (v_t_hat, g_t)
          - outputs:   (v_t_hat, g_t, rho_t, c_t, ratio) for logging/analysis
        """
        v_tp1_hat, g_tp1 = carry  # each (B,)
        next_done, v_tp1, v_t, r_t, logp_t, histlogp_t = inp

        next_nonterminal = 1.0 - next_done.astype(jnp.float32)

        # Importance ratio with symmetric log clip for stability
        log_ratio = jnp.clip(logp_t - histlogp_t, a_min=-log_ratio_clip, a_max=+log_ratio_clip)
        ratio = jnp.exp(log_ratio)

        rho_t = jnp.minimum(rho_bar, ratio)
        c_t = gae_lambda * jnp.minimum(c_bar, ratio)  # c_t is λ * min(c_bar, ratio)

        # TD error (untruncated delta; truncation enters via rho_t in actor, and via c_t in recursion)
        delta_t = r_t + gamma * next_nonterminal * v_tp1 - v_t

        # Critic target
        v_t_hat = v_t + rho_t * delta_t + gamma * next_nonterminal * c_t * (v_tp1_hat - v_tp1)

        if weighted_advantage:
            # V-trace policy-gradient advantage (IMPALA); not compatible with ppo off-policy corrections
            g_t = rho_t * delta_t + gamma * next_nonterminal * c_t * g_tp1
        else:
            g_t = r_t + gamma * next_nonterminal * v_tp1_hat - v_t

        new_carry = (v_t_hat, g_t)
        outs = (v_t_hat, g_t, rho_t, c_t, ratio)
        return new_carry, outs

    @staticmethod
    def _compute_v_trace(
            agent_state,
            next_obs,
            next_done,
            storage,
            args,
            actor_conv,
            actor_fc,
            critic,
            critic_conv,
            critic_fc,
            compute_vtrace_once,
    ):
        """
        Writes:
          - returns    := V-trace targets v_t_hat   (for the critic)
          - advantages := V-trace PG adv g_t        (for the actor; matches IMPALA)
        """
        # Get bootstrap value V_{T} from next_obs (your code path preserved)
        actor_conv_params, actor_fc_params = agent_state.params.agent.actor_network_params
        critic_conv_params, critic_fc_params = agent_state.params.agent.critic_network_params

        if args.decoupled_repr:
            z_prime = critic_conv.apply(critic_conv_params, next_obs)
            next_hidden = critic_fc.apply(critic_fc_params, z_prime)
        else:
            z_prime = actor_conv.apply(actor_conv_params, next_obs)
            next_hidden = actor_fc.apply(actor_fc_params, z_prime)

        next_value = critic.apply(agent_state.params.agent.critic_params, next_hidden).squeeze()

        # Concatenate for bootstrap alignment (T+1, B)
        dones = jnp.concatenate([storage.dones, next_done[None, :]], axis=0)  # (T+1, B)
        values = jnp.concatenate([storage.values, next_value[None, :]], axis=0)  # (T+1, B)

        # Prepare scan inputs (T, B) each: use time indices 0..T-1 with next at 1..T
        xs = (
            dones[1:],  # next_done_t
            values[1:],  # V_{t+1}
            values[:-1],  # V_t
            storage.rewards,  # r_t
            storage.logprobs,  # log π_t(a_t|s_t)
            storage.hist_logprobs,  # log μ_t(a_t|s_t)
        )

        # Initial carry is (v_{T}^, g_{T}) where:
        #   v_{T}^ = V_{T} (bootstrap)
        #   g_{T}  = 0     (no future advantage beyond horizon)
        init_vhat_T = values[-1]  # (B,)
        init_g_T = jnp.zeros_like(init_vhat_T)  # (B,)
        init_carry = (init_vhat_T, init_g_T)

        # Hyperparameters
        rho_bar = c_bar = 1.0
        log_ratio_clip = getattr(args, "vtrace_log_ratio_clip", 7.0)
        gamma = args.gamma

        def scan_fun(carry, inp):
            return compute_vtrace_once(
                carry, inp, gamma, rho_bar, c_bar, log_ratio_clip, args.gae_lambda,
                # we don't use the weighted advantages with off-policy PPO correction
                args.drift_formulation)

        # Reverse-time scan from T-1 down to 0
        (_, _), (v_targets, adv, rhos, cs, ratios) = jax.lax.scan(
            scan_fun,
            init_carry,
            xs,
            reverse=True
        )

        # Write back: critic uses V-trace targets; actor uses V-trace PG advantages
        storage = storage.replace(
            advantages=adv,  # (T, B)
            returns=v_targets,  # (T, B)
        )
        return storage

    @staticmethod
    def _update_maximizer(
            agent_state: TrainState,
            storage: Storage,  # Storage with shapes (T, B, ...), where T is the number of steps and B is the batch size
            key: jax.random.PRNGKey,
            # to be fixed
            args: DreamSPIArgs,
            mean_discrepancy_grad_fn: Callable,
            actor_conv: NetworkConv,
            transition_network: DiscreteActionTransitionNetwork,
            num_minibatches: int = 1,  # Number of minibatches to use for PPO update
    ):
        flat = jax.tree_map(flatten_first_two, storage)
        N = flat.obs.shape[0]
        _build_minibatches = partial(build_minibatches, N=N, num_minibatches=num_minibatches)

        def update_minibatches(carry, mb):
            agent_state, key, grads = carry
            z = actor_conv.apply(agent_state.params.agent.actor_network_params[0], mb.obs)
            z_prime = actor_conv.apply(agent_state.params.agent.actor_network_params[0], mb.next_obs)
            transition_dist = transition_network.apply(
                agent_state.params.world_model.transition_network_params, (z, mb.actions))
            key, subkey = jax.random.split(key)
            z_prime_sampled = transition_dist.sample(seed=subkey)
            loss, grads = mean_discrepancy_grad_fn(
                agent_state.params,
                mb.obs,
                mb.actions,
                z,
                z_prime,
                z_prime_sampled,
            )
            agent_state = agent_state.apply_gradients(grads=grads)
            return (agent_state, key, grads), loss

        # outer scan: over epochs (returns per-epoch averages)
        def update_epoch(carry, _):
            agent_state, key, _ = carry
            key, subkey = jax.random.split(key)
            shuffled = _build_minibatches(subkey, flat)

            init_grads = jax.tree_map(lambda x: jnp.zeros_like(x), agent_state.params)
            (agent_state, key, grads), loss = jax.lax.scan(
                update_minibatches, (agent_state, key, init_grads), shuffled)

            return (agent_state, key, grads), loss

        init_last_grads = jax.tree_map(lambda x: jnp.zeros_like(x), agent_state.params)
        (agent_state, key, last_grads), loss = jax.lax.scan(
            update_epoch, (agent_state, key, init_last_grads), xs=None, length=args.n_max_updates
        )

        return agent_state, loss, last_grads, key



    @staticmethod
    def _get_by_path(tree, path):
        """Navigate a (possibly FrozenDict/dataclass) pytree following a tuple path."""
        from flax.serialization import to_state_dict as _to_state
        node = _to_state(tree)
        for k in path:
            if isinstance(node, dict):
                if k in node:
                    node = node[k]
                elif isinstance(k, int) and str(k) in node:
                    node = node[str(k)]
                else:
                    raise KeyError(f"Path not found: {'/'.join(map(str, path))} (missing key: {k})")
            elif isinstance(node, (list, tuple)):
                if isinstance(k, int) and 0 <= k < len(node):
                    node = node[k]
                else:
                    raise KeyError(f"Bad index into sequence at {'/'.join(map(str, path))}: {k}")
            else:
                raise KeyError(
                    f"Cannot descend into non-container at {'/'.join(map(str, path))}; "
                    f"stuck on {k} ({type(node)})"
                )
        return node

    @classmethod
    def _assert_unchanged(cls, params_before, params_after, paths, *, who):
        """Assert that the subtrees at each path have not changed (L2 diff == 0).
        Works for leaves or whole subtrees (dicts/FrozenDicts/tuples).
        """
        from flax.serialization import to_state_dict as _to_state

        def _collect_leaves(x, base=()):
            # Return {relative_path_tuple: jnp.ndarray or None} of all array leaves.
            out = {}
            # Normalize containers first (e.g., Flax structs)
            if not isinstance(x, (dict, tuple, list)):
                try:
                    x = _to_state(x)
                except Exception:
                    pass
            if isinstance(x, dict):
                for k, v in x.items():
                    out.update(_collect_leaves(v, base + (k,)))
            elif isinstance(x, (list, tuple)):
                for i, v in enumerate(x):
                    out.update(_collect_leaves(v, base + (i,)))
            else:
                if x is None:
                    out[base] = None
                else:
                    out[base] = jnp.asarray(x)
            return out

        for path in paths:
            a = cls._get_by_path(params_before, path)
            b = cls._get_by_path(params_after, path)
            a_leaves = _collect_leaves(a)
            b_leaves = _collect_leaves(b)

            # Same set of leaves?
            if set(a_leaves.keys()) != set(b_leaves.keys()):
                missing_a = sorted(set(b_leaves.keys()) - set(a_leaves.keys()))
                missing_b = sorted(set(a_leaves.keys()) - set(b_leaves.keys()))
                raise AssertionError(
                    f"[{who}] structure changed at {'/'.join(map(str, path))}:\n"
                    f"  missing in BEFORE: {['/'.join(map(str, k)) for k in missing_a[:5]]}\n"
                    f"  missing in AFTER:  {['/'.join(map(str, k)) for k in missing_b[:5]]}"
                )

            # Compare leaves
            for rel in a_leaves.keys():
                aval = a_leaves[rel]
                bval = b_leaves[rel]
                if aval is None and bval is None:
                    continue
                if (aval is None) != (bval is None):
                    raise AssertionError(f"[{who}] None vs non-None at {'/'.join(map(str, path + rel))}")
                if aval.shape != bval.shape:
                    raise AssertionError(
                        f"[{who}] shape mismatch at {'/'.join(map(str, path + rel))}: {aval.shape} vs {bval.shape}"
                    )
                diff = jnp.linalg.norm((aval - bval).reshape(-1))
                if float(diff) != 0.0:
                    raise AssertionError(
                        f"[{who}] parameters changed at {'/'.join(map(str, path + rel))}; L2 diff = {float(diff):.3e}"
                    )

    def check_update(self, prev_params, new_params, who, dyna_style=False):
        """Check that the parameters have not changed in unexpected ways after an update."""
        {
            # Verify: Agent step must NOT change world model params
            'agent': lambda: self._assert_unchanged(
                prev_params, new_params,
                paths=[
                    ("world_model", "transition_network_params",),
                    ("world_model", "reward_network_params",),
                    ("world_model", "done_predictor_params",),
                    ("discriminator",),
                ],
                who="agent_step",
            ),
            # Verify: representation/world model step must NOT change agent params nor discriminator params
            "repr_wm": lambda: self._assert_unchanged(
                prev_params, new_params,
                paths=[("discriminator", ), ],
                who="repr_and_world_model_learning_step",
            ) if dyna_style else self._assert_unchanged(
                prev_params, new_params,
                paths=[
                    ("discriminator", ),
                    ("agent", "actor_network_params", 1),
                    ("agent", "actor_params"),
                    ("agent", "critic_network_params"),
                    ("agent", "critic_params"),
                ],
                who="repr_and_world_model_learning_step",
            ),
            # Verify: Maximizer step must NOT change world model nor agent params
            "maximizer": lambda: self._assert_unchanged(
                prev_params, new_params,
                paths=[
                    ("agent", ),
                    ("world_model", "transition_network_params"),
                    ("world_model", "reward_network_params"),
                    ("world_model", "done_predictor_params"),
                ],
                who="maximizer_step",
            ),
        }[who]()

    @staticmethod
    def alpha_imagine(global_step, max_steps, start=0., end=1.):
        """Linear schedule for the alpha parameter controlling the mix of real and imagined data."""
        progress = jnp.clip(global_step / max_steps, 0., 1.)
        return start + (end - start) * progress

    @staticmethod
    def alpha_imagine_exp(global_step, max_steps, start=0., end=1.):
        """Exponential schedule for the alpha parameter controlling the mix of real and imagined data."""
        progress = jnp.clip(global_step / max_steps, 0., 1.)
        # avoid log(0) if start=0, so shift slightly or clamp
        eps = 1e-8
        start_ = jnp.maximum(start, eps)
        end_ = jnp.maximum(end, eps)
        return start_ * (end_ / start_) ** progress


def get_latent_states_and_dones(
        obs: jnp.ndarray,
        dones: jnp.ndarray,
        actor_network_params: Tuple,
        actor_conv: NetworkConv,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Extract latent states and dones from the storage."""
    # Reshape the observations to be 2D (B, H, W) for the actor_conv
    obs = flatten_first_two(obs)
    actor_conv_params, _ = actor_network_params
    z = actor_conv.apply(actor_conv_params, obs)
    dones = dones.reshape((-1,)).astype(jnp.int32)
    return z, dones


def sync_train_states(
        encoder_state: TrainState,
        world_model_state: TrainState,
        actor_state: TrainState,
        critic_state: TrainState,
        maximizer_state: Optional[TrainState],
        args: DreamSPIArgs,
) -> Tuple[TrainState, TrainState, TrainState, TrainState, Optional[TrainState]]:
    """Sync the parameters of all train states."""
    world_model_params = world_model_state.params.world_model
    actor_conv_params, _ = encoder_state.params.agent.actor_network_params
    _, actor_fc_params = actor_state.params.agent.actor_network_params
    actor_params = actor_state.params.agent.actor_params
    if args.decoupled_repr:
        # even if the decoupled representation setting, the critic conv is tied to actor conv;
        # the critic fc is separate, meaning the representation is "semi"-decoupled.
        _, critic_fc_params = critic_state.params.agent.critic_network_params
    else:
        critic_fc_params = actor_fc_params
    critic_params = critic_state.params.agent.critic_params
    if maximizer_state is not None:
        discriminator_params = maximizer_state.params.discriminator
    else:
        discriminator_params = None

    train_states = []
    for train_state in (encoder_state, world_model_state, actor_state, critic_state, maximizer_state):
        if train_state is None:
            train_states.append(None)
            continue
        params = train_state.params
        new_params = params.replace(
            world_model=world_model_params,
            agent=params.agent.replace(
                actor_network_params=(actor_conv_params, actor_fc_params),
                actor_params=actor_params,
                critic_network_params=(actor_conv_params, critic_fc_params),
                critic_params=critic_params,
            ),
        )
        if maximizer_state is not None:
            new_params = new_params.replace(discriminator=discriminator_params)
        new_train_state = train_state.replace(params=new_params)
        train_states.append(new_train_state)

    return tuple(train_states)  # type: ignore


def sample_from_rb(
    replay_buffer: Union[ReplayBuffer, DeviceReplayBuffer, tuple],
    rng: jax.random.PRNGKey,
    batch_width: int,
    batch_length: int,
    *,
    avoid_cross_terminal: bool = False,
    device: Optional[jax.Device] = None,
    rb_lies_on_gpu: bool = False,
) -> Storage:
    if not rb_lies_on_gpu:
        return replay_buffer.sample(
            rng=rng,
            width=batch_width,
            length=batch_length,
            avoid_cross_terminal=avoid_cross_terminal,
            compute_next_from_obs=True,  # do not store next_*; derive from obs/dones
            device=device,
        )
    elif isinstance(replay_buffer, DeviceReplayBuffer):
        return rb_gpu_sample_batch(
            replay_buffer=replay_buffer,
            rng=rng,
            batch_width=int(batch_width),
            batch_length=int(batch_length),
            avoid_cross_terminal=avoid_cross_terminal,
        )
    else:  # flashbax
        return sample_from_fbx_rb(replay_buffer, rng, batch_width, batch_length, avoid_cross_terminal)


def train(
        agent: DreamSPIAgent,
        envs: gym.vector.VectorEnv,
        handle,
        args: DreamSPIArgs,
        writer: SummaryWriter,
        key: jax.random.PRNGKey
) -> DreamSPIAgent:
    global_step = 0
    start_time = time.time()
    next_obs = envs.reset()
    next_done = jnp.zeros(args.num_envs, dtype=jax.numpy.bool_)

    episode_stats = EpisodeStatistics(
        episode_returns=jnp.zeros(args.num_envs, dtype=jnp.float32),
        episode_lengths=jnp.zeros(args.num_envs, dtype=jnp.int32),
        returned_episode_returns=jnp.zeros(args.num_envs, dtype=jnp.float32),
        returned_episode_lengths=jnp.zeros(args.num_envs, dtype=jnp.int32),
    )

    # Get the cleanRL ppo scores
    csv_path = args.compare_scores_csv
    score_col = args.compare_scores_column
    scores_mean, scores_sigma = load_env_mean_std(csv_path, score_col)
    max_avg_return = -np.infty
    encoder_state = agent.repr_train_state
    world_model_state = agent.world_model_train_state
    actor_state = agent.actor_train_state
    critic_state = agent.critic_train_state
    maximizer_state = agent.maximizer_train_state

    get_latent_states_and_dones_fn = partial(get_latent_states_and_dones, actor_conv=agent.actor_conv)
    get_latent_states_and_dones_fn = jax.jit(get_latent_states_and_dones_fn)

    for iteration in range(1, args.num_iterations + 1):
        outer_start = time.time()
        # Outer rollout on real env
        world_model_state, episode_stats, next_obs, next_done, storage, key, handle = agent.rollout(
            world_model_state, episode_stats, next_obs, next_done, key, handle
        )
        global_step += args.num_steps * args.num_envs
        # get the alpha value for the current global step
        if args.progressive:
            if args.progression_scheme == 'linear':
                alpha = agent.alpha_imagine(global_step, args.total_timesteps)
            else:  # exponential
                alpha = agent.alpha_imagine_exp(global_step, args.total_timesteps)
        else:
            alpha = 1.
        # Compute GAE for the world model
        storage = agent.compute_gae_for_wm(world_model_state, next_obs, next_done, storage)
        # Update the world model
        _params_before_wm_update = world_model_state.params

        if args.dyna_style:
            train_states = (encoder_state, world_model_state, actor_state, critic_state)
        else:
            train_states = (encoder_state, world_model_state)
        (
            train_states, loss, pg_loss, v_loss, entropy_loss, approx_kl, drift_penalty_mean,
            transition_loss, reward_loss, gradient_penalty, grads_wm, key
        ) = agent.update_world_model(train_states, storage, key)
        if args.dyna_style:
            encoder_state, world_model_state, actor_state, critic_state = train_states
        else:
            encoder_state, world_model_state = train_states

        if args.debug:
            # Debugging: check that the actor/critic parameters did not change during
            # the world model update (except the actor conv)
            agent.check_update(
                _params_before_wm_update, world_model_state.params, who="repr_wm", dyna_style=args.dyna_style)

        # sync the parameters of all train states with the new world model and representation parameters
        encoder_state, world_model_state, actor_state, critic_state, maximizer_state = sync_train_states(
            encoder_state, world_model_state, actor_state, critic_state, maximizer_state, args)

        if maximizer_state is not None:
            # Update the Wasserstein discriminator
            _params_before_max_update = maximizer_state.params
            maximizer_state, maximizer_loss, grads_maximizer, key = agent.update_maximizer(
                maximizer_state, storage, key)

            if args.debug:
                # Debugging: check that the world model parameters did not change during the maximizer update
                agent.check_update(_params_before_max_update, maximizer_state.params, "maximizer")

            # update the parameters with the new discriminator parameters
            encoder_state, world_model_state, actor_state, critic_state, maximizer_state = sync_train_states(
                encoder_state, world_model_state, actor_state, critic_state, maximizer_state, args)
        else:
            maximizer_loss = None

        avg_episodic_return = np.mean(jax.device_get(episode_stats.returned_episode_returns))
        max_avg_return = np.max([max_avg_return, avg_episodic_return])

        log(
            global_step, avg_episodic_return, max_avg_return, world_model_state,
            v_loss, pg_loss, entropy_loss, approx_kl, drift_penalty_mean,
            transition_loss, reward_loss, scores_mean, gradient_penalty,
            start_time, outer_start, episode_stats, loss, args, writer, grads_wm,
            prefix='env_step' if args.dyna_style else 'repr_wm_step',
            multiple_lr=True, print_logs=True, maximizer_loss=maximizer_loss, alpha=alpha)

        # gather the latent states and dones from the storage and start imagination rollouts from those latent states
        z_prime, im_next_done = get_latent_states_and_dones_fn(
            storage.obs, storage.dones, actor_state.params.agent.actor_network_params)

        # *latent imagination*
        inner_start = time.time()
        # Duplicate the starting latent states/dones so we can run all imagination rollouts in one pass
        n_rollouts = int(args.n_imagination_rollouts)
        for rollout in range(n_rollouts):
            z_start = z_prime
            d_start = im_next_done

            # Perform imagination rollouts once (batched). This returns the final next_obs/next_done to feed into GAE.
            # Note that we provide actor_state while both actor and critic params are used. It doesn't matter as
            # the actor_state and critic_state parameters have just been synced before.
            actor_state, z_after, im_next_done_after, imagination_storage, key = agent.imagine(
                actor_state, z_start, d_start, key)

            # Compute GAE for the imagination storage using the final state from the imagination scan
            # Same here; we use actor_state which is already synced with critic_state
            imagination_storage = agent.compute_gae(actor_state, z_after, im_next_done_after, imagination_storage)

            # Apply a single PPO update on the aggregated imagination storage
            _params_before_agent_update = actor_state.params
            (
                (actor_state, critic_state), loss, pg_loss, v_loss, entropy_loss, approx_kl, drift_penalty_mean,
                _, _, _, grads_agent, key
            ) = agent.update_ppo(
                alpha,
                (actor_state, critic_state),
                imagination_storage,
                key,
            )
            if args.debug:
                # Debugging: check that the world model parameters did not change during the agent update
                agent.check_update(_params_before_agent_update, actor_state.params, "agent")

            # Sync/update the parameters of all train states with the new actor/critic parameters
            encoder_state, world_model_state, actor_state, critic_state, maximizer_state = sync_train_states(
                encoder_state, world_model_state, actor_state, critic_state, maximizer_state, args)

        if n_rollouts > 0:
            inner_steps = args.num_envs * args.num_steps * args.imagination_horizon * args.n_imagination_rollouts
            log(global_step, avg_episodic_return, max_avg_return, actor_state,
                v_loss, pg_loss, entropy_loss, approx_kl, drift_penalty_mean,
                None, None, scores_mean, None,
                start_time, inner_start, None, loss, args, writer, grads_agent, prefix='imagination_step', multiple_lr=True,
                intermediate_step=inner_steps, print_logs=False, alpha=alpha)

    return agent.replace(
        repr_train_state=encoder_state,
        world_model_train_state=world_model_state,
        actor_train_state=actor_state,
        critic_train_state=critic_state,
        maximizer_train_state=maximizer_state,)

def train_with_rb(
        agent: DreamSPIAgent,
        envs: gym.vector.VectorEnv,
        handle,
        args: DreamSPIArgs,
        writer: SummaryWriter,
        key: jax.random.PRNGKey
) -> DreamSPIAgent:
    global_step = 0
    start_time = time.time()
    next_obs_env = envs.reset()
    next_done_env = jnp.zeros(args.num_envs, dtype=jax.numpy.bool_)

    episode_stats = EpisodeStatistics(
        episode_returns=jnp.zeros(args.num_envs, dtype=jnp.float32),
        episode_lengths=jnp.zeros(args.num_envs, dtype=jnp.int32),
        returned_episode_returns=jnp.zeros(args.num_envs, dtype=jnp.float32),
        returned_episode_lengths=jnp.zeros(args.num_envs, dtype=jnp.int32),
    )

    # Get the cleanRL ppo scores
    csv_path = args.compare_scores_csv
    score_col = args.compare_scores_column
    scores_mean, scores_sigma = load_env_mean_std(csv_path, score_col)
    max_avg_return = -np.infty
    encoder_state = agent.repr_train_state
    world_model_state = agent.world_model_train_state
    actor_state = agent.actor_train_state
    critic_state = agent.critic_train_state
    maximizer_state = agent.maximizer_train_state

    encoder_state, world_model_state, actor_state, critic_state, maximizer_state = sync_train_states(
        encoder_state, world_model_state, actor_state, critic_state, maximizer_state, args)

    rb: Optional[Union[ReplayBuffer, DeviceReplayBuffer, tuple]] = None

    get_latent_states_and_dones_fn = partial(get_latent_states_and_dones, actor_conv=agent.actor_conv)
    get_latent_states_and_dones_fn = jax.jit(get_latent_states_and_dones_fn)

    for iteration in range(1, args.num_iterations + 1):
        outer_start = time.time()
        # Outer rollout on real env
        world_model_state, episode_stats, next_obs_env, next_done_env, storage, key, handle = agent.rollout(
            world_model_state, episode_stats, next_obs_env, next_done_env, key, handle
        )
        if args.rb_type == 'flashbax':
            if rb is None:
                example = storage.replace(returns=None, advantages=None, next_obs=None, next_dones=None)
                rb = _make_flashbax_rb(
                    args, example,
                    n_envs=args.num_envs,
                    action_n_classes=envs.single_action_space.n,  # <= 256 → uint8
                    prioritize_ends=args.prioritize_new_samples
                )
            fbx_buffer, fbx_state, fbx_add_rollout, fbx_sample, fbx_can_sample, meta = rb
            fbx_state = fbx_add_rollout(fbx_state, storage)
            rb = (fbx_buffer, fbx_state, fbx_add_rollout, fbx_sample, fbx_can_sample, meta)
        elif args.rb_type == 'gpu':
            if rb is None:
                example = storage.replace(returns=None, advantages=None, next_obs=None, next_dones=None)
                config = RBConfig(
                    T_cap=args.replay_buffer_capacity // args.num_envs,
                    W_cap=args.num_envs,
                    store_next_obs=False,
                    store_logprobs=True,
                    obs_dtype=jnp.uint8,
                    reward_dtype=jnp.float32,
                    action_dtype=None,
                    action_n_classes=envs.single_action_space.n,
                )
                rb = init_replay_buffer(config, example, device=jax.devices("gpu")[0])
            rb, info = rb_gpu_add_to_buffer(rb, storage)
        else:  # numpy or device replay buffer
            if rb is None:
                example = storage.replace(returns=None, advantages=None, next_obs=None, next_dones=None)
                rb = ReplayBuffer(
                    T_cap=args.replay_buffer_capacity // args.num_envs,
                    example=example,
                )
            rb.add(storage)

        global_step += args.num_steps * args.num_envs
        num_wm_it = args.num_envs * args.num_steps // args.env_steps_per_wm_update
        for wm_it in range(num_wm_it):
            wm_start = time.time()

            if args.rb_type == 'flashbax':
                _, fbx_state, _, _, fbx_can_sample, _ = rb
                if not bool(jax.device_get(fbx_can_sample(fbx_state))):
                    break
            else:
                if rb.size < args.rb_prefill:
                    break

            key, subkey = jax.random.split(key)
            batch = sample_from_rb(
                rb, subkey, args.rb_batch_size, args.rb_sequence_length,
                avoid_cross_terminal=True,
                rb_lies_on_gpu=args.rb_type in ('gpu', 'flashbax'),
            )
            batch = batch.replace(hist_logprobs=batch.logprobs)
            # re-compute action and value for the sampled batch
            logprobs, values = agent.get_logprobs_and_value(
                world_model_state, flatten_first_two(batch.obs), flatten_first_two(batch.actions))
            logprobs = logprobs.reshape((args.rb_sequence_length, args.rb_batch_size))
            values = values.reshape((args.rb_sequence_length, args.rb_batch_size))
            batch = batch.replace(logprobs=logprobs, values=values)

            # Compute GAE for the world model
            next_obs = batch.next_obs[-1]
            next_done = batch.next_dones[-1]
            if args.use_v_trace:
                # compute V-traces for the sampled batch
                storage = agent.compute_v_trace(world_model_state, next_obs, next_done, batch)
            else:
                # otherwise, fall back to standard GAE
                storage = agent.compute_gae_for_wm(world_model_state, next_obs, next_done, batch)

            # Update the world model
            _params_before_wm_update = world_model_state.params

            if args.dyna_style:
                train_states = (encoder_state, world_model_state, actor_state, critic_state)
            else:
                train_states = (encoder_state, world_model_state)
            (
                train_states, loss, pg_loss, v_loss, entropy_loss, approx_kl, drift_penalty_mean,
                transition_loss, reward_loss, gradient_penalty, grads_wm, key
            ) = agent.update_world_model(train_states, storage, key)
            if args.dyna_style:
                encoder_state, world_model_state, actor_state, critic_state = train_states
            else:
                encoder_state, world_model_state = train_states

            if args.debug:
                # Debugging: check that the actor/critic parameters did not change during
                # the world model update (except the actor conv)
                agent.check_update(
                    _params_before_wm_update, world_model_state.params, who="repr_wm", dyna_style=args.dyna_style)

            # sync the parameters of all train states with the new world model and representation parameters
            encoder_state, world_model_state, actor_state, critic_state, maximizer_state = sync_train_states(
                encoder_state, world_model_state, actor_state, critic_state, maximizer_state, args)

            if maximizer_state is not None:
                # Update the Wasserstein discriminator
                _params_before_max_update = maximizer_state.params
                maximizer_state, maximizer_loss, grads_maximizer, key = agent.update_maximizer(
                    maximizer_state, storage, key)

                if args.debug:
                    # Debugging: check that the world model parameters did not change during the maximizer update
                    agent.check_update(_params_before_max_update, maximizer_state.params, "maximizer")

                # update the parameters with the new discriminator parameters
                encoder_state, world_model_state, actor_state, critic_state, maximizer_state = sync_train_states(
                    encoder_state, world_model_state, actor_state, critic_state, maximizer_state, args)
            else:
                maximizer_loss = None

            avg_episodic_return = np.mean(jax.device_get(episode_stats.returned_episode_returns))
            max_avg_return = np.max([max_avg_return, avg_episodic_return])

            if wm_it == num_wm_it - 1:
                log(
                    global_step, avg_episodic_return, max_avg_return, world_model_state,
                    v_loss, pg_loss, entropy_loss, approx_kl, drift_penalty_mean,
                    transition_loss, reward_loss, scores_mean, gradient_penalty,
                    start_time, outer_start, episode_stats, loss, args, writer, grads_wm,
                    prefix='world_model_update' if args.dyna_style else 'repr_wm_step',
                    multiple_lr=True, print_logs=True, maximizer_loss=maximizer_loss,)

            # gather the latent states and dones from the storage and start imagination rollouts from those latent states
            z_prime, im_next_done = get_latent_states_and_dones_fn(
                storage.obs, storage.dones, actor_state.params.agent.actor_network_params)

            # *latent imagination*
            inner_start = time.time()
            n_rollouts = int(
                args.n_imagination_rollouts *  args.rb_batch_size * args.rb_sequence_length * args.imagination_horizon
            ) // (args.minibatch_size * args.num_minibatches)
            for rollout in range(n_rollouts):

                if n_rollouts == 1:
                    z_start = z_prime
                    d_start = im_next_done
                else:
                    key, subkey = jax.random.split(key)
                    idx = jax.random.randint(
                        subkey, (args.minibatch_size * args.num_minibatches,), 0, z_prime.shape[0]
                    )
                    z_start = z_prime[idx, ...]
                    d_start = im_next_done[idx, ...]

                # Perform imagination rollouts once (batched). This returns the final next_obs/next_done to feed into GAE.
                # Note that we provide actor_state while both actor and critic params are used. It doesn't matter as
                # the actor_state and critic_state parameters have just been synced before.
                actor_state, z_after, im_next_done_after, imagination_storage, key = agent.imagine(
                    actor_state, z_start, d_start, key)

                # Compute GAE for the imagination storage using the final state from the imagination scan
                # Same here; we use actor_state which is already synced with critic_state
                imagination_storage = agent.compute_gae(actor_state, z_after, im_next_done_after, imagination_storage)

                # Apply a single PPO update on the aggregated imagination storage
                _params_before_agent_update = actor_state.params
                (
                    (actor_state, critic_state), loss, pg_loss, v_loss, entropy_loss, approx_kl, drift_penalty_mean,
                    _, _, _, grads_agent, key
                ) = agent.update_ppo(
                    1.,  # alpha is always 1 with replay buffer
                    (actor_state, critic_state),
                    imagination_storage,
                    key,
                )
                if args.debug:
                    # Debugging: check that the world model parameters did not change during the agent update
                    agent.check_update(_params_before_agent_update, actor_state.params, "agent")

                # Sync/update the parameters of all train states with the new actor/critic parameters
                encoder_state, world_model_state, actor_state, critic_state, maximizer_state = sync_train_states(
                    encoder_state, world_model_state, actor_state, critic_state, maximizer_state, args)

            inner_steps = args.num_envs * args.num_steps * args.imagination_horizon * args.n_imagination_rollouts

            if args.rb_print_stats:
                print(f"Iteration {iteration}, WM it {wm_it+1}/{num_wm_it}, "
                      f"AvgReturn {avg_episodic_return:.2f}, MaxAvgReturn {max_avg_return:.2f}, "
                      f"Time {time.time() - wm_start:.2f}s",)
            if wm_it == num_wm_it - 1:
                log(global_step, avg_episodic_return, max_avg_return, actor_state,
                    v_loss, pg_loss, entropy_loss, approx_kl, drift_penalty_mean,
                    None, None, scores_mean, None,
                    start_time, inner_start, None, loss, args, writer, grads_agent,
                    prefix='imagination_update', multiple_lr=True, intermediate_step=inner_steps, print_logs=False)

    return agent.replace(
        repr_train_state=encoder_state,
        world_model_train_state=world_model_state,
        actor_train_state=actor_state,
        critic_train_state=critic_state,
        maximizer_train_state=maximizer_state,)

if __name__ == "__main__":
    args = tyro.cli(DreamSPIArgs)
    check_and_process_args(args)

    if args.use_replay_buffer:
        args.batch_size = int(args.rb_batch_size * args.rb_sequence_length)
        args.minibatch_size = int(args.batch_size // args.num_minibatches)

    args.wm_num_minibatches = args.num_minibatches
    if args.imagination_num_minibatches is None:
        args.num_minibatches *= args.imagination_horizon * args.n_imagination_rollouts

    if args.global_anneal_lr:
        args.encoder_anneal_lr = True
        args.world_model_anneal_lr = True
        args.actor_anneal_lr = True
        args.critic_anneal_lr = True

    if args.use_replay_buffer and not args.drift_formulation:
        warnings.warn("Using V-trace with replay buffer. Setting use_v_trace=True.")
        args.use_v_trace = True

    if args.progressive:
        if not args.dyna_style:
            warnings.warn("Progressive training is only available with dyna_style=True."
                          "Setting dyna_style=True.")
            args.dyna_style = True
        assert (not args.use_replay_buffer), "Progressive training is not compatible with replay buffer."

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
            tags=args.wandb_tags
        )
        wandb.config.update(vars(args), allow_val_change=True)

        if args.hp_tuning_mode:
            # Log the final args, including any changes made by your code
            for k, v in vars(args).items():
                wandb.run.summary[f"hyperparam/{k}"] = v

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)

    # env setup
    envs = make_env(args.env_id, args.seed, args.num_envs, args.reward_clip, args.stochastic_env)()
    handle, recv, send, step_env = envs.xla()
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = DreamSPIAgent.create(
        args=args,
        envs=envs,
        raw_step_env=step_env,
        key=key)

    if args.use_replay_buffer:
        agent = train_with_rb(agent, envs, handle, args, writer, key)
    else:
        agent = train(agent, envs, handle, args, writer, key)

    envs.close()
    writer.close()
