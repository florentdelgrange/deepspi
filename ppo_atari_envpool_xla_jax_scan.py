# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_envpool_xla_jaxpy
import os
import random
import time
import warnings
from dataclasses import dataclass, field
from functools import partial
from typing import Tuple, Optional

import envpool
import flax
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from flax.training.train_state import TrainState
from torch.utils.tensorboard import SummaryWriter

from utils.gp import lipschitz_gp, wgan_gp
from utils.log import flatten_dict
from utils.loss import smooth_l1_loss, hamming_distance, CategoricalCost
from networks.architectures import NetworkConv, NetworkFCOutput, Actor, Critic, DiscreteActionTransitionCNN, \
    DiscreteActionRewardCNN, LipDiscreteActionTransitionCNN, LipDiscreteActionRewardNetwork, \
    CategoricalEncoder, NetworkAttentionOutput, DiscreteActionTransitionNetworkSoftMoE, \
    DiscreteActionRewardNetworkSoftMoE, NetType, DiscreteActionTransitionNetwork, DiscreteActionRewardNetwork, \
    AutoregressiveDiscreteActionTransitionTransformer
from utils.distributions import TransitionDensity
from utils.scores import human_normalized_score, load_env_mean_std, atari_human_normalized_scores, ratio_vs_baseline, \
    rel_improvement, z_score_vs_baseline

# Fix weird OOM https://github.com/google/jax/discussions/6332#discussioncomment-1279991
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6"
# Fix CUDNN non-determinisim; https://github.com/google/jax/issues/4823#issuecomment-952835771
os.environ["TF_XLA_FLAGS"] = "--xla_gpu_autotune_level=2 --xla_gpu_deterministic_reductions"
os.environ["TF_CUDNN DETERMINISTIC"] = "1"


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    track_params: bool = False
    """if toggled, parameters and grads will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    wandb_tags: list[str] = field(default_factory=list)
    """the tags of the wandb's run"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""
    stochastic_env: bool = False
    """whether to use stochastic environment (default is False, which means deterministic environment)"""
    reward_clip: bool = True
    """whether to clip the reward to [-1, 1] (default is True, which means rewards are clipped)"""
    compare_scores_csv: str = "utils/cleanrl_result_table.csv"
    """the csv file to save the comparison scores of this experiment with other experiments"""
    compare_scores_column: str = "CleanRL's ppo_atari_envpool_xla_jax.py"
    """the column name of the comparison scores in the csv file"""
    compare_scores_max: bool = False
    """whether to compare scores with the max value reported (default is False, which means *last* score reported)"""
    hp_tuning_mode: bool = False
    """if toggled, the script will run in parameter tuning mode, which means it will not run the full experiment"""

    # Algorithm specific arguments
    env_id: str = "Breakout-v5"
    """the id of the environment"""
    random_env_id: bool = False
    """if toggled, a random environment id will be used from the envpool registry; ignore the env_id argument"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 8
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    parallel_envs_config: str = None
    """string of the form 'n_env=8,n_steps=128' to override num_envs and num_steps"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    decoupled_repr: bool = False
    """whether to use decoupled actor and critic representations"""
    drift_formulation: bool = False
    """whether to use the drift formulation of PPO (default is the standard PPO)"""
    drift_coef: float = 1.0
    """drift coefficient drif_coef * D_{π_n}(π_{n + 1} | s) (only used if using drift formulation)"""
    transition_loss_coef: float = 1.0
    """coefficient for the transition loss (only if using world model)"""
    reward_loss_coef: float = 1.0
    """coefficient for the reward loss (only if using world model)"""
    transition_density: TransitionDensity = TransitionDensity.DETERMINISTIC
    """Choice of transition distribution: DETERMINISTIC | NORMAL | MIXTURE_NORMAL | CATEGORICAL"""
    lambda_gp: float = 0.01
    """coefficient for the gradient penalty (only if using world model)"""
    use_wgan_gp: bool = False
    """whether to use WGAN gradient penalty for enforcing the Lipschitzness of the world model"""
    lipschitz_nets: bool = False
    """whether to use Lipschitz networks for the world model (only if using world model)"""
    use_gumbel_softmax: bool = False
    """whether to use Gumbel-Softmax for the actor network (only if transition_density is CATEGORICAL)"""
    categorical_cost: CategoricalCost = CategoricalCost.L2
    """Choice of the categorical cost: L2 | CROSS_ENTROPY | HAMMING | JENSEN_SHANNON (only if transition_density is CATEGORICAL)"""
    use_attention: bool = False
    """whether to use attention in the actor network (only if transition_density is CATEGORICAL)"""
    auxiliary_task_net_type: NetType = NetType.FC
    """Type of auxiliary task network: CONV | FC | SOFTMOE | TRANSFORMER (only if transition_density is CATEGORICAL)"""
    layer_norm_cnn_output: bool = False
    """whether to use layer normalization before the CNN output"""
    use_feature_group: bool = False
    """whether to use feature group convolution in the CNN transition network (only if auxiliary_task_net_type is CONV)"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

def make_env(env_id, seed, num_envs):
    def thunk():
        envs = envpool.make(
            env_id,
            env_type="gym",
            num_envs=num_envs,
            episodic_life=True,
            reward_clip=args.reward_clip,
            seed=seed,
            repeat_action_probability=0.3 if args.stochastic_env else 0.,
            noop_max=60 if args.stochastic_env else 30,
        )
        envs.num_envs = num_envs
        envs.single_action_space = envs.action_space
        envs.single_observation_space = envs.observation_space
        envs.is_vector_env = True
        return envs

    return thunk


@flax.struct.dataclass
class AgentParams:
    actor_network_params: Tuple[flax.core.FrozenDict, flax.core.FrozenDict]
    critic_network_params: Tuple[flax.core.FrozenDict, flax.core.FrozenDict]
    actor_params: flax.core.FrozenDict
    critic_params: flax.core.FrozenDict

@flax.struct.dataclass
class WorldModelParams:
    transition_network_params: flax.core.FrozenDict
    reward_network_params: flax.core.FrozenDict

@flax.struct.dataclass
class FullParams:
    agent: AgentParams
    world_model: WorldModelParams

@flax.struct.dataclass
class Storage:
    obs: jnp.array
    actions: jnp.array
    logprobs: jnp.array
    dones: jnp.array
    values: jnp.array
    advantages: jnp.array
    returns: jnp.array
    rewards: jnp.array
    next_obs: jnp.array


@flax.struct.dataclass
class EpisodeStatistics:
    episode_returns: jnp.array
    episode_lengths: jnp.array
    returned_episode_returns: jnp.array
    returned_episode_lengths: jnp.array


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    if args.parallel_envs_config:
        # parse parallel_envs_config string
        config = dict(item.split('=') for item in args.parallel_envs_config.split(','))
        args.num_envs = int(config.get('n_env', args.num_envs))
        args.num_steps = int(config.get('n_steps', args.num_steps))
        args.batch_size = int(args.num_envs * args.num_steps)
        args.minibatch_size = int(args.batch_size // args.num_minibatches)
        args.num_iterations = args.total_timesteps // args.batch_size

    if args.random_env_id:
        env_list = list(atari_human_normalized_scores.keys())
        args.env_id = env_list[np.random.randint(len(env_list))]
        print(f'Environment drawn: {args.env_id}')

    if args.lipschitz_nets and args.lambda_gp > 0.:
        # raise warning if using both lipschitz_nets and lambda_gp > 0
        warnings.warn("You should not use both `lipschitz_nets` and `lambda_gp > 0` at the same time. "
                      "Setting `lambda_gp` to 0 to avoid unnecessary computation.")
        args.lambda_gp = 0

    if args.transition_density in [TransitionDensity.MIXTURE_NORMAL, TransitionDensity.NORMAL]:
        args.use_gumbel_softmax = False

    if args.transition_density == TransitionDensity.CATEGORICAL:
        if args.lambda_gp > 0. or args.lipschitz_nets:
            # raise warning if using both CATEGORICAL transition density and lambda_gp > 0
            warnings.warn("You should not use `CATEGORICAL` transition density with `lambda_gp > 0` or Lipschitz networks. "
                          "Setting `lambda_gp` to 0 and lipschitz_net to False"
                          " to avoid unnecessary computation.")
            args.lipschitz_nets = False
            args.lambda_gp = 0.
        if args.layer_norm_cnn_output:
            # raise warning if using layer_norm_cnn_output with CATEGORICAL transition density
            warnings.warn("Using `layer_norm_cnn_output` with `CATEGORICAL` transition density is not forbidden. "
                          "Setting `layer_norm_cnn_output` to False.")
            args.layer_norm_cnn_output = False

    if args.transition_density != TransitionDensity.CATEGORICAL and args.auxiliary_task_net_type != NetType.CONV:
        # raise warning if using soft_moe with non-CATEGORICAL transition density
        warnings.warn("For non-CATEGORICAL transition density, only `CONV` is implemented for `auxiliary_task_net_type`")
        args.auxiliary_task_net_type = NetType.CONV

    if not args.drift_formulation:
        args.drift_coef = 1.

    if args.hp_tuning_mode:
        if args.transition_density == TransitionDensity.MIXTURE_NORMAL:
            args.use_wgan_gp = False
            args.lipschitz_nets = True  # otherwise too slow

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

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, network_key, actor_key, critic_key, transition_key, reward_key = jax.random.split(key, 6)

    # env setup
    envs = make_env(args.env_id, args.seed, args.num_envs)()
    episode_stats = EpisodeStatistics(
        episode_returns=jnp.zeros(args.num_envs, dtype=jnp.float32),
        episode_lengths=jnp.zeros(args.num_envs, dtype=jnp.int32),
        returned_episode_returns=jnp.zeros(args.num_envs, dtype=jnp.float32),
        returned_episode_lengths=jnp.zeros(args.num_envs, dtype=jnp.int32),
    )
    handle, recv, send, step_env = envs.xla()

    def step_env_wrappeed(episode_stats, handle, action):
        handle, (next_obs, reward, next_done, info) = step_env(handle, action)
        new_episode_return = episode_stats.episode_returns + info["reward"]
        new_episode_length = episode_stats.episode_lengths + 1
        episode_stats = episode_stats.replace(
            episode_returns=(new_episode_return) * (1 - info["terminated"]) * (1 - info["TimeLimit.truncated"]),
            episode_lengths=(new_episode_length) * (1 - info["terminated"]) * (1 - info["TimeLimit.truncated"]),
            # only update the `returned_episode_returns` if the episode is done
            returned_episode_returns=jnp.where(
                info["terminated"] + info["TimeLimit.truncated"], new_episode_return, episode_stats.returned_episode_returns
            ),
            returned_episode_lengths=jnp.where(
                info["terminated"] + info["TimeLimit.truncated"], new_episode_length, episode_stats.returned_episode_lengths
            ),
        )
        return episode_stats, handle, (next_obs, reward, next_done, info)

    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    def linear_schedule(count):
        # anneal learning rate linearly after one training iteration which contains
        # (args.num_minibatches * args.update_epochs) gradient updates
        frac = 1.0 - (count // (args.num_minibatches * args.update_epochs)) / args.num_iterations
        return args.learning_rate * frac

    dummy_input = np.array([envs.single_observation_space.sample()])

    if args.transition_density == TransitionDensity.CATEGORICAL:
        actor_conv_encoder_cls = lambda: flax.linen.Sequential(
            [NetworkConv(args.layer_norm_cnn_output), CategoricalEncoder(),])
        if args.use_attention:
            actor_fc_encoder_cls = NetworkAttentionOutput
        else:
            actor_fc_encoder_cls = NetworkFCOutput
        if args.decoupled_repr:
            conv_encoder_cls = lambda: NetworkConv(args.layer_norm_cnn_output)
            fc_encoder_cls = NetworkFCOutput
        else:
            conv_encoder_cls = actor_conv_encoder_cls
            fc_encoder_cls =  actor_fc_encoder_cls
    else:
        actor_conv_encoder_cls = conv_encoder_cls = lambda: NetworkConv(args.layer_norm_cnn_output)
        actor_fc_encoder_cls = fc_encoder_cls = NetworkFCOutput

    if args.lipschitz_nets:
        transition_network = LipDiscreteActionTransitionCNN(
            num_actions=envs.action_space.n,
            density=TransitionDensity(args.transition_density),
            feature_group=args.use_feature_group,)
        reward_network = LipDiscreteActionRewardNetwork(num_actions=envs.action_space.n, )
    elif args.transition_density == TransitionDensity.CATEGORICAL and args.auxiliary_task_net_type == NetType.SOFTMOE:
        transition_network = DiscreteActionTransitionNetworkSoftMoE(
            num_actions=envs.action_space.n,
            num_experts=2 * envs.action_space.n,
            gumbel_softmax=args.use_gumbel_softmax)
        reward_network = DiscreteActionRewardNetworkSoftMoE(
            num_actions=envs.action_space.n,
            num_experts=2 * envs.action_space.n,)
    elif args.transition_density == TransitionDensity.CATEGORICAL and args.auxiliary_task_net_type == NetType.FC:
        transition_network = DiscreteActionTransitionNetwork(
            num_actions=envs.action_space.n,
            density=TransitionDensity(args.transition_density),
            gumbel_softmax=args.use_gumbel_softmax,)
        reward_network = DiscreteActionRewardNetwork(num_actions=envs.action_space.n,)
    elif args.transition_density == TransitionDensity.CATEGORICAL and args.auxiliary_task_net_type == NetType.TRANSFORMER:
        transition_network = AutoregressiveDiscreteActionTransitionTransformer(
            num_actions=envs.action_space.n,
            density=TransitionDensity(args.transition_density),
            gumbel_softmax=args.use_gumbel_softmax, )
        reward_network = DiscreteActionRewardNetwork(num_actions=envs.action_space.n,)
    else:
        transition_network = DiscreteActionTransitionCNN(
            num_actions=envs.action_space.n,
            density=TransitionDensity(args.transition_density),
            gumbel_softmax=args.use_gumbel_softmax,
            feature_group=args.use_feature_group,)
        reward_network = DiscreteActionRewardNetwork(
            num_actions=envs.action_space.n,
            use_embedding=False)

    def _initialize_world_model_params(conv_out):
        transition_network_params = transition_network.init(transition_key, (conv_out, jnp.zeros((conv_out.shape[0],), dtype=jnp.int32)))
        reward_network_params = reward_network.init(reward_key, (conv_out, jnp.zeros((conv_out.shape[0],), dtype=jnp.int32)))
        return WorldModelParams(
            transition_network_params=transition_network_params,
            reward_network_params=reward_network_params,
        )

    if args.decoupled_repr:
        # Decoupled actor/critic encoders
        actor_conv = actor_conv_encoder_cls()
        actor_fc = actor_fc_encoder_cls()
        critic_conv = conv_encoder_cls()
        critic_fc = fc_encoder_cls()

        actor = Actor(action_dim=envs.single_action_space.n)
        critic = Critic()

        # Initialize actor encoder
        actor_conv_params = actor_conv.init(network_key, dummy_input)
        actor_conv_out = actor_conv.apply(actor_conv_params, dummy_input)
        actor_fc_params = actor_fc.init(actor_key, actor_conv_out)
        actor_features = actor_fc.apply(actor_fc_params, actor_conv_out)

        # Initialize critic encoder
        critic_conv_params = critic_conv.init(network_key, dummy_input)
        critic_conv_out = critic_conv.apply(critic_conv_params, dummy_input)
        critic_fc_params = critic_fc.init(critic_key, critic_conv_out)
        critic_features = critic_fc.apply(critic_fc_params, critic_conv_out)

        world_model_params = _initialize_world_model_params(actor_conv_out)
        agent_params = AgentParams(
            actor_network_params=(actor_conv_params, actor_fc_params),
            critic_network_params=(critic_conv_params, critic_fc_params),
            actor_params=actor.init(actor_key, actor_features),
            critic_params=critic.init(critic_key, critic_features),
        )

        if args.transition_density == TransitionDensity.CATEGORICAL and args.auxiliary_task_net_type == NetType.TRANSFORMER:
            agent_state = TrainState.create(
                apply_fn=None,
                params=FullParams(agent=agent_params, world_model=world_model_params),
                tx=optax.chain(
                    optax.clip_by_global_norm(args.max_grad_norm),
                    optax.inject_hyperparams(optax.adamw)(
                    learning_rate=linear_schedule if args.anneal_lr else args.learning_rate, eps=1e-5)))
        else:
            agent_state = TrainState.create(
                apply_fn=None,
                params=FullParams(agent=agent_params, world_model=world_model_params),
                tx=optax.chain(
                    optax.clip_by_global_norm(args.max_grad_norm),
                    optax.inject_hyperparams(optax.adam)(
                        learning_rate=linear_schedule if args.anneal_lr else args.learning_rate, eps=1e-5)))

        # JIT all applies
        actor_conv.apply = jax.jit(actor_conv.apply)
        actor_fc.apply = jax.jit(actor_fc.apply)
        critic_conv.apply = jax.jit(critic_conv.apply)
        critic_fc.apply = jax.jit(critic_fc.apply)

    else:
        # Shared encoder (same conv+fc for both actor and critic)
        shared_conv = conv_encoder_cls()
        shared_fc = fc_encoder_cls()
        actor = Actor(action_dim=envs.single_action_space.n)
        critic = Critic()

        conv_params = shared_conv.init(network_key, dummy_input)
        conv_out = shared_conv.apply(conv_params, dummy_input)
        fc_params = shared_fc.init(actor_key, conv_out)  # just once

        features = shared_fc.apply(fc_params, conv_out)

        world_model_params = _initialize_world_model_params(conv_out)
        agent_params = AgentParams(
            actor_network_params=(conv_params, fc_params),
            critic_network_params=(conv_params, fc_params),  # reuse
            actor_params=actor.init(actor_key, features),
            critic_params=critic.init(critic_key, features),
        )

        if args.transition_density == TransitionDensity.CATEGORICAL and args.auxiliary_task_net_type == NetType.TRANSFORMER:
            agent_state = TrainState.create(
                apply_fn=None,
                params=FullParams(agent=agent_params, world_model=world_model_params),
                tx=optax.chain(
                    optax.clip_by_global_norm(args.max_grad_norm),
                    optax.inject_hyperparams(optax.adamw)(
                        learning_rate=linear_schedule if args.anneal_lr else args.learning_rate, eps=1e-5)))
        else:
            agent_state = TrainState.create(
                apply_fn=None,
                params=FullParams(agent=agent_params, world_model=world_model_params),
                tx=optax.chain(
                    optax.clip_by_global_norm(args.max_grad_norm),
                    optax.inject_hyperparams(optax.adam)(
                        learning_rate=linear_schedule if args.anneal_lr else args.learning_rate, eps=1e-5),))

        # Alias for consistent access
        actor_conv = critic_conv = shared_conv
        actor_fc = critic_fc = shared_fc

        actor_conv.apply = jax.jit(actor_conv.apply)
        actor_fc.apply = jax.jit(actor_fc.apply)

    # JIT transition and reward networks apply methods
    transition_network.apply = jax.jit(transition_network.apply)
    reward_network.apply = jax.jit(reward_network.apply)

    def pi_sample(
        params: FullParams,
        actor_hidden: np.ndarray,
        key: jax.random.PRNGKey,
    ):
        logits = actor.apply(params.agent.actor_params, actor_hidden)

        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey, shape=logits.shape)
        action = jnp.argmax(logits - jnp.log(-jnp.log(u)), axis=1)
        logprob = jax.nn.log_softmax(logits)[jnp.arange(action.shape[0]), action]

        return action, logprob, key

    @jax.jit
    def get_action_and_value(
        agent_state: TrainState,
        obs: np.ndarray,
        key: jax.random.PRNGKey,
    ):
        """sample action, calculate value, logprob, entropy, and update storage"""
        actor_conv_params, actor_fc_params = agent_state.params.agent.actor_network_params
        critic_conv_params, critic_fc_params = agent_state.params.agent.critic_network_params

        z = actor_conv.apply(actor_conv_params, obs)
        actor_hidden = actor_fc.apply(actor_fc_params, z)

        if not args.decoupled_repr:
            critic_hidden = actor_hidden
        else:
            critic_hidden = critic_conv.apply(critic_conv_params, obs)
            critic_hidden = critic_fc.apply(critic_fc_params, critic_hidden)

        action, logprob, key = pi_sample(agent_state.params, actor_hidden, key)

        value = critic.apply(agent_state.params.agent.critic_params, critic_hidden).squeeze(1)
        return action, logprob, value, key

    @jax.jit
    def compute_auxiliary_losses(
        params: FullParams,
        z: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        z_prime: np.ndarray,
        key: jax.random.PRNGKey,
    ):
        """Compute auxiliary losses for the world model"""
        transition_network_params = params.world_model.transition_network_params
        reward_network_params = params.world_model.reward_network_params

        # Transition loss
        dist = transition_network.apply(transition_network_params, (z, action))

        dont_sample_now = args.transition_density == TransitionDensity.CATEGORICAL and \
                          args.categorical_cost in [CategoricalCost.CROSS_ENTROPY, CategoricalCost.JENSEN_SHANNON]
        if dont_sample_now:
            z_prime_sampled = None
        else:
            key, subkey = jax.random.split(key)
            z_prime_sampled = dist.sample(seed=subkey)

        if args.transition_loss_coef == 0.:
            transition_loss = 0.
        else:
            if args.transition_density in [TransitionDensity.NORMAL, TransitionDensity.MIXTURE_NORMAL] \
                    and not args.lipschitz_nets:
                # closed form loss for Normal/MixtureNormal of E_{z_sampled ~ p(z_sampled|z,a)}[z' - z_sampled]
                # see https://en.wikipedia.org/wiki/Folded_normal_distribution
                if args.transition_density == TransitionDensity.NORMAL:
                    mu, sigma = dist.distribution.loc, dist.distribution.scale  # (B, H, W, C)
                    _z_prime = z_prime
                else:
                    mu, sigma = dist.loc, dist.scale  # (B, K, H, W, C); K is number of components
                    _z_prime = z_prime[:, None, ...]

                delta = (_z_prime - mu) / (sigma + 1e-8)
                term_1 = delta * (2. * jax.scipy.stats.norm.cdf(delta) - 1.)
                term_2 = jnp.sqrt(2. / jnp.pi) * jnp.exp(-.5 * delta**2.)
                expected_abs_diff = sigma * (term_1 + term_2)
                if args.transition_density == TransitionDensity.NORMAL:
                    transition_loss =  smooth_l1_loss(
                        expected_abs_diff, 0., reduction='sum', axis=[1, 2, 3]) # norm; (B, )
                else:
                    weights = dist.weights  # (B, K,)
                    # sum over components K
                    weighted_sum = jnp.einsum('bkhwc,bk->bhwc', expected_abs_diff, weights)
                    # norm; (B, H, W, C) -> (B, )
                    transition_loss = smooth_l1_loss(weighted_sum,0., reduction='sum', axis=[1, 2, 3])
            elif args.transition_density == TransitionDensity.CATEGORICAL:
                sample_now = dont_sample_now  # True if CategoricalCost is CROSS_ENTROPY or JENSEN_SHANNON
                key, subkey = jax.random.split(key) if sample_now else (key, key)
                transition_loss = {
                    CategoricalCost.HAMMING: lambda: hamming_distance(z_prime_sampled, z_prime),
                    CategoricalCost.CROSS_ENTROPY: lambda: jnp.sum(dist.relaxed_cross_entropy(z, subkey), axis=-1),
                    CategoricalCost.JENSEN_SHANNON: lambda: jnp.sum(dist.relaxed_js_distance(z, subkey), axis=-1),
                    CategoricalCost.L2: lambda: smooth_l1_loss(z_prime_sampled, z_prime, reduction='sum', axis=[1, 2]) / z_prime.shape[1],
                }[args.categorical_cost]()
            else:
                transition_loss = smooth_l1_loss(z_prime_sampled, z_prime, reduction='sum', axis=[1, 2, 3])

        # Reward loss
        if args.reward_loss_coef == 0.:
            reward_loss = 0.
        else:
            reward_pred = reward_network.apply(reward_network_params, (z, action))
            reward_loss = smooth_l1_loss(reward_pred, reward, reduction='none')

        # Gradient penalty
        if args.lambda_gp > 0.:
            B, H, W, C = z.shape
            flat_size = H * W * C
            A = envs.action_space.n

            def flatten_za(z, a):
                z_flat = jnp.reshape(z, (B, flat_size))
                a_onehot = jax.nn.one_hot(a, A)
                return jnp.concatenate([z_flat, a_onehot], -1)

            def decode(za_vec):
                z_flat = za_vec[..., :flat_size]  # (3136, )
                z_conv = z_flat.reshape((H, W, C))[None, ...]  # (H, W, C)
                a_int = jnp.argmax(za_vec[..., flat_size:], axis=-1)[None]  # (1,) int actions
                return z_conv, a_int

            def transition_network_apply(_za, key):
                _z, _a = decode(_za)
                dist = transition_network.apply(transition_network_params, (_z, _a))
                return dist.sample(seed=key)

            def reward_network_apply(_za, key):
                _z, _a = decode(_za)
                return reward_network.apply(reward_network_params, (_z, _a))

            if args.use_wgan_gp:
                actor_hidden_1 = actor_fc.apply(params.agent.actor_network_params[1], z_prime)
                actor_hidden_2 = actor_fc.apply(params.agent.actor_network_params[1], z_prime_sampled)
                action_1, _, key = pi_sample(params, actor_hidden_1, key)
                action_2, _, key = pi_sample(params, actor_hidden_2, key)
                z_prime_a = flatten_za(z_prime, jax.lax.stop_gradient(action_1))
                z_prime_a = jax.lax.stop_gradient(z_prime_a)
                z_prime_sampled_a = flatten_za(z_prime_sampled, jax.lax.stop_gradient(action_2))
                z_prime_sampled_a = jax.lax.stop_gradient(z_prime_sampled_a)
                transition_gp, key = wgan_gp(
                    transition_network_apply,
                    z_prime_a, z_prime_sampled_a,
                    key)
                reward_gp, key = wgan_gp(
                    reward_network_apply,
                    z_prime_a, z_prime_sampled_a,
                    key)
            else:
                z_a = flatten_za(z, action)
                z_a = jax.lax.stop_gradient(z_a)
                key, sub = jax.random.split(key)
                keys = jax.random.split(sub, z_a.shape[0])  # (B, 2)

                transition_gp = lipschitz_gp(
                    transition_network_apply,
                    z_a, keys)
                reward_gp = lipschitz_gp(
                    reward_network_apply,
                    z_a, keys)

            gradient_penalty = transition_gp + reward_gp
        else:
            gradient_penalty = 0.

        return transition_loss, reward_loss, gradient_penalty, key

    @jax.jit
    def get_action_and_value2(
            params: FullParams,
            x: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            next_obs: np.ndarray,
            key: jax.random.PRNGKey = jax.random.PRNGKey(0),
    ):
        actor_conv_params, actor_fc_params = params.agent.actor_network_params
        critic_conv_params, critic_fc_params = params.agent.critic_network_params

        z = actor_conv.apply(actor_conv_params, x)
        z_prime = actor_conv.apply(actor_conv_params, next_obs)
        actor_hidden = actor_fc.apply(actor_fc_params, z)

        if not args.decoupled_repr:
            critic_hidden = actor_hidden
        else:
            critic_hidden = critic_conv.apply(critic_conv_params, x)
            critic_hidden = critic_fc.apply(critic_fc_params, critic_hidden)

        logits = actor.apply(params.agent.actor_params, actor_hidden)
        logprob = jax.nn.log_softmax(logits)[jnp.arange(action.shape[0]), action]

        # Entropy
        logits = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
        logits = logits.clip(min=jnp.finfo(logits.dtype).min)
        p_log_p = logits * jax.nn.softmax(logits)
        entropy = -p_log_p.sum(-1)

        # Auxiliary losses
        if args.transition_loss_coef == 0. and args.reward_loss_coef == 0.:
            transition_loss = jnp.array(0.0)
            reward_loss = jnp.array(0.0)
            gradient_penalty = jnp.array(0.0)
        else:
            transition_loss, reward_loss, gradient_penalty, key = compute_auxiliary_losses(
                params, z, action, reward, z_prime, key)

        value = critic.apply(params.agent.critic_params, critic_hidden).squeeze()
        return logprob, entropy, value, transition_loss, reward_loss, gradient_penalty, key

    def compute_gae_once(carry, inp, gamma, gae_lambda):
        advantages = carry
        nextdone, nextvalues, curvalues, reward = inp
        nextnonterminal = 1.0 - nextdone

        delta = reward + gamma * nextvalues * nextnonterminal - curvalues
        advantages = delta + gamma * gae_lambda * nextnonterminal * advantages
        return advantages, advantages

    compute_gae_once = partial(compute_gae_once, gamma=args.gamma, gae_lambda=args.gae_lambda)

    @jax.jit
    def compute_gae(
            agent_state: TrainState,
            next_obs: np.ndarray,
            next_done: np.ndarray,
            storage: Storage,
    ):
        critic_conv_params, critic_fc_params = agent_state.params.agent.critic_network_params

        next_hidden = critic_conv.apply(critic_conv_params, next_obs)
        next_hidden = critic_fc.apply(critic_fc_params, next_hidden)
        next_value = critic.apply(agent_state.params.agent.critic_params, next_hidden).squeeze()

        advantages = jnp.zeros((args.num_envs,))
        dones = jnp.concatenate([storage.dones, next_done[None, :]], axis=0)
        values = jnp.concatenate([storage.values, next_value[None, :]], axis=0)
        _, advantages = jax.lax.scan(
            compute_gae_once, advantages, (dones[1:], values[1:], values[:-1], storage.rewards), reverse=True
        )
        storage = storage.replace(
            advantages=advantages,
            returns=advantages + storage.values,
        )
        return storage

    def ppo_loss(params, x, a, logp, mb_advantages, mb_returns, reward, next_obs, key):
        newlogprob, entropy, newvalue, transition_loss, reward_loss, gradient_penalty, key = get_action_and_value2(
            params, x, a, reward, next_obs, key)
        logratio = newlogprob - logp
        ratio = jnp.exp(logratio)
        approx_kl = ((ratio - 1) - logratio).mean()
        transition_loss = transition_loss.mean()
        reward_loss = reward_loss.mean()
        auxiliary_loss = args.reward_loss_coef * reward_loss + args.transition_loss_coef * transition_loss

        if args.norm_adv:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

        clip_low = 1.0 - args.clip_coef
        clip_high = 1.0 + args.clip_coef
        clipped_ratio = jnp.clip(ratio, clip_low, clip_high)

        if args.drift_formulation:
            weighted_advantage = ratio * mb_advantages
            # Apply drift penalty \mathfrak{D}_{π_n}(π_{n + 1} | s)
            drift_penalty = jax.nn.relu((ratio - clipped_ratio) * (mb_advantages - auxiliary_loss))
            # Subtract the drift term from the utility
            pg_loss = (-weighted_advantage + args.drift_coef * drift_penalty).mean()
        else:
            # Standard clipped PPO
            pg_loss1 = -(mb_advantages - auxiliary_loss) * ratio
            pg_loss2 = -(mb_advantages - auxiliary_loss) * jnp.clip(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
            pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

        # Value loss
        v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

        if args.drift_formulation:
            drift_penalty_mean = drift_penalty.mean()
        else:
            drift_penalty_mean = jnp.array(0.0)

        entropy_loss = entropy.mean()
        loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + args.lambda_gp * gradient_penalty
        return loss, (pg_loss, v_loss, entropy_loss, jax.lax.stop_gradient(approx_kl), drift_penalty_mean, transition_loss, reward_loss, gradient_penalty, key)

    ppo_loss_grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)

    @jax.jit
    def update_ppo(
        agent_state: TrainState,
        storage: Storage,
        key: jax.random.PRNGKey,
    ):
        def update_epoch(carry, unused_inp):
            agent_state, key = carry
            key, subkey = jax.random.split(key)

            def flatten(x):
                return x.reshape((-1,) + x.shape[2:])

            # taken from: https://github.com/google/brax/blob/main/brax/training/agents/ppo/train.py
            def convert_data(x: jnp.ndarray):
                x = jax.random.permutation(subkey, x)
                x = jnp.reshape(x, (args.num_minibatches, -1) + x.shape[1:])
                return x

            flatten_storage = jax.tree_map(flatten, storage)
            shuffled_storage = jax.tree_map(convert_data, flatten_storage)

            def update_minibatch(carry, minibatch):
                agent_state, key = carry
                (loss, (pg_loss, v_loss, entropy_loss, approx_kl, drift_penalty_mean, transition_loss, reward_loss, gradient_penalty, key)), grads = ppo_loss_grad_fn(
                    agent_state.params,
                    minibatch.obs,
                    minibatch.actions,
                    minibatch.logprobs,
                    minibatch.advantages,
                    minibatch.returns,
                    minibatch.rewards,
                    minibatch.next_obs,
                    key,
                )
                agent_state = agent_state.apply_gradients(grads=grads)
                return (agent_state, key), (loss, pg_loss, v_loss, entropy_loss, approx_kl, drift_penalty_mean, transition_loss, reward_loss, gradient_penalty, grads)

            (agent_state, key), (loss, pg_loss, v_loss, entropy_loss, approx_kl, drift_penalty_mean, transition_loss, reward_loss, gradient_penalty, grads) = jax.lax.scan(
                update_minibatch, (agent_state, key), shuffled_storage
            )
            return (agent_state, key), (loss, pg_loss, v_loss, entropy_loss, approx_kl, drift_penalty_mean, transition_loss, reward_loss, gradient_penalty, grads)

        (agent_state, key), (loss, pg_loss, v_loss, entropy_loss, approx_kl, drift_penalty_mean, transition_loss, reward_loss, gradient_penalty, grads) = jax.lax.scan(
            update_epoch, (agent_state, key), (), length=args.update_epochs
        )
        return agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, drift_penalty_mean, transition_loss, reward_loss, gradient_penalty, grads, key

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = envs.reset()
    next_done = jnp.zeros(args.num_envs, dtype=jax.numpy.bool_)

    # based on https://github.dev/google/evojax/blob/0625d875262011d8e1b6aa32566b236f44b4da66/evojax/sim_mgr.py
    def step_once(carry, step, env_step_fn):
        agent_state, episode_stats, obs, done, key, handle = carry
        action, logprob, value, key = get_action_and_value(agent_state, obs, key)

        episode_stats, handle, (next_obs, reward, next_done, _) = env_step_fn(episode_stats, handle, action)
        storage = Storage(
            obs=obs,
            actions=action,
            logprobs=logprob,
            dones=done,
            values=value,
            rewards=reward,
            returns=jnp.zeros_like(reward),
            advantages=jnp.zeros_like(reward),
            next_obs=next_obs,
        )
        return ((agent_state, episode_stats, next_obs, next_done, key, handle), storage)

    def rollout(agent_state, episode_stats, next_obs, next_done, key, handle, step_once_fn, max_steps):
        (agent_state, episode_stats, next_obs, next_done, key, handle), storage = jax.lax.scan(
            step_once_fn, (agent_state, episode_stats, next_obs, next_done, key, handle), (), max_steps
        )
        return agent_state, episode_stats, next_obs, next_done, storage, key, handle

    rollout = partial(rollout, step_once_fn=partial(step_once, env_step_fn=step_env_wrappeed), max_steps=args.num_steps)

    # Get the cleanRL ppo scores
    csv_path = args.compare_scores_csv
    score_col = args.compare_scores_column
    scores_mean, scores_sigma = load_env_mean_std(csv_path, score_col)
    max_avg_return = -np.infty

    for iteration in range(1, args.num_iterations + 1):
        iteration_time_start = time.time()
        agent_state, episode_stats, next_obs, next_done, storage, key, handle = rollout(
            agent_state, episode_stats, next_obs, next_done, key, handle
        )
        global_step += args.num_steps * args.num_envs
        storage = compute_gae(agent_state, next_obs, next_done, storage)
        agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, drift_penalty_mean, transition_loss, reward_loss, gradient_penalty, grads, key = update_ppo(
            agent_state,
            storage,
            key,
        )
        avg_episodic_return = np.mean(jax.device_get(episode_stats.returned_episode_returns))
        max_avg_return = np.max([max_avg_return, avg_episodic_return])
        print(f"global_step={global_step}, avg_episodic_return={avg_episodic_return}")

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/avg_episodic_return", avg_episodic_return, global_step)
        if args.env_id in atari_human_normalized_scores:
            writer.add_scalar(
                "charts/human_normalized_score", human_normalized_score(args.env_id, avg_episodic_return), global_step)
        if args.env_id in scores_mean:
            score = max_avg_return if args.compare_scores_max else avg_episodic_return
            rel_imp = rel_improvement(score, args.env_id, scores_mean)
            writer.add_scalar("charts/relative_improvement", rel_imp, global_step)
            if args.env_id in atari_human_normalized_scores:
                rel_imp_hns = rel_improvement(
                    human_normalized_score(args.env_id, score), args.env_id,
                    {args.env_id: human_normalized_score(args.env_id, scores_mean[args.env_id])})
                writer.add_scalar("charts/relative_improvement_hns", rel_imp_hns, global_step)
        writer.add_scalar(
            "charts/avg_episodic_length", np.mean(jax.device_get(episode_stats.returned_episode_lengths)),
            global_step)
        writer.add_scalar("charts/learning_rate", agent_state.opt_state[1].hyperparams["learning_rate"].item(), global_step)
        writer.add_scalar("losses/value_loss", v_loss[-1, -1].item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss[-1, -1].item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss[-1, -1].item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl[-1, -1].item(), global_step)
        writer.add_scalar("losses/loss", loss[-1, -1].item(), global_step)
        writer.add_scalar("losses/transition_loss", transition_loss[-1, -1].item(), global_step)
        writer.add_scalar("losses/reward_loss", reward_loss[-1, -1].item(), global_step)
        writer.add_scalar("losses/gradient_penalty", gradient_penalty[-1, -1].item(), global_step)
        if args.drift_formulation:
            writer.add_scalar("losses/drift", drift_penalty_mean[-1, -1].item(), global_step)

        # if tracking weights and gradients is enabled, log them for the transition network
        if args.track_params:
            # Log transition network parameters
            transition_params = flatten_dict(agent_state.params.world_model.transition_network_params)
            for name, value in transition_params.items():
                wandb.log({f"params/transition_network/{name}": jnp.linalg.norm(value).item()}, step=global_step)

            # Log gradients
            grads = jax.tree_map(lambda x: x[-1, -1], grads)  # final minibatch, final epoch
            grad_transition = flatten_dict(grads.world_model.transition_network_params)
            for name, value in grad_transition.items():
                wandb.log({f"grads/transition_network/{name}": jnp.linalg.norm(value).item()}, step=global_step)

            wandb.log({f"params/transition_network/{name}": wandb.Histogram(value)}, step=global_step)

            # Log CategoricalEncoder parameters if using it
            if args.transition_density == TransitionDensity.CATEGORICAL:
                encoder_params = flatten_dict(agent_state.params.agent.actor_network_params[0])  # conv encoder
                for name, value in encoder_params.items():
                    wandb.log({f"params/encoder/{name}": jnp.linalg.norm(value).item()}, step=global_step)

                encoder_grads = flatten_dict(grads.agent.actor_network_params[0])
                for name, value in encoder_grads.items():
                    wandb.log({f"grads/encoder/{name}": jnp.linalg.norm(value).item()}, step=global_step)


        print(
            "SPS:", int(global_step / (time.time() - start_time)),
            f'transition_loss: {transition_loss[-1, -1].item():.6g}',
            f'reward_loss: {reward_loss[-1, -1].item():.6g}',
            f'gradient_penalty: {gradient_penalty[-1, -1].item():.6g}',
        )
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        writer.add_scalar(
            "charts/SPS_update", int(args.num_envs * args.num_steps / (time.time() - iteration_time_start)), global_step
        )

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        with open(model_path, "wb") as f:
            f.write(
                flax.serialization.to_bytes(
                    [
                        vars(args),
                        [
                            agent_state.params.agent.network_params,
                            agent_state.params.agent.actor_params,
                            agent_state.params.agent.critic_params,
                        ],
                    ]
                )
            )
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.ppo_envpool_jax_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=(NetworkConv, NetworkFCOutput, Actor, Critic),
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "PPO", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()
