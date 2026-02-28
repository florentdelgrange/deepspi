import envpool
import numpy as np
from tqdm import trange


def run_random_agent(game_id="Kaboom-v5", num_episodes=100):
    # Create the environment
    env = envpool.make(
        game_id,
        env_type="gym",
        num_envs=1,
        episodic_life=True,
        # full_action_space=True,
    )

    env.reset()
    total_rewards = []
    episode_reward = 0

    for _ in trange(num_episodes):
        done = False
        env.reset()
        episode_reward = 0

        while not done:
            action = np.array([env.action_space.sample()], dtype=np.int32)
            obs, reward, done, info = env.step(action)
            episode_reward += reward

        total_rewards.append(episode_reward)

    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    print(f"[{game_id}] Random agent over {num_episodes} episodes:")
    print(f"  Mean reward: {mean_reward:.2f}, Std: {std_reward:.2f}")

    return mean_reward, std_reward


games = ['Kaboom-v5', 'LaserGates-v5', 'MarioBros-v5', 'Et-v5']

results = {}
for game in games:
    try:
        mean, std = run_random_agent(game, num_episodes=100)
        results[game] = (mean, std)
    except Exception as e:
        print(f"⚠️ Could not run {game}: {e}")

print("\n✅ Finished all games.")
for game, (mean, std) in results.items():
    print(f"{game}: Mean={mean:.2f}, Std={std:.2f}")