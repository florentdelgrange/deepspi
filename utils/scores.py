# For Atari environments HNS:
# ==============================================================================
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Costa: Note the data is not the same as Mnih et al., 2015
# Note the random agent score on Video Pinball is sometimes greater than the
# human score under other evaluation methods.

from typing import Mapping, Optional, Union
import pandas as pd
from pathlib import Path
from collections import defaultdict

atari_human_normalized_scores = {
    "Alien-v5": (227.8, 7127.7),
    "Amidar-v5": (5.8, 1719.5),
    "Assault-v5": (222.4, 742.0),
    "Asterix-v5": (210.0, 8503.3),
    "Asteroids-v5": (719.1, 47388.7),
    "Atlantis-v5": (12850.0, 29028.1),  # note our Envpool + PPO only gets 25 as the base return
    "BankHeist-v5": (14.2, 753.1),  # note our Envpool + PPO only gets 0 as the base return
    "BattleZone-v5": (2360.0, 37187.5),
    "BeamRider-v5": (363.9, 16926.5),
    "Berzerk-v5": (123.7, 2630.4),
    "Bowling-v5": (23.1, 160.7),
    "Boxing-v5": (0.1, 12.1),
    "Breakout-v5": (1.7, 30.5),
    "Centipede-v5": (2090.9, 12017.0),
    "ChopperCommand-v5": (811.0, 7387.8),
    "CrazyClimber-v5": (10780.5, 35829.4),
    "Defender-v5": (2874.5, 18688.9),  ## TODO: where is defender in the original DQN paper?
    "DemonAttack-v5": (152.1, 1971.0),
    "DoubleDunk-v5": (-18.6, -16.4),
    "Enduro-v5": (0.0, 860.5),
    "FishingDerby-v5": (-91.7, -38.7),
    "Freeway-v5": (0.0, 29.6),
    "Frostbite-v5": (65.2, 4334.7),
    "Gopher-v5": (257.6, 2412.5),
    "Gravitar-v5": (173.0, 3351.4),
    "Hero-v5": (1027.0, 30826.4),
    "IceHockey-v5": (-11.2, 0.9),
    "Jamesbond-v5": (29.0, 302.8),
    "Kangaroo-v5": (52.0, 3035.0),
    "Krull-v5": (1598.0, 2665.5),
    "KungFuMaster-v5": (258.5, 22736.3),
    "MontezumaRevenge-v5": (0.0, 4753.3),
    "MsPacman-v5": (307.3, 6951.6),
    "NameThisGame-v5": (2292.3, 8049.0),
    "Phoenix-v5": (761.4, 7242.6),  ## TODO: where is Phoenix in the original DQN paper?
    "Pitfall-v5": (-229.4, 6463.7),  ## TODO: where is Pitfall in the original DQN paper?
    "Pong-v5": (-20.7, 14.6),
    "PrivateEye-v5": (24.9, 69571.3),
    "Qbert-v5": (163.9, 13455.0),
    "Riverraid-v5": (1338.5, 17118.0),
    "RoadRunner-v5": (11.5, 7845.0),
    "Robotank-v5": (2.2, 11.9),
    "Seaquest-v5": (68.4, 42054.7),
    "Skiing-v5": (
        -17098.1,
        -4336.9,
    ),  # note our Envpool + PPO only gets -28500 as the base return ## TODO: where is Skiing in the original DQN paper?
    "Solaris-v5": (1236.3, 12326.7),  ## TODO: where is Solaris in the original DQN paper?
    "SpaceInvaders-v5": (148.0, 1668.7),
    "StarGunner-v5": (664.0, 10250.0),
    "Surround-v5": (-10.0, 6.5),  ## TODO: where is Surround in the original DQN paper?
    "Tennis-v5": (-23.8, -8.3),
    "TimePilot-v5": (3568.0, 5229.2),
    "Tutankham-v5": (11.4, 167.6),
    "UpNDown-v5": (533.4, 11693.2),
    "Venture-v5": (0.0, 1187.5),
    "VideoPinball-v5": (16256.9, 17667.9),
    "WizardOfWor-v5": (563.5, 4756.5),  # note our Envpool + PPO only gets 0 as the base return
    "YarsRevenge-v5": (3092.9, 54576.9),  ## TODO: where is YarsRevenge in the original DQN paper?
    "Zaxxon-v5": (32.5, 9173.3),
    # new games with stochastic dynamics (we picked the best scores from the leaderboards, EMU track):
    "Kaboom-v5": (2.49, 39101),  # https://www.twingalaxies.com/games/leaderboard-details/Kaboom/atari-2600-vcs
    "LaserGates-v5": (139.55, 87511), # https://www.twingalaxies.com/games/leaderboard-details/Laser-gates/atari-2600-vcs
    "MarioBros-v5": (120, 778700), # https://www.twingalaxies.com/games/leaderboard-details/Mario-bros/atari-2600-vcs
    "Et-v5": (19.60, 19127), # https://www.twingalaxies.com/games/leaderboard-details/Et/atari-2600-vcs
}

def human_normalized_score(env_name: str, score: float) -> float:
    if env_name not in atari_human_normalized_scores:
        raise ValueError(f"No human score data for {env_name}")
    random, human = atari_human_normalized_scores[env_name]
    return (score - random) / (human - random)

def ratio_vs_baseline(
        score: float,
        env_name: str,
        baseline_scores: Mapping[str, float],
) -> float:
    if env_name not in baseline_scores:
        raise ValueError(f"No baseline score for environment '{env_name}'")
    return score / baseline_scores[env_name]


def rel_improvement(
        score: float,
        env_name: str,
        baseline_scores: Mapping[str, float],
) -> float:
    if env_name not in baseline_scores:
        raise ValueError(f"No baseline score for environment '{env_name}'")
    return (score - baseline_scores[env_name]) / abs(baseline_scores[env_name])

def load_env_mean_dict(
        csv_path: str | Path,
        score_column: str = "CleanRL's ppo_atari_envpool_xla_jax.py",
) -> dict[str, float]:
    """
    Return {environment_name: mean_score} extracted from the CleanRL
    `result_table.csv`.

    Parameters
    ----------
    csv_path : str or Path
        Path to `result_table.csv`.
    score_column : str
        The column that holds the "mean ± std" strings you care about.

    Returns
    -------
    dict
        Mapping from environment id (e.g. "Alien-v5") to mean episodic return
        (float).
    """
    # ---------- 1. read CSV ----------
    # We first read with *no* index, then (optionally) promote the first column
    # to an index *only* if it is numeric (0,1,2,…) and thus useless.
    df = pd.read_csv(csv_path)

    # If the first column is numeric (0, 1, 2, …) and carries no information,
    # turn it into the index so that the real env names become the index later.
    if pd.api.types.is_integer_dtype(df.iloc[:, 0]):
        df = pd.read_csv(csv_path, index_col=0)

    # ---------- 2. locate environment names ----------
    # Case A – env names are already the index -------------------------------
    if score_column in df.columns and not pd.api.types.is_numeric_dtype(df.index):
        env_names = df.index
        score_series = df[score_column]

    # Case B – env names are inside some *other* column ----------------------
    else:
        # Heuristic: pick the first *non-numeric* column that is NOT the
        # score column itself as env-name column.
        env_col = next(
            col for col in df.columns
            if col != score_column and df[col].dtype == object
        )
        env_names = df[env_col]
        score_series = df[score_column]

    # ---------- 3. build the dictionary ----------
    def mean_from_cell(cell: str | float) -> float | None:
        """Extract the number before '±'. Returns None on failure."""
        try:
            if "±" in  str(cell):
                return float(str(cell).split("±")[0].strip())
            else:
                return float(cell)
        except Exception:
            return None

    env_mean_dict = {
        env.strip(): mean_from_cell(cell)
        for env, cell in zip(env_names, score_series)
        if pd.notna(cell) and mean_from_cell(cell) is not None
    }
    return env_mean_dict

def load_env_mean_std(
    csv_path: str | Path,
    score_column: str
) -> tuple[dict[str, float], dict[str, float]]:
    """Return ({env: μ}, {env: σ}) from a 'mean ± std' column."""
    df = pd.read_csv(csv_path, index_col=0)

    if score_column not in df.columns:
        raise ValueError(f"Column '{score_column}' not found.")

    def parse(cell: Union[str, float]) -> tuple[float, Optional[float]] | None:
        try:
            if type(cell) is str and "±" in cell:
                mean_str, std_str = cell.split("±")
                return float(mean_str.strip()), float(std_str.strip())
            elif type(cell) is float:
                return cell, None
            else:
                return float(cell.strip()), None
        except Exception:
            return None

    mu, sigma = {}, {}
    for env, cell in df[score_column].dropna().items():
        parsed = parse(cell)
        if parsed:
            mu[env], sigma[env] = parsed
    return mu, sigma

def z_score_vs_baseline(score, env_name: str, mu, sigma):
    """{env: (agent − μ₀) / σ₀}"""
    if env_name not in mu or env_name not in sigma:
        raise ValueError(f"No data for environment '{env_name}'")
    else:
        return (score - mu[env_name]) / sigma[env_name]

def gather_scores_for_stochastic_envs(entity: str, project: str, tag: str) -> pd.DataFrame:
    """
    Gather scores for stochastic environments from WandB runs.

    Parameters
    ----------
    entity : str
        WandB entity (e.g. "cleanrl").
    project : str
        WandB project name (e.g. "ppo-atari").
    tag : str
        Special tag used to filter the correct runs.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: env_id, mean_max_return, individual_max_returns, n_runs.
    """
    import wandb

    # Ensure you have the wandb library installed and logged in.
    wandb.login()

    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}", filters={"tags": {"$in": [tag]}})

    env_to_max_returns = defaultdict(list)

    for run in runs:
        if run.agent_state != "finished":
            continue

        env_id = run.config.get("env_id") or run.config.get("game") or run.config.get("env_name")
        if not env_id:
            print(f"Run {run.name} missing env_id. Skipping.")
            continue

        try:
            history = run.history(keys=["charts/avg_episodic_return"], pandas=True)
            if "charts/avg_episodic_return" in history:
                max_return = history["charts/avg_episodic_return"].max()
                env_to_max_returns[env_id].append(max_return)
        except Exception as e:
            print(f"Failed loading run {run.name}: {e}")

    # Aggregate
    results = []
    for env_id, scores in env_to_max_returns.items():
        scores = sorted(scores, reverse=True)
        mean_max = sum(scores) / len(scores)

        try:
            normalized_scores = [human_normalized_score(env_id, s) for s in scores]
            mean_normalized = sum(normalized_scores) / len(normalized_scores)
        except ValueError as e:
            print(f"Skipping normalized score for {env_id}: {e}")
            mean_normalized = None

        results.append({
            "env_id": env_id,
            "mean_max_return": mean_max,
            "individual_max_returns": scores,
            "mean_max_normalized_human_score": mean_normalized,
            "n_runs": len(scores)
        })

    df = pd.DataFrame(results).sort_values("mean_max_return", ascending=False)
    print(df.to_string(index=False))
    df.to_csv("ppo_atari_max_avg_return.csv", index=False)

    return df
