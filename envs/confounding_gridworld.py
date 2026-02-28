from typing import Dict, Tuple, Any

import numpy as np
import numpy as onp
import jax
import jax.numpy as jnp
from gym import spaces
from jax import lax, random
import jax.image as jimage
from matplotlib import pyplot as plt
import flax
from flax import struct

# Global debug flag
debug: bool = False

# ============================================================
# 1. Env / MDP specification
# ============================================================

@flax.struct.dataclass
class EnvParams:
    n_path: int                      # number of '>' cells in each corridor
    epsilon: float = 0.1             # hazard prob to go to bottom
    frame_size: Tuple[int, int, int] = (1, 84, 84)
    max_steps: int = 200             # truncation horizon
    gamma: float = .99               # discount factor



@flax.struct.dataclass
class MDPEnvSpec:
    params: EnvParams
    P: jnp.ndarray              # (S, A, S)
    R: jnp.ndarray              # (S, A)
    state_coords: jnp.ndarray   # (S, 2) int32 (row, col)
    is_terminal: jnp.ndarray    # (S,) bool
    initial_state: jnp.ndarray  # scalar int32
    critical_states: Tuple[jnp.ndarray, ...]
    grid_intensity: jnp.ndarray # (n_rows, n_cols) uint8



@flax.struct.dataclass
class EnvState:
    s: jnp.ndarray      # state index (scalar int32)
    done: jnp.ndarray   # bool
    step: jnp.ndarray   # int32



# ============================================================
# 2. Symbolic grid helper (Python-side)
# ============================================================

def _create_symbolic_grid(n_path: int) -> onp.ndarray:
    """
    Create a 5 x n_cols char grid with '#', 'I', '@', '>', '!', 'G', 'F', 'W'.

    Layout:

      row 0: walls except G at col = n_path
      row 1: '#' at col 0, then '>' * (n_path-1), '!' at col n_path, 'W' at col n_path+1
      row 2: 'I' at col 0, '@' at col 1, walls elsewhere
      row 3: symmetric to row 1
      row 4: walls except F at col = n_path
    """
    n_cols = n_path + 3
    grid = onp.full((5, n_cols), '#', dtype='<U1')

    # Row 0: all walls except G
    grid[0, n_path] = 'G'

    # Row 1: # >...> ! W
    grid[1, 1:n_path] = '>'
    grid[1, n_path] = '!'
    grid[1, n_path + 1] = 'W'

    # Row 2: I @ ####
    grid[2, 0] = 'I'
    grid[2, 1] = '@'

    # Row 3: same as row 1
    grid[3, 1:n_path] = '>'
    grid[3, n_path] = '!'
    grid[3, n_path + 1] = 'W'

    # Row 2: walls except F
    grid[2, n_path] = 'F'

    return grid


# ============================================================
# 3. Build analytic MDP + rendering spec
# ============================================================

def make_confounding_mdp_spec(
    params: EnvParams,
    dtype=jnp.float32
) -> MDPEnvSpec:
    """
    Build the confounding gridworld MDP + rendering information.

    Returns an MDPEnvSpec with:
        - P: (S, A, S)
        - R: (S, A)
        - state_coords: (S,2)
        - is_terminal: (S,)
        - initial_state: scalar
        - grid_intensity: (n_rows, n_cols) uint8 background
    """
    n_path = params.n_path
    epsilon = params.epsilon
    gamma = params.gamma

    grid = _create_symbolic_grid(n_path)
    n_rows, n_cols = grid.shape
    A = 4  # 0=UP,1=DOWN,2=LEFT,3=RIGHT

    # 1) Assign indices to non-wall cells
    state_index: Dict[Tuple[int, int], int] = {}
    coords = []
    s = 0
    for r in range(n_rows):
        for c in range(n_cols):
            if grid[r, c] != '#':
                state_index[(r, c)] = s
                coords.append((r, c))
                s += 1
    S = s
    state_coords_np = onp.array(coords, dtype=onp.int32)

    # 2) Allocate P and R (NumPy)
    P_np = onp.zeros((S, A, S), dtype=onp.float64)
    R_np = onp.zeros((S, A), dtype=onp.float64)

    def is_terminal_sym(sym: str) -> bool:
        return sym in ['G', 'W', 'F']

    # Hazard successors
    top_start = state_index[(1, 1)]
    bot_start = state_index[(3, 1)]

    critical_states = set()

    # Fill transitions
    for s_idx in range(S):
        r, c = state_coords_np[s_idx]
        sym = grid[r, c]

        if is_terminal_sym(sym):
            # Absorbing
            for a in range(A):
                P_np[s_idx, a, s_idx] = 1.0
                R_np[s_idx, a] = 0.0
            continue

        for a in range(A):
            # Default: self-loop, 0 reward
            P_np[s_idx, a, s_idx] = 1.0
            R_np[s_idx, a] = 0.0

            # I (start)
            if sym == 'I':
                if a == 3:  # RIGHT
                    s_next = state_index[(2, 1)]  # '@'
                    P_np[s_idx, a, :] = 0.0
                    P_np[s_idx, a, s_next] = 1.0
                    R_np[s_idx, a] = 0.0
                continue

            # Hazard '@'
            if sym == '@':
                P_np[s_idx, a, :] = 0.0
                P_np[s_idx, a, top_start] = 1.0 - epsilon
                P_np[s_idx, a, bot_start] = epsilon
                R_np[s_idx, a] = 0.0
                continue

            # Path '>'
            if sym == '>':
                if a == 3:  # RIGHT
                    r2, c2 = r, c + 1
                    if 0 <= r2 < n_rows and 0 <= c2 < n_cols and grid[r2, c2] != '#':
                        s_next = state_index[(r2, c2)]
                        P_np[s_idx, a, :] = 0.0
                        P_np[s_idx, a, s_next] = 1.0
                        R_np[s_idx, a] = 1.0
                # other actions: self-loop
                continue

            # Choice '!'
            if sym == '!':
                critical_states.add(s_idx)
                if a == 3:  # RIGHT -> W
                    r2, c2 = r, c + 1
                    s_next = state_index[(r2, c2)]
                    P_np[s_idx, a, :] = 0.0
                    P_np[s_idx, a, s_next] = 1.0
                    R_np[s_idx, a] = 1.0
                elif a == 0:  # UP -> G or F
                    if r == 1:  # top corridor
                        r2, c2 = 0, n_path
                        s_next = state_index[(r2, c2)]
                        P_np[s_idx, a, :] = 0.0
                        P_np[s_idx, a, s_next] = 1.0
                        R_np[s_idx, a] = n_path / gamma**n_path
                    elif r == 3:  # bottom corridor
                        r2, c2 = 2, n_path
                        s_next = state_index[(r2, c2)]
                        P_np[s_idx, a, :] = 0.0
                        P_np[s_idx, a, s_next] = 1.0
                        R_np[s_idx, a] = - (2. - epsilon) * n_path / (epsilon * gamma**n_path)
                # other actions: self-loop
                continue

            # No other non-wall symbols should appear here.

    # 3) Build intensity grid (background) for rendering
    grid_intensity_np = onp.zeros((n_rows, n_cols), dtype=onp.uint8)
    # defaults: walls 50
    grid_intensity_np[grid == '#'] = 50
    grid_intensity_np[grid == '>'] = 100
    grid_intensity_np[grid == '!'] = 100  # '!' and '>' have the same intensity.
    grid_intensity_np[grid == '@'] = 180
    grid_intensity_np[grid == 'I'] = 120
    grid_intensity_np[grid == 'G'] = 220
    grid_intensity_np[grid == 'W'] = 200
    grid_intensity_np[grid == 'F'] = 220  # G and F are not distinguishable from their color

    # 4) Terminal mask and initial state index
    is_terminal_np = onp.zeros((S,), dtype=onp.bool_)
    for s_idx in range(S):
        r, c = state_coords_np[s_idx]
        if is_terminal_sym(grid[r, c]):
            is_terminal_np[s_idx] = True

    initial_state = state_index[(2, 0)]  # 'I' at (2,0)

    # 5) Convert everything to JAX arrays
    P = jnp.asarray(P_np, dtype=dtype)
    R = jnp.asarray(R_np, dtype=dtype)
    state_coords = jnp.asarray(state_coords_np, dtype=jnp.int32)
    is_terminal = jnp.asarray(is_terminal_np)
    initial_state = jnp.asarray(initial_state, dtype=jnp.int32)
    grid_intensity = jnp.asarray(grid_intensity_np, dtype=jnp.uint8)

    spec = MDPEnvSpec(
        params=params,
        P=P,
        R=R,
        state_coords=state_coords,
        is_terminal=is_terminal,
        initial_state=initial_state,
        grid_intensity=grid_intensity,
        critical_states=tuple(critical_states),
    )
    return spec


# ============================================================
# 4. Rendering: state -> 84x84 frame
# ============================================================

def render_obs(spec: MDPEnvSpec, state: EnvState) -> jnp.ndarray:
    """
    Render current state as a single (H,W) uint8 frame.
    Background intensities from spec.grid_intensity; agent at (row,col) is 255.
    """
    grid_intensity = spec.grid_intensity  # (n_rows, n_cols)
    n_rows, n_cols = grid_intensity.shape
    stack, frame_h, frame_w = spec.params.frame_size

    # Convert state index -> (row, col)
    rc = spec.state_coords[state.s]    # (2,)
    row = rc[0]
    col = rc[1]

    # Start from background
    base = grid_intensity

    # Overlay agent (if not done)
    def draw_agent(im):
        return im.at[row, col].set(255)

    base_with_agent = lax.cond(
        state.done,
        lambda im: im,
        draw_agent,
        base
    )

    # Resize to Atari frame
    obs = jimage.resize(
        base_with_agent,
        shape=(frame_h, frame_w),
        method="nearest"
    )
    if stack == 1:
        obs = obs[None]
    else:
        obs = jnp.repeat(obs[None], stack, axis=0)

    return obs.astype(jnp.uint8)


# ============================================================
# 5. Single-env reset / step
# ============================================================

def reset(
    key: jax.Array,
    spec: MDPEnvSpec
) -> Tuple[EnvState, jnp.ndarray]:
    """
    Reset environment to initial state ('I').
    """
    del key  # no randomness on reset

    state = EnvState(
        s=spec.initial_state,
        done=jnp.array(False),
        step=jnp.int32(0),
    )
    obs = render_obs(spec, state)
    return state, obs


def step(
    key: jax.Array,
    spec: MDPEnvSpec,
    state: EnvState,
    action: jnp.ndarray
) -> Tuple[EnvState, jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
    """
    One environment step.
    - Transitions follow analytic MDP: P[s,a,:]
    - Reward from R[s,a]
    - Termination when entering terminal state or hitting max_steps
    """
    def done_branch(_):
        obs = render_obs(spec, state)
        reward = jnp.array(0.0, dtype=spec.R.dtype)
        done = jnp.array(True)
        info: Dict[str, Any] = {}
        return state, obs, reward, done, info

    def active_branch(_):
        s = state.s
        # Reward from R
        reward = spec.R[s, action]

        # Sample next state ~ P[s, a, :]
        probs = spec.P[s, action]               # (S,)
        s_next = jax.random.choice(key, probs.shape[0], p=probs)

        # Determine termination
        terminal = spec.is_terminal[s_next]
        new_step = state.step + jnp.int32(1)
        trunc = new_step >= spec.params.max_steps
        done = terminal | trunc

        new_state = EnvState(
            s=s_next,
            done=done,
            step=new_step,
        )

        obs = render_obs(spec, new_state)
        info: Dict[str, Any] = {}
        return new_state, obs, reward, done, info

    new_state, obs, reward, done, info = lax.cond(
        state.done,
        done_branch,
        active_branch,
        operand=None
    )

    return new_state, obs, reward, done, info


# ============================================================
# 6. Vectorized multi-env wrappers
# ============================================================

def reset_vec(
    keys: jax.Array,
    spec: MDPEnvSpec,
) -> Tuple[EnvState, jnp.ndarray]:
    """
    Vectorized reset over num_envs.

    Returns:
        env_keys: (num_envs, 2) PRNGKeys
        states: EnvState with batched fields (num_envs,)
        obs: (num_envs, H, W) uint8
    """
    def _reset(k):
        return reset(k, spec)

    states, obs = jax.vmap(_reset)(keys)
    return states, obs


def step_vec(
    keys: jax.Array,
    spec: MDPEnvSpec,
    states: EnvState,
    actions: jnp.ndarray
) -> Tuple[EnvState, jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
    """
    Vectorized step over num_envs.

    Args:
        keys: (num_envs, 2) PRNGKeys
        spec: MDPEnvSpec
        states: EnvState (batched)
        actions: (num_envs,) int32 in {0,1,2,3}

    Returns:
        new_states: EnvState (batched)
        obs: (num_envs, H, W) uint8
        rewards: (num_envs,) float32
        dones: (num_envs,) bool
        infos: {} (empty dict)
    """
    def _step(k, s, a):
        return step(k, spec, s, a)

    new_states, obs, rewards, dones, infos = jax.vmap(_step)(keys, states, actions)
    return new_states, obs, rewards, dones, infos

# ============================================================
# 7. Policy evaluation
# ============================================================

def evaluate_policy_exact(
    P: jnp.ndarray,
    R: jnp.ndarray,
    pi: jnp.ndarray,
    gamma: float,
) -> jnp.ndarray:
    """
    Exact policy evaluation for an analytic MDP.

    Args:
        P: (S, A, S) transition probabilities, P[s, a, s'] = P(s' | s, a)
        R: (S, A) rewards, R[s, a] = E[r | s, a]
        pi: (S, A) policy, pi[s, a] = π(a | s)
             (can be deterministic one-hot or stochastic)
        gamma: discount factor in (0,1)

    Returns:
        V: (S,) state-value function under policy π
    """
    # P^π[s, s'] = sum_a π[s,a] P[s,a,s']
    P_pi = jnp.einsum("sa,saz->sz", pi, P)    # (S,S)

    # r^π[s] = sum_a π[s,a] R[s,a]
    r_pi = jnp.einsum("sa,sa->s", pi, R)      # (S,)

    S = P_pi.shape[0]
    I = jnp.eye(S, dtype=P_pi.dtype)

    # Solve (I - γ P^π) V = r^π
    V = jnp.linalg.solve(I - gamma * P_pi, r_pi)
    return V


def evaluate_policy_iterative(
    P: jnp.ndarray,
    R: jnp.ndarray,
    pi: jnp.ndarray,
    gamma: float,
    num_iters: int = 1000,
) -> jnp.ndarray:
    """
    Iterative policy evaluation via fixed-point iteration:

        V_{k+1} = r^π + γ P^π V_k

    This gives the same limit as evaluate_policy_exact, but via iterations.

    Args:
        P: (S, A, S) transition probabilities
        R: (S, A) rewards
        pi: (S, A) policy
        gamma: discount factor
        num_iters: number of iterations

    Returns:
        V: (S,) approximate value function after num_iters updates
    """
    # Precompute P^π and r^π
    P_pi = jnp.einsum("sa,saz->sz", pi, P)   # (S,S)
    r_pi = jnp.einsum("sa,sa->s", pi, R)     # (S,)

    S = P_pi.shape[0]
    V0 = jnp.zeros((S,), dtype=P_pi.dtype)

    def body(_, V):
        return r_pi + gamma * (P_pi @ V)

    V = lax.fori_loop(0, num_iters, body, V0)
    return V

# ============================================================
# 8. CLI gameplay loop
# ============================================================

def action_from_key(key_str: str) -> int:
    """
    Map user key to action index.
    0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
    """
    key_str = key_str.strip().lower()
    if key_str == 'w':
        return 0
    elif key_str == 's':
        return 1
    elif key_str == 'a':
        return 2
    elif key_str == 'd':
        return 3
    else:
        return -1  # invalid

# ============================================================
# 9. EnvPool-like environment
# ============================================================
class ConfoundingGridEnvPoolLike:
    """
    Minimal adapter to look like an EnvPool env for DeepSPIAgent.
    """
    def __init__(self, num_envs: int, seed: int, n_path: int = 5, epsilon: float = 0.05, gamma: float = 0.99):
        self.num_envs = num_envs
        self.is_vector_env = True

        # mimic Atari obs: (1, 84, 84) uint8
        self.single_observation_space = spaces.Box(
            low=0, high=255, shape=(1, 84, 84), dtype=np.uint8
        )
        # 4 discrete actions: up, down, left, right
        self.single_action_space = spaces.Discrete(4)
        self.action_space = self.single_action_space

        # build analytic MDP spec
        self.params = EnvParams(
            n_path=n_path,
            epsilon=epsilon,
            frame_size=(1, 84, 84),
            max_steps=200,
            gamma=gamma)
        self.spec = make_confounding_mdp_spec(self.params)

        # JAX RNG
        self._rng_key = jax.random.PRNGKey(seed)

        # Do an initial vectorized reset, store handle + obs
        subkeys = jax.random.split(self._rng_key, self.num_envs)
        states, obs = reset_vec(subkeys, self.spec)
        self._initial_handle = (self._rng_key, states)
        self._initial_obs = obs  # shape (num_envs, 1, 84, 84)
        if debug:
            symbol_map = {
                "#": 0,
                ">": 1,
                "I": 2,
                "@": 3,
                "!": 4,
                "W": 5,
                "G": 6,
                "F": 7,
            }
            # action_map = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
            print("DEBUG: Symbol map:", symbol_map)
            sym_grid = _create_symbolic_grid(n_path)
            for r in range(sym_grid.shape[0]):
                for c in range(sym_grid.shape[1]):
                    sym = sym_grid[r, c]
                    sym_grid[r, c] = symbol_map[sym]
            sym_grid = sym_grid.astype("uint8")
            self._symbolic_grid = jnp.asarray(sym_grid)
            self._state_coords = jnp.asarray(self.spec.state_coords)
        else:
            self._symbolic_grid = None

    def xla(self):
        # start handle: already contains env_keys, states from an initial reset
        handle = self._initial_handle

        def step_env(handle, actions):
            rng_key, states = handle

            # Step the MDP
            rng_key, step_key = jax.random.split(rng_key)
            step_keys = jax.random.split(step_key, self.num_envs)
            new_states, obs, rewards, dones, infos = step_vec(
                step_keys, self.spec, states, actions
            )

            # DEBUG: post-step, BEFORE reset
            if debug:
                r1, c1 = self.spec.state_coords[states.s[0]]
                sym1 = self._symbolic_grid[r1, c1]
                r2, c2 = self.spec.state_coords[new_states.s[0]]
                sym2 = self._symbolic_grid[r2, c2]
                jax.debug.print(
                    "t={t}, s={s} [sym: {sym1}], a={a}  -->  s'={sp} [sym: {sym2}], r={r}, done={d}",
                    t=states.step[0],
                    s=states.s[0],
                    sym1=sym1,
                    a=actions[0],
                    sp=new_states.s[0],
                    sym2=sym2,
                    r=rewards[0],
                    d=dones[0],
                )

            # Auto-reset any envs where done=True
            #    This is the EnvPool-like behaviour:
            #    - the (obs, reward, done) we return correspond to the terminal transition
            #    - but the *next* call will start from a reset state.
            terminated = dones  # shape (num_envs,)

            # Always compute reset candidates for all envs, then merge with mask
            rng_key, reset_key = jax.random.split(rng_key)
            reset_keys = jax.random.split(reset_key, self.num_envs)
            reset_states, reset_obs = reset_vec(
                reset_keys, self.spec
            )

            def merge(new, reset):
                # new, reset: (num_envs, ...)
                # broadcast mask to all non-batch dims
                mask = terminated.reshape((self.num_envs,) +
                                          (1,) * (new.ndim - 1))
                return jnp.where(mask, reset, new)

            # states is a dataclass of arrays: apply merge to each field
            merged_states = jax.tree_map(merge, new_states, reset_states)
            # obs: (num_envs, 1, 84, 84) – return *post-reset* obs
            merged_obs = merge(obs, reset_obs)

            # Build new handle with already-reset states
            new_handle = (rng_key, merged_states)

            # Info: done is about the *transition*, not about the next obs
            info = {
                "reward": rewards,
                "terminated": dones,
                "TimeLimit.truncated": jnp.zeros_like(dones, dtype=jnp.bool_),
            }
            return new_handle, (merged_obs, rewards, dones, info)

        recv = send = None
        return handle, recv, send, step_env

    def reset(self):
        """
        Mimic EnvPool.reset(): return batched observations.
        We simply return the initial obs we computed in __init__.
        """
        # If you want a fresh reset each time, you could call reset_vec again,
        # but for this use case train() only calls reset() once at the start.
        return np.array(self._initial_obs)  # or jnp.array(self._initial_obs)

    def log(
            self,
            pi: jnp.ndarray,
            use_value_iteration: bool = False,
            gamma: float = 0.99,
            num_iters: int = 1000,
    ) -> Dict[str, np.ndarray]:
        """
        Convenience wrapper: outputs V(initial_state) and the full V.
        """
        if use_value_iteration:
            V = evaluate_policy_iterative(
                P=self.spec.P,
                R=self.spec.R,
                pi=pi,
                gamma=gamma,
                num_iters=num_iters,
            )
        else:
            V = evaluate_policy_exact(
                P=self.spec.P,
                R=self.spec.R,
                pi=pi,
                gamma=gamma,
            )

        init = self.spec.initial_state
        return {
            'value_s_init': np.asarray(V[init]),
            'value_full': np.asarray(V),
        }

    def close(self):
        pass


def main():
    # ----- Environment setup -----
    params = EnvParams(
        n_path=5,
        epsilon=0.2,
        frame_size=(1, 84, 84),
        max_steps=200,
    )
    spec = make_confounding_mdp_spec(params)

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    # Testing value estimation
    P = spec.P
    R = spec.R
    S = P.shape[0]
    gamma = params.gamma
    actions = jnp.full((S, ), RIGHT, dtype=jnp.int32)
    pi_right = jnp.eye(4, dtype=P.dtype)[actions]
    V_exact = evaluate_policy_exact(P, R, pi_right, gamma)
    V_iterative = evaluate_policy_iterative(P, R, pi_right, gamma)

    s0 = spec.initial_state
    print("V_exact(initial) =", float(V_exact[s0]))
    print("V_iter(initial)  =", float(V_iterative[s0]))

    # testing confounding PU
    pi_bad = pi_right
    for state in spec.critical_states:
        pi_bad = pi_bad.at[state].set(jnp.eye(4, dtype=P.dtype)[UP])
    pi_good = pi_right.at[spec.critical_states[-1]].set(jnp.eye(4, dtype=P.dtype)[RIGHT])

    V_exact_bad = evaluate_policy_exact(P, R, pi_bad, gamma)
    V_exact_good = evaluate_policy_exact(P, R, pi_good, gamma)
    print("V_exact_bad(initial) =", float(V_exact_bad[s0]))
    print("V_exact_good(initial) =", float(V_exact_good[s0]))

    key = random.PRNGKey(0)
    state, obs = reset(key, spec)  # obs: (1, 84, 84)

    total_reward = 0.0

    # ----- Matplotlib setup -----
    fig, ax = plt.subplots()

    # Always pass a 2D array (H, W) to imshow
    first_frame = onp.array(obs[0])  # shape: (84, 84)
    img = ax.imshow(first_frame, cmap='gray', vmin=0, vmax=255)
    ax.set_title("Confounding Gridworld")
    plt.show()
    plt.close()

    print("Controls: w=UP, s=DOWN, a=LEFT, d=RIGHT, q=quit")
    print("Starting episode...\n")

    # ----- Gameplay loop -----
    while True:
        fig, ax = plt.subplots()
        # Update display (again as a 2D array)
        ax.imshow(onp.array(obs[0]), cmap='gray', vmin=0, vmax=255)
        ax.set_xlabel(
            f"Step: {int(state.step)}, "
            f"Done: {bool(state.done)}, "
            f"Total reward: {total_reward:.2f}"
        )
        fig.canvas.draw()
        plt.show()
        plt.pause(0.001)
        ax.clear()
        plt.close()

        # If episode finished, ask to reset or quit
        if bool(state.done):
            print(f"Episode finished. Total reward: {total_reward:.2f}")
            cmd = input("Press [r] to reset, [q] to quit: ").strip().lower()
            if cmd == 'q':
                break
            elif cmd == 'r':
                key, subkey = random.split(key)
                state, obs = reset(subkey, spec)
                total_reward = 0.0
                continue
            else:
                print("Unknown command, quitting.")
                break

        # Ask for user action
        cmd = input("Action (w/a/s/d, q=quit): ").strip().lower()
        if cmd == 'q':
            print("Quitting.")
            break

        a = action_from_key(cmd)
        if a < 0:
            print("Invalid action. Use w/a/s/d or q.")
            continue

        # Step the environment
        key, subkey = random.split(key)
        key, state, obs, reward, done, info = step(subkey, spec, state, jnp.int32(a))
        total_reward += float(reward)

    plt.ioff()
    plt.close(fig)


if __name__ == "__main__":
    main()
