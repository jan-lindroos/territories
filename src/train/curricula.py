from collections import deque

import torch
import territories


STAGES = [
    # Stage 1: Learn basic mechanics (1 agent, small grid)
    #   Decaying trail bonus encourages short excursions, -5 death, territory capped at 10
    {"width": 10, "height": 10, "num_agents": 1, "max_steps": 30,
     "death_penalty": -5.0, "explore_bonus": 1.0,
     "terr_clamp": (0, 10), "advance_territory_per_capture": 4.5},

    # Stage 2: Learn competitive play (4 agents, small grid)
    #   Small explore bonus retained, -10 death, territory changes capped [-10, 20]
    {"width": 10, "height": 10, "num_agents": 4, "max_steps": 40,
     "death_penalty": -10.0, "explore_bonus": 0.3,
     "terr_clamp": (-10, 20), "advance_territory_per_capture": 4.0},

    # Stage 3: Full game (10 agents, big grid)
    #   -10 death, territory uncapped
    {"width": 40, "height": 40, "num_agents": 10, "max_steps": 200,
     "death_penalty": -10.0, "explore_bonus": 0.0,
     "terr_clamp": (-10, None), "advance_territory_per_capture": None},
]

# Advance after the rolling average over this many iterations exceeds threshold
ADVANCE_WINDOW = 10


class Curriculum:
    """Wraps VectorEnv with shaped rewards, auto-reset, and staged difficulty.

    Stage 1: 1 agent, 10x10 — learn to leave territory and close loops.
    Stage 2: 4 agents, 10x10 — learn competitive play with territory changes.
    Stage 3: 10 agents, 40x40 — full game, uncapped territory.
    """

    def __init__(self, num_envs=64, window_radius=5, stage=0):
        self.num_envs = num_envs
        self.window_radius = window_radius
        self.grid_side = 2 * window_radius + 1
        self.obs_size = self.grid_side ** 2
        self.num_channels = 1
        self.num_actions = 4
        self.stage = stage
        self.step_counts = None
        self.prev_alive = None
        self.prev_trail_counts = None
        self.reward_history = deque(maxlen=ADVANCE_WINDOW)
        self._apply_stage(stage)

    def _apply_stage(self, stage):
        self.stage = stage
        cfg = STAGES[stage]
        self.num_agents = cfg["num_agents"]
        self.max_steps = cfg["max_steps"]
        self.advance_threshold = cfg["advance_territory_per_capture"]
        self.death_penalty = cfg["death_penalty"]
        self.explore_bonus = cfg["explore_bonus"]
        self.terr_clamp = cfg["terr_clamp"]
        self.reward_history.clear()
        self.vec_env = territories.VectorEnv(
            self.num_envs, self.num_agents,
            cfg["width"], cfg["height"], self.window_radius,
        )

    def should_advance(self, metrics):
        """Record metrics and return True if it's time to advance to the next stage."""
        self.reward_history.append(metrics["avg_territory_per_capture"])
        if self.advance_threshold is None:
            return False
        if len(self.reward_history) < ADVANCE_WINDOW:
            return False
        return sum(self.reward_history) / len(self.reward_history) >= self.advance_threshold

    def advance_stage(self):
        """Move to the next difficulty stage. Returns True if advanced, False if already at max."""
        if self.stage + 1 >= len(STAGES):
            return False
        self._apply_stage(self.stage + 1)
        return True

    def _get_alive(self):
        return torch.tensor(self.vec_env.get_alive()).view(self.num_envs, self.num_agents)

    def _get_trail_counts(self):
        return torch.tensor(self.vec_env.get_trail_counts()).view(self.num_envs, self.num_agents)

    def _get_territory_counts(self):
        return torch.tensor(self.vec_env.get_territory_counts()).view(self.num_envs, self.num_agents)

    def _encode_obs(self):
        arr = self.vec_env.get_observations_numpy()                   # [E, A, H*W] int32 numpy
        obs = torch.from_numpy(arr).float()                           # zero-copy int->tensor, then float
        obs = (obs + 4.0) / 7.0
        return obs.view(self.num_envs, self.num_agents, 1, self.grid_side, self.grid_side)

    def reset(self):
        self.vec_env.reset()
        self.step_counts = torch.zeros(self.num_envs, dtype=torch.long)
        self.prev_alive = self._get_alive()
        self.prev_trail_counts = self._get_trail_counts()
        # Metrics accumulators (reset each rollout via get_metrics)
        self.total_deaths = 0
        self.total_captures = 0          # successful trail closes
        self.total_territory_gained = 0  # total cells gained from captures
        self.total_alive_steps = 0       # agent-steps while alive
        return self._encode_obs()

    def step(self, actions):
        """Step all envs. Dead agents receive action 0 (ignored by C++ env).

        Returns:
            obs:          [E, A, 1, H, W] float tensor
            rewards:      [E, A] float tensor
            dones:        [E, A] bool tensor — True on the step an agent dies
            action_mask:  [E, A] bool tensor — True for agents alive before this step
        """
        if isinstance(actions, torch.Tensor):
            actions = actions.int()
        else:
            actions = torch.tensor(actions, dtype=torch.int)

        prev_territory = self._get_territory_counts()
        territory_deltas = torch.tensor(self.vec_env.step(actions.tolist()))
        cur_alive = self._get_alive()
        cur_trails = self._get_trail_counts()
        cur_territory = self._get_territory_counts()

        action_mask = self.prev_alive.bool()
        just_died = action_mask & ~cur_alive.bool()
        still_alive = action_mask & cur_alive.bool()

        # Track game-skill metrics
        self.total_deaths += just_died.sum().item()
        self.total_alive_steps += still_alive.sum().item()
        gained = (cur_territory - prev_territory).clamp(min=0)
        captured_mask = gained > 0
        self.total_captures += captured_mask.sum().item()
        self.total_territory_gained += gained.sum().item()

        rewards = torch.zeros(self.num_envs, self.num_agents)
        rewards[just_died] = self.death_penalty

        # Exploration bonus: decays with trail length to encourage short excursions
        if self.explore_bonus > 0:
            trail_delta = (cur_trails - self.prev_trail_counts).clamp(min=0).float()
            exploring = still_alive & (trail_delta > 0)
            if exploring.any():
                decay = self.explore_bonus / cur_trails[exploring].float()
                rewards[exploring] += decay

        # Territory changes — clamp per stage config
        terr_lo, terr_hi = self.terr_clamp
        if terr_hi is not None:
            terr = territory_deltas.clamp(min=terr_lo, max=terr_hi).float()
        else:
            terr = territory_deltas.clamp(min=terr_lo).float()
        rewards[still_alive] += terr[still_alive]

        # Auto-reset envs where everyone is dead or max steps reached
        self.step_counts += 1
        all_dead = ~cur_alive.bool().any(dim=1)
        timed_out = self.step_counts >= self.max_steps
        needs_reset = all_dead | timed_out

        # Mark all agents as done on timeout so GAE doesn't bootstrap across episode boundary
        dones = ~cur_alive.bool()
        dones[timed_out] = True
        for e in needs_reset.nonzero(as_tuple=False).squeeze(-1).tolist():
            self.vec_env.get_env(e).reset()

        if needs_reset.any():
            self.step_counts[needs_reset] = 0
            cur_alive = self._get_alive()
            cur_trails = self._get_trail_counts()

        self.prev_alive = cur_alive
        self.prev_trail_counts = cur_trails

        return self._encode_obs(), rewards, dones, action_mask

    def get_metrics(self):
        """Return game-skill metrics for the period since last call, then reset."""
        metrics = {
            "deaths": self.total_deaths,
            "captures": self.total_captures,
            "territory_gained": self.total_territory_gained,
            "avg_territory_per_capture": self.total_territory_gained / max(self.total_captures, 1),
            "survival_rate": self.total_alive_steps / max(self.total_alive_steps + self.total_deaths, 1),
            "alive_steps": self.total_alive_steps,
        }
        self.total_deaths = 0
        self.total_captures = 0
        self.total_territory_gained = 0
        self.total_alive_steps = 0
        return metrics
