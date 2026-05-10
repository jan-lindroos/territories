import numpy as np
import torch

import territories


class Curriculum:
    def __init__(
        self,
        num_envs: int = 1024,
        num_agents: int = 10,
        env_size: int = 50,
        max_steps: int = 200,
        window_radius: int = 7,
        terr_delta_coef: float = 1.0,
        terr_delta_min: float = 0,
        terr_delta_max: float = 100,
        death_penalty: float = 0.0,
        leave_terr_bonus: float = 0.0,
        kill_reward: float = 0.0,
        seed: int = 0
    ):
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.env_size = env_size
        self.max_steps = max_steps
        self.current_step = 0
        self.window_radius = window_radius
        self._terr_delta_coef = terr_delta_coef
        self._terr_delta_min = terr_delta_min
        self._terr_delta_max = terr_delta_max
        self._death_penalty = death_penalty
        self._leave_terr_bonus = leave_terr_bonus
        self._kill_reward = kill_reward

        self._envs = [
            territories.Env(
                num_agents, env_size, env_size,
                window_radius, seed=seed + i
            )
            for i in range(num_envs)
        ]

        self.reset()

    def get_obs_shape(self) -> tuple:
        side = 2 * self.window_radius + 1
        return (1, side, side)

    def get_obs(self):
        obs = torch.from_numpy(np.stack([env.get_obs() for env in self._envs]).astype(np.float32)) / 4.0
        return obs.view(len(self._envs), self.num_agents, *self.get_obs_shape())

    def _get_territory(self):
        return torch.tensor([
            [a.territory for a in env.get_agents()]
            for env in self._envs
        ], dtype=torch.float32)

    def get_alive(self):
        return torch.tensor([
            [a.is_alive for a in env.get_agents()]
            for env in self._envs
        ], dtype=torch.bool)

    def _get_has_trail(self):
        return torch.tensor([
            [a.has_trail for a in env.get_agents()]
            for env in self._envs
        ], dtype=torch.bool)

    def _get_kills(self):
        return torch.tensor([
            [a.kills for a in env.get_agents()]
            for env in self._envs
        ], dtype=torch.float32)

    def reset(self):
        self.current_step = 0
        for env in self._envs:
            env.reset()

    def step(self, actions: torch.Tensor):
        prev_terr = self._get_territory()
        prev_alive = self.get_alive()
        prev_kills = self._get_kills()

        for i, env in enumerate(self._envs):
            env.step(actions[i].int().tolist())

        curr_terr = self._get_territory()
        curr_alive = self.get_alive()
        curr_has_trail = self._get_has_trail()
        curr_kills = self._get_kills()

        rewards = self._terr_delta_coef * (curr_terr - prev_terr)
        rewards = np.clip(rewards, self._terr_delta_min, self._terr_delta_max)

        died = prev_alive & ~curr_alive
        rewards[died] += self._death_penalty

        rewards[curr_alive & curr_has_trail] += self._leave_terr_bonus

        rewards += self._kill_reward * (curr_kills - prev_kills)

        self.current_step += 1
        return rewards

