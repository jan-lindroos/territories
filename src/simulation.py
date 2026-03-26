import random
from enum import IntEnum

import territories


class Action(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class Cell(IntEnum):
    WALL = -4
    OWN_HEAD = -3
    OWN_TRAIL = -2
    OWN_TERRITORY = -1
    EMPTY = 0
    ENEMY_TERRITORY = 1
    ENEMY_TRAIL = 2
    ENEMY_HEAD = 3


class Simulation:
    def __init__(
        self,
        num_agents: int = 2,
        width: int = 20,
        height: int = 20,
        window_radius: int = 7,
        seed: int | None = None,
    ):
        if seed is None:
            seed = random.randint(0, 99999)
        self.env = territories.Env(num_agents, width, height, window_radius, seed)

    @property
    def num_agents(self) -> int:
        return self.env.num_agents

    @property
    def width(self) -> int:
        return self.env.width

    @property
    def height(self) -> int:
        return self.env.height

    @property
    def window_radius(self) -> int:
        return self.env.window_radius

    def reset(self) -> list[list[Cell]]:
        self.env.reset()
        return self.get_observations()

    def step(self, actions: list[Action]) -> tuple[list[int], list[list[Cell]]]:
        scores = self.env.step([int(a) for a in actions])
        return scores, self.get_observations()

    def get_observations(self) -> list[list[Cell]]:
        return [[Cell(c) for c in obs] for obs in self.env.get_observations()]

    def get_snake(self, agent_id: int) -> territories.Snake:
        return self.env.snakes[agent_id]

    def is_alive(self, agent_id: int) -> bool:
        return self.env.snakes[agent_id].alive

    def get_territory_count(self, agent_id: int) -> int:
        return sum(1 for c in self.env.territory_grid if c == agent_id)

    def get_trail_count(self, agent_id: int) -> int:
        return sum(1 for c in self.env.trail_grid if c == agent_id)
