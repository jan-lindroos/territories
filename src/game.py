import curses
import time
from pathlib import Path

import numpy as np
import torch

import territories
from rl.policy import Policy
from ui import init_colors, draw


def load_policy(checkpoint: Path) -> tuple[Policy, int]:
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    window_radius = ckpt.get("window_radius", 7) if isinstance(ckpt, dict) else 7
    state_dict = ckpt["policy"] if isinstance(ckpt, dict) and "policy" in ckpt else ckpt
    policy = Policy(window_radius=window_radius)
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy, window_radius


def get_actions(policy: Policy, env, n: int, side: int) -> list[int]:
    obs = torch.from_numpy(np.array(env.get_obs(), dtype=np.float32)).view(n, 1, side, side) / 4.0
    with torch.no_grad():
        logits = policy.actor(policy.backbone(obs))
    return logits.argmax(dim=-1).tolist()


def run(stdscr, checkpoint: Path, w: int = 30, h: int = 30, n: int = 6) -> None:
    init_colors()
    curses.curs_set(0)
    stdscr.nodelay(True)

    policy, window_radius = load_policy(checkpoint)
    side = 2 * window_radius + 1
    seed = 0

    def new_env():
        nonlocal seed
        env = territories.Env(n, w, h, window_radius, seed=seed)
        seed += 1
        env.reset()
        return env

    env = new_env()
    scores = [0] * n
    prev_territory = [a.territory for a in env.get_agents()]
    paused = False
    episode = 1

    while True:
        draw(stdscr, env.get_agents(), env.get_territories(), env.get_trails(), episode, paused, scores, w, h)

        ch = stdscr.getch()
        if ch == ord("q"):
            break
        if ch == ord(" "):
            paused = not paused
        if ch == ord("r"):
            env = new_env()
            scores = [0] * n
            prev_territory = [a.territory for a in env.get_agents()]
            episode += 1
            continue

        if paused:
            time.sleep(0.05)
            continue

        env.step(get_actions(policy, env, n, side))
        agents = env.get_agents()
        for i, agent in enumerate(agents):
            delta = agent.territory - prev_territory[i]
            if delta > 0:
                scores[i] += delta
        prev_territory = [a.territory for a in agents]

        if not any(a.is_alive for a in agents):
            time.sleep(0.3)
            env = new_env()
            scores = [0] * n
            prev_territory = [a.territory for a in env.get_agents()]
            episode += 1

        time.sleep(0.05)
