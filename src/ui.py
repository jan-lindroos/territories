"""Terminal UI for the Territories game."""

import curses
import time
from pathlib import Path

import torch

from simulation import Action, Simulation
from train.ppo import Policy

PLAYER_COLORS = []  # populated at runtime by init_colors()
CP_PLAYERS = 1
MODEL_PATH = Path(__file__).with_name("train") / "best.pt"
GRID_SIDE = 11  # 2 * window_radius + 1

# 256-color extended palette (color index, r, g, b as 0-1000 curses values)
_EXTENDED_COLORS = [
    (50, 1000, 500, 0),    # orange
    (51, 400, 800, 1000),  # sky blue
    (52, 1000, 400, 700),  # pink
    (53, 500, 1000, 500),  # lime
    (54, 700, 400, 1000),  # purple
    (55, 1000, 800, 300),  # gold
    (56, 0, 800, 600),     # teal
    (57, 900, 300, 300),   # coral
]


def init_colors():
    curses.start_color()
    curses.use_default_colors()

    base = [
        curses.COLOR_BLUE, curses.COLOR_RED, curses.COLOR_GREEN,
        curses.COLOR_YELLOW, curses.COLOR_MAGENTA, curses.COLOR_CYAN,
    ]

    if curses.COLORS >= 256 and curses.can_change_color():
        for idx, r, g, b in _EXTENDED_COLORS:
            curses.init_color(idx, r, g, b)
            base.append(idx)

    PLAYER_COLORS.extend(base)

    for p, fg in enumerate(PLAYER_COLORS):
        curses.init_pair(CP_PLAYERS + p * 2, curses.COLOR_WHITE, fg)
        curses.init_pair(CP_PLAYERS + p * 2 + 1, fg, -1)


def put(win, r, c, text, attrs=0):
    try:
        win.addstr(r, c, text, attrs)
    except curses.error:
        pass


def load_ai(path: Path = MODEL_PATH) -> "callable":
    """Load a trained Policy and return an AI callable(sim) -> list[Action]."""
    policy = Policy(num_channels=1, grid_side=GRID_SIDE, num_actions=4)
    ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()

    @torch.no_grad()
    def ai(sim: Simulation) -> list[Action]:
        observations = sim.get_observations()
        batch = torch.stack([
            torch.tensor([(c + 4.0) / 7.0 for c in obs], dtype=torch.float32)
                 .view(1, GRID_SIDE, GRID_SIDE)
            for obs in observations
        ])  # [N, 1, H, W]
        logits, _ = policy(batch)
        return [Action(a.item()) for a in logits.argmax(dim=-1)]

    return ai


def draw(stdscr, sim: Simulation, episode, paused, scores):
    stdscr.erase()
    _, max_c = stdscr.getmaxyx()
    h, w, n = sim.height, sim.width, sim.num_agents

    header = f" Territories  ep={episode}  {'PAUSED' if paused else 'running'}"
    put(stdscr, 0, 0, header.ljust(max_c - 1), curses.A_REVERSE)

    put(stdscr, 1, 0, "+" + "-" * (w * 2) + "+")
    put(stdscr, h + 2, 0, "+" + "-" * (w * 2) + "+")
    for r in range(h):
        put(stdscr, 2 + r, 0, "|")
        put(stdscr, 2 + r, w * 2 + 1, "|")

    tgrid = sim.env.territory_grid
    rgrid = sim.env.trail_grid

    for y in range(h):
        for x in range(w):
            idx = y * w + x
            head_owner = -1
            for i in range(n):
                if sim.is_alive(i):
                    s = sim.get_snake(i)
                    if s.x == x and s.y == y:
                        head_owner = i
                        break

            if head_owner >= 0:
                ci = head_owner % len(PLAYER_COLORS)
                put(stdscr, 2 + y, 1 + x * 2, "██",
                    curses.color_pair(CP_PLAYERS + ci * 2 + 1) | curses.A_BOLD)
            elif rgrid[idx] >= 0:
                ci = rgrid[idx] % len(PLAYER_COLORS)
                put(stdscr, 2 + y, 1 + x * 2, "░░",
                    curses.color_pair(CP_PLAYERS + ci * 2 + 1))
            elif tgrid[idx] >= 0:
                ci = tgrid[idx] % len(PLAYER_COLORS)
                put(stdscr, 2 + y, 1 + x * 2, "  ",
                    curses.color_pair(CP_PLAYERS + ci * 2))

    lc = w * 2 + 4
    put(stdscr, 2, lc, "Players:", curses.A_BOLD)
    for i in range(n):
        ci = i % len(PLAYER_COLORS)
        status = "dead " if not sim.is_alive(i) else "alive"
        cp = curses.color_pair(CP_PLAYERS + ci * 2) | curses.A_BOLD
        put(stdscr, 3 + i, lc, f"  P{i}  score={scores[i]:5d}  {status}", cp)

    put(stdscr, h + 4, 0, " q quit  space pause  r reset ", curses.A_DIM)
    stdscr.refresh()


def run(stdscr, h=30, w=30, n=6):
    init_colors()
    curses.curs_set(0)
    stdscr.nodelay(True)

    ai = load_ai()
    window_radius = (GRID_SIDE - 1) // 2
    sim = Simulation(num_agents=n, width=w, height=h, window_radius=window_radius)
    sim.reset()
    scores = [0] * n
    paused = False
    episode = 1

    while True:
        draw(stdscr, sim, episode, paused, scores)

        ch = stdscr.getch()
        if ch == ord("q"):
            break
        if ch == ord(" "):
            paused = not paused
        if ch == ord("r"):
            sim = Simulation(num_agents=n, width=w, height=h, window_radius=window_radius)
            sim.reset()
            scores = [0] * n
            episode += 1
            continue
        if paused:
            time.sleep(0.05)
            continue

        actions = ai(sim)
        deltas, _ = sim.step(actions)
        for i in range(n):
            scores[i] += deltas[i]

        if not any(sim.is_alive(i) for i in range(n)):
            time.sleep(0.1)
            sim = Simulation(num_agents=n, width=w, height=h, window_radius=window_radius)
            sim.reset()
            scores = [0] * n
            episode += 1

        time.sleep(0.05)


if __name__ == "__main__":
    curses.wrapper(run)
