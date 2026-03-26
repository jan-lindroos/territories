"""Terminal UI for the Territories game."""

import curses
import random
import time

from simulation import Action, Simulation

AI = None  # Set to a callable(sim) -> list[Action] to use a custom policy.

PLAYER_COLORS = [
    curses.COLOR_BLUE, curses.COLOR_RED, curses.COLOR_GREEN,
    curses.COLOR_YELLOW, curses.COLOR_MAGENTA, curses.COLOR_CYAN,
]
CP_PLAYERS = 1

# Directions: UP=0(dx=0,dy=-1), RIGHT=1(1,0), DOWN=2(0,1), LEFT=3(-1,0)
DX = [0, 1, 0, -1]
DY = [-1, 0, 1, 0]


def init_colors():
    curses.start_color()
    curses.use_default_colors()
    for p, fg in enumerate(PLAYER_COLORS):
        curses.init_pair(CP_PLAYERS + p * 2, curses.COLOR_WHITE, fg)
        curses.init_pair(CP_PLAYERS + p * 2 + 1, fg, -1)


def put(win, r, c, text, attrs=0):
    try:
        win.addstr(r, c, text, attrs)
    except curses.error:
        pass


def random_safe_actions(sim: Simulation) -> list[Action]:
    """Pick random actions that avoid immediate wall death when possible."""
    actions = []
    for i in range(sim.num_agents):
        if not sim.is_alive(i):
            actions.append(Action.UP)
            continue
        s = sim.get_snake(i)
        safe = []
        for a in Action:
            nx = s.x + DX[a]
            ny = s.y + DY[a]
            if 1 <= nx < sim.width - 1 and 1 <= ny < sim.height - 1:
                safe.append(a)
        actions.append(random.choice(safe) if safe else random.choice(list(Action)))
    return actions


def draw(stdscr, sim: Simulation, episode, paused, scores):
    stdscr.erase()
    _, max_c = stdscr.getmaxyx()
    h, w, n = sim.height, sim.width, sim.num_agents

    header = f" Territories  ep={episode}  {'PAUSED' if paused else 'running'}"
    put(stdscr, 0, 0, header.ljust(max_c - 1), curses.A_REVERSE)

    put(stdscr, 1, 0, "+" + "-" * w + "+")
    put(stdscr, h + 2, 0, "+" + "-" * w + "+")
    for r in range(h):
        put(stdscr, 2 + r, 0, "|")
        put(stdscr, 2 + r, w + 1, "|")

    tgrid = sim.env.territory_grid
    rgrid = sim.env.trail_grid

    for y in range(h):
        for x in range(w):
            idx = y * w + x
            # Check for head
            head_owner = -1
            for i in range(n):
                if sim.is_alive(i):
                    s = sim.get_snake(i)
                    if s.x == x and s.y == y:
                        head_owner = i
                        break

            if head_owner >= 0:
                ci = head_owner % len(PLAYER_COLORS)
                put(stdscr, 2 + y, 1 + x, "@",
                    curses.color_pair(CP_PLAYERS + ci * 2 + 1) | curses.A_BOLD)
            elif rgrid[idx] >= 0:
                ci = rgrid[idx] % len(PLAYER_COLORS)
                put(stdscr, 2 + y, 1 + x, ".",
                    curses.color_pair(CP_PLAYERS + ci * 2 + 1) | curses.A_DIM)
            elif tgrid[idx] >= 0:
                ci = tgrid[idx] % len(PLAYER_COLORS)
                put(stdscr, 2 + y, 1 + x, " ",
                    curses.color_pair(CP_PLAYERS + ci * 2) | curses.A_BOLD)

    lc = w + 4
    put(stdscr, 2, lc, "Players:", curses.A_BOLD)
    for i in range(n):
        ci = i % len(PLAYER_COLORS)
        status = "dead " if not sim.is_alive(i) else "alive"
        cp = curses.color_pair(CP_PLAYERS + ci * 2) | curses.A_BOLD
        put(stdscr, 3 + i, lc, f"  P{i}  score={scores[i]:5d}  {status}", cp)

    put(stdscr, h + 4, 0, " q quit  space pause ", curses.A_DIM)
    stdscr.refresh()


def run(stdscr):
    init_colors()
    curses.curs_set(0)
    stdscr.nodelay(True)

    h, w, n = 30, 60, 4
    sim = Simulation(num_agents=n, width=w, height=h)
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
        if paused:
            time.sleep(0.05)
            continue

        actions = AI(sim) if AI else random_safe_actions(sim)
        deltas, _ = sim.step(actions)
        for i in range(n):
            scores[i] += deltas[i]

        if not any(sim.is_alive(i) for i in range(n)):
            time.sleep(0.1)
            sim = Simulation(num_agents=n, width=w, height=h)
            sim.reset()
            scores = [0] * n
            episode += 1

        time.sleep(0.05)


if __name__ == "__main__":
    curses.wrapper(run)
