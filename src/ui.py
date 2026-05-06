import curses

PLAYER_COLORS: list[int] = []
CP_BASE = 1

_EXTENDED_COLORS = [
    (50, 1000, 500, 0),
    (51, 400, 800, 1000),
    (52, 1000, 400, 700),
    (53, 500, 1000, 500),
    (54, 700, 400, 1000),
    (55, 1000, 800, 300),
    (56, 0, 800, 600),
    (57, 900, 300, 300),
]


def init_colors() -> None:
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
        curses.init_pair(CP_BASE + p * 2, curses.COLOR_WHITE, fg)
        curses.init_pair(CP_BASE + p * 2 + 1, fg, -1)


def put(win, r: int, c: int, text: str, attrs: int = 0) -> None:
    try:
        win.addstr(r, c, text, attrs)
    except curses.error:
        pass


def draw(stdscr, agents, territory: list[int], trail: list[int],
         episode: int, paused: bool, scores: list[int], w: int, h: int) -> None:
    stdscr.erase()
    _, max_c = stdscr.getmaxyx()

    header = f" Territories  ep={episode}  {'PAUSED' if paused else 'running'}"
    put(stdscr, 0, 0, header.ljust(max_c - 1), curses.A_REVERSE)
    put(stdscr, 1, 0, "+" + "-" * (w * 2) + "+")
    put(stdscr, h + 2, 0, "+" + "-" * (w * 2) + "+")
    for r in range(h):
        put(stdscr, 2 + r, 0, "|")
        put(stdscr, 2 + r, w * 2 + 1, "|")

    heads = {(a.x, a.y): i for i, a in enumerate(agents) if a.is_alive}

    for y in range(h):
        for x in range(w):
            idx = y * w + x
            if (x, y) in heads:
                ci = heads[(x, y)] % len(PLAYER_COLORS)
                put(stdscr, 2 + y, 1 + x * 2, "██",
                    curses.color_pair(CP_BASE + ci * 2) | curses.A_BOLD)
            elif trail[idx] >= 0:
                ci = trail[idx] % len(PLAYER_COLORS)
                put(stdscr, 2 + y, 1 + x * 2, "░░",
                    curses.color_pair(CP_BASE + ci * 2 + 1))
            elif territory[idx] >= 0:
                ci = territory[idx] % len(PLAYER_COLORS)
                put(stdscr, 2 + y, 1 + x * 2, "  ",
                    curses.color_pair(CP_BASE + ci * 2))

    lc = w * 2 + 4
    put(stdscr, 2, lc, "Players:", curses.A_BOLD)
    for i, agent in enumerate(agents):
        ci = i % len(PLAYER_COLORS)
        status = "dead " if not agent.is_alive else "alive"
        put(stdscr, 3 + i, lc, f"  P{i}  score={scores[i]:5d}  {status}",
            curses.color_pair(CP_BASE + ci * 2) | curses.A_BOLD)

    put(stdscr, h + 4, 0, " q quit  space pause  r reset ", curses.A_DIM)
    stdscr.refresh()
