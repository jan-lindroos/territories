import argparse
import sys


def main():
    parser = argparse.ArgumentParser(prog="territories")
    sub = parser.add_subparsers(dest="command")

    show_p = sub.add_parser("show")
    show_p.add_argument("-H", "--height", type=int, default=30)
    show_p.add_argument("-W", "--width", type=int, default=30)
    show_p.add_argument("-n", "--num-agents", type=int, default=6)

    args = parser.parse_args()

    if args.command == "show":
        import curses
        from ui import run
        curses.wrapper(lambda stdscr: run(stdscr, h=args.height, w=args.width, n=args.num_agents))
    else:
        parser.print_help()
        sys.exit(1)
