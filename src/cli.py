import argparse
import sys
from pathlib import Path

_DEFAULT_CHECKPOINT = Path(__file__).with_name("rl") / "checkpoints" / "stage_2_best.pt"


def main():
    parser = argparse.ArgumentParser(prog="territories")
    sub = parser.add_subparsers(dest="command")

    play_p = sub.add_parser("play", help="watch a trained checkpoint play")
    play_p.add_argument(
        "-c", "--checkpoint", type=Path, default=_DEFAULT_CHECKPOINT,
        metavar="PATH",
        help="path to .pt checkpoint (default: rl/checkpoints/stage_2_best.pt)",
    )
    play_p.add_argument("-H", "--height", type=int, default=40)
    play_p.add_argument("-W", "--width", type=int, default=40)
    play_p.add_argument("-n", "--num-agents", type=int, default=10)

    args = parser.parse_args()

    if args.command == "play":
        import curses
        from game import run
        curses.wrapper(
            lambda stdscr: run(
                stdscr,
                checkpoint=args.checkpoint,
                w=args.width,
                h=args.height,
                n=args.num_agents,
            )
        )
    else:
        parser.print_help()
        sys.exit(1)
