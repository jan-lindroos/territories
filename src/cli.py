import sys


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "show":
        import curses
        from ui import run
        curses.wrapper(run)
    else:
        print("Usage: territories show")
        sys.exit(1)
