# Territories

A multi-agent grid-world where players carve out territory by drawing trails and closing loops, à la Splatoon / paper.io. Agents are trained from scratch with PPO and watched in a terminal UI.

## How the game works

- Each agent moves on a grid. Cells inside your enclosed area are your territory.
- Step outside your territory and you leave a trail behind you.
- Close the loop back into your territory and the enclosed region is captured.
- Crossing your own trail, or being hit by another agent while you have a trail out, kills you.

## Install

The environment is a C++ extension built via pybind11.

```bash
pip install -e .
```

Requires Python ≥ 3.9, a C++17 compiler, PyTorch, NumPy, tqdm.

## Watch a trained policy

```bash
territories play
# or
territories play -c src/rl/checkpoints/stage_2_best.pt -H 40 -W 40 -n 10
```

Controls: `space` pause, `r` reset, `q` quit.

## Train

Open `src/rl/train.ipynb`. Training uses PPO (`src/rl/ppo.py`) with a small conv policy (`src/rl/policy.py`) over a curriculum of reward-shaping stages (`src/rl/curricula.py`). Checkpoints land in `src/rl/checkpoints/`.
