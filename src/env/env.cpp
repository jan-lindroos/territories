#include "env.h"

#include <queue>

namespace territories {

constexpr int kDx[] = {0, 1, 0, -1};
constexpr int kDy[] = {-1, 0, 1, 0};

Env::Env(int num_agents, int width, int height, int window_radius, int seed)
    : num_agents_(num_agents),
      width_(width),
      height_(height),
      window_radius_(window_radius),
      seed_(seed) {
  Reset();
}

void Env::Reset() {
  int size = width_ * height_;
  territory_grid_.assign(size, -1);
  trail_grid_.assign(size, -1);
  snakes_.clear();
  rng_.seed(seed_ >= 0 ? seed_ : std::random_device{}());
  SpawnSnakes();
}

void Env::SpawnSnakes() {
  std::uniform_int_distribution<int> x_dist(2, width_ - 3);
  std::uniform_int_distribution<int> y_dist(1, height_ - 2);
  for (int i = 0; i < num_agents_; i++) {
    int x, y;
    do {
      x = x_dist(rng_);
      y = y_dist(rng_);
    } while (territory_grid_[y * width_ + x] != -1);
    snakes_.push_back({x, y, true});
    for (int dx = -1; dx <= 1; dx++) {
      territory_grid_[y * width_ + x + dx] = i;
    }
  }
}

void Env::KillSnake(int id) {
  snakes_[id].alive = false;
  int size = width_ * height_;
  for (int i = 0; i < size; i++) {
    if (territory_grid_[i] == id) territory_grid_[i] = -1;
    if (trail_grid_[i] == id) trail_grid_[i] = -1;
  }
}

void Env::ClaimTerritory(int id) {
  int size = width_ * height_;
  for (int i = 0; i < size; i++) {
    if (trail_grid_[i] == id) {
      trail_grid_[i] = -1;
      territory_grid_[i] = id;
    }
  }

  // Flood fill from borders to find cells NOT enclosed by this snake.
  std::vector<bool> reachable(size, false);
  std::queue<int> q;
  for (int x = 0; x < width_; x++) {
    for (int y : {0, height_ - 1}) {
      int idx = y * width_ + x;
      if (territory_grid_[idx] != id && !reachable[idx]) {
        reachable[idx] = true;
        q.push(idx);
      }
    }
  }
  for (int y = 1; y < height_ - 1; y++) {
    for (int x : {0, width_ - 1}) {
      int idx = y * width_ + x;
      if (territory_grid_[idx] != id && !reachable[idx]) {
        reachable[idx] = true;
        q.push(idx);
      }
    }
  }
  while (!q.empty()) {
    int idx = q.front();
    q.pop();
    int x = idx % width_, y = idx / width_;
    for (int d = 0; d < 4; d++) {
      int nx = x + kDx[d], ny = y + kDy[d];
      if (nx < 0 || nx >= width_ || ny < 0 || ny >= height_) continue;
      int nidx = ny * width_ + nx;
      if (!reachable[nidx] && territory_grid_[nidx] != id) {
        reachable[nidx] = true;
        q.push(nidx);
      }
    }
  }

  for (int i = 0; i < size; i++) {
    if (!reachable[i]) territory_grid_[i] = id;
  }
}

std::vector<int> Env::Step(const std::vector<int>& actions) {
  int size = width_ * height_;

  // Snapshot territory counts for scoring.
  std::vector<int> old_counts(num_agents_, 0);
  for (int cell : territory_grid_) {
    if (cell >= 0) old_counts[cell]++;
  }

  // Move.
  for (int i = 0; i < num_agents_; i++) {
    if (!snakes_[i].alive) continue;
    snakes_[i].x += kDx[actions[i]];
    snakes_[i].y += kDy[actions[i]];
  }

  // Wall deaths.
  for (int i = 0; i < num_agents_; i++) {
    if (!snakes_[i].alive) continue;
    if (snakes_[i].x <= 0 || snakes_[i].x >= width_ - 1 ||
        snakes_[i].y <= 0 || snakes_[i].y >= height_ - 1) {
      KillSnake(i);
    }
  }

  // Trail collisions: stepping on a trail kills the trail owner.
  for (int i = 0; i < num_agents_; i++) {
    if (!snakes_[i].alive) continue;
    int trail_owner = trail_grid_[snakes_[i].y * width_ + snakes_[i].x];
    if (trail_owner >= 0) KillSnake(trail_owner);
  }

  // Head-on collisions.
  for (int i = 0; i < num_agents_; i++) {
    if (!snakes_[i].alive) continue;
    for (int j = i + 1; j < num_agents_; j++) {
      if (!snakes_[j].alive) continue;
      if (snakes_[i].x == snakes_[j].x && snakes_[i].y == snakes_[j].y) {
        KillSnake(i);
        KillSnake(j);
      }
    }
  }

  // Territory claims and trail placement.
  for (int i = 0; i < num_agents_; i++) {
    if (!snakes_[i].alive) continue;
    int idx = snakes_[i].y * width_ + snakes_[i].x;
    bool has_trail = false;
    for (int j = 0; j < size && !has_trail; j++) {
      has_trail = (trail_grid_[j] == i);
    }
    if (territory_grid_[idx] == i && has_trail) {
      ClaimTerritory(i);
    } else if (territory_grid_[idx] != i) {
      trail_grid_[idx] = i;
    }
  }

  // Score = net territory change.
  std::vector<int> scores(num_agents_, 0);
  for (int cell : territory_grid_) {
    if (cell >= 0) scores[cell]++;
  }
  for (int i = 0; i < num_agents_; i++) {
    scores[i] -= old_counts[i];
  }
  return scores;
}

std::vector<std::vector<int>> Env::GetObservations() {
  int side = 2 * window_radius_ + 1;
  std::vector<std::vector<int>> obs(
      num_agents_, std::vector<int>(side * side, 0));

  for (int i = 0; i < num_agents_; i++) {
    if (!snakes_[i].alive) continue;
    for (int dy = -window_radius_; dy <= window_radius_; dy++) {
      for (int dx = -window_radius_; dx <= window_radius_; dx++) {
        int wx = snakes_[i].x + dx, wy = snakes_[i].y + dy;
        int oi = (dy + window_radius_) * side + (dx + window_radius_);

        if (wx <= 0 || wx >= width_ - 1 || wy <= 0 || wy >= height_ - 1) {
          obs[i][oi] = -4;
          continue;
        }

        int gi = wy * width_ + wx;

        // Check heads first.
        bool head = false;
        for (int j = 0; j < num_agents_; j++) {
          if (snakes_[j].alive && snakes_[j].x == wx && snakes_[j].y == wy) {
            obs[i][oi] = (j == i) ? -3 : 3;
            head = true;
            break;
          }
        }
        if (head) continue;

        if (trail_grid_[gi] >= 0) {
          obs[i][oi] = (trail_grid_[gi] == i) ? -2 : 2;
        } else if (territory_grid_[gi] >= 0) {
          obs[i][oi] = (territory_grid_[gi] == i) ? -1 : 1;
        }
      }
    }
  }
  return obs;
}

}  // namespace territories
