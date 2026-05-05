#include "env.h"
#include <algorithm>
#include <queue>

namespace territories {

Env::Env(int num_agents, int width, int height, int window_radius, int seed)
    : rng_(seed),
      num_agents_(num_agents),
      width_(width),
      height_(height),
      window_radius_(window_radius) {
    reset();
}

int Env::cell(int x, int y) const {
    return y * width_ + x;
}

int Env::grid_size() const {
    return width_ * height_;
}

void Env::apply_action(int action, int& x, int& y) {
    switch (action) {
        case 0: y--; break;
        case 1: x++; break;
        case 2: y++; break;
        case 3: x--; break;
    }
}

void Env::kill(std::optional<int> killer, int victim) {
    agents_[victim].is_alive = false;
    if (killer) 
        agents_[*killer].kills++;

    int grid_id = victim + 1;
    for (int i = 0; i < grid_size(); ++i) {
        if (trails_[i] == grid_id) 
            trails_[i] = 0;
        if (territories_[i] == grid_id) 
            territories_[i] = 0;
    }
}

void Env::capture_territory(int agent_idx) {
    if (!agents_[agent_idx].has_trail)
        return;

    int grid_id = agent_idx + 1;

    // Collect trail cells and convert trail to territory
    std::vector<int> trail_cells;
    for (int i = 0; i < grid_size(); ++i) {
        if (trails_[i] == grid_id) {
            trails_[i] = 0;
            territories_[i] = grid_id;
            trail_cells.push_back(i);
        }
    }

    // For each trail cell, try to flood fill non-territory neighbors.
    // If a fill reaches the grid border, it's exterior — discard.
    // If it stays contained, it's interior — claim it.
    std::vector<bool> visited(grid_size(), false);
    for (int tc : trail_cells) visited[tc] = true;

    for (int tc : trail_cells) {
        int tx = tc % width_, ty = tc / width_;
        for (int d = 0; d < 4; ++d) {
            int sx = tx, sy = ty;
            apply_action(d, sx, sy);
            if (sx < 0 || sx >= width_ || sy < 0 || sy >= height_) continue;
            int sc = cell(sx, sy);
            if (visited[sc] || territories_[sc] == grid_id) continue;

            // BFS from this seed
            std::vector<int> filled;
            std::queue<int> q;
            bool hit_border = false;
            visited[sc] = true;
            q.push(sc);
            while (!q.empty()) {
                int c = q.front(); q.pop();
                filled.push_back(c);
                int cx = c % width_, cy = c / width_;
                if (cx == 0 || cx == width_ - 1 || cy == 0 || cy == height_ - 1)
                    hit_border = true;
                for (int dd = 0; dd < 4; ++dd) {
                    int nx = cx, ny = cy;
                    apply_action(dd, nx, ny);
                    if (nx < 0 || nx >= width_ || ny < 0 || ny >= height_) continue;
                    int nc = cell(nx, ny);
                    if (!visited[nc] && territories_[nc] != grid_id) {
                        visited[nc] = true;
                        q.push(nc);
                    }
                }
            }
            if (!hit_border) {
                for (int c : filled) territories_[c] = grid_id;
            }
        }
    }

    // Kill any agent caught inside our new territory
    for (int j = 0; j < num_agents_; ++j) {
        if (j == agent_idx || !agents_[j].is_alive) continue;
        if (territories_[cell(agents_[j].x, agents_[j].y)] == grid_id) {
            kill(agent_idx, j);
        }
    }
}

void Env::update_counts() {
    for (int i = 0; i < num_agents_; ++i) {
        agents_[i].territory = 0;
        agents_[i].has_trail = false;
    }
    for (int i = 0; i < grid_size(); ++i) {
        if (territories_[i] != 0) agents_[territories_[i] - 1].territory++;
        if (trails_[i] != 0) agents_[trails_[i] - 1].has_trail = true;
    }
}

void Env::reset() {
    int side = 2 * window_radius_ + 1;
    territories_.assign(grid_size(), 0);
    trails_.assign(grid_size(), 0);
    obs_.assign(num_agents_ * side * side, 0);
    agents_.assign(num_agents_, Agent{});

    std::uniform_int_distribution<int> dist_x(2, width_ - 3);
    std::uniform_int_distribution<int> dist_y(2, height_ - 3);

    for (int i = 0; i < num_agents_; ++i) {
        agents_[i].x = dist_x(rng_);
        agents_[i].y = dist_y(rng_);
        agents_[i].is_alive = true;

        // 3x3 starting territory
        int id = i + 1;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                territories_[cell(agents_[i].x + dx, agents_[i].y + dy)] = id;
            }
        }
    }
    update_counts();
}

void Env::move_agents(const std::vector<int>& actions) {
    for (int i = 0; i < num_agents_; ++i) {
        if (!agents_[i].is_alive) continue;
        apply_action(std::clamp(actions[i], 0, 3), agents_[i].x, agents_[i].y);
        agents_[i].x = std::clamp(agents_[i].x, 0, width_ - 1);
        agents_[i].y = std::clamp(agents_[i].y, 0, height_ - 1);
        agents_[i].alive_steps++;
    }
}

void Env::resolve_collisions() {
    for (int i = 0; i < num_agents_; ++i) {
        if (!agents_[i].is_alive) continue;
        for (int j = i + 1; j < num_agents_; ++j) {
            if (!agents_[j].is_alive) continue;
            if (agents_[i].x == agents_[j].x && agents_[i].y == agents_[j].y) {
                kill(std::nullopt, i);
                kill(std::nullopt, j);
            }
        }
    }
}

void Env::resolve_trails() {
    for (int i = 0; i < num_agents_; ++i) {
        if (!agents_[i].is_alive) continue;
        int c = cell(agents_[i].x, agents_[i].y);
        int id = i + 1;

        if (trails_[c] == id) {
            kill(std::nullopt, i);
            continue;
        }

        if (trails_[c] != 0 && trails_[c] != id) {
            kill(i, trails_[c] - 1);
        }

        if (territories_[c] == id) {
            capture_territory(i);
        } else {
            trails_[c] = id;
        }
    }
}

void Env::step(const std::vector<int>& actions) {
    move_agents(actions);
    resolve_collisions();
    resolve_trails();
    update_counts();
}

const std::vector<int>& Env::get_obs() {
    int side = 2 * window_radius_ + 1;
    int obs_size = side * side;
    std::fill(obs_.begin(), obs_.end(), EMPTY);

    for (int i = 0; i < num_agents_; ++i) {
        if (!agents_[i].is_alive) 
            continue;

        int grid_id = i + 1;
        int base = i * obs_size;
        int idx = 0;
        for (int dy = -window_radius_; dy <= window_radius_; ++dy) {
            for (int dx = -window_radius_; dx <= window_radius_; ++dx) {
                int wx = agents_[i].x + dx;
                int wy = agents_[i].y + dy;
                if (wx < 0 || wx >= width_ || wy < 0 || wy >= height_) {
                    obs_[base + idx] = WALL;
                } else if (dx == 0 && dy == 0) {
                    obs_[base + idx] = OWN_HEAD;
                } else {
                    int c = cell(wx, wy);
                    bool enemy_head = false;
                    for (int j = 0; j < num_agents_; ++j) {
                        if (j != i && agents_[j].is_alive && agents_[j].x == wx && agents_[j].y == wy) {
                            enemy_head = true;
                            break;
                        }
                    }
                    if (enemy_head)                      
                        obs_[base + idx] = ENEMY_HEAD;
                    else if (territories_[c] == grid_id)      
                        obs_[base + idx] = OWN_TERRITORY;
                    else if (trails_[c] == grid_id)           
                        obs_[base + idx] = OWN_TRAIL;
                    else if (territories_[c] != 0)       
                        obs_[base + idx] = ENEMY_TERRITORY;
                    else if (trails_[c] != 0)            
                        obs_[base + idx] = ENEMY_TRAIL;
                }
                ++idx;
            }
        }
    }
    return obs_;
}

const std::vector<Agent>& Env::get_agents() const {
    return agents_;
}

const std::vector<int>& Env::get_territories() const {
    return territories_;
}

const std::vector<int>& Env::get_trails() const {
    return trails_;
}

}
