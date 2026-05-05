#pragma once
#include <optional>
#include <random>
#include <vector>

namespace territories {

enum Cell : int {
    EMPTY            = 0,
    OWN_HEAD         = 1,
    OWN_TERRITORY    = 2,
    OWN_TRAIL        = 3,
    ENEMY_HEAD       = 4,
    ENEMY_TERRITORY  = 5,
    ENEMY_TRAIL      = 6,
    WALL             = 7,
};

struct Agent {
    int x = 0, y = 0;
    bool is_alive = false;
    int alive_steps = 0;
    int kills = 0;
    int territory = 0;
    bool has_trail = false;
};

class Env {
public:
    Env(int num_agents, int width, int height, int window_radius, int seed);
    void reset();
    void step(const std::vector<int>& actions);
    const std::vector<int>& get_obs();
    const std::vector<Agent>& get_agents() const;
    const std::vector<int>& get_territories() const;
    const std::vector<int>& get_trails() const;

private:
    void move_agents(const std::vector<int>& actions);
    void resolve_collisions();
    void resolve_trails();
    void kill(std::optional<int> killer, int victim);
    void capture_territory(int agent_idx);
    void update_counts();
    int cell(int x, int y) const;
    int grid_size() const;
    static void apply_action(int action, int& x, int& y);

    std::mt19937 rng_;
    const int num_agents_, width_, height_, window_radius_;
    std::vector<int> territories_, trails_, obs_;
    std::vector<Agent> agents_;
};

}
