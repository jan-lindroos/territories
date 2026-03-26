#pragma once
#include <random>
#include <vector>

namespace territories {

struct Snake {
  int x, y;
  bool alive;
};

class Env {
 public:
  const int num_agents_, width_, height_, window_radius_;
  int seed_;
  std::vector<Snake> snakes_;
  std::vector<int> territory_grid_;  // -1 = empty, else agent_id
  std::vector<int> trail_grid_;      // -1 = empty, else agent_id
  std::mt19937 rng_;

  Env(int num_agents, int width, int height, int window_radius, int seed = -1);

  void Reset();
  std::vector<int> Step(const std::vector<int>& actions);
  std::vector<std::vector<int>> GetObservations();

 private:
  void SpawnSnakes();
  void KillSnake(int id);
  void ClaimTerritory(int id);
};

}  // namespace territories
