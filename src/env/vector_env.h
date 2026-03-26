#pragma once
#include <vector>

#include "env.h"

namespace territories {

class VectorEnv {
 public:
  const int num_envs_, num_agents_, width_, height_, window_radius_;

  VectorEnv(int num_envs, int num_agents, int width, int height,
            int window_radius, int seed = -1);

  void Reset();
  std::vector<std::vector<int>> Step(
      const std::vector<std::vector<int>>& actions);
  std::vector<std::vector<std::vector<int>>> GetObservations();

  Env& GetEnv(int i) { return envs_[i]; }

 private:
  std::vector<Env> envs_;
};

}  // namespace territories
