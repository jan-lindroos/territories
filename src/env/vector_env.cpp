#include "vector_env.h"

#include <thread>

namespace territories {

VectorEnv::VectorEnv(int num_envs, int num_agents, int width, int height,
                     int window_radius, int seed)
    : num_envs_(num_envs),
      num_agents_(num_agents),
      width_(width),
      height_(height),
      window_radius_(window_radius) {
  envs_.reserve(num_envs);
  for (int i = 0; i < num_envs; i++) {
    envs_.emplace_back(num_agents, width, height, window_radius,
                       seed >= 0 ? seed + i : -1);
  }
}

void VectorEnv::Reset() {
  std::vector<std::thread> threads;
  threads.reserve(num_envs_);
  for (auto& env : envs_) {
    threads.emplace_back([&env]() { env.Reset(); });
  }
  for (auto& t : threads) t.join();
}

std::vector<std::vector<int>> VectorEnv::Step(
    const std::vector<std::vector<int>>& actions) {
  std::vector<std::vector<int>> scores(num_envs_);
  std::vector<std::thread> threads;
  threads.reserve(num_envs_);
  for (int i = 0; i < num_envs_; i++) {
    threads.emplace_back([this, &actions, &scores, i]() {
      scores[i] = envs_[i].Step(actions[i]);
    });
  }
  for (auto& t : threads) t.join();
  return scores;
}

std::vector<std::vector<std::vector<int>>> VectorEnv::GetObservations() {
  std::vector<std::vector<std::vector<int>>> obs(num_envs_);
  std::vector<std::thread> threads;
  threads.reserve(num_envs_);
  for (int i = 0; i < num_envs_; i++) {
    threads.emplace_back([this, &obs, i]() {
      obs[i] = envs_[i].GetObservations();
    });
  }
  for (auto& t : threads) t.join();
  return obs;
}

}  // namespace territories
