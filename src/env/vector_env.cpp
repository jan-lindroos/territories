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

std::vector<int> VectorEnv::GetObservationsFlat() {
  int obs_size = (2 * window_radius_ + 1) * (2 * window_radius_ + 1);
  std::vector<int> flat(num_envs_ * num_agents_ * obs_size);

  std::vector<std::thread> threads;
  threads.reserve(num_envs_);
  for (int i = 0; i < num_envs_; i++) {
    threads.emplace_back([this, &flat, obs_size, i]() {
      auto obs = envs_[i].GetObservations();
      int base = i * num_agents_ * obs_size;
      for (int a = 0; a < num_agents_; a++) {
        std::copy(obs[a].begin(), obs[a].end(),
                  flat.begin() + base + a * obs_size);
      }
    });
  }
  for (auto& t : threads) t.join();
  return flat;
}

std::vector<bool> VectorEnv::GetAlive() {
  std::vector<bool> alive(num_envs_ * num_agents_);
  for (int i = 0; i < num_envs_; i++) {
    for (int a = 0; a < num_agents_; a++) {
      alive[i * num_agents_ + a] = envs_[i].snakes_[a].alive;
    }
  }
  return alive;
}

std::vector<int> VectorEnv::GetTrailCounts() {
  std::vector<int> counts(num_envs_ * num_agents_, 0);
  for (int i = 0; i < num_envs_; i++) {
    int size = width_ * height_;
    for (int j = 0; j < size; j++) {
      int owner = envs_[i].trail_grid_[j];
      if (owner >= 0) counts[i * num_agents_ + owner]++;
    }
  }
  return counts;
}

std::vector<int> VectorEnv::GetTerritoryCounts() {
  std::vector<int> counts(num_envs_ * num_agents_, 0);
  for (int i = 0; i < num_envs_; i++) {
    int size = width_ * height_;
    for (int j = 0; j < size; j++) {
      int owner = envs_[i].territory_grid_[j];
      if (owner >= 0) counts[i * num_agents_ + owner]++;
    }
  }
  return counts;
}

}  // namespace territories
