import math
import time

import torch
import torch.nn as nn
from torch.distributions import Categorical

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def _layer_init(layer, std=math.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Policy(nn.Module):
    def __init__(self, num_channels, grid_side, num_actions):
        super().__init__()
        self.conv = nn.Sequential(
            _layer_init(nn.Conv2d(num_channels, 32, 3, padding=1)), nn.ReLU(),
            _layer_init(nn.Conv2d(32, 64, 3, padding=1)), nn.ReLU(),
            _layer_init(nn.Conv2d(64, 64, 3, stride=2, padding=1)), nn.ReLU(),
            nn.Flatten(),
        )
        flat_dim = 64 * ((grid_side + 1) // 2) ** 2
        self.shared = nn.Sequential(
            _layer_init(nn.Linear(flat_dim, 128)), nn.ReLU(),
        )
        self.actor = _layer_init(nn.Linear(128, num_actions), std=0.1)
        self.critic = _layer_init(nn.Linear(128, 1), std=1.0)

    def forward(self, obs):
        h = self.shared(self.conv(obs))  # [B, 128]
        return self.actor(h), self.critic(h).squeeze(-1)

    def act(self, obs):
        logits, value = self(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value

    def evaluate(self, obs, actions):
        logits, value = self(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value


class RolloutBuffer:
    """Stores one rollout of [T, E, A, ...] tensors and computes GAE."""

    def __init__(self):
        self.obs, self.actions, self.log_probs = [], [], []
        self.values, self.rewards, self.dones, self.masks = [], [], [], []

    def store(self, obs, actions, log_probs, values, rewards, dones, action_mask):
        self.obs.append(obs)
        self.actions.append(actions)
        self.log_probs.append(log_probs)
        self.values.append(values)
        self.rewards.append(rewards)
        self.dones.append(dones)
        self.masks.append(action_mask)

    def clear(self):
        self.__init__()

    def get(self, last_values, gamma=0.99, gae_lambda=0.95):
        T = len(self.obs)
        values = torch.stack(self.values)       # [T, E, A]
        rewards = torch.stack(self.rewards)     # [T, E, A]
        dones = torch.stack(self.dones)         # [T, E, A]
        masks = torch.stack(self.masks)         # [T, E, A]

        advantages = torch.zeros_like(rewards)
        gae = 0
        for t in reversed(range(T)):
            next_val = values[t + 1] if t + 1 < T else last_values
            not_done = (~dones[t]).float()
            delta = rewards[t] + gamma * next_val * not_done - values[t]
            gae = delta + gamma * gae_lambda * not_done * gae
            advantages[t] = gae

        returns = advantages + values

        # Flatten to only transitions where the agent was alive and acting
        valid = masks.bool()
        flat_obs = torch.stack(self.obs)[valid].detach()
        flat_act = torch.stack(self.actions)[valid].detach()
        flat_lp = torch.stack(self.log_probs)[valid].detach()
        flat_ret = returns[valid].detach()
        flat_adv = advantages[valid].detach()

        return flat_obs, flat_act, flat_lp, flat_ret, flat_adv


class PPOTrainer:

    def __init__(self, num_channels, grid_side, num_actions, lr=2.5e-4, clip_eps=0.2,
                 entropy_coef=0.05, value_coef=0.5, epochs=8, batch_size=256,
                 clip_vloss=True, anneal_lr=True, num_iterations=1000):
        self.device = DEVICE
        self.policy = Policy(num_channels, grid_side, num_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)
        self.lr = lr
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.epochs = epochs
        self.batch_size = batch_size
        self.clip_vloss = clip_vloss
        self.anneal_lr = anneal_lr
        self.num_iterations = num_iterations
        self.iteration = 0

    def update(self, buffer, last_values, gamma=0.99, gae_lambda=0.95):
        self.iteration += 1

        # Anneal learning rate
        if self.anneal_lr:
            frac = 1.0 - (self.iteration - 1.0) / self.num_iterations
            self.optimizer.param_groups[0]["lr"] = frac * self.lr

        obs, actions, old_lp, returns, advantages = buffer.get(last_values, gamma, gae_lambda)
        if obs.numel() == 0:
            return 0.0

        obs, actions = obs.to(self.device), actions.to(self.device)
        old_lp, returns, advantages = old_lp.to(self.device), returns.to(self.device), advantages.to(self.device)

        # Store old values for value clipping
        with torch.no_grad():
            _, old_values = self.policy(obs)

        total_loss, n = 0.0, 0
        for _ in range(self.epochs):
            for idx in torch.randperm(len(obs)).split(self.batch_size):
                new_lp, entropy, vals = self.policy.evaluate(obs[idx], actions[idx])
                logratio = new_lp - old_lp[idx]
                ratio = logratio.exp()

                # Per-minibatch advantage normalization
                adv = advantages[idx]
                if adv.numel() > 1:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                # Policy loss
                surr1 = -adv * ratio
                surr2 = -adv * ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps)
                policy_loss = torch.max(surr1, surr2).mean()

                # Value loss
                if self.clip_vloss:
                    v_loss_unclipped = (vals - returns[idx]) ** 2
                    v_clipped = old_values[idx] + torch.clamp(
                        vals - old_values[idx], -self.clip_eps, self.clip_eps
                    )
                    v_loss_clipped = (v_clipped - returns[idx]) ** 2
                    value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    value_loss = 0.5 * (vals - returns[idx]).pow(2).mean()

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.optimizer.step()

                total_loss += loss.item()
                n += 1

        buffer.clear()
        return total_loss / max(n, 1)

    def collect_rollout(self, curriculum, obs, rollout_steps=128):
        buffer = RolloutBuffer()
        E, A = curriculum.num_envs, curriculum.num_agents
        t0 = time.perf_counter()

        self.policy.eval()
        with torch.no_grad():
            for _ in range(rollout_steps):
                flat_obs = obs.view(E * A, *obs.shape[2:]).to(self.device)
                actions, log_probs, values = self.policy.act(flat_obs)
                actions = actions.cpu().view(E, A)
                log_probs = log_probs.cpu().view(E, A)
                values = values.cpu().view(E, A)

                next_obs, rewards, dones, action_mask = curriculum.step(actions)
                buffer.store(obs, actions, log_probs, values, rewards, dones, action_mask)
                obs = next_obs

            # Bootstrap value for GAE at rollout boundary
            flat_obs = obs.view(E * A, *obs.shape[2:]).to(self.device)
            _, last_values = self.policy(flat_obs)
            last_values = last_values.cpu().view(E, A)
        self.policy.train()

        elapsed = time.perf_counter() - t0
        total_steps = rollout_steps * E * A
        fps = total_steps / elapsed

        all_rewards = torch.stack(buffer.rewards)   # [T, E, A]
        all_masks = torch.stack(buffer.masks)        # [T, E, A]
        mean_reward = all_rewards[all_masks].mean().item() if all_masks.any() else 0.0

        return buffer, obs, mean_reward, last_values, fps

    def save(self, path):
        torch.save({
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "iteration": self.iteration,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, weights_only=True)
        self.policy.load_state_dict(checkpoint["policy"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.iteration = checkpoint.get("iteration", 0)
