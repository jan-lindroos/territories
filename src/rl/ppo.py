import torch
from torch import nn
import numpy as np

from rl.curricula import Curriculum
from rl.policy import Policy


def train_epoch(
    c: Curriculum,
    policy: Policy,
    optimizer: torch.optim.Optimizer,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    update_epochs: int = 4,
    num_minibatches: int = 32,
    clip_coef: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    device: str = "cpu"
):
    buffer_shape = (c.max_steps, c.num_envs, c.num_agents)
    obs_shape = c.get_obs_shape()

    obs = torch.zeros(buffer_shape + obs_shape).to(device)
    actions = torch.zeros(buffer_shape).to(device)
    logprobs = torch.zeros(buffer_shape).to(device)
    rewards = torch.zeros(buffer_shape).to(device)
    dones = torch.zeros(buffer_shape).to(device)
    values = torch.zeros(buffer_shape).to(device)

    for i in range(c.max_steps):
        obs[i] = c.get_obs().to(device)
        obs_batch = obs[i].reshape(-1, *obs_shape)
        with torch.no_grad():
            action, logprob, _, value = policy.get_action_and_value(obs_batch)

        actions[i] = action.reshape(c.num_envs, c.num_agents)
        logprobs[i] = logprob.reshape(c.num_envs, c.num_agents)
        values[i] = value.reshape(c.num_envs, c.num_agents)

        rewards[i] = c.step(actions[i].cpu()).to(device)
        dones[i] = (~c.get_alive()).to(device)

    # Bootstrap from final state before reset
    with torch.no_grad():
        next_values = policy.get_value(c.get_obs().to(device).reshape(-1, *obs_shape))
        next_values = next_values.reshape(c.num_envs, c.num_agents)
    c.reset()

    advantages = torch.zeros_like(rewards).to(device)
    lastgaelam = 0
    for t in reversed(range(c.max_steps)):
        if t != c.max_steps - 1:
            next_values = values[t + 1]
        delta = rewards[t] + gamma * next_values * (1 - dones[t]) - values[t]
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * (1 - dones[t]) * lastgaelam
    returns = advantages + values

    b_obs = obs.reshape((-1, *obs_shape))
    b_logprobs = logprobs.reshape(-1)
    b_actions = actions.reshape(-1)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)

    batch_size = c.max_steps * c.num_envs * c.num_agents
    minibatch_size = int(batch_size / num_minibatches)
    b_idx = np.arange(batch_size)
    for _ in range(update_epochs):
        np.random.shuffle(b_idx)
        for start in range(0, batch_size, minibatch_size):
            mb_idx = b_idx[start:start + minibatch_size]

            _, newlogprob, entropy, newvalue = policy.get_action_and_value(b_obs[mb_idx], b_actions.long()[mb_idx])
            newvalue = newvalue.view(-1)
            logratio = newlogprob - b_logprobs[mb_idx]
            ratio = logratio.exp()

            mb_advantages = b_advantages[mb_idx]
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            v_loss = 0.5 * ((newvalue - b_returns[mb_idx]) ** 2).mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

    y_pred, y_true = b_returns.cpu().numpy(), values.reshape(-1).cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

    return {
        "policy_loss": pg_loss.item(),
        "value_loss": v_loss.item(),
        "entropy": entropy_loss.item(),
        "mean_reward": rewards.sum(dim=0).mean().item(),
        "explained_variance": explained_var,
    }
 