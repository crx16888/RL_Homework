import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import os

from models import Actor, Critic
from buffer import ReplayBuffer

# PPO算法实现
class PPO:
    def __init__(self, state_dim, action_dim, device='cpu',
                 lr_actor=3e-4, lr_critic=3e-4, gamma=0.99,
                 gae_lambda=0.95, clip_epsilon=0.2, ent_coef=0.01,
                 batch_size=64, n_epochs=10):
        
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.ent_coef = ent_coef
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.device = device
        
    def save_model(self, path):
        """保存模型参数"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'optimizer_actor': self.optimizer_actor.state_dict(),
            'optimizer_critic': self.optimizer_critic.state_dict()
        }, path)
        
    def load_model(self, path):
        """加载模型参数"""
        if not os.path.exists(path):
            print(f"Model file {path} does not exist!")
            return
            
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.optimizer_actor.load_state_dict(checkpoint['optimizer_actor'])
        self.optimizer_critic.load_state_dict(checkpoint['optimizer_critic'])

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            mu, std = self.actor(state)
            dist = Normal(mu, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum()
            value = self.critic(state)
        return action.cpu().numpy(), log_prob.item(), value.item()

    def compute_gae(self, rewards, masks, values):
        returns = []
        gae = 0
        next_value = 0
        
        for step in reversed(range(len(rewards))):
            if step < len(rewards) - 1:
                next_value = values[step + 1]
            delta = rewards[step] + self.gamma * next_value * masks[step] - values[step]
            gae = delta + self.gamma * self.gae_lambda * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns

    def update(self, buffer):
        states = torch.FloatTensor(np.array(buffer.states)).to(self.device)
        actions = torch.FloatTensor(np.array(buffer.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(buffer.log_probs)).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array(buffer.rewards)).unsqueeze(1).to(self.device)
        masks = torch.FloatTensor(np.array(buffer.masks)).unsqueeze(1).to(self.device)
        values = torch.FloatTensor(np.array(buffer.values)).unsqueeze(1).to(self.device)

        # 计算GAE和returns
        returns = self.compute_gae(rewards.cpu().numpy().flatten(),
                                  masks.cpu().numpy().flatten(),
                                  values.cpu().numpy().flatten())
        returns = torch.FloatTensor(returns).unsqueeze(1).to(self.device)
        advantages = returns - values

        # 标准化优势函数
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 创建数据集
        dataset = torch.utils.data.TensorDataset(states, actions, old_log_probs, returns, advantages)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # 记录损失值
        avg_policy_loss = 0
        avg_value_loss = 0
        update_count = 0

        # 多epoch更新
        for _ in range(self.n_epochs):
            for batch in dataloader:
                state_b, action_b, old_log_prob_b, return_b, advantage_b = batch

                # 计算新策略的概率和熵
                mu, std = self.actor(state_b)
                dist = Normal(mu, std)
                new_log_prob = dist.log_prob(action_b).sum(dim=1, keepdim=True)
                entropy = dist.entropy().mean()

                # 计算策略损失
                ratio = (new_log_prob - old_log_prob_b).exp()
                surr1 = ratio * advantage_b
                surr2 = torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon) * advantage_b
                policy_loss = -torch.min(surr1, surr2).mean()

                # 计算价值损失
                current_value = self.critic(state_b)
                value_loss = F.mse_loss(current_value, return_b)

                # 总损失
                loss = policy_loss + 0.5 * value_loss - self.ent_coef * entropy

                # 更新网络
                self.optimizer_actor.zero_grad()
                self.optimizer_critic.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_actor.step()
                self.optimizer_critic.step()
                
                # 累加损失值
                avg_policy_loss += policy_loss.item()
                avg_value_loss += value_loss.item()
                update_count += 1

        buffer.clear()
        
        # 返回平均损失值
        if update_count > 0:
            return avg_policy_loss / update_count, avg_value_loss / update_count
        return None