import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
import numpy as np
import os

from model import Actor, Critic
from buffer import ReplayBuffer

# TRPO算法实现
class TRPO:
    def __init__(self, state_dim, action_dim, device='cpu',
                 lr_critic=1e-4, gamma=0.99,  # 降低学习率以提高稳定性
                 gae_lambda=0.97, max_kl=0.008, damping=0.1, ent_coef=0.02,  # 调整GAE参数和KL约束
                 batch_size=128, n_epochs=5):  # 增大批量，减少更新次数
        
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.max_kl = max_kl  # KL散度约束
        self.damping = damping  # 共轭梯度法的阻尼系数
        self.ent_coef = ent_coef
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.device = device
        
    def save_model(self, path):
        """保存模型参数"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
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

    def flat_grad(self, grads):
        flat_grads = []
        for grad in grads:
            flat_grads.append(grad.view(-1))
        return torch.cat(flat_grads)

    def get_kl(self, states):
        # 计算当前策略与旧策略之间的KL散度
        mu, std = self.actor(states)
        mu_old, std_old = mu.detach(), std.detach()
        
        dist = Normal(mu, std)
        dist_old = Normal(mu_old, std_old)
        
        kl = kl_divergence(dist_old, dist).sum(1, keepdim=True).mean()
        return kl

    def conjugate_gradient(self, states, b, nsteps=10, residual_tol=1e-10):
        # 使用共轭梯度法求解Ax=b，其中A是Fisher信息矩阵
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rdotr = torch.dot(r, r)
        
        for i in range(nsteps):
            Ap = self.fisher_vector_product(states, p)
            alpha = rdotr / (torch.dot(p, Ap) + 1e-8)
            x += alpha * p
            r -= alpha * Ap
            new_rdotr = torch.dot(r, r)
            if new_rdotr < residual_tol:
                break
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
        return x

    def fisher_vector_product(self, states, v):
        # 计算Fisher信息矩阵与向量v的乘积
        kl = self.get_kl(states)
        grads = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True)
        flat_grad_kl = self.flat_grad(grads)
        
        kl_v = (flat_grad_kl * v).sum()
        grads = torch.autograd.grad(kl_v, self.actor.parameters())
        flat_grad_grad_kl = self.flat_grad(grads)
        
        return flat_grad_grad_kl + self.damping * v

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

        # 改进的优势函数标准化
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # 标准化回报

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

                # 更新价值网络
                current_value = self.critic(state_b)
                value_loss = F.mse_loss(current_value, return_b)
                
                self.optimizer_critic.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()
                
                # 计算策略梯度
                mu, std = self.actor(state_b)
                dist = Normal(mu, std)
                log_probs = dist.log_prob(action_b).sum(dim=1, keepdim=True)
                entropy = dist.entropy().mean()
                
                # 改进的策略目标函数
                ratio = (log_probs - old_log_prob_b).exp()
                surrogate = (ratio * advantage_b).mean() + self.ent_coef * entropy  # 添加熵正则化
                
                # 计算策略梯度
                grads = torch.autograd.grad(surrogate, self.actor.parameters())
                policy_gradient = self.flat_grad(grads)
                
                # 使用共轭梯度法计算搜索方向
                search_dir = self.conjugate_gradient(state_b, policy_gradient.detach())
                
                # 计算步长
                gHg = (self.fisher_vector_product(state_b, search_dir) * search_dir).sum(0, keepdim=True)
                step_size = torch.sqrt(2 * self.max_kl / (gHg + 1e-8))
                
                # 更新策略参数
                params = self.flat_grad([param for param in self.actor.parameters()])
                new_params = params + step_size * search_dir
                
                # 将新参数应用到策略网络
                index = 0
                for param in self.actor.parameters():
                    param_size = param.numel()
                    param.data.copy_(new_params[index:index + param_size].view(param.size()))
                    index += param_size
                
                # 累加损失值
                avg_policy_loss += -surrogate.item()
                avg_value_loss += value_loss.item()
                update_count += 1

        buffer.clear()
        
        # 返回平均损失值
        if update_count > 0:
            return avg_policy_loss / update_count, avg_value_loss / update_count
        return None