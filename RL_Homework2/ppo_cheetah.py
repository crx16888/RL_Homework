import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import os
from collections import deque
from torch.utils.tensorboard.writer import SummaryWriter
import matplotlib.pyplot as plt
from datetime import datetime

# 神经网络定义
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))  # 可学习的对数标准差

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.mu_head(x))  # 输出在[-1,1]之间
        return mu, self.log_std.exp()

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.value_head(x)

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

# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.masks = []
        self.log_probs = []
        self.values = []

    def add(self, state, action, reward, mask, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.masks.append(mask)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.masks = []
        self.log_probs = []
        self.values = []

# 训练函数
def train(model_save_path=r'C:\Users\95718\Desktop\vscode\Program\RL_Homework\RL_Homework2\models', save_interval=10, max_episodes=100):
    # 训练时不渲染
    env = gym.make('HalfCheetah-v4')
    # 具体奖励如下，如果要修改可以使用Wrapper包装器来修改
    # reward = forward_reward - ctrl_cost
    # forward_reward = x_velocity  # 奖励x方向上的速度
    # ctrl_cost = 0.1 * np.sum(np.square(action))  # 动作惩罚项，防止智能体使用过大的动作；action也可以理解为力矩，力矩影响速度
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # 创建模型保存目录
    import os
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    
    # 设置TensorBoard和图表保存路径
    log_dir = os.path.join(os.path.dirname(model_save_path), 'runs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)
    
    # 创建图表
    plt.style.use('seaborn')
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle('Training Progress')
    
    # 初始化数据列表
    episodes = []
    rewards = []
    policy_losses = []
    value_losses = []
    
    ppo = PPO(state_dim, action_dim, device=device)
    buffer = ReplayBuffer()
    
    update_interval = 2048
    total_steps = 0
    episode = 0
    best_reward = -float('inf')
    
    # 记录训练指标
    episode_rewards = []
    policy_losses = []
    value_losses = []
    
    while episode < max_episodes: #训练轮数
        state, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        episode_policy_losses = []
        episode_value_losses = []
        episode_steps = 0
        
        # 采集数据
        while not (done or truncated) and episode_steps < 1000:
            action, log_prob, value = ppo.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            mask = 0.0 if done else 1.0
            
            buffer.add(state, action, reward, mask, log_prob, value)
            state = next_state
            episode_reward += reward
            total_steps += 1
            episode_steps += 1
            
            if total_steps % update_interval == 0:
                # 更新时记录损失值
                losses = ppo.update(buffer)
                if losses:
                    policy_loss, value_loss = losses
                    episode_policy_losses.append(policy_loss)
                    episode_value_losses.append(value_loss)
                    writer.add_scalar("Loss/Policy", policy_loss, total_steps)
                    writer.add_scalar("Loss/Value", value_loss, total_steps)
        
        # 记录训练数据
        episode_rewards.append(episode_reward)
        writer.add_scalar("Reward/Episode", episode_reward, episode)
        writer.add_scalar("Steps/Episode", total_steps, episode)
        
        # 计算平均损失值并记录
        if episode_policy_losses:
            avg_policy_loss = sum(episode_policy_losses) / len(episode_policy_losses)
            policy_losses.append(avg_policy_loss)
            writer.add_scalar("Loss/PolicyAvg", avg_policy_loss, episode)
        
        if episode_value_losses:
            avg_value_loss = sum(episode_value_losses) / len(episode_value_losses)
            value_losses.append(avg_value_loss)
            writer.add_scalar("Loss/ValueAvg", avg_value_loss, episode)
        
        print(f"Episode {episode} | Reward: {episode_reward:.1f} | Steps: {episode_steps}")
        
        # 更新图表
        episodes.append(episode)
        rewards.append(episode_reward)
        if episode_policy_losses:
            policy_losses.append(sum(episode_policy_losses) / len(episode_policy_losses))
        if episode_value_losses:
            value_losses.append(sum(episode_value_losses) / len(episode_value_losses))
            
        # 绘制图表
        ax1.clear()
        ax2.clear()
        ax3.clear()
        
        ax1.plot(episodes, rewards, 'b-')
        ax1.set_title('Episode Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        
        if policy_losses:
            ax2.plot(episodes[-len(policy_losses):], policy_losses, 'r-')
            ax2.set_title('Policy Loss')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Loss')
        
        if value_losses:
            ax3.plot(episodes[-len(value_losses):], value_losses, 'g-')
            ax3.set_title('Value Loss')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Loss')
        
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, 'training_progress.png'))
        
        # 保存模型
        if episode % save_interval == 0:
            ppo.save_model(f"{model_save_path}/ppo_cheetah_episode_{episode}.pt")
            print(f"Model saved at episode {episode}")
        
        # 保存最佳模型
        if episode_reward > best_reward:
            best_reward = episode_reward
            ppo.save_model(f"{model_save_path}/ppo_cheetah_best.pt")
            print(f"Best model saved with reward {best_reward:.1f}")
        
        episode += 1
    
    # 保存最终模型
    ppo.save_model(f"{model_save_path}/ppo_cheetah_final.pt")
    print("Final model saved")
    
    env.close()
    writer.close()

# 测试函数
def test(model_path, episodes=5):
    # 测试时渲染
    env = gym.make('HalfCheetah-v4', render_mode='human')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    ppo = PPO(state_dim, action_dim, device=device)
    ppo.load_model(model_path)
    
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            action, _, _ = ppo.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            state = next_state
            episode_reward += reward
            env.render()
        
        print(f"Test Episode {episode} | Reward: {episode_reward:.1f}")
    
    env.close()

# 主函数
def main():
    # import argparse
    # parser = argparse.ArgumentParser(description='PPO HalfCheetah')
    # parser.add_argument('--mode', type=str, default='train', help='train or test')
    # parser.add_argument('--model_path', type=str, default='models/ppo_cheetah_best.pt', help='path to model for testing')
    # parser.add_argument('--save_interval', type=int, default=10, help='save model every n episodes')
    # args = parser.parse_args()
    
    # if args.mode == 'train':
    #     train(save_interval=args.save_interval, max_steps=1000)
    # elif args.mode == 'test':
    #     test(args.model_path,episodes=5)
    # else:
    #     print("Invalid mode. Use 'train' or 'test'")
    train(save_interval=100, max_episodes=1000)
    # test(model_path=r'C:\Users\95718\Desktop\vscode\Program\RL_Homework\RL_Homework2\models\ppo_cheetah_best.pt',episodes=5)

if __name__ == "__main__":
    main()