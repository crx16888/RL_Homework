import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard.writer import SummaryWriter
import time

from env import CheetahEnv
from ppo import PPO
from trpo import TRPO
from buffer import ReplayBuffer

# 训练函数
def train(algorithm='ppo', model_save_path='models', save_interval=100, max_episodes=1000):
    # 创建环境
    env = CheetahEnv()
    device = env.get_device()
    
    # 创建模型保存目录
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    
    # 设置TensorBoard和图表保存路径
    log_dir = os.path.join(os.path.dirname(model_save_path), f'runs_{algorithm}')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)
    
    # 创建图表
    plt.style.use('seaborn')
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle(f'{algorithm.upper()} Training Progress')
    
    # 初始化数据列表
    episodes = []
    rewards = []
    policy_losses = []
    value_losses = []
    
    # 初始化算法和缓冲区
    if algorithm.lower() == 'ppo':
        agent = PPO(env.state_dim, env.action_dim, device=device)
    elif algorithm.lower() == 'trpo':
        agent = TRPO(env.state_dim, env.action_dim, device=device)
    else:
        raise ValueError(f"不支持的算法: {algorithm}，请选择 'ppo' 或 'trpo'")
        
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
        state = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        episode_policy_losses = []
        episode_value_losses = []
        episode_steps = 0
        
        # 采集数据
        while not (done or truncated) and episode_steps < 1000:
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            mask = 0.0 if done else 1.0
            
            buffer.add(state, action, reward, mask, log_prob, value)
            state = next_state
            episode_reward += reward
            total_steps += 1
            episode_steps += 1
            
            if total_steps % update_interval == 0:
                # 更新时记录损失值
                losses = agent.update(buffer)
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
        
        # 更新数据列表（每回合都收集数据）
        episodes.append(episode)
        rewards.append(episode_reward)
        if episode_policy_losses:
            policy_losses.append(sum(episode_policy_losses) / len(episode_policy_losses))
        if episode_value_losses:
            value_losses.append(sum(episode_value_losses) / len(episode_value_losses))
            
        # 每100个回合更新一次图表，或者是最后一个回合
        if episode % 100 == 0 or episode == max_episodes - 1:
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
            print(f"图表已更新，保存至 {os.path.join(log_dir, 'training_progress.png')}")
        
        # 保存模型
        if episode % save_interval == 0:
            agent.save_model(f"{model_save_path}/{algorithm}_cheetah_episode_{episode}.pt")
            print(f"Model saved at episode {episode}")
        
        # 保存最佳模型
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save_model(f"{model_save_path}/{algorithm}_cheetah_best.pt")
            print(f"Best model saved with reward {best_reward:.1f}")
        
        episode += 1
    
    # 保存最终模型
    agent.save_model(f"{model_save_path}/{algorithm}_cheetah_final.pt")
    print("Final model saved")
    
    # 返回训练结果，用于性能对比
    return {
        'episodes': episodes,
        'rewards': rewards,
        'policy_losses': policy_losses,
        'value_losses': value_losses
    }
    
    env.close()
    writer.close()

# 测试函数
def test(model_path, algorithm='ppo', episodes=5):
    # 创建环境（测试时渲染）
    env = CheetahEnv(render_mode='human')
    device = env.get_device()
    
    # 创建并加载模型
    if algorithm.lower() == 'ppo':
        agent = PPO(env.state_dim, env.action_dim, device=device)
    elif algorithm.lower() == 'trpo':
        agent = TRPO(env.state_dim, env.action_dim, device=device)
    else:
        raise ValueError(f"不支持的算法: {algorithm}，请选择 'ppo' 或 'trpo'")
        
    agent.load_model(model_path)
    
    total_reward = 0
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            action, _, _ = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            state = next_state
            episode_reward += reward
        
        total_reward += episode_reward
        print(f"Test Episode {episode} | Reward: {episode_reward:.1f}")
    
    avg_reward = total_reward / episodes
    print(f"Average Reward over {episodes} episodes: {avg_reward:.1f}")
    
    env.close()
    
    return avg_reward