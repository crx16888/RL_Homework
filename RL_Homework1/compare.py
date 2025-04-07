import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import shutil

from train import train, test

def compare_algorithms(max_episodes=1000, save_interval=100):
    # 创建保存目录
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # 创建比较结果保存目录
    comparison_dir = os.path.join(os.path.dirname(__file__), 'comparison_results')
    if not os.path.exists(comparison_dir):
        os.makedirs(comparison_dir)
    
    # 训练PPO算法
    print("\n开始训练PPO算法...")
    start_time = time.time()
    ppo_results = train(algorithm='ppo', max_episodes=max_episodes)
    ppo_time = time.time() - start_time
    print(f"PPO训练完成，耗时: {ppo_time:.2f}秒")
    
    # 训练TRPO算法
    print("\n开始训练TRPO算法...")
    start_time = time.time()
    trpo_results = train(algorithm='trpo', max_episodes=max_episodes)
    trpo_time = time.time() - start_time
    print(f"TRPO训练完成，耗时: {trpo_time:.2f}秒")
    
    # 绘制奖励对比图
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(range(len(ppo_results)), ppo_results, 'b-', label='PPO')
    plt.plot(range(len(trpo_results)), trpo_results, 'r-', label='TRPO')
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    
    # 绘制训练时间对比
    plt.subplot(2, 2, 4)
    algorithms = ['PPO', 'TRPO']
    times = [ppo_time, trpo_time]
    plt.bar(algorithms, times, color=['blue', 'red'])
    plt.title('Training Time')
    plt.ylabel('Time (seconds)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'algorithm_comparison.png'))
    plt.close()
    
    # 测试两种算法的性能
    print("\n测试PPO算法性能...")
    ppo_model_path = os.path.join(model_dir, 'ppo_maze_best.pth')
    ppo_rewards = test(model_path=ppo_model_path, algorithm='ppo')
    ppo_avg_reward = sum(ppo_rewards) / len(ppo_rewards) if ppo_rewards else 0
    
    print("\n测试TRPO算法性能...")
    trpo_model_path = os.path.join(model_dir, 'trpo_maze_best.pth')
    trpo_rewards = test(model_path=trpo_model_path, algorithm='trpo')
    trpo_avg_reward = sum(trpo_rewards) / len(trpo_rewards) if trpo_rewards else 0
    
    # 绘制测试性能对比图
    plt.figure(figsize=(8, 6))
    plt.bar(['PPO', 'TRPO'], [ppo_avg_reward, trpo_avg_reward], color=['blue', 'red'])
    plt.title('Test Performance Comparison')
    plt.ylabel('Average Reward')
    plt.grid(True)
    plt.savefig(os.path.join(comparison_dir, 'test_performance_comparison.png'))
    plt.close()
    
    print("\n性能对比完成！")
    print(f"PPO平均奖励: {ppo_avg_reward:.1f}")
    print(f"TRPO平均奖励: {trpo_avg_reward:.1f}")
    print(f"对比结果已保存到 {comparison_dir} 目录")

# 添加滑动平均函数，用于平滑曲线
def moving_average(data, window_size):
    if len(data) < window_size:
        return data
    ret = np.cumsum(data, dtype=float)
    ret[window_size:] = ret[window_size:] - ret[:-window_size]
    return ret[window_size - 1:] / window_size

# 绘制平滑后的对比图
def plot_smoothed_comparison(ppo_results, trpo_results, window_size=5):
    comparison_dir = os.path.join(os.path.dirname(__file__), 'comparison_results')
    if not os.path.exists(comparison_dir):
        os.makedirs(comparison_dir)
    
    # 平滑奖励曲线
    if len(ppo_results) >= window_size and len(trpo_results) >= window_size:
        ppo_smooth_rewards = moving_average(ppo_results, window_size)
        trpo_smooth_rewards = moving_average(trpo_results, window_size)
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(ppo_smooth_rewards)), ppo_smooth_rewards, 'b-', label='PPO')
        plt.plot(range(len(trpo_smooth_rewards)), trpo_smooth_rewards, 'r-', label='TRPO')
        plt.title('Smoothed Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward (Moving Average)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(comparison_dir, 'smoothed_rewards_comparison.png'))
        plt.close()

if __name__ == "__main__":
    # 设置随机种子，确保结果可复现
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 比较PPO和TRPO算法
    compare_algorithms(max_episodes=500)  # 可以根据需要调整训练轮数