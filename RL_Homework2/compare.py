import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import time

from train import train, test

def compare_algorithms(max_episodes=1000, save_interval=100):
    # 创建保存目录
    model_save_path = 'models'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    
    # 创建比较结果保存目录
    comparison_dir = 'comparison_results'
    if not os.path.exists(comparison_dir):
        os.makedirs(comparison_dir)
    
    # 训练PPO算法
    print("\n开始训练PPO算法...")
    start_time = time.time()
    ppo_results = train(algorithm='ppo', model_save_path=model_save_path, 
                       save_interval=save_interval, max_episodes=max_episodes)
    ppo_time = time.time() - start_time
    print(f"PPO训练完成，耗时: {ppo_time:.2f}秒")
    
    # 训练TRPO算法
    print("\n开始训练TRPO算法...")
    start_time = time.time()
    trpo_results = train(algorithm='trpo', model_save_path=model_save_path, 
                        save_interval=save_interval, max_episodes=max_episodes)
    trpo_time = time.time() - start_time
    print(f"TRPO训练完成，耗时: {trpo_time:.2f}秒")
    
    # 绘制奖励对比图
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(ppo_results['episodes'], ppo_results['rewards'], 'b-', label='PPO')
    plt.plot(trpo_results['episodes'], trpo_results['rewards'], 'r-', label='TRPO')
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    
    # 绘制策略损失对比图
    plt.subplot(2, 2, 2)
    if ppo_results['policy_losses'] and trpo_results['policy_losses']:
        ppo_episodes = ppo_results['episodes'][-len(ppo_results['policy_losses']):]
        trpo_episodes = trpo_results['episodes'][-len(trpo_results['policy_losses']):]
        plt.plot(ppo_episodes, ppo_results['policy_losses'], 'b-', label='PPO')
        plt.plot(trpo_episodes, trpo_results['policy_losses'], 'r-', label='TRPO')
    plt.title('Policy Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 绘制价值损失对比图
    plt.subplot(2, 2, 3)
    if ppo_results['value_losses'] and trpo_results['value_losses']:
        ppo_episodes = ppo_results['episodes'][-len(ppo_results['value_losses']):]
        trpo_episodes = trpo_results['episodes'][-len(trpo_results['value_losses']):]
        plt.plot(ppo_episodes, ppo_results['value_losses'], 'b-', label='PPO')
        plt.plot(trpo_episodes, trpo_results['value_losses'], 'r-', label='TRPO')
    plt.title('Value Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
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
    ppo_avg_reward = test(model_path=f"{model_save_path}/ppo_cheetah_best.pt", algorithm='ppo')
    
    print("\n测试TRPO算法性能...")
    trpo_avg_reward = test(model_path=f"{model_save_path}/trpo_cheetah_best.pt", algorithm='trpo')
    
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
def plot_smoothed_comparison(ppo_results, trpo_results, window_size=5, save_path='comparison_results'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # 平滑奖励曲线
    if len(ppo_results['rewards']) >= window_size and len(trpo_results['rewards']) >= window_size:
        ppo_smooth_rewards = moving_average(ppo_results['rewards'], window_size)
        trpo_smooth_rewards = moving_average(trpo_results['rewards'], window_size)
        
        plt.figure(figsize=(10, 6))
        plt.plot(ppo_results['episodes'][window_size-1:], ppo_smooth_rewards, 'b-', label='PPO')
        plt.plot(trpo_results['episodes'][window_size-1:], trpo_smooth_rewards, 'r-', label='TRPO')
        plt.title('Smoothed Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward (Moving Average)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_path, 'smoothed_rewards_comparison.png'))
        plt.close()

if __name__ == "__main__":
    # 设置随机种子，确保结果可复现
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 比较PPO和TRPO算法
    compare_algorithms(max_episodes=100)  # 可以根据需要调整训练轮数