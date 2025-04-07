import argparse
import os
from train import train, test
from compare import compare_algorithms

def main():
    parser = argparse.ArgumentParser(description='强化学习迷宫环境训练与比较')
    
    # 添加命令行参数
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'compare'],
                        help='运行模式: train (训练), test (测试), compare (比较算法)')
    parser.add_argument('--algorithm', type=str, default='ppo', choices=['ppo', 'trpo'],
                        help='强化学习算法: ppo 或 trpo')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='训练回合数')
    parser.add_argument('--model', type=str, default=None,
                        help='测试模式下使用的模型路径，如果不指定则使用最新的模型')
    
    args = parser.parse_args()
    
    # 创建模型目录
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    if args.mode == 'train':
        # 训练模式
        print(f"开始使用 {args.algorithm.upper()} 算法训练...")
        rewards = train(algorithm=args.algorithm, max_episodes=args.episodes)
        print(f"训练完成，共 {len(rewards)} 个回合")
    
    elif args.mode == 'test':
        # 测试模式
        if args.model is None:
            # 如果没有指定模型，使用最新的模型
            model_files = [f for f in os.listdir(model_dir) 
                          if f.startswith(f"{args.algorithm}_maze_") and f.endswith('.pth')]
            if not model_files:
                print(f"没有找到 {args.algorithm.upper()} 算法的模型，请先训练")
                return
            latest_model = max(model_files, key=lambda x: os.path.getmtime(os.path.join(model_dir, x)))
            model_path = os.path.join(model_dir, latest_model)
        else:
            model_path = args.model
        
        print(f"测试 {args.algorithm.upper()} 模型: {model_path}")
        rewards = test(model_path, algorithm=args.algorithm)
        print(f"测试完成，平均奖励: {sum(rewards)/len(rewards):.2f}")
    
    elif args.mode == 'compare':
        # 比较模式
        print("比较 PPO 和 TRPO 算法性能...")
        compare_algorithms(max_episodes=args.episodes)
        print("比较完成，结果已保存到 comparison_results 目录")

if __name__ == "__main__":
    main()