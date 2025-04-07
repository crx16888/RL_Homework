import os
import argparse
from train import train, test
from compare import compare_algorithms

def main():
    parser = argparse.ArgumentParser(description='Train or test RL algorithms on HalfCheetah')
    parser.add_argument('--mode', type=str, default='train', 
                        choices=['train', 'test', 'compare'], 
                        help='train, test, or compare algorithms')
    parser.add_argument('--algorithm', type=str, default='ppo', 
                        choices=['ppo', 'trpo'], 
                        help='RL algorithm to use (ppo or trpo)')
    parser.add_argument('--model_path', type=str, default=None, 
                        help='path to model for testing')
    parser.add_argument('--episodes', type=int, default=1000, 
                        help='number of episodes for training or comparison')
    args = parser.parse_args()
    
    if args.mode == 'train':
        print(f"训练 {args.algorithm.upper()} 算法...")
        train(algorithm=args.algorithm, max_episodes=args.episodes)
    elif args.mode == 'test':
        if args.model_path is None:
            args.model_path = f'models/{args.algorithm}_cheetah_best.pt'
        print(f"测试 {args.algorithm.upper()} 算法，模型路径: {args.model_path}")
        test(model_path=args.model_path, algorithm=args.algorithm)
    elif args.mode == 'compare':
        print(f"比较 PPO 和 TRPO 算法性能...")
        compare_algorithms(max_episodes=args.episodes) # 如果不在命令行指定回合数，默认为default

if __name__ == "__main__":
    main()