import argparse
import os
from train import train, test

# 主函数
def main():
    parser = argparse.ArgumentParser(description='PPO HalfCheetah')
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--model_path', type=str, default='models/ppo_cheetah_best.pt', help='path to model for testing')
    parser.add_argument('--save_interval', type=int, default=100, help='save model every n episodes')
    parser.add_argument('--max_episodes', type=int, default=1000, help='maximum number of episodes')
    parser.add_argument('--test_episodes', type=int, default=5, help='number of test episodes')
    args = parser.parse_args()
    
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_save_path = os.path.join(current_dir, 'models')
    
    if args.mode == 'train':
        train(model_save_path=model_save_path, save_interval=args.save_interval, max_episodes=args.max_episodes)
    elif args.mode == 'test':
        model_path = os.path.join(current_dir, args.model_path)
        test(model_path=model_path, episodes=args.test_episodes)
    else:
        print("Invalid mode. Use 'train' or 'test'")

if __name__ == "__main__":
    main()