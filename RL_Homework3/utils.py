import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard.writer import SummaryWriter

class ExperimentLogger:
    def __init__(self):
        # 创建保存目录
        self.log_dir = os.path.join(os.path.dirname(__file__), 'logs')
        self.model_dir = os.path.join(os.path.dirname(__file__), 'models')
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        
        # 初始化tensorboard writer
        self.writer = SummaryWriter(self.log_dir)
        
        # 存储训练数据
        self.data = {}
    
    def log(self, algorithm, env_name, epoch, reward):
        """记录每个epoch的奖励"""
        key = f'{algorithm}_{env_name}'
        if key not in self.data:
            self.data[key] = []
        self.data[key].append(reward)
        
        # 记录到tensorboard
        self.writer.add_scalar(f'{key}/reward', reward, epoch)
    
    def plot_results(self, filename):
        """绘制训练曲线"""
        plt.figure(figsize=(10, 6))
        for key, rewards in self.data.items():
            plt.plot(rewards, label=key)
        plt.xlabel('Epoch')
        plt.ylabel('Reward')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.log_dir, filename))
        plt.close()
    
    def save_results(self, filename):
        """保存训练数据"""
        np.save(os.path.join(self.log_dir, filename), self.data)
    
    def save_model(self, model, algorithm, env_name, epoch):
        """保存模型"""
        model_path = os.path.join(self.model_dir, f'{algorithm}_{env_name}_epoch_{epoch}.pth')
        torch.save(model.state_dict(), model_path)
        return model_path
    
    def load_model(self, model, model_path):
        """加载模型"""
        model.load_state_dict(torch.load(model_path))
        return model
    
    def close(self):
        """关闭tensorboard writer"""
        self.writer.close()