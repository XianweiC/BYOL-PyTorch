import random
import numpy as np
import argparse
from typing import List
import matplotlib.pyplot as plt

import torch
from torch.backends import cudnn


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    args = parser.parse_args()
    return args


class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def display_class_accuracies(class_accuracies, index_to_class):
    """
    Args:
        class_accuracies (dict): 类别索引和对应准确率的字典。
        index_to_class (list): 按索引映射到类别名称的列表。
    """
    print("\nPer-Class Accuracy:\n")
    for index, class_name in enumerate(index_to_class):
        accuracy = class_accuracies.get(index, None)
        if accuracy is not None:
            print(f"{class_name:10s}: {accuracy:.2f}%")
        else:
            print(f"{class_name:10s}: No samples")

def setup_system(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        cudnn.enabled = True
        cudnn.benchmark = True
        cudnn.deterministic = False
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.mps.is_available():
        torch.mps.manual_seed(seed)
    return seed

@torch.no_grad()
def get_features_from_encoder(encoder, loader, device):
    """
    Get the features from the pre-trained model
    """
    x_train = []
    y_train = []
    for i, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        feature_vector = encoder(x)
        x_train.extend(feature_vector)
        y_train.extend(y.cpu().numpy())
    x_train = torch.stack(x_train)
    y_train = torch.tensor(y_train)
    return x_train, y_train


# def plot_n_polygon(n, values, labels=None):
#     """
#     Plots an n-sided polygon chart with given values and optional labels.
    
#     Parameters:
#         n (int): Number of sides (indicators) for the polygon.
#         values (list or array): Values for each side, range [0, 1] representing percentage (0%-100%).
#         labels (list, optional): Labels for each side. Defaults to None.
#     """
#     # Ensure the values and labels are appropriately sized
#     assert len(values) == n, "Number of values must match number of sides (n)."
#     if labels:
#         assert len(labels) == n, "Number of labels must match number of sides (n)."
#     else:
#         labels = [f'Indicator {i+1}' for i in range(n)]
    
#     # Create angles for the n-sided polygon
#     angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
#     values = np.append(values, values[0])  # Close the polygon by repeating the first value
#     angles += angles[:1]  # Close the angle by repeating the first angle
    
#     # Plot the polygon
#     fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
#     ax.fill(angles, values, color='blue', alpha=0.25)
#     ax.plot(angles, values, color='blue', linewidth=2)
    
#     # Add labels
#     # ax.set_yticks([0.25, 0.5, 0.75, 1.0])  # Scale: 25%, 50%, 75%, 100%
#     ax.set_yticklabels(['25%', '50%', '75%', '100%'])
#     # ax.set_xticks(angles[:-1])  # Ignore closing angle
#     ax.set_xticklabels(labels)
    
#     plt.title(f'{n}-Sided Polygon Chart')
#     plt.show()


def plot_n_polygon(labels: List[str], values: List[float]):
    num_vars = len(labels)

    # 设置角度
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # 闭合雷达图

    # 闭合数据点
    values += values[:1]

    # 创建图表
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # 绘制雷达图
    ax.fill(angles, values, color='blue', alpha=0.25)  # 填充区域
    ax.plot(angles, values, color='blue', linewidth=2)  # 绘制边框线

    # 添加标签
    ax.set_yticks([0, 25, 50, 75, 100])  # 设置刻度
    ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"], fontsize=10)  # 设置刻度标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)
    
    # 标注每个点的具体值
    for angle, value in zip(angles, values):
        x = angle
        y = value
        ax.text(x, y + 0.2, f'{value:.2f}%', fontsize=10,
                ha='center', va='center', color='black')  # 偏移调整更明显

    # 设置标题或调整样式
    ax.set_title("Imbalanced result", fontsize=15, pad=20)
    plt.show()