import sys
import os

# 获取当前脚本所在目录的父目录
script_dir = os.path.dirname(os.path.abspath(__file__))
src_root = os.path.abspath(os.path.join(script_dir, '../../'))
sys.path.append(src_root)
# General imports
import json
import collections

# Data science imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import scipy.sparse as sp

from src.datasets.lazega_lawyers_dataset import LazegaLawyersDataset
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from torch_geometric.utils import add_self_loops
from torch_geometric.utils import remove_self_loops
from torch_geometric.utils import is_undirected
from torch_geometric.utils import to_networkx
from torch_geometric.utils import from_networkx
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GINConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import global_add_pool



import numpy.random
import torch
import random
import time
from typing import Union, List

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU

import torch_geometric.transforms as T

import numpy as np

import networkx as nx

import matplotlib.pyplot as plt

from src.config.global_config import SEED
from src.config.global_config import CUDA
from src.config.global_config import neg_sampling_ratio
from src.utils.visualize import GraphVisualization

import tqdm
from tqdm.auto import trange

pyg.seed_everything(SEED)
# 设置全局随机种子
torch.manual_seed(SEED)

# 设置CUDA随机种子
torch.cuda.manual_seed(SEED)

# 可选：为所有CUDA设备设置随机种子
torch.cuda.manual_seed_all(SEED)

numpy.random.seed(SEED)

device = torch.device(CUDA) if torch.cuda.is_available() else torch.device("cpu")

EMBEDDING_DIM = 32
LR = 0.01

dataset_path = "../../data/TUDataset"
dataset = TUDataset(root=dataset_path, name='MUTAG')

def create_graph(graph):
    g = to_networkx(graph)
    pos = nx.spring_layout(g)
    vis = GraphVisualization(
        g, pos, node_text_position='top left', node_size=20,
    )
    fig = vis.create_figure()
    return fig

dataset = dataset.shuffle()

# train_dataset = dataset[:150]
# test_dataset = dataset[150:]
train_dataset = dataset[len(dataset) // 10:]
test_dataset = dataset[:len(dataset) // 10]

print(f'Number of training graphs: {len(train_dataset)}')
print(f'Example graph[0]:')
print(train_dataset[0])
print(f'Example graph[1]:')
print(train_dataset[1])
print(f'Number of test graphs: {len(test_dataset)}')


train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

for step, data in enumerate(train_loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of graphs in the current batch: {data.num_graphs}')
    print(data)
    print()


class Net(torch.nn.Module):
    def __init__(self, in_channels, dim, out_channels):
        super(Net, self).__init__()

        self.conv1 = GINConv(
            Sequential(Linear(in_channels, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, dim), ReLU()))

        self.conv2 = GINConv(
            Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, dim), ReLU()))

        self.conv3 = GINConv(
            Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, dim), ReLU()))

        self.conv4 = GINConv(
            Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, dim), ReLU()))

        self.conv5 = GINConv(
            Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, dim), ReLU()))

        self.lin1 = Linear(dim, dim)
        self.lin2 = Linear(dim, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        x = self.conv4(x, edge_index)
        x = self.conv5(x, edge_index)
        x = global_add_pool(x, batch)
        x = self.lin1(x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)


model = Net(dataset.num_features, EMBEDDING_DIM, dataset.num_classes).to(device)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
    model.eval()

    total_correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        total_correct += int((out.argmax(-1) == data.y).sum())
    return total_correct / len(loader.dataset)

best_test_acc = 0.0  # 初始化最佳测试准确率为0.0
for epoch in range(1, 101):
    loss = train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f} '
          f'Test Acc: {test_acc:.4f}')
    # 检查当前测试准确率是否优于最佳测试准确率
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        # 保存最佳模型
        torch.save(model, f"GIN_best_epoch{epoch}_batch128_lr{LR}_dropout0.5_hidden{EMBEDDING_DIM}_seed{SEED}.pt")
        print(f'Best model saved with Test Acc: {best_test_acc:.4f}')
