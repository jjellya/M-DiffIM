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
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GINConv
from torch_geometric.nn import global_mean_pool



import numpy.random
import torch
import random
import time
from typing import Union, List

import torch.nn as nn
import torch.optim as optim

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

train_dataset = dataset[:150]
test_dataset = dataset[150:]

print(f'Number of training graphs: {len(train_dataset)}')
print(f'Example graph[0]:')
print(train_dataset[0])
print(f'Example graph[1]:')
print(train_dataset[1])
print(f'Number of test graphs: {len(test_dataset)}')


train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

for step, data in enumerate(train_loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of graphs in the current batch: {data.num_graphs}')
    print(data)
    print()


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


model = GCN(hidden_channels=64).to(device)
# print(model)

# model = GCN(hidden_channels=64)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()
    total_loss = 0
    for data in train_loader:  # Iterate in batches over the training dataset.
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
        total_loss += loss.item()
    return total_loss / len(train_loader)


def test(loader):
    model.eval()
    correct = 0
    loss_ = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss_ += loss.item()
        pred = out.argmax(dim=1)  # Use the class with highest probability.

        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset), loss_ / len(loader.dataset)


for epoch in trange(1, 171):
    train_loss = train()
    train_acc, _ = test(train_loader)
    test_acc, test_loss = test(test_loader)

    print(f'\nEpoch {epoch:03d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

torch.save(model, f"graph_classification_model_epoch{epoch}_batch64_lr0.001_dropout0.5_hidden64_seed{SEED}.pt")

