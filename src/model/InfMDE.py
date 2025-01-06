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
from torch_geometric.utils import degree
from torch_geometric.utils import to_undirected

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

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

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
EMB_LR = 0.01
R_LR = 0.001
DROP_OUT = 0.5
BATCH_SIZE = 128
NUM_SIMULATIONS = 100

num_epochs = 100
hidden_channels = [8]
regress_hidden = [8, 4, 2]
out_channels = 8
num_layers = 3
is_train = True

dataset_path = "../../data/TUDataset"
dataset = TUDataset(root=dataset_path, name='MUTAG')
num_nodes = dataset[0].num_nodes


def get_beta_threshold(data: pyg.data.Data, type='simple')->float:
    # 处理节点自环和多重边
    edge_index = remove_self_loops(data.edge_index)
    edge_index = to_undirected(edge_index)
    # 获取节点度
    dc_list = np.array(list(dict(degree(edge_index,data.num_nodes)).values()))
    if type == 'complex':
        beta = float(dc_list.mean())/(float((dc_list**2).mean())-float(dc_list.mean()))
    else:
        beta = 1.0 / float(dc_list.mean())
    print('beta:', beta)
    return beta


# TODO 待验证
def simulate_sir(data, beta=0.1, gamma=1, num_simulations=NUM_SIMULATIONS):
    """
    使用 SIR 模型对每个节点作为种子进行蒙特卡洛模拟，返回影响节点数。
    :param data: PyG 数据对象
    :param beta: 传播率
    :param gamma: 恢复率
    :param num_simulations: 模拟次数
    :return: 每个节点的平均影响节点数
    """
    num_nodes = data.num_nodes
    influence_counts = torch.zeros(num_nodes)

    # pbar = tqdm(range(num_nodes))
    pbar = trange(num_nodes)

    for seed in pbar:
        pbar.set_description("Processing Node %s 's true labels " % seed)  # 设置描述
        for _ in range(num_simulations):
            # SIR 模拟
            infected = set([seed])
            recovered = set()
            while infected:
                new_infected = set()
                for node in infected:
                    neighbors = data.edge_index[1][data.edge_index[0] == node].tolist()
                    for neighbor in neighbors:
                        if neighbor not in infected and neighbor not in recovered:
                            if torch.rand(1).item() < beta:
                                new_infected.add(neighbor)
                    if torch.rand(1).item() < gamma:
                        recovered.add(node)
                infected = new_infected
            influence_counts[seed] += len(infected) + len(recovered)

    return influence_counts / num_simulations

# 生成真实标签
def get_true_labels(data: pyg.data.Data, file_name='true_labels.json')->torch.Tensor:
    # 生成真实标签
    if not os.path.exists(file_name):
        print('Generating true labels...')
        labels = simulate_sir(data)
        with open(file_name, 'w') as f:
            json.dump(labels.tolist(), f)
        return labels
    else:
        print('True labels are exist, and then loading ...')
        with open(file_name, 'r') as f:
            labels = torch.tensor(json.load(f))
        return labels

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


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

for step, data in enumerate(train_loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of graphs in the current batch: {data.num_graphs}')
    print(data)
    print()


class InfGIN(torch.nn.Module):
    def __init__(self, in_channels, dim, out_channels=2):
        super(InfGIN, self).__init__()

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
        x = F.dropout(x, p=DROP_OUT, training=self.training)
        return x  # 直接返回 `dim` 维的 `embedding`
        # x = self.lin2(x)
        # return F.log_softmax(x, dim=-1)


class Regressor(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: list, out_channels: int = 1, dropout=0):
        super(Regressor, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(nn.Linear(in_channels, hidden_channels[0]))
        num_layers = len(hidden_channels) - 1 + 2
        for i in range(num_layers - 2):
            self.convs.append(nn.Linear(hidden_channels[i], hidden_channels[i + 1]))
        self.convs.append(nn.Linear(hidden_channels[-1], out_channels))
        self.dropout = dropout

    def forward(self, x):
        for lin in self.convs[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x)
        return x


class InfMDE(torch.nn.Module):
    def __init__(self, emb_model, regressor, num_communities=7, dist_community_threshold = 1.0):
        super().__init__()
        self.emb_model = emb_model
        self.regressor = regressor

        self.num_communities = num_communities
        self.dist_community_threshold = dist_community_threshold

    def forward(self, x, edge_index, batch):
        # Step 1: 嵌入生成部分
        embedding = self.emb_model(x, edge_index, batch)

        # Step 2: 社区发现（K-Means 聚类）
        kmeans = KMeans(n_clusters=self.num_communities, random_state=0)
        community_labels = kmeans.fit_predict(embedding.cpu().detach().numpy())
        community_centers = torch.tensor(kmeans.cluster_centers_, device=embedding.device)

        # Step 3: 计算社区特性
        # 3.1 社区大小
        community_sizes = torch.tensor(
            [torch.sum(torch.tensor(community_labels == i)).item() for i in range(self.num_communities)],
            dtype=torch.float,
            device=embedding.device
        )

        # 3.2 节点所属社区的中心
        node_community_center = community_centers[community_labels]

        # 3.3 社区间距离矩阵
        community_distances = torch.tensor(
            euclidean_distances(community_centers.cpu().numpy()),
            dtype=torch.float,
            device=embedding.device
        )

        # 3.4 节点到其他社区中心的距离
        node_to_other_communities = community_distances[community_labels]

        # 3.5 节点所属多个社区的数量 (软社区划分)
        # 假设通过嵌入向量与所有社区中心的距离，取阈值确定软社区划分
        node_multi_community_count = torch.sum(node_to_other_communities < self.dist_community_threshold, dim=1)  # 1.0 为阈值，可调整

        # 3.6 节点对节点的距离
        node_distances = torch.cdist(embedding, embedding)

        # Step 4: 特征组合
        # 将嵌入和上述特性拼接成特征向量
        combined_features = torch.cat(
            [
                embedding,  # 原始嵌入
                node_community_center,  # 所属社区中心
                node_to_other_communities,  # 到其他社区中心的距离
                node_multi_community_count.unsqueeze(-1),  # 多社区数量
                node_distances.mean(dim=1, keepdim=True),  # 节点间平均距离
                community_sizes[community_labels].unsqueeze(-1)  # 所属社区大小
            ],
            dim=-1
        )

        # Step 5: 回归部分
        prediction = self.regressor(combined_features)
        return prediction


emb_model = InfGIN(dataset.num_features, EMBEDDING_DIM, dataset.num_classes).to(device)
regressor = Regressor(out_channels, regress_hidden, 1, dropout=DROP_OUT).to(device)
inf_model = InfMDE(emb_model, regressor).to(device)
# print(emb_model)
# print(regressor)
print(inf_model)

# optimizer = torch.optim.Adam(list(emb_model.parameters()) + list(regressor.parameters()), lr=LR)
# criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam([
    {'params': emb_model.parameters(), 'lr': EMB_LR},  # 嵌入模型使用较高学习率
    {'params': regressor.parameters(), 'lr': R_LR}  # 回归模型使用较低学习率
])
criterion = torch.nn.L1Loss()


def train():
    inf_model.train()
    total_loss = 0
    for data_item in train_loader:
        data_item = data_item.to(device)
        optimizer.zero_grad()
        prediction  = inf_model(data_item.x, data_item.edge_index, data_item.batch)
        loss = criterion(prediction, data_item.y)
        loss.backward()
        optimizer.step()
        # total_loss += float(loss) * data.num_graphs
        total_loss += float(loss)
    return total_loss / len(num_nodes)
    # return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
    inf_model.eval()
    # total_correct = 0
    total_mse = 0
    for data in loader:
        data = data.to(device)
        out = inf_model(data.x, data.edge_index, data.batch)
        # total_correct += int((out.argmax(-1) == data.y).sum())
    # return total_correct / len(loader.dataset)
        mse = criterion(out, data.y.float())
        total_mse += float(mse) * data.num_graphs
    return total_mse / len(loader.dataset)


best_test_acc = 0.0  # 初始化最佳测试准确率为0.0
best_test_mse = float('inf')    #  初始化最佳测试 MSE 为无穷大

if is_train():

    for epoch in range(1, num_epochs+1):
        loss = train()
        train_mse = test(train_loader)
        test_mse = test(test_loader)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train MSE: {train_mse:.4f} '
              f'Test MSE: {test_mse:.4f}')
        # 检查当前测试准确率是否优于最佳测试准确率
        if test_mse < best_test_mse:
            best_test_mse = test_mse
            checkpoint_filename = f'checkpoint_epoch{epoch}_batch{BATCH_SIZE}_emb_lr{EMB_LR}_dropout{DROP_OUT}_hidden{EMBEDDING_DIM}_seed{SEED}.pth'
            torch.save({
                'model_state_dict': inf_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_filename)
            print(f'Best model saved with Test MSE: {best_test_mse:.4f}')

else:
    # 加载模型参数和优化器状态
    best_epoch = num_epochs # TODO  选择最佳模型
    checkpoint_filename = f'checkpoint_epoch{best_epoch}_batch{BATCH_SIZE}_emb_lr{EMB_LR}_dropout{DROP_OUT}_hidden{EMBEDDING_DIM}_seed{SEED}.pth'
    checkpoint = torch.load(checkpoint_filename)
    inf_model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


