import torch
import random

from src.datasets.lazega_lawyers_dataset import LazegaLawyersDataset
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils

import torch_geometric.transforms as T

import numpy as np

import networkx as nx

import matplotlib.pyplot as plt

from src.config.global_config import SEED
from src.config.global_config import CUDA
from src.config.global_config import neg_sampling_ratio

pyg.seed_everything(SEED)

device = torch.device(CUDA) if torch.cuda.is_available() else torch.device("cpu")

# 加载数据集
transform = T.Compose([
    T.NormalizeFeatures(),
    T.ToDevice(device),
    # T.RandomNodeSplit(num_val=0.1, num_test=0.1),
    # T.RandomLinkSplit(num_val=0.1, num_test=0.2, is_undirected=True, split_labels=True),
    # T.ToSparseTensor(remove_edge_index=False),
])

dataset = LazegaLawyersDataset(root='../data/LazegaLawyers', transform=transform)
data_adv = dataset[0]
data_fri = dataset[1]
data_work = dataset[2]
print(data_adv)
print(data_fri)
print(data_work)

transform_adj = T.ToSparseTensor(remove_edge_index=False)
data_work = transform_adj(data_work)
dense_adj = data_work.adj_t.to_dense()
print(dense_adj)

# 将 Pytorch Geometric 数据转换为 NetworkX 图
G = pyg.utils.to_networkx(data_adv, to_undirected=True)

# 设置图节点的位置（可选）
pos = nx.spring_layout(G)

# # 绘制图形
# plt.figure(figsize=(32, 32))
# nx.draw(G, pos, with_labels=True, node_color='#FF7F0E', edge_color='#1f78b4', node_size=500, font_size=16, font_color='white')
#
# # 显示图形
# plt.show()

# 创建一个图形和一个坐标轴对象
fig, ax = plt.subplots()

# 画出图形
nx.draw(G, pos, with_labels=False, node_color='lightgreen', edge_color='gray', node_size=100, font_size=16, ax=ax)

# 设置坐标轴以适应图形内容
ax.set_xlim(min(x[0] for x in pos.values()) - 0.1, max(x[0] for x in pos.values()) + 0.1)
ax.set_ylim(min(x[1] for x in pos.values()) - 0.1, max(x[1] for x in pos.values()) + 0.1)

# 去除坐标轴的刻度和标签
ax.set_xticks([])
ax.set_yticks([])

# 显示图形
plt.show()

# model = Node2Vec(
#     data.edge_index,
#     embedding_dim=128,
#     walks_per_node=10,
#     walk_length=20,
#     context_size=10,
#     p=1.0,
#     q=1.0,
#     num_negative_samples=1,
# ).to(device)
#
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def multi_relation_random_walk(data_list, start_node, walk_length):
    """
    Perform a random walk on multi-relation networks.

    Parameters:
    - data_list: List of PyG Data objects representing different relation networks.
    - start_node: The starting node for the random walk.
    - walk_length: The length of the random walk.

    Returns:
    - walk: List of nodes visited during the random walk.
    """
    num_nodes = data_list[0].num_nodes
    walk = [start_node]

    for _ in range(walk_length - 1):
        current_node = walk[-1]

        # Randomly choose a relation network
        relation_network = random.choice(data_list)
        edge_index = relation_network.edge_index

        # Find neighbors of the current node in the chosen relation network
        neighbors = edge_index[1][edge_index[0] == current_node].tolist()

        if neighbors:
            # Randomly choose a neighbor to move to
            next_node = random.choice(neighbors)
            walk.append(next_node)
        else:
            # If the current node has no neighbors in the chosen network, stay at the current node
            walk.append(current_node)

    return walk


# Example usage

# Assuming edge_index is in the format of a 2xE torch tenso

# List of relation networks
data_list = [data_adv, data_fri, data_work]

# Perform a random walk starting from node 0 with a walk length of 10
walk = multi_relation_random_walk(data_list, start_node=0, walk_length=10)
print(walk)
