import numpy.random
import torch
import random
import time

from src.datasets.lazega_lawyers_dataset import LazegaLawyersDataset
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from torch_geometric.utils import add_self_loops


import torch.nn as nn
import torch.optim as optim

import random

import torch_geometric.transforms as T

import numpy as np

import networkx as nx

import matplotlib.pyplot as plt

from src.config.global_config import SEED
from src.config.global_config import CUDA
from src.config.global_config import neg_sampling_ratio

pyg.seed_everything(SEED)
# 设置全局随机种子
torch.manual_seed(SEED)

# 设置CUDA随机种子
torch.cuda.manual_seed(SEED)

# 可选：为所有CUDA设备设置随机种子
torch.cuda.manual_seed_all(SEED)

numpy.random.seed(SEED)

device = torch.device(CUDA) if torch.cuda.is_available() else torch.device("cpu")

# 加载数据集
transform = T.Compose([
    T.NormalizeFeatures(),
    T.ToDevice(device),
    T.AddSelfLoops()
    # T.RandomNodeSplit(num_val=0.1, num_test=0.1),
    # T.RandomLinkSplit(num_val=0.1, num_test=0.2, is_undirected=True, split_labels=True),
    # T.ToSparseTensor(remove_edge_index=False),
])

dataset = LazegaLawyersDataset(root='../../data/LazegaLawyers', transform=transform)
data_adv = dataset[0]
data_fri = dataset[1]
data_work = dataset[2]
print(data_adv)
print(data_fri)
print(data_work)




# 1. 多关系随机游走
def multi_relation_random_walk(data_list, walk_length, start_node):
    edge_index_list = []
    edge_type_list = []

    for idx, data in enumerate(data_list):
        edge_index, _ = add_self_loops(data.edge_index)
        edge_index_list.append(edge_index)
        edge_type_list.append(torch.full((edge_index.size(1),), idx, dtype=torch.long))

    edge_index = torch.cat(edge_index_list, dim=1)
    edge_type = torch.cat(edge_type_list, dim=0)

    adj_list = {i: [] for i in range(data_list[0].num_nodes)}
    for src, dst, etype in zip(edge_index[0], edge_index[1], edge_type):
        adj_list[src.item()].append(dst.item())

    walk = [start_node]
    current_node = start_node
    for _ in range(walk_length - 1):
        if current_node not in adj_list or len(adj_list[current_node]) == 0:
            break
        next_node = random.choice(adj_list[current_node])
        walk.append(next_node)
        current_node = next_node

    return walk


# 2. 生成随机游走路径
def generate_walks(data_list, num_walks, walk_length):
    walks = []
    for node in range(data_list[0].num_nodes):
        for _ in range(num_walks):
            walk = multi_relation_random_walk(data_list, walk_length, start_node=node)
            walks.append(walk)
    return walks


# 3. 构建 Skip-gram 数据集
def build_skipgram_dataset(walks, window_size):
    pairs = []
    for walk in walks:
        for i, center in enumerate(walk):
            for j in range(-window_size, window_size + 1):
                if j == 0 or i + j < 0 or i + j >= len(walk):
                    continue
                context = walk[i + j]
                pairs.append((center, context))
    return pairs


# 4. 定义 Skip-gram 模型
class SkipGramModel(nn.Module):
    def __init__(self, num_nodes, embedding_dim):
        super(SkipGramModel, self).__init__()
        self.embeddings = nn.Embedding(num_nodes, embedding_dim)
        self.output_embeddings = nn.Embedding(num_nodes, embedding_dim)

    def forward(self, center_nodes, context_nodes):
        center_embeds = self.embeddings(center_nodes)  # [batch_size, embedding_dim]
        context_embeds = self.output_embeddings(context_nodes)  # [batch_size, embedding_dim]
        scores = torch.sum(center_embeds * context_embeds, dim=1)  # [batch_size]
        return scores


# 5. 训练节点嵌入
def train_node_embeddings(data_list, num_walks, walk_length, window_size, embedding_dim, epochs, lr, patience=10):
    num_nodes = data_list[0].num_nodes
    walks = generate_walks(data_list, num_walks, walk_length)
    pairs = build_skipgram_dataset(walks, window_size)

    # 将数据转换为 Tensor
    pairs = torch.tensor(pairs, dtype=torch.long).to(device)
    center_nodes = pairs[:, 0]
    context_nodes = pairs[:, 1]

    # 定义模型和优化器
    model = SkipGramModel(num_nodes, embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    # 创建标签 (1 表示真实关系)
    labels = torch.ones(center_nodes.size(0), dtype=torch.float32).to(device)

    best_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    # 训练模型
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        scores = model(center_nodes, context_nodes)
        loss = loss_fn(scores, labels)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

        # 检查是否需要更新最优模型
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_model_state = model.state_dict()  # 保存最优模型状态
            patience_counter = 0  # 重置耐心计数器
        else:
            patience_counter += 1
            print(f"patience counter = {patience_counter}")

        # 检查是否早停
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}, best loss: {best_loss:.4f}")
            break

    # 加载最优模型状态
    model.load_state_dict(best_model_state)
    return model.embeddings.weight.data.cpu()
# 训练嵌入
embedding_dim = 128
num_walks = 10
walk_length = 10
window_size = 5
epochs = 500
lr = 0.01

node_embeddings = train_node_embeddings(
    [data_adv, data_fri, data_work],
    num_walks,
    walk_length,
    window_size,
    embedding_dim,
    epochs,
    lr
)

print("节点嵌入形状:", node_embeddings.shape)


# 创建随机游走函数
# def multi_relation_random_walk(data_list, walk_length, start_node):
#     """
#     多关系随机游走算法
#     Args:
#         data_list: List of PyG Data objects, 每个Data对象表示一个层网络
#         walk_length: int, 随机游走的步长
#         start_node: int, 起始节点编号
#     Returns:
#         walk: List[int], 随机游走的路径
#     """
#     # 合并边信息并标注边类型
#     edge_index_list = []
#     edge_type_list = []
#
#     for idx, data in enumerate(data_list):
#         # edge_index, _ = T.AddSelfLoops(data.edge_index)
#         data = transform(data) # 添加自环
#         edge_index = data.edge_index
#         edge_index_list.append(edge_index)
#         edge_type_list.append(torch.full((edge_index.size(1),), idx, dtype=torch.long))  # 用 idx 标识边类型
#
#     # 拼接所有网络的边
#     edge_index = torch.cat(edge_index_list, dim=1)
#     edge_type = torch.cat(edge_type_list, dim=0)
#
#     # 构建邻接表
#     adj_list = {i: [] for i in range(data_list[0].num_nodes)}
#     for src, dst, etype in zip(edge_index[0], edge_index[1], edge_type):
#         adj_list[src.item()].append((dst.item(), etype.item()))
#
#     # 使用独立的随机生成器
#     rand_gen = torch.Generator()
#     rand_gen.manual_seed(SEED)  # 使用当前时间作为种子
#
#     # 随机游走
#     walk = [start_node]
#     current_node = start_node
#     for _ in range(walk_length - 1):
#         if current_node not in adj_list or len(adj_list[current_node]) == 0:
#             break  # 无法继续游走
#         # 从当前节点的邻居中随机选择
#         next_node, next_edge_type = adj_list[current_node][torch.randint(0, len(adj_list[current_node]), (1,)).item()]
#         walk.append(next_node)
#         current_node = next_node
#
#     return walk
#
#
# # 多关系随机游走示例
# walk_length = 20
# start_node = 0
# walk = multi_relation_random_walk([data_adv, data_fri, data_work], walk_length, start_node)
# print("随机游走路径:", walk)
