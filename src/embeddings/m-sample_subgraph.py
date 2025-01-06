import sys
import os

# 获取当前脚本所在目录的父目录
script_dir = os.path.dirname(os.path.abspath(__file__))
src_root = os.path.abspath(os.path.join(script_dir, '../../'))
sys.path.append(src_root)

import numpy.random
import torch
import random
import time
from typing import Union, List

from src.datasets.lazega_lawyers_dataset import LazegaLawyersDataset
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from torch_geometric.utils import add_self_loops
from torch_geometric.utils import remove_self_loops
from torch_geometric.utils import is_undirected
from torch_geometric.utils import to_networkx


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

import tqdm

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
data_adv.edge_index,_ = remove_self_loops(data_adv.edge_index)
data_fri.edge_index,_ = remove_self_loops(data_fri.edge_index)
data_work.edge_index,_ = remove_self_loops(data_work.edge_index)
print(data_adv)
print(data_fri)
print(data_work)

def weighted_random_choice(items, weights):
    """根据权重随机选择一个元素"""
    total_weight = sum(weights)
    rand_num = random.uniform(0, total_weight)
    cumulative_weight = 0.0
    for item, weight in zip(items, weights):
        cumulative_weight += weight
        if rand_num < cumulative_weight:
            return item
    return items[-1]

# TODO 待验证
def multi_relation_random_walk_with_weights(data_list, walk_length, start_node, weights=None):
    edge_index_list = []
    edge_weight_list = []

    if weights is None:
        weights = [1.0 / len(data_list)] * len(data_list)

    for idx, data in enumerate(data_list):
        edge_index, _ = add_self_loops(data.edge_index)
        edge_weight_list.extend([weights[idx]] * edge_index.size(1))
        edge_index_list.append(edge_index)

    edge_index = torch.cat(edge_index_list, dim=1)
    edge_weight = torch.tensor(edge_weight_list, dtype=torch.float)

    adj_dict = {}
    for src, dst, weight in zip(edge_index[0], edge_index[1], edge_weight):
        src_item, dst_item = src.item(), dst.item()
        if src_item not in adj_dict:
            adj_dict[src_item] = ([], [])
        adj_dict[src_item][0].append(dst_item)
        adj_dict[src_item][1].append(weight.item())

    walk = [start_node]
    current_node = start_node
    for _ in range(walk_length - 1):
        if current_node not in adj_dict or len(adj_dict[current_node][0]) == 0:
            break
        neighbors, neighbor_weights = adj_dict[current_node]
        next_node = weighted_random_choice(neighbors, neighbor_weights)
        walk.append(next_node)
        current_node = next_node

    return walk


def plot_subgraphs(subgraphs):
    """Plot the given subgraphs using matplotlib."""
    for idx, sg in enumerate(subgraphs):
        nx_graph = to_networkx(sg, to_undirected=True)
        fig, ax = plt.subplots()  # 显式创建 Figure 和 Axes 对象
        print(nx_graph)
        nx.draw(nx_graph, with_labels=True, ax=ax)
        ax.set_title(f'Subgraph {idx}')
        plt.show()

#    提取子图
def extract_subgraph(data_list, active_nodes, merge_edges=True, layers=None)->Union[pyg.data.Data, List]:
    """
    根据激活的节点历史记录提取子网
    :param data_list: 包含多层图的数据列表
    :param active_nodes: 多层网络中被激活节点 [{},{},...,{}] Layers 个元素
    :param merge_edges: 是否需要合并所有层的边索引
    :param layers: 指定提取的层（如果为None则提取所有层）
    :return: 提取的子网（torch_geometric.data.Data）
    """
    if layers is None:
        layers = range(len(data_list))

    # 根据激活节点提取子网
    if merge_edges:
        # 合并所有层的边
        edge_index_list = []
        for layer in layers:
            mask = torch.tensor([node in active_nodes[layer] for node in range(data_list[layer].num_nodes)],dtype=torch.bool)   # 创建一个布尔张量，用于标记哪些节点被激活
            sub_edge_index = pyg_utils.subgraph(mask, data_list[layer].edge_index.cpu())[0]  # 根据布尔张量提取子图
            edge_index_list.append(sub_edge_index)
        merged_edge_index = torch.cat(edge_index_list, dim=1) # 合并所有层的边
        subgraph = pyg.data.Data(edge_index=merged_edge_index)
    else:
        # 分层提取子网
        subgraphs = []
        for layer in layers:
            mask = torch.tensor([node in active_nodes[layer] for node in range(data_list[layer].num_nodes)],
                                dtype=torch.bool)
            sub_edge_index = pyg_utils.subgraph(mask, data_list[layer].edge_index.cpu())[0]
            subgraphs.append(pyg.data.Data(edge_index=sub_edge_index))
        return subgraphs

    return subgraph

# 多层独立级联扩散算法
#TODO 有问题啊,激活过还会再激活
def independent_cascade_multilayer(data_list, start_node, max_steps, p_list, cross_layer_activation_prob=0.1, initial_layer=0):
    """
    多层独立级联扩散算法
    :param data_list: 包含多个图的数据列表
    :param start_node: 起始节点
    :param max_steps: 最大扩散步数
    :param p_list: 在 data_list中 各层激活邻居的概率
    :param cross_layer_activation_prob: 在某一层被激活后，激活其他层相同节点的概率
    :param initial_layer: 初始激活的层索引（0: data_adv, 1: data_fri, 2: data_work）
    :return: 每一步的活跃节点列表
    """
    edge_index_list = [data.edge_index for data in data_list]
    num_nodes = data_list[0].num_nodes

    # 初始化邻接字典
    adj_dict = {}
    for layer, edge_index in enumerate(edge_index_list):
        for src, dst in zip(edge_index[0], edge_index[1]):
            src_item, dst_item = src.item(), dst.item()
            if src_item not in adj_dict:
                adj_dict[src_item] = {}
            if dst_item not in adj_dict[src_item]:
                adj_dict[src_item][dst_item] = [0.0, 0.0, 0.0]  # 初始化三层的概率为0
            adj_dict[src_item][dst_item][layer] = 1.0  # 设置当前层的概率为1

    # 初始化活跃节点集合
    past_active_nodes = [{}, {}, {}] # 曾经被激活过的节点, 不允许再被激活
    active_nodes = [{}, {}, {}]
    active_nodes[initial_layer][start_node] = True
    past_active_nodes[initial_layer][start_node] = True
    # 初始化活跃节点激活其他层的相同节点
    for other_layer in range(3):
        if other_layer != initial_layer:
            if random.random() < cross_layer_activation_prob:
                active_nodes[other_layer][start_node] = True
                past_active_nodes[other_layer][start_node] = True
                print(f'第 {other_layer} 层的节点 {start_node} 被其他层激活')
    active_history = [active_nodes.copy()]

    for step in tqdm.tqdm(range(max_steps)):
        new_active_nodes = [{}, {}, {}]
        for layer, active_layer in enumerate(active_nodes):
            for node in active_layer:
                if node not in adj_dict:
                    continue
                for neighbor, probs in adj_dict[node].items():
                    if neighbor in past_active_nodes[layer]: # 如果节点已经被激活过，则跳过
                        continue
                    # 根据概率选择是否激活邻居节点
                    if probs[layer] == 1.0 and random.random() < p_list[layer]:
                        new_active_nodes[layer][neighbor] = True
                        past_active_nodes[layer][neighbor] = True
                        print(f'第 {layer} 层的节点 {neighbor} 被节点{node} 激活')
                        # 激活其他层的相同节点
                        for other_layer in range(3):
                            if other_layer != layer:
                                if neighbor in past_active_nodes[other_layer]:  # 如果节点已经被激活过，则跳过
                                    continue
                                if random.random() < cross_layer_activation_prob:
                                    new_active_nodes[other_layer][neighbor] = True
                                    past_active_nodes[other_layer][neighbor] = True
                                    print(f'第 {other_layer} 层的节点 {neighbor} 被其他层激活')

        if not any(new_active_nodes):
            break
        for layer in range(3):
            # active_nodes[layer].update(new_active_nodes[layer])
            active_nodes[layer] = new_active_nodes[layer]
        active_history.append(active_nodes.copy())
        # layer_subgraphs = extract_subgraph(data_list, active_history[-1], merge_edges=False)
        # plot_subgraphs(layer_subgraphs)
    return active_history

def process_active_history(active_history):
    # 初始化 set_of_active_nodes 为一个空列表
    set_of_active_nodes = []

    # 根据 active_history 的第一项确定层数
    if active_history:
        num_layers = len(active_history[0])
        set_of_active_nodes = [set() for _ in range(num_layers)]

    # 遍历 active_history 中的每一项
    for step_active_nodes in active_history:
        for layer in range(num_layers):
            set_of_active_nodes[layer] = set_of_active_nodes[layer].union(step_active_nodes[layer])

    return set_of_active_nodes


# 示例使用
data_list = [data_adv, data_fri, data_work]
start_node = 0
max_steps = 2
p1 = 0.2  # 在 data_adv 层激活邻居的概率
p2 = 0.4  # 在 data_fri 层激活邻居的概率
p3 = 0.2  # 在 data_work 层激活邻居的概率
cross_layer_activation_prob = 0.3  # 在某一层被激活后，激活其他层相同节点的概率
initial_layer = 0  # 初始激活的层索引（0: data_adv, 1: data_fri, 2: data_work）
p_list = [p1, p2, p3]

active_history = independent_cascade_multilayer(data_list, start_node, max_steps, p_list, cross_layer_activation_prob, initial_layer)
for step, active_nodes in enumerate(active_history):
    print(f"Step {step}: 共激活了{len(active_nodes[0])}个节点,Active Nodes in data_adv: {active_nodes[0]}")
    print(f"Step {step}: 共激活了{len(active_nodes[1])}个节点,Active Nodes in data_fri: {active_nodes[1]}")
    print(f"Step {step}: 共激活了{len(active_nodes[2])}个节点,Active Nodes in data_work: {active_nodes[2]}")
    # activated_nodes = set().union(*active_nodes)
    activated_nodes = set(active_node for layer in active_nodes for active_node in layer)
    # print(f"Step {step}: Active Nodes set: {activated_nodes}")

set_of_active_nodes = process_active_history(active_history)
print(f"set_of_active_nodes：{set_of_active_nodes}")
# 扩散后提取子网
subgraph = extract_subgraph(data_list, set_of_active_nodes, merge_edges=True)
# print(f"子网边索引：{subgraph.edge_index}")

subgraphs = []
# 提取单层子图
layer_subgraphs = extract_subgraph(data_list, set_of_active_nodes, merge_edges=False)
for idx, sg in enumerate(layer_subgraphs):
    print(f"第 {idx} 层子图边索引：{sg.edge_index}")
    # subgraphs.append(to_networkx(sg, to_undirected=True))
    subgraphs.append(sg)

plot_subgraphs(subgraphs)