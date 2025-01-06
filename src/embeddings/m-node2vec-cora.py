import os

import torch
import random

from src.datasets.lazega_lawyers_dataset import LazegaLawyersDataset
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from tqdm.notebook import tqdm
from sklearn.manifold import TSNE

import torch_geometric.transforms as T

from torch_geometric.nn import Node2Vec

import numpy as np

import networkx as nx

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from src.config.global_config import SEED
from src.config.global_config import CUDA
from src.config.global_config import neg_sampling_ratio

from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN

# Hyperparameters
pyg.seed_everything(SEED)

device = torch.device(CUDA) if torch.cuda.is_available() else torch.device("cpu")

EMBEDDING_DIM = 128
WALKS_PER_NODE = 10
WALK_LENGTH = 20
CONTEXT_SIZE = 10
P = 1.0
Q = 1.0
NUM_NEGATIVE_SAMPLES = 1
LR = 0.1
BATCH_SIZE = 512
EPOCHS = 200 # TODO 昨晚调了这里，你应该接着搭模型而不是陷进局部优化中

# 加载数据集
transform = T.Compose([
    T.NormalizeFeatures(),
    T.ToDevice(device),
    T.RandomNodeSplit(num_val=0.1, num_test=0.1),
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

data_adv = transform(data_adv).to(device)
print('=========data_adv=========')
# print(data_adv.train_mask)

# Training

# 模型定义
model = Node2Vec(
    data_adv.edge_index,
    embedding_dim=EMBEDDING_DIM,
    walks_per_node=WALKS_PER_NODE,
    walk_length=WALK_LENGTH,
    context_size=CONTEXT_SIZE,
    p=P,
    q=Q,
    num_negative_samples=NUM_NEGATIVE_SAMPLES,
    sparse=True
).to(device)

optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=LR)
loader = model.loader(batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

for idx, (pos_rw, neg_rw) in enumerate(loader):
    print(idx, pos_rw.shape, neg_rw.shape)

idx, (pos_rw, neg_rw) = next(enumerate(loader))
print(f"idx = {idx}, pos_rw.shape = {pos_rw.shape}, neg_rw.shape = {neg_rw.shape}")

edge_tuples = [tuple(x) for x in data_adv.edge_index.cpu().numpy().transpose()]
G = nx.from_edgelist(edge_tuples)
pos = nx.spring_layout(G, center=[0.5, 0.5])
nx.set_node_attributes(G, pos, 'pos')

nodelist = next(enumerate(loader))[1][0][0].tolist()
walk = nx.path_graph(len(nodelist))
nx.set_node_attributes(walk, {idx: pos[node_id] for idx, node_id in enumerate(nodelist)}, 'pos')

fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(1, 2, 1)
nx.draw_networkx_nodes(G,
                       ax=ax,
                       pos=nx.get_node_attributes(G, 'pos'),
                       node_size=5,
                       alpha=0.3,
                       node_color='b')
nx.draw(walk,
        node_size=40,
        node_color='r',
        ax=ax,
        pos=nx.get_node_attributes(walk, 'pos'),
        width=2,
        edge_color='r')
ax = fig.add_subplot(1, 2, 2)
nx.draw(walk,
        node_size=40,
        node_color='r',
        ax=ax,
        pos=nx.get_node_attributes(walk, 'pos'),
        width=2,
        edge_color='r')
plt.show()

# Train the model
def train():
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in tqdm(loader):
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device)).to(device)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def validate():
    model.eval()
    total_val_loss = 0
    for pos_rw, neg_rw in loader:
        val_loss = model.loss(pos_rw.to(device), neg_rw.to(device)).item()
        total_val_loss += val_loss
    return total_val_loss / len(loader)
# 测试函数

# print(f"data = {data_adv}")
# print(f"data.y = {data_adv.y}")
# print(f"data.y.unique() = {data_adv.y.unique()}")


# @torch.no_grad()
# def test():
#     model.eval()
#     z = model().to(device)
#     acc = model.test(z[data_adv.train_mask], data_adv.y[data_adv.train_mask],
#                      z[data_adv.test_mask], data_adv.y[data_adv.test_mask],
#                      max_iter=150)
#     return acc

@torch.no_grad()
def test():
    model.eval()
    z = model().to(device)

    # 获取图对象
    data = data_adv  # data_adv 是一个 PyG Data 对象

    # 获取图中的所有节点
    num_nodes = data.num_nodes

    # 生成正样本（真实存在的边）
    positive_edges = data.edge_index.t().cpu().numpy().tolist()

    # 生成负样本（不存在的边）
    negative_edges = []
    while len(negative_edges) < len(positive_edges):
        node1 = random.randint(0, num_nodes - 1)
        node2 = random.randint(0, num_nodes - 1)
        if node1 != node2 and (node1, node2) not in positive_edges and (node2, node1) not in positive_edges:
            negative_edges.append((node1, node2))

    # 将正样本和负样本合并
    edges = positive_edges + negative_edges
    labels = [1] * len(positive_edges) + [0] * len(negative_edges)

    # 计算嵌入向量的相似度
    embeddings = z.cpu().numpy()
    scores = []
    for edge in edges:
        node1, node2 = edge
        score = torch.dot(torch.tensor(embeddings[node1]), torch.tensor(embeddings[node2])).item()
        scores.append(score)

    # 计算 ROC-AUC 分数
    auc_score = roc_auc_score(labels, scores)

    # 计算 Accuracy
    threshold = 0.5  # 你可以调整这个阈值
    predicted_labels = [1 if score >= threshold else 0 for score in scores]
    acc = accuracy_score(labels, predicted_labels)

    # 计算 Loss
    pos_rw = torch.tensor([list(e) for e in positive_edges], dtype=torch.long).to(device)
    neg_rw = torch.tensor([list(e) for e in negative_edges], dtype=torch.long).to(device)
    loss = model.loss(pos_rw, neg_rw).item()

    return auc_score, acc, loss


# 训练

best_val_loss = float('inf')
best_model_state_dict = None

node2vec_model_name = f"./storage/node2dev_model_dim{EMBEDDING_DIM}_lr{LR}_epochs{EPOCHS}.pth"

if not os.path.exists(node2vec_model_name):
    for epoch in range(1, EPOCHS + 1):
        loss = train()
        val_loss = validate()
        auc_score, acc, test_loss = test()
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}, AUC: {auc_score:.4f}, Acc: {acc:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state_dict = model.state_dict()
            print(f'New best validation loss: {best_val_loss:.4f}')

    # 在测试集上评估最终模型
    final_auc_score, final_acc, final_test_loss = test()
    print(f'Final Test Loss: {final_test_loss:.4f}, Final AUC Score: {final_auc_score:.4f}, Final Acc: {final_acc:.4f}')
    torch.save(best_model_state_dict, node2vec_model_name)
else:
    best_model_state_dict = torch.load(node2vec_model_name)
# 加载最佳模型
model.load_state_dict(best_model_state_dict)

# 可视化
@torch.no_grad()
def plot_points(colors):
    model.eval()
    z = model(torch.arange(data_adv.num_nodes, device=device))
    z = TSNE(n_components=2).fit_transform(z.cpu().numpy())
    y = data_adv.y.cpu().numpy()

    plt.figure(figsize=(8, 8))
    for i in range(dataset.num_classes):
        plt.scatter(z[y == i, 0], z[y == i, 1], s=20, color=colors[i])
    plt.axis('off')
    plt.show()

colors = [
    '#ffc0cb', '#bada55', '#008080', '#420420', '#7fe5f0', '#065535',
    '#ffd700'
]

# 如果 data_adv.y 为 None，则不进行可视化
if data_adv.y is not None:
    plot_points(colors)
else:
    print("No labels available for visualization.")
plt.draw()


# 假设您已经计算了节点嵌入向量 z
@torch.no_grad()
def get_embeddings(model, data):
    model.eval()
    z = model(torch.arange(data.num_nodes, device=device)).cpu()
    return z

# 获取节点嵌入向量
z = get_embeddings(model, data_adv)
print(z)
# 保存向量到 .pt 或 .pth 文件
z_file_name = f"./storage/embedding_vector_dim{EMBEDDING_DIM}_lr{LR}_epochs{EPOCHS}.pt"
torch.save(z, z_file_name)

# 加载向量
loaded_z = torch.load(z_file_name)
'''
使用谱聚类进行社区发现
'''
# num_clusters = 7  # 您可以根据需要调整聚类的数量
# spectral_clustering = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors', assign_labels='kmeans',
#                                          random_state=SEED)
# cluster_labels = spectral_clustering.fit_predict(z)
#
# # 将聚类标签添加到图中
# G = pyg.utils.to_networkx(data_adv, to_undirected=True)
# nx.set_node_attributes(G, dict(zip(G.nodes(), cluster_labels)), 'community')


# 可视化社区发现结果
# def plot_communities(G, colors):
#     pos = nx.spring_layout(G, seed=SEED)
#     plt.figure(figsize=(12, 12))
#
#     # 绘制节点
#     for i in range(num_clusters):
#         nodes = [node for node, attr in G.nodes(data=True) if attr['community'] == i]
#         nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_size=50, node_color=colors[i], label=f'Community {i}')
#
#     # 绘制边
#     nx.draw_networkx_edges(G, pos, alpha=0.3)
#
#     # 去除坐标轴的刻度和标签
#     plt.axis('off')
#     plt.legend()
#     plt.show()


# colors = ['#FF7F0E', '#1f78b4', '#33a02c', '#e31a1c', '#ff7f00', '#6a3d9a', '#b15928']
# plot_communities(G, colors)

z_np = z.numpy()  # 将 tensor 转换为 numpy 数组

# 2. 确定最优聚类数量
def find_optimal_clusters_silhouette(z, max_clusters=15):
    silhouette_scores = []
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=SEED)
        cluster_labels = kmeans.fit_predict(z)
        silhouette_avg = silhouette_score(z, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        print(f"For n_clusters = {n_clusters}, the average silhouette_score is : {silhouette_avg}")

    # 找到轮廓系数最高的聚类数量
    optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
    return optimal_clusters, silhouette_scores

optimal_clusters, silhouette_scores = find_optimal_clusters_silhouette(z_np)

# 绘制轮廓系数曲线
plt.figure(figsize=(10, 6))
plt.plot(range(2, len(silhouette_scores) + 2), silhouette_scores, marker='o')
plt.title('Silhouette Score Method For Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.show()

print(f"Optimal number of clusters using Silhouette Score: {optimal_clusters}")

# 3. 应用 K-Means 聚类
kmeans = KMeans(n_clusters=optimal_clusters, random_state=SEED)
cluster_labels = kmeans.fit_predict(z_np)

# 4. 将社区标签添加到图中
G = pyg_utils.to_networkx(data_adv, to_undirected=True)
nx.set_node_attributes(G, {i: label for i, label in enumerate(cluster_labels)}, "community")

# 5. 可视化社区发现结果
def plot_communities(G, colors):
    if not isinstance(G, nx.Graph):
        raise ValueError("G must be a valid networkx Graph object.")

    if 'community' not in next(iter(G.nodes(data=True)))[1]:
        raise ValueError("Nodes in G must have a 'community' attribute.")

    num_clusters = len(set(nx.get_node_attributes(G, 'community').values()))
    if len(colors) < num_clusters:
        raise ValueError(f"Colors array length ({len(colors)}) is less than the number of clusters ({num_clusters}).")

    pos = nx.spring_layout(G, seed=SEED)
    plt.figure(figsize=(12, 12))

    # 绘制节点
    for i in range(num_clusters):
        nodes = [node for node, attr in G.nodes(data=True) if attr['community'] == i]
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_size=50, node_color=colors[i], label=f'Community {i}')

    # 绘制边
    nx.draw_networkx_edges(G, pos, alpha=0.3)

    # 去除坐标轴的刻度和标签
    plt.axis('off')
    plt.title("K-Means Community Detection", fontsize=16)
    plt.legend()
    plt.show()

colors = ['#FF7F0E', '#1f78b4', '#33a02c', '#e31a1c', '#ff7f00', '#6a3d9a', '#b15928', '#984ea3', '#999999', '#e41a1c', '#dede00', '#a65628', '#f781bf', '#99d8c9', '#8da0cb']
plot_communities(G, colors[:optimal_clusters])

# 6. 评估聚类结果
# 计算轮廓系数
silhouette_avg = silhouette_score(z_np, cluster_labels)
print(f"Average Silhouette Score: {silhouette_avg}")

# 其他评估指标（例如：轮廓系数的标准差）
silhouette_std = np.std([silhouette_score(z_np, cluster_labels, sample_size=1000, random_state=SEED) for _ in range(10)])
print(f"Standard Deviation of Silhouette Score: {silhouette_std}")

