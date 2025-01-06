import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU

import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from torch_geometric.data import DataLoader
from torch_geometric.utils import negative_sampling

import matplotlib.pyplot as plt  # needed to visualize loss curves

import numpy as np

import torch_geometric.transforms as T

from torch_geometric.utils import train_test_split_edges
from torch_geometric.utils import is_undirected
from torch_geometric.utils import from_networkx
from torch_geometric.utils import to_networkx
from torch_geometric.utils import to_undirected

from torch_geometric.data import HeteroData

from torch import Tensor
import networkx as nx

import scipy.stats as stats
from tqdm import tqdm

from src.config.global_config import SEED
from src.config.global_config import CUDA
from src.config.global_config import neg_sampling_ratio

from src.utils.networkx_plus import drop_weights
from src.utils.pyg_plus import to_undirected


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

split_func = T.RandomLinkSplit(num_val=0.1, num_test=0.2, is_undirected=True, split_labels=True,
                               neg_sampling_ratio=neg_sampling_ratio)

node_split_func = T.RandomNodeSplit(num_val=0.1, num_test=0.2)

transform_adj = T.ToSparseTensor(remove_edge_index=False)

feature_norm_func = T.NormalizeFeatures()

# TODO Dataset Process------------------------------------------

'''
-----------------Lazega Lawyers Dataset----------------------------------------
'''
# df = pd.read_csv('../data/Facebook/tvshow_edges.csv')
# nx_g_facebook = nx.from_pandas_edgelist(df, source='node_1', target='node_2')
# 以二进制格式读取邻接矩阵
adjacency_matrix_adv = np.fromfile('../../data/LazegaLawyers/ELadv.dat', dtype=np.int32, sep=' ')
adjacency_matrix_fri = np.fromfile('../../data/LazegaLawyers/ELfriend.dat', dtype=np.int32, sep=' ')
adjacency_matrix_work = np.fromfile('../../data/LazegaLawyers/ELwork.dat', dtype=np.int32, sep=' ')

feature_matrix = np.fromfile('../../data/LazegaLawyers/ELattr.dat', dtype=np.float64, sep=' ')

# 假设是方阵，你可能需要调整形状
# 如果是非方阵，需要知道矩阵的行列数
num_nodes = int(np.sqrt(len(adjacency_matrix_adv)))
num_features = 8
adjacency_matrix_adv = adjacency_matrix_adv.reshape((num_nodes, num_nodes))
adjacency_matrix_fri = adjacency_matrix_fri.reshape((num_nodes, num_nodes))
adjacency_matrix_work = adjacency_matrix_work.reshape((num_nodes, num_nodes))
feature_matrix = feature_matrix.reshape((num_nodes, num_features))

# print('feature->')
# print(feature_matrix)

# print(np.sum(adjacency_matrix_adv))
# print(np.sum(adjacency_matrix_fri))
# print(np.sum(adjacency_matrix_work))

nx_g_lawyers_adv_directed = nx.from_numpy_matrix(adjacency_matrix_adv, create_using=nx.DiGraph)
nx_g_lawyers_fri_directed = nx.from_numpy_matrix(adjacency_matrix_fri, create_using=nx.DiGraph)
nx_g_lawyers_work_directed = nx.from_numpy_matrix(adjacency_matrix_work, create_using=nx.DiGraph)

# print(nx.number_of_edges(nx_g_lawyers_adv_directed))
# print(nx.number_of_edges(nx_g_lawyers_fri_directed))
# print(nx.number_of_edges(nx_g_lawyers_work_directed))

# 获取有向图中的自环边
selfloop_edges_adv = list(nx.selfloop_edges(nx_g_lawyers_adv_directed))
selfloop_edges_fri = list(nx.selfloop_edges(nx_g_lawyers_fri_directed))
selfloop_edges_work = list(nx.selfloop_edges(nx_g_lawyers_work_directed))
# 判断是否有自环
if selfloop_edges_adv or selfloop_edges_fri or selfloop_edges_work:
    print("有向图中存在自环边：", selfloop_edges_adv+selfloop_edges_fri+selfloop_edges_work)
else:
    print("有向图中不存在自环边。")

# print(nx.number_of_edges(nx_g_lawyers_adv))
# print('nx_g_lawyers_adv is_directed  = %s' % nx.is_directed(nx_g_lawyers_adv))

# nx.draw(nx_g_lawyers_adv)

# print('nx_g_lawyers_adv_directed is_directed  = %s' % nx.is_directed(nx_g_lawyers_adv_directed))
# print('nx_g_lawyers_fri_directed is_directed  = %s' % nx.is_directed(nx_g_lawyers_fri_directed))
# print('nx_g_lawyers_work_directed is_directed  = %s' % nx.is_directed(nx_g_lawyers_work_directed))

nx_g_lawyers_adv_directed = drop_weights(nx_g_lawyers_adv_directed)  # 去除权重
nx_g_lawyers_fri_directed = drop_weights(nx_g_lawyers_fri_directed)  # 去除权重
nx_g_lawyers_work_directed = drop_weights(nx_g_lawyers_work_directed)  # 去除权重


# TODO 两种方案
# 方案一： 分开存储多张图
data_lawyers_adv = from_networkx(nx_g_lawyers_adv_directed)
data_lawyers_fri = from_networkx(nx_g_lawyers_fri_directed)
data_lawyers_work = from_networkx(nx_g_lawyers_work_directed)

data_lawyers_adv.edge_index = to_undirected(data_lawyers_adv.edge_index)
data_lawyers_fri.edge_index = pyg_utils.to_undirected(data_lawyers_fri.edge_index)
data_lawyers_work.edge_index = pyg_utils.to_undirected(data_lawyers_work.edge_index)

data_lawyers_adv.x = torch.tensor(feature_matrix)
data_lawyers_fri.x = torch.tensor(feature_matrix)
data_lawyers_work.x = torch.tensor(feature_matrix)

print(data_lawyers_adv)
print(data_lawyers_fri)
print(data_lawyers_work)

nx_g_lawyers_adv_undirected = to_networkx(data_lawyers_adv, to_undirected=True)
nx_g_lawyers_fri_undirected = to_networkx(data_lawyers_fri, to_undirected=True)
nx_g_lawyers_work_undirected = to_networkx(data_lawyers_work, to_undirected=True)
# 去除自环
nx_g_lawyers_adv_undirected.remove_edges_from(nx.selfloop_edges(nx_g_lawyers_adv_undirected))
nx_g_lawyers_fri_undirected.remove_edges_from(nx.selfloop_edges(nx_g_lawyers_fri_undirected))
nx_g_lawyers_work_undirected.remove_edges_from(nx.selfloop_edges(nx_g_lawyers_work_undirected))

# TODO 验证数据集的正确性
print('-----------------Lazega Lawyers advice-----------------------')
print(nx_g_lawyers_adv_undirected)
print('-----------------Lazega Lawyers friend-----------------------')
print(nx_g_lawyers_fri_undirected)
print('-----------------Lazega Lawyers work-----------------------')
print(nx_g_lawyers_work_undirected)
# print('data_lawyers_adv.is_undirected = %s' % data_lawyers_adv.is_undirected())
# print('data_lawyers_fri.is_undirected = %s' % data_lawyers_fri.is_undirected())
# print('data_lawyers_work.is_undirected = %s' % data_lawyers_work.is_undirected())
# data_email.edge_index = to_undirected(data_email.edge_index)  # ensure the graph is undirected

torch.save(data_lawyers_adv, '../../data/LazegaLawyers/processed/data_lawyers_adv.pt')
torch.save(data_lawyers_fri, '../../data/LazegaLawyers/processed/data_lawyers_fri.pt')
torch.save(data_lawyers_work, '../../data/LazegaLawyers/processed/data_lawyers_work.pt')

# TODO 2.方案二： 用异质图数据结构存储 

data_Lawyers_hetero = HeteroData()
# Create the same node types "member" holding a feature matrix:
data_Lawyers_hetero['member'].x = torch.tensor(feature_matrix)

# Create an edge type "(author, writes, paper)" and building the
# graph connectivity:
data_Lawyers_hetero['member', 'consult', 'member'].edge_index = data_lawyers_adv.edge_index  # [2, num_edges]
data_Lawyers_hetero['member', 'friend', 'member'].edge_index = data_lawyers_fri.edge_index  # [2, num_edges]
data_Lawyers_hetero['member', 'work', 'member'].edge_index = data_lawyers_work.edge_index  # [2, num_edges]

# print(data_Lawyers_hetero['member'].x)


'''

'''
# Considering the facebook TV shows dataset has not node features, add the heuristic feature.
'''
# TODO data_facebook.x ; data_facebook.y
# degree_feature = generate_degree_feature(data_facebook.edge_index, data_facebook.num_nodes)
# betweenness_feature = generate_betweenness_centrality_feature(nx_g_facebook)
# closeness_feature = generate_closeness_centrality_feature(nx_g_facebook)
# k_shell_feature = generate_k_shell_feature(nx_g_facebook)
# print('k shell shape = %s, value =%s' % (k_shell_feature.shape, k_shell_feature))
# feature_tensor = torch.stack([degree_feature.cpu(), betweenness_feature.cpu(), closeness_feature.cpu(), k_shell_feature.cpu()])
# torch.save(feature_tensor, '../data/Facebook/feature.pt')
# torch.save(betweenness_feature, '../data/Facebook/betweenness_feature.pt')
feature_tensor = torch.load('../data/Facebook/feature.pt')
betweenness_feature = torch.load('../data/Facebook/betweenness_feature.pt')
print('generate heuristic feature success..')

# feature_tensor = torch.load('../data/PowerGrid/feature.pt')  # [4,4941]
prob_matrix = load_matrix_from_csv('../data/Facebook/Node-SIR-facebook_0_counter_betath1.csv')
norm_prob_matrix = torch.nn.functional.normalize(prob_matrix, p=1, dim=1)
# feature_tensor = torch.concat((feature_tensor.t(), norm_prob_matrix), dim=1)
data_facebook.x = feature_tensor.t().to(device)
label = load_label_from_csv('../data/Facebook/Node-SIR-facebook_0.csv', first_col='Node', beta=1)
print('label.shape = %s, value = %s' % (label.shape, label))
data_facebook.y = label
transform_adj(data_facebook)  # get the adj_t of graph
print(data_facebook)
print(data_facebook.adj_t.to_dense())
# prob_matrix = load_matrix_from_csv('../data/Hamster/hamster-Node-SIR_counter_n100_step1000.csv')
# normalize
# 对矩阵进行行归一化

# norm_prob_matrix = prob_matrix
prob_matrix = prob_matrix.t()

# TODO 验证此处代码正确性
print(norm_prob_matrix.sum(dim=1))

high_order_matrix_facebook = generate_high_order_structure(data_facebook.adj_t.to_dense().to(device),
                                                          norm_prob_matrix.to(device), eta=eta)
if high_order_matrix_norm:
    high_order_matrix_facebook = norm_row_matrix(high_order_matrix_facebook)
un_attach_edge_index, un_attach_edge_weight = pyg_utils.dense_to_sparse(high_order_matrix_facebook)
# data_hamster.edge_index = un_attach_edge_index
# data_hamster.edge_weight = un_attach_edge_weight

# TODO 待完善,有Bug
print('is_undirected = %s' % is_undirected(data_facebook.edge_index))
node_split_func(data_facebook)


data_facebook_split = data_facebook.clone()
data_facebook_split_edges = train_test_split_edges(data_facebook_split, val_ratio=0.1, test_ratio=0.2)
'''
# Finished splitting the edges
'''
# TODO 验证数据集的正确性
data_facebook.to(device)
print(data_facebook)
# print(train_data_hamster)
# print(val_data_hamster)
# print(test_data_hamster)

'''
'''
----------------Lazega Lawyers Dataset Finished--------------------------------
'''