import numpy as np

from src.utils.equation_extractor import EquationExtractor

import numpy as np
import scipy.sparse as sp
import torch

from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, to_undirected


def split_data_list(data_list: list, target: str = 'DATA:'):
    """
    使用列表推导和字符串方法找到target的位置并分割列表
    :param data_list: list
    :param target: str
    :return: list, list
    """
    # 使用列表推导和字符串方法找到target的位置并分割列表
    split_points = [i for i, s in enumerate(data_list) if s.startswith(target)]
    splitted_list = []
    start = 0
    if len(split_points) == 0:
        return data_list, None
    for split_point in split_points:
        splitted_list.append(data_list[start:split_point])
        start = split_point + 1  # 跳过分隔符
    # 添加最后一部分，不包含分隔符
    return splitted_list[0], data_list[start:]
    # splitted_list.append(data_list[start:])
    # return splitted_list


# splitted_list = split_data_list(data_list)

def read_dat(path):
    # with np.load(path) as f:
    with open(path, 'r') as f:
        return parse_dat(f)


def parse_dat(f, is_undirected=True):
    raw_data = f.read()

    # x = sp.csr_matrix((f['attr_data'], f['attr_indices'], f['attr_indptr']),
    #                   f['attr_shape']).todense()
    # x = torch.from_numpy(x).to(torch.float)
    # x[x > 0] = 1
    # DL是一种html语言，标签定义了定义列表（definition list），用于结合 （定义列表中的项目）和（描述列表中的项目）。
    SPLIT_SIGN = '\n'
    eq_extractor = EquationExtractor()
    description, matrix_data_serialized = split_data_list(raw_data.split())

    # description, _ = split_data_list()
    N = int(eq_extractor.get_value_by_sign(description[1])[0])
    layers = int(eq_extractor.get_value_by_sign(description[2])[0])
    raw_description, _ = split_data_list(raw_data.split(SPLIT_SIGN))

    print('N：', N, 'layers：', layers)

    print('数据描述：', description)
    print('len(description)', len(description))
    # 将读取的数据分割并转换为浮点数
    # numbers = np.array(data.split()[len(description):], dtype=np.float)
    numbers = np.array(matrix_data_serialized, dtype=np.float32)
    print('numbers = ', numbers)

    # 如果数据是二维的，比如一个矩阵，你可能需要重新调整数组的形状
    # 假设数据是 N 行 M 列，每行的分隔符是空格
    matrixs = numbers.reshape(layers, N, N)

    print('数据形状：', matrixs.shape)
    print('matrix[0] = \n', matrixs[0])

    # adj = sp.csr_matrix((f['adj_data'], f['adj_indices'], f['adj_indptr']),
    #                     f['adj_shape']).tocoo()
    adjs = []
    dataset = []
    for i, matrix in enumerate(matrixs):
        csr_adj = sp.csr_matrix(matrix).tocoo()
        adjs.append(csr_adj)
        row = torch.from_numpy(csr_adj.row).to(torch.long)
        col = torch.from_numpy(csr_adj.col).to(torch.long)
        weight = torch.from_numpy(csr_adj.data).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)  # TODO: 已验证edge_index为正确.
        edge_index, _ = remove_self_loops(edge_index)
        if is_undirected:
            edge_index = to_undirected(edge_index, num_nodes=N)

        # y = torch.from_numpy(f['labels']).to(torch.long)   # TODO: 找到dat文件中'ROW LABELS:'后面对应的标签并且编码.

        # _, raw_y = split_data_list(raw_description, 'ROW LABELS:')
        # raw_y, _ = split_data_list(raw_y, 'COLUMN LABELS:')
        # y = torch.from_numpy(np.array(raw_y[0].split()[1:])).to(torch.long) if raw_y is not None else None
        x = torch.eye(N, dtype=torch.float)  # 创建一个对角矩阵，‌其中对角线上的元素为1，‌其余元素为0

        dataset.append(Data(x=x, edge_index=edge_index, edge_weight=weight))  # y先不加进去
    return dataset



def load_data():
    pass


# 打开文件并读取数据
# with open('../../data/KarateClub/zachary.dat', 'r') as file:
#     data = file.read()

# # DL是一种html语言，标签定义了定义列表（definition list），用于结合 （定义列表中的项目）和（描述列表中的项目）。
# eq_extractor = EquationExtractor()
# # description = data.split()[0:13]
# description, matrix_data_serialized = split_data_list(data.split())
# N = int(eq_extractor.get_value_by_sign(description[1])[0])
# layers = int(eq_extractor.get_value_by_sign(description[2])[0])
#
# print('N：', N, 'layers：', layers)
#
# print('数据描述：', description)
# print('len(description)', len(description))
# # 将读取的数据分割并转换为浮点数
# # numbers = np.array(data.split()[len(description):], dtype=np.float)
# numbers = np.array(matrix_data_serialized, dtype=np.float32)
# print('numbers = ', numbers)
#
# # 如果数据是二维的，比如一个矩阵，你可能需要重新调整数组的形状
# # 假设数据是 N 行 M 列，每行的分隔符是空格
# matrix = numbers.reshape(layers, N, N)
#
# print('数据形状：', matrix.shape)
# print('matrix[0] = \n', matrix[0])
# pyg_data = read_dat('../../data/UCINet/KapfererTailorShop/raw/kaptail.dat')