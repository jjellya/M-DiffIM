from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor

import numpy as np

from torch_geometric.typing import OptTensor
from torch_geometric.utils import coalesce, sort_edge_index
from torch_geometric.utils.num_nodes import maybe_num_nodes

MISSING = '???'


def to_undirected(edge_index: Tensor,
                  edge_attr: Union[OptTensor, List[Tensor], str] = MISSING,
                  num_nodes: Optional[int] = None,
                  reduce: str = 'add',
                  ) -> Union[Tensor, Tuple[OptTensor, Tensor], Tuple[Tensor, List[Tensor]]]:
    r"""Converts the graph given by :attr:`edge_index` to an undirected graph
    such that :math:`(j,i) \in \mathcal{E}` for every edge :math:`(i,j) \in
    \mathcal{E}`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor or List[Tensor], optional): Edge weights or multi-
            dimensional edge features.
            If given as a list, will remove duplicates for all its entries.
            (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        reduce (str, optional): The reduce operation to use for merging edge
            features (:obj:`"add"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`,
            :obj:`"mul"`). (default: :obj:`"add"`)

    :rtype: :class:`LongTensor` if :attr:`edge_attr` is not passed, else
        (:class:`LongTensor`, :obj:`Optional[Tensor]` or :obj:`List[Tensor]]`)

    .. warning::

        From :pyg:`PyG >= 2.3.0` onwards, this function will always return a
        tuple whenever :obj:`edge_attr` is passed as an argument (even in case
        it is set to :obj:`None`).
    """
    # Maintain backward compatibility to `to_undirected(edge_index, num_nodes)`
    if isinstance(edge_attr, int):
        edge_attr = MISSING
        num_nodes = edge_attr

    row, col = edge_index[0], edge_index[1]
    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    edge_index = torch.stack([row, col], dim=0)

    if isinstance(edge_attr, Tensor):
        edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
    elif isinstance(edge_attr, (list, tuple)):
        edge_attr = [torch.cat([e, e], dim=0) for e in edge_attr]

    you_have_problem = False
    if you_have_problem:
        # 转换为 NumPy 数组
        edge_index_np = edge_index.numpy()

        # 找到重复项
        unique_edges, counts = np.unique(edge_index_np, axis=1, return_counts=True)
        duplicates = unique_edges[:, counts > 1]

        # 输出重复项
        print("重复的边：", duplicates)
        print("duplicates ' shape", len(duplicates[0]))

    return coalesce(edge_index, edge_attr, num_nodes, reduce)
