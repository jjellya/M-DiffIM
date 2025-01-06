import os.path as osp
from typing import Callable, Optional

import torch

from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.io import read_npz

from src.utils.load_dat import read_dat




class UCINetDataset(InMemoryDataset):
    r"""The  multi-relational network datasets :obj:`"THURMAN OFFICE"`, :obj:`"KAPFERER TAILOR SHOP"` and
        :obj:`"ZACHARY KARATE CLUB"` from the `"UCINet IV Datasets"
        <http://vlado.fmf.uni-lj.si/pub/networks/data/ucinet/ucidata.htm>`_ Network Data Repository,

            :obj:`"THURMAN OFFICE"



            :obj:`"ZACHARY KARATE CLUB"
                DESCRIPTION Two 34×34 matrices.
                    `ZACHE` symmetric, binary.

                    `ZACHC` symmetric, valued.

                    Attention: In the UnDiGraph Visualization, the edge are only 78 = 156 / 2.

                    Every node is labeled by one of four classes obtained via modularity-based
                    clustering, following the `《Semi-supervised Classification with Graph
                    Convolutional Networks》 <https://arxiv.org/abs/1609.02907>`_ paper.
                    Training is based on a single labeled example per class, *i.e.* a total
                    number of 4 labeled nodes.

                BACKGROUND
                    These are data collected from the members of a university karate club by Wayne Zachary.
                    The ZACHE matrix represents the presence or absence of ties among the members of the club;
                    the ZACHC matrix indicates the relative strength of the associations
                    (number of situations in and outside the club in which interactions occurred).

                REFERENCE
                    Zachary W. (1977). An information flow model for conflict and fission in small groups. Journal of Anthropological Research, 33, 452-473.
            Nodes represent documents and edges represent citation links.
            Training, validation and test splits are given by binary masks.

        Args:
            root (str): Root directory where the dataset should be saved.
            name (str): The name of the dataset (:obj:`"ThurmanOffice"`, :obj:`"KapfererTailorShop"`,
                :obj:`"KarateClub"`).
            transform (callable, optional): A function/transform that takes in an
                :obj:`torch_geometric.data.Data` object and returns a transformed
                version. The data object will be transformed before every access.
                (default: :obj:`None`)
            pre_transform (callable, optional): A function/transform that takes in
                an :obj:`torch_geometric.data.Data` object and returns a
                transformed version. The data object will be transformed before
                being saved to disk. (default: :obj:`None`)
            force_reload (bool, optional): Whether to re-process the dataset.
                (default: :obj:`False`)

        **STATS:**

        .. list-table::
            :widths: 10 10 10 10 10 10
            :header-rows: 1

            * - Name
              - #layers
              - #nodes
              - #edges
              - #features
              - #classes
            * - Thurman Office
              - 2
              - 15
              - 10,556
              - 1,433
              - 7
            * - KAPFERER TAILOR SHOP
              - 4
              - 39
              - 1,018
              - 0
              - 0
            * - KarateClub
              - 2
              - 34
              - 156
              - 34
              - 4
        """
    # url = 'https://github.com/kimiyoung/planetoid/raw/master/data'
    # geom_gcn_url = ('https://raw.githubusercontent.com/graphdml-uiuc-jlu/'
    #                 'geom-gcn/master')

    def __init__(self, root: str, name: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name

        # x = torch.eye(y.size(0), dtype=torch.float)# 创建一个对角矩阵，‌其中对角线上的元素为1，‌其余元素为0

        assert name.lower() in ['thurman_office', 'kapferer_tailor_shop', 'karate_club']
        if name.lower() == 'thurman_office':
            self.name = 'ThurmanOffice'
        elif name.lower() == 'kapferer_tailor_shop':
            self.name = 'KapfererTailorShop'
        else:
            self.name = 'KarateClub'

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> str:
        # return f'ms_academic_{self.name[:3].lower()}.npz'
        return f'{self.name.lower()}.dat'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        download_url(self.url + self.raw_file_names, self.raw_dir)

    def process(self):
        print('self.raw_paths[0]', self.raw_paths[0])  # TODO Test output of raw_paths[0]
        data = read_dat(self.raw_paths[0])
        # data = read_npz(self.raw_paths[0])
        data = data if self.pre_transform is None else self.pre_transform(data)
        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}{self.name}()'


