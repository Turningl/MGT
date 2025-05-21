# -*- coding: utf-8 -*-
# @Author : liang
# @File : dataset.py


import math, os, random
import torch
import numpy as np
from numpy import dtype
from torch.utils.data import Dataset, DataLoader
from jarvis.core.atoms import Atoms
from tqdm import tqdm
import pandas as pd
from torch.utils.data.sampler import SubsetRandomSampler
from transformers import PreTrainedTokenizerFast
from jarvis.core.graphs import nearest_neighbor_edges
from torch_geometric.data.batch import Batch
from torch_geometric.data import Data
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor, PairTensor


class CrystalDataset(Dataset):
    def __init__(self, graph_input, target):

        # self.atom_text = atom_text
        self.graph_input = graph_input
        self.target = target

    def __len__(self):
        return len(self.graph_input)

    def __getitem__(self, item):

        # text_prompt = self.atom_text[item]
        graph_prompt = self.graph_input[item]

        if self.target is not None:
            label = self.target[item]

            if isinstance(graph_prompt, Data): graph_prompt = (graph_prompt, graph_prompt)

            (se3_graph_prompt,
             so3_graph_prompt) = graph_prompt

            return se3_graph_prompt, so3_graph_prompt, label

        else:
            if isinstance(graph_prompt, Data):
                graph_prompt = (graph_prompt, graph_prompt)
            else:
                raise TypeError("graph_prompt must be an instance of torch_geometric.data.Data")

            # grad
            edge_nei_angle_noise = graph_prompt[0].edge_nei_angle_noise.requires_grad_()
            origin_angle = graph_prompt[0].edge_nei_angle + 0 * edge_nei_angle_noise.sum()

            edge_attr_noise = graph_prompt[1].edge_attr_noise.requires_grad_()
            origin_edge_attr = graph_prompt[1].edge_attr + 0 * edge_attr_noise.sum()

            # se3 graph
            se3_graph_prompt = (
                Data(x=graph_prompt[0].x,
                     edge_index=graph_prompt[0].edge_index,
                     edge_attr=graph_prompt[0].edge_attr,
                     edge_nei_angle=edge_nei_angle_noise,
                     edge_nei_len=graph_prompt[0].edge_nei_len,

                     origin_angle=origin_angle
                     ))

            # so3 graph
            so3_graph_prompt = (
                Data(x=graph_prompt[1].x,
                     edge_index=graph_prompt[1].edge_index,
                     edge_attr=edge_attr_noise,

                     origin_edge_attr=origin_edge_attr,
                     ))

            return se3_graph_prompt, so3_graph_prompt

    # def build_undirected_edgedata(self,
    #         atoms=None, edges={},
    # ):
    #     """Build undirected graph data from edge set.
    #
    #     edges: dictionary mapping (src_id, dst_id) to set of dst_image
    #     r: cartesian displacement vector from src -> dst
    #     """
    #     # second pass: construct *undirected* graph
    #     # import pprint
    #
    #     frac_coords = torch.tensor(atoms.frac_coords, dtype=torch.float32, requires_grad=True)
    #     latt = torch.tensor(atoms.lattice_mat, dtype=torch.float32, requires_grad=True)
    #
    #     cart_coords = torch.matmul(frac_coords, latt)
    #
    #     u, v, r_list = [], [], []
    #     for (src_id, dst_id), images in edges.items():
    #
    #         for dst_image in images:
    #             dst_image_tensor = torch.tensor(dst_image, dtype=torch.float32)
    #
    #             # fractional coordinate for periodic image of dst
    #             dst_coord = frac_coords[dst_id] + dst_image_tensor
    #             # cartesian displacement vector pointing from src -> dst
    #
    #             diff = dst_coord - frac_coords[src_id]
    #
    #             d = torch.matmul(diff.unsqueeze(0), latt).squeeze(0)
    #
    #             # if np.linalg.norm(d)!=0:
    #             # print ('jv',dst_image,d)
    #             # add edges for both directions
    #             for uu, vv, dd in [(src_id, dst_id, d), (dst_id, src_id, -d)]:
    #                 u.append(uu)
    #                 v.append(vv)
    #                 r_list.append(dd)
    #
    #     u = torch.tensor(u, dtype=torch.long)
    #     v = torch.tensor(v, dtype=torch.long)
    #     r = torch.stack(r_list)
    #     r = r + 0 * cart_coords.sum()
    #
    #     # a = are_on_same_graph(r, cart_coords)
    #
    #     # print("cart_coords grad_fn:", cart_coords.grad_fn)
    #     # print("r grad_fn:", r.grad_fn)
    #     # print(a)
    #
    #     return u, v, r, cart_coords

    @staticmethod
    def collate_fn(samples):
        """Dataloader helper to batch graphs cross `samples`."""

        if len(samples[0]) == 3:
            se3_graph_prompt, so3_graph_prompt, labels = map(list, zip(*samples))

            # text_prompt = torch.tensor(text_prompt, dtype=torch.long)
            # text_mask = torch.tensor(text_mask, dtype=torch.long)
            se3_batched_graph = Batch.from_data_list(se3_graph_prompt)
            so3_batched_graph = Batch.from_data_list(so3_graph_prompt)
            labels = torch.tensor(labels, dtype=torch.float32)

            return se3_batched_graph, so3_batched_graph, labels

        else:
            se3_graph_prompt, so3_graph_prompt = map(list, zip(*samples))
            se3_batched_graph = Batch.from_data_list(se3_graph_prompt)
            so3_batched_graph = Batch.from_data_list(so3_graph_prompt)
            # labels = torch.tensor(labels, dtype=torch.float32)

            return se3_batched_graph, so3_batched_graph


class CrystalDataLoader:
    def __init__(self,
                 root: str =None,
                 name: str=None,
                 target=None,
                 batch_size=None,
                 num_workers=None,
                 train_size=None,
                 valid_size=None,
                 test_size=None,
                 normalize=True,
                 random_seed=123,
                 idx_save_file=None,
                 ):
        super(CrystalDataLoader, self).__init__()

        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_size = train_size
        self.val_size = valid_size
        self.test_size = test_size
        self.target = target
        self.name = name
        self.normalize = normalize
        self.random_seed = random_seed
        self.idx_save_file = idx_save_file

        self.std = 1.0  # std
        self.mean = 0.0  # mean
        self.target_array = None
        self.target_list = None
        self.strain_tensor = None

    def get_data_loaders(self):
        if self.target:
            self.dataset = torch.load(os.path.join(self.root, self.name))

            self.target_list = []
            self.graph_input = []

            if self.target in ['adsorption energy label']:
                self.strain_tensor = []

                for i in range(len(self.dataset[self.target])):
                    if (
                            self.dataset[self.target][i] is not None
                            and self.dataset[self.target][i] != "na"
                            and not math.isnan(self.dataset[self.target][i])
                    ):

                        self.graph_input.append(self.dataset['graph_input'][i])
                        self.target_list.append(self.dataset[self.target][i])
                        self.strain_tensor.append(self.dataset['strain tensor'][i])

            else:
                for i in range(len(self.dataset[self.target])):
                    if (
                            self.dataset[self.target][i] is not None
                            and self.dataset[self.target][i] != "na"
                            and not math.isnan(self.dataset[self.target][i])
                    ):
                        # self.atom_text.append(self.dataset['atom_txt'][i])
                        self.graph_input.append(self.dataset['graph_input'][i])
                        self.target_list.append(self.dataset[self.target][i])

            self.target_array = np.array(self.target_list)

        else:
            print('Loading pre-training dataset, please wait!!!')
            self.dataset = torch.load(os.path.join(self.root, self.name))

            # self.atom_text = self.dataset['atom_txt']
            self.graph_input = self.dataset['graph_input']

            print('Loading pre-training dataset with success !!!')

        dataset = CrystalDataset(graph_input=self.graph_input,
                                 target=self.target_list)

        train_loader, valid_loader, test_loader = self.get_train_validation_data_loaders(dataset)

        return train_loader, valid_loader, test_loader

    def get_train_validation_data_loaders(self, dataset):
        # obtain training indices that will be used for validation

        idx_save_path = (os.path.join(self.root,
                                      self.target + self.idx_save_file + str(self.random_seed) + '.pt')
                         if self.target
                         else os.path.join(self.root, self.idx_save_file + '.pt'))

        if os.path.exists(idx_save_path):
            ids_train_val_test = torch.load(idx_save_path)

            train_idx = ids_train_val_test['train_idx']
            valid_idx = ids_train_val_test['valid_idx']
            test_idx = ids_train_val_test['test_idx']

            print(f'Load from saved {self.target} train_val_test idx !!!')

        else:
            num_data = len(dataset)

            random.seed(self.random_seed)
            np.random.seed(self.random_seed)
            torch.manual_seed(self.random_seed)
            torch.cuda.manual_seed(self.random_seed)
            torch.cuda.manual_seed_all(self.random_seed)

            indices = list(range(num_data))

            if self.target not in ['bulk modulus', 'shear modulus',
                                   'adsorption energy label']:
                random.shuffle(indices)

            # shuffled = dataset.shuffle()

            if self.train_size < 1.0:
                self.test_size = int(num_data * self.test_size)

                train_idx = indices[ :-(self.test_size + self.test_size)]
                valid_idx = indices[-(self.test_size + self.test_size): -self.test_size]
                test_idx = indices[-self.test_size: ]

            else:

                train_idx = indices[ :self.train_size]
                valid_idx = indices[-(self.val_size + self.test_size): -self.test_size]
                test_idx = indices[-self.test_size: ]

            assert self.test_size - 1 <= len(test_idx) <= self.test_size + 1, 'split is wrong, please check!'

            ids_train_val_test = {
                'train_idx': train_idx,
                'valid_idx': valid_idx,
                'test_idx': test_idx
            }

            print(f'Save {self.target} train_val_test idx !!!')
            torch.save(ids_train_val_test, idx_save_path)

        if self.target and self.normalize:

            if self.target not in ['bulk modulus', 'shear modulus',
                                   'adsorption energy label']:
                print(f'Normalized {self.target} !!!')

                self.train_target_array = self.target_array[train_idx]
                self.mean, self.std = np.mean(self.train_target_array), np.std(self.train_target_array)

                self.target_array = (self.target_array - self.mean) / self.std

            else:
                print(f'Not normalized with {self.target} !!!')

            dataset.target = self.target_array.tolist()

        # define samplers for obtaining training, validation and testing batches

        (train_sampler,
         valid_sampler,
         test_sampler) = (
            SubsetRandomSampler(train_idx),
            SubsetRandomSampler(valid_idx),
            SubsetRandomSampler(test_idx)
        )

        train_loader = DataLoader(dataset,
                                  batch_size=self.batch_size,
                                  sampler=train_sampler,
                                  collate_fn=CrystalDataset.collate_fn,
                                  num_workers=self.num_workers,
                                  drop_last=True,
                                  shuffle=False)

        valid_loader = DataLoader(dataset,
                                  batch_size=self.batch_size,
                                  sampler=valid_sampler,
                                  collate_fn=CrystalDataset.collate_fn,
                                  num_workers=self.num_workers,
                                  drop_last=True,
                                  shuffle=False)

        test_loader = DataLoader(dataset,
                                 batch_size=self.batch_size,
                                 sampler=test_sampler,
                                 collate_fn=CrystalDataset.collate_fn,
                                 num_workers=self.num_workers,
                                 drop_last=True,
                                 shuffle=False)

        return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    from torch_geometric.data import Data, Batch

    # dataset = CrystalDataset(root='/home/zl/CrystalCLR2/dataset/dataset',
    #                          name='megnet',
    #                          target='e_form',
    #                          cutoff=8.0,
    #                          atom_features="atomic_number",
    #                          max_neighbors=12,
    #                          compute_line_graph=False,
    #                          use_canonize=False,
    #                          use_lattice=False,
    #                          use_angle=False,
    #                          normalize=True,
    #                          transform=None,
    #                          pre_transform=None,
    #                          pre_filter=None)

    max_len = 1280

    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        f'/home/zl/CrystalCLR6/tokenizer',
        max_len=max_len
    )

    datawrapper = CrystalDataLoader(root='/home/zl/CrystalCLR6/dataset/pretrained/processed/',
                                    name='dft_3d_processed.pkl',
                                    target='ehull',
                                    train_size=0.8,
                                    valid_size=0.1,
                                    test_size=0.1,
                                    batch_size=32,
                                    num_workers=0,
                                    normalize=True,
                                    idx_save_file = '_ids_train_val_test.pt',
                                    )

    train_loader, val_loader, test_loader = datawrapper.get_data_loaders()

    # test_dataloader = DataLoader(test_dataset, batch_size=32)

    for i, data in tqdm(enumerate(train_loader)):
        print(data)