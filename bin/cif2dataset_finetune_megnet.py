# -*- coding: utf-8 -*-
# @Author : liang
# @File : cif2dataset_finetune_megnet.py

import gzip
import sys, os

import argparse, copy
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm

from utils.cryutils import atoms_to_str, atoms_to_point
# from utils import algorithm
import torch
import itertools
import gzip, pickle
from jarvis.db.figshare import data as jdata
from jarvis.core.atoms import Atoms
from torch_geometric.data import Data, InMemoryDataset
from typing import Optional, Union, List, Tuple
from collections import defaultdict
from jarvis.core.specie import chem_data, get_node_attributes
from torch_geometric.data import Data, DataLoader

def convert(args):
    ############################megnet######################################
    dataset = jdata(args.dataset)
    atoms_list = []
    id_list = []
    graphs_list = []
    e_form_list = []
    gap_pbe_list = []

    for atoms in tqdm(dataset):

        structure = Atoms.from_dict(atoms["atoms"])
        structure_ase = structure.ase_converter()

        atoms_list.append(atoms_to_str(structure_ase, tagged=1))
        id_list.append(atoms['id'])

        e_form_list.append(atoms['e_form'])
        gap_pbe_list.append(atoms['gap pbe'])

        graph = atom_multigraph(structure,
                                     cutoff=args.cutoff,
                                     atom_features=args.atom_features,
                                     neighbor_strategy=args.neighbor_strategy,
                                     max_neighbors=args.max_neighbors,
                                     compute_line_graph=args.compute_line_graph,
                                     use_canonize=args.use_canonize,
                                     use_lattice=args.use_lattice,
                                     use_angle=args.use_angle,
                                     infinite_funcs=args.infinite_funcs,
                                     infinite_params=args.infinite_params
                                     )
        graphs_list.append(graph)

    # e_form_list = (np.array(e_form_list) - np.mean(e_form_list)) / np.std(e_form_list)
    # gap_pbe_list = (np.array(gap_pbe_list) - np.mean(gap_pbe_list)) / np.std(gap_pbe_list)

    torch.save({'id': id_list,
                'atom_txt': atoms_list,
                'graph_input': graphs_list,
                'e_form': e_form_list,
                'gap pbe': gap_pbe_list,
                },
               args.save_path + 'megnet_processed.pt')

def atom_multigraph(atoms=None,
                    cutoff=4.0,
                    max_neighbors=25,
                    neighbor_strategy="k-nearest",
                    atom_features="cgcnn",
                    max_attempts=3,
                    compute_line_graph: bool = True,
                    use_canonize: bool = False,
                    use_lattice: bool = False,
                    use_angle: bool = False,
                    target=None,
                    infinite_funcs=None,
                    infinite_params=None
                    ):

    # build up atom attribute tensor
    sps_features = []
    for ii, s in enumerate(atoms.elements):
        # atom features
        atom_feats  = list(get_node_attributes(s, atom_features=atom_features))
        sps_features.append(atom_feats)

    sps_features = np.array(sps_features)
    node_features = torch.tensor(sps_features).type(
        torch.get_default_dtype()
    )

    # u = torch.arange(0, node_features.size(0), 1).unsqueeze(1).repeat((1, node_features.size(0))).flatten().long()
    # v = torch.arange(0, node_features.size(0), 1).unsqueeze(0).repeat((node_features.size(0), 1)).flatten().long()
    # edge_index = torch.stack([u, v])

    # cgcnn node attr
    z = torch.tensor(
        _get_attribute_lookup()[node_features.type(torch.IntTensor).squeeze()]
    ).type(torch.FloatTensor)
    # print(z.size())
    if z.dim() == 1: z = z.unsqueeze(0)
    node_features = z

    lattice_mat = atoms.lattice_mat.astype(dtype=np.double)

    edges, a, b, c = nearest_neighbor_edges_submit(
        atoms=atoms,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        id=id,
        use_canonize=use_canonize,
        use_lattice=use_lattice,
        use_angle=use_angle,
    )
    u, v, edge_attr, l, edge_nei, atom_lat = build_undirected_edgedata(atoms, edges, a, b, c)

    edge_index = torch.cat((u.unsqueeze(0), v.unsqueeze(0)), dim=0).long()
    edge_nei_angle = bond_cosine(edge_nei, edge_attr.unsqueeze(1).repeat(1, 3, 1))
    edge_nei_len = -0.75 / torch.norm(edge_nei, dim=-1)

    graph = Data(x=node_features, edge_attr=edge_attr, edge_index=edge_index,
                 edge_nei_angle=edge_nei_angle, edge_nei_len=edge_nei_len)

    return graph


def bond_cosine(r1, r2):
    bond_cosine = torch.sum(r1 * r2, dim=-1) / (
        torch.norm(r1, dim=-1) * torch.norm(r2, dim=-1)
    )
    bond_cosine = torch.clamp(bond_cosine, -1, 1)
    return bond_cosine

def build_undirected_edgedata(
    atoms=None,
    edges={},
    a=None,
    b=None,
    c=None,
):
    """Build undirected se3_graph data from edge set.

    edges: dictionary mapping (src_id, dst_id) to set of dst_image
    r: cartesian displacement vector from src -> dst
    """
    # second pass: construct *undirected* se3_graph
    # import pprint
    u, v, r, l, nei, angle, atom_lat = [], [], [], [], [], [], []
    v1, v2, v3 = atoms.lattice.cart_coords(a), atoms.lattice.cart_coords(b), atoms.lattice.cart_coords(c)
    # atom_lat.append([v1, v2, v3, -v1, -v2, -v3])
    atom_lat.append([v1, v2, v3])
    for (src_id, dst_id), images in edges.items():

        for dst_image in images:
            # fractional coordinate for periodic image of dst
            dst_coord = atoms.frac_coords[dst_id] + dst_image
            # cartesian displacement vector pointing from src -> dst
            d = atoms.lattice.cart_coords(
                dst_coord - atoms.frac_coords[src_id]
            )
            for uu, vv, dd in [(src_id, dst_id, d), (dst_id, src_id, -d)]:
                u.append(uu)
                v.append(vv)
                r.append(dd)
                # nei.append([v1, v2, v3, -v1, -v2, -v3])
                nei.append([v1, v2, v3])
                # angle.append([compute_bond_cosine(dd, v1), compute_bond_cosine(dd, v2), compute_bond_cosine(dd, v3)])

    u = torch.tensor(u)
    v = torch.tensor(v)
    r = torch.tensor(np.array(r)).type(torch.get_default_dtype())
    l = torch.tensor(l).type(torch.int)
    nei = torch.tensor(np.array(nei)).type(torch.get_default_dtype())
    atom_lat = torch.tensor(np.array(atom_lat)).type(torch.get_default_dtype())
    # nei_angles = torch.tensor(angle).type(torch.get_default_dtype())
    return u, v, r, l, nei, atom_lat


def nearest_neighbor_edges_submit(
        atoms=None,
        cutoff=4.0,
        max_neighbors=25,
        id=None,
        use_canonize=False,
        use_lattice=False,
        use_angle=False,
):
    """Construct k-NN edge list."""
    # returns List[List[Tuple[site, distance, index, image]]]
    lat = atoms.lattice
    all_neighbors_now = atoms.get_all_neighbors(r=cutoff)
    min_nbrs = min(len(neighborlist) for neighborlist in all_neighbors_now)

    attempt = 0
    if min_nbrs < max_neighbors:
        lat = atoms.lattice
        if cutoff < max(lat.a, lat.b, lat.c):
            r_cut = max(lat.a, lat.b, lat.c)
        else:
            r_cut = 2 * cutoff
        attempt += 1
        return nearest_neighbor_edges_submit(
            atoms=atoms,
            use_canonize=use_canonize,
            cutoff=r_cut,
            max_neighbors=max_neighbors,
            id=id,
            use_lattice=use_lattice,
        )

    edges = defaultdict(set)
    # lattice correction process
    r_cut = max(lat.a, lat.b, lat.c) + 1e-2
    all_neighbors = atoms.get_all_neighbors(r=r_cut)
    neighborlist = all_neighbors[0]
    neighborlist = sorted(neighborlist, key=lambda x: x[2])
    ids = np.array([nbr[1] for nbr in neighborlist])
    images = np.array([nbr[3] for nbr in neighborlist])
    images = images[ids == 0]
    lat1 = images[0]
    # finding lat2
    start = 1
    for i in range(start, len(images)):
        lat2 = images[i]
        if not same_line(lat1, lat2):
            start = i
            break
    # finding lat3
    for i in range(start, len(images)):
        lat3 = images[i]
        if not same_plane(lat1, lat2, lat3):
            break
    # find the invariant corner
    if angle_from_array(lat1, lat2, lat.matrix) > 90.0:
        lat2 = - lat2
    if angle_from_array(lat1, lat3, lat.matrix) > 90.0:
        lat3 = - lat3
    # find the invariant coord system
    if not correct_coord_sys(lat1, lat2, lat3, lat.matrix):
        lat1 = - lat1
        lat2 = - lat2
        lat3 = - lat3

    # if not correct_coord_sys(lat1, lat2, lat3, lat.matrix):
    #     print(lat1, lat2, lat3)
    # lattice correction end
    for site_idx, neighborlist in enumerate(all_neighbors_now):

        # sort on distance
        neighborlist = sorted(neighborlist, key=lambda x: x[2])
        distances = np.array([nbr[2] for nbr in neighborlist])
        ids = np.array([nbr[1] for nbr in neighborlist])
        images = np.array([nbr[3] for nbr in neighborlist])

        # find the distance to the k-th nearest neighbor
        max_dist = distances[max_neighbors - 1]
        ids = ids[distances <= max_dist]
        images = images[distances <= max_dist]
        distances = distances[distances <= max_dist]
        for dst, image in zip(ids, images):
            src_id, dst_id, src_image, dst_image = canonize_edge(
                site_idx, dst, (0, 0, 0), tuple(image)
            )
            if use_canonize:
                edges[(src_id, dst_id)].add(dst_image)
            else:
                edges[(site_idx, dst)].add(tuple(image))

        if use_lattice:
            edges[(site_idx, site_idx)].add(tuple(lat1))
            edges[(site_idx, site_idx)].add(tuple(lat2))
            edges[(site_idx, site_idx)].add(tuple(lat3))

    return edges, lat1, lat2, lat3

def canonize_edge(
    src_id,
    dst_id,
    src_image,
    dst_image,
):
    """Compute canonical edge representation.

    Sort vertex ids
    shift periodic images so the first vertex is in (0,0,0) image
    """
    # store directed edges src_id <= dst_id
    if dst_id < src_id:
        src_id, dst_id = dst_id, src_id
        src_image, dst_image = dst_image, src_image

    # shift periodic images so that src is in (0,0,0) image
    if not np.array_equal(src_image, (0, 0, 0)):
        shift = src_image
        src_image = tuple(np.subtract(src_image, shift))
        dst_image = tuple(np.subtract(dst_image, shift))

    assert src_image == (0, 0, 0)

    return src_id, dst_id, src_image, dst_image


def angle_from_array(a, b, lattice):
    a_new = np.dot(a, lattice)
    b_new = np.dot(b, lattice)
    assert a_new.shape == a.shape
    value = sum(a_new * b_new)
    length = (sum(a_new ** 2) ** 0.5) * (sum(b_new ** 2) ** 0.5)
    cos = value / length
    angle = np.arccos(cos)
    return angle / np.pi * 180.0

def correct_coord_sys(a, b, c, lattice):
    a_new = np.dot(a, lattice)
    b_new = np.dot(b, lattice)
    c_new = np.dot(c, lattice)
    assert a_new.shape == a.shape
    plane_vec = np.cross(a_new, b_new)
    value = sum(plane_vec * c_new)
    length = (sum(plane_vec ** 2) ** 0.5) * (sum(c_new ** 2) ** 0.5)
    cos = value / length
    angle = np.arccos(cos)
    return (angle / np.pi * 180.0 <= 90.0)

def same_line(a, b):
    a_new = a / (sum(a ** 2) ** 0.5)
    b_new = b / (sum(b ** 2) ** 0.5)
    flag = False
    if abs(sum(a_new * b_new) - 1.0) < 1e-5:
        flag = True
    elif abs(sum(a_new * b_new) + 1.0) < 1e-5:
        flag = True
    else:
        flag = False
    return flag

def same_plane(a, b, c):
    flag = False
    if abs(np.dot(np.cross(a, b), c)) < 1e-5:
        flag = True
    return flag

def _get_attribute_lookup(atom_features: str = "cgcnn"):
    """Build a lookup array indexed by atomic number."""
    max_z = max(v["Z"] for v in chem_data.values())

    # get feature shape (referencing Carbon)
    template = get_node_attributes("C", atom_features)

    features = np.zeros((1 + max_z, len(template)))

    for element, v in chem_data.items():
        z = v["Z"]
        x = get_node_attributes(element, atom_features)

        if x is not None:
            features[z, :] = x

    return features

def parse_args():
    parser = argparse.ArgumentParser(description='convert data to string and graph')

    # parser.add_argument('--name', type=str, help='name of dataset', default='train_round_range_6')
    parser.add_argument('--dataset', type=str, help='path to atomistic data to convert',
                        default='megnet')
    parser.add_argument('--save_path', type=str, help='path to atomistic data to convert',
                        default='/home/zl/DGM/dataset/finetune/processed/')
    parser.add_argument('--dst_path', type=str, help='path to data to save', default='')
    parser.add_argument('--data_type', type=str, help='lmdb or ase', default='ase', choices=['ase', 'lmdb'])

    parser.add_argument('--cutoff', type=int, default=4.0)
    parser.add_argument('--atom_features', type=str, default="atomic_number")
    parser.add_argument('--neighbor_strategy', type=str, default='k-nearest')
    parser.add_argument('--max_neighbors', type=int, default=25)
    parser.add_argument('--compute_line_graph', type=bool, default=False)
    parser.add_argument('--use_canonize', type=bool, default=False)
    parser.add_argument('--use_lattice', type=bool, default=False)
    parser.add_argument('--use_angle', type=bool, default=False)
    # parser.add_argument('--infinite_funcs', default=["zeta", "zeta", "exp"])
    # parser.add_argument('--infinite_params', default=[0.5, 3.0, 3.0])

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    convert(args)
    print('convert completed!!!')










