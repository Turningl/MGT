# -*- coding: utf-8 -*-
# @Author : liang
# @File : cryutils.py


from math import ceil, sqrt, cos, sin, radians, acos
from ase import Atoms
from ase.visualize import view
from ase.geometry import get_distances
import numpy as np
from ast import literal_eval
from pymatgen.core import Structure, Element, Lattice


def type_check(string):
    try:
        literal_eval(string)
    except ValueError:
        return 'str'
    return 'num'


def from_parameters_to_cell(a, b, c, alpha, beta, gamma):
    """
    Create a Lattice using parameters a, b and c as box lengths
    and corresponging angles alpha, beta, gamma

    :param a: (float): *a* lattice length.
    :param b: (float): *b* lattice length.
    :param c: (float): *c* lattice length.
    :param alpha: (float): *alpha* angle in degrees.
    :param beta: (float): *beta* angle in degrees.
    :param gamma: (float): *gamma* angle in degrees.

    :rtype : (Lattice) object
    """

    alpha_radians = radians(alpha)
    beta_radians = radians(beta)
    gamma_radians = radians(gamma)
    val = (cos(alpha_radians) * cos(beta_radians) - cos(gamma_radians)) / (sin(alpha_radians) * sin(beta_radians))
    # Sometimes rounding errors result in values slightly > 1.
    val = val if abs(val) <= 1 else val / abs(val)
    gamma_star = acos(val)
    cell = np.zeros((3, 3))
    cell[0] = [a * sin(beta_radians), 0.0, a * cos(beta_radians)]
    cell[1] = [-b * sin(alpha_radians) * cos(gamma_star),
               b * sin(alpha_radians) * sin(gamma_star),
               b * cos(alpha_radians)]
    cell[2] = [0.0, 0.0, float(c)]
    return cell


def lattice_validity(lattice_str):
    if len(lattice_str) != 6:
        return False
    try:
        lattice = [float(i) * 180 for i in lattice_str]
    except ValueError:
        return False
    if 0. in lattice:
        return False
    return True


def str_to_atoms(atoms_str, lat_idx=1, early_stop=True, skip_fail=False, return_str=False):
    """
    Convert string to atoms

    Args:
      atoms_str : string represenation of atoms
      lat_idx : starting index of string represenation, if model generates string with '<bos>' token, lat_idx = 1, else 0
      early_stop : if True, the conversion is stop when atomic overlapping is detected
      skip_fail : if True, skip the overlapped atoms
      return_str : if True, return string representation of atoms
    """

    atoms_str_return = []
    atoms_str = atoms_str.split(' ')
    struct_checksum = True
    valid_gen_checksum = True

    # ads = atoms_str[0]
    lattice_str = atoms_str[lat_idx:lat_idx + 6]

    # Lattice check
    if not lattice_validity(lattice_str):
        valid_gen_checksum = False
        if return_str:
            return None, struct_checksum, valid_gen_checksum, None
        else:
            return None, struct_checksum, valid_gen_checksum

    a, b, c, alp, bet, gam = [float(i) * 180 for i in lattice_str]
    cell = from_parameters_to_cell(a, b, c, alp, bet, gam)

    atoms = Atoms(
        cell=cell,
        pbc=True
    )

    coordinates = atoms_str[lat_idx + 6:]

    # Add atoms to structure
    atoms_str_return.extend(lattice_str)
    for i in range(int(len(coordinates) / 4)):
        symbol = coordinates[i * 4]

        try:
            Element(symbol)
        except:
            valid_gen_checksum = False
            break

        try:
            x, y, z = [float(i) for i in coordinates[i * 4 + 1:i * 4 + 4]]
        except:
            valid_gen_checksum = False
            break

        scaled_pos = np.array([x, y, z])
        pos = np.dot(scaled_pos, cell)

        if i != 0:
            if any(get_distances(pos, atoms.positions, cell=cell, pbc=True)[1][0] - 0.5 < 0):
                if early_stop:
                    struct_checksum = False
                    break
                else:
                    if skip_fail:
                        continue
                    else:
                        struct_checksum = False
                        pass

        atoms += Atoms(symbol, cell=cell, scaled_positions=[scaled_pos])
        atoms_str_return.extend(coordinates[i * 4:i * 4 + 4])

    if return_str:
        return atoms, struct_checksum, valid_gen_checksum, ' '.join(atoms_str_return)
    else:
        return atoms, struct_checksum, valid_gen_checksum


def atoms_to_str(atoms, tagged=True, round_range=4):
    lattice = [f'{np.round(i, round_range):.4f}' for i in atoms.cell.cellpar() / 180]
    # elements = []
    surf = []
    bulk = []
    ads = []

    if tagged:
        for atom, pos in zip(atoms, atoms.get_scaled_positions(wrap=True)):
            tag = atom.tag

        #     if tag == 0:
        #         bulk.append(atom.symbol)
        #         bulk.extend([f'{np.round(i, round_range):.4f}' for i in pos])
        #     elif tag == 1:
        #         surf.append(atom.symbol)
        #         surf.extend([f'{np.round(i, round_range):.4f}' for i in pos])
        #
        # return ' '.join(lattice + bulk + surf)


            if tag == 0:
                bulk.append(atom.symbol)
                bulk.extend([f'{np.round(i, round_range):.4f}' for i in pos])
            elif tag == 1:
                surf.append(atom.symbol)
                surf.extend([f'{np.round(i, round_range):.4f}' for i in pos])
            elif tag == 2:
                ads.append(atom.symbol)
                ads.extend([f'{np.round(i, round_range):.4f}' for i in pos])
        return ' '.join(lattice + bulk + surf + ads)

    else:
        for atom, pos in zip(atoms, atoms.get_scaled_positions(wrap=True)):
            surf.append(atom.symbol)
            surf.extend([f'{np.round(i, round_range):.4f}' for i in pos])
        return ' '.join(lattice + surf)


def atoms_to_point(atoms, tagged=True, round_range=6):
    lattice = []

    for i in atoms.cell.cellpar() / 180 :
        lattice.extend(list(f'{np.round(i, round_range):.6f}'))

    surf = []
    bulk = []
    ads = []

    if tagged:
        for atom, pos in zip(atoms, atoms.get_scaled_positions(wrap=True)):
            tag = atom.tag

            if tag == 0:
                bulk.append(atom.symbol)
                bulk_coordinate = [f'{np.round(i, round_range):.6f}' for i in pos]
                bulk.extend([sym for coord in bulk_coordinate for sym in coord])

            elif tag == 1:
                surf.append(atom.symbol)
                surf_coordinate = [f'{np.round(i, round_range):.6f}' for i in pos]
                surf.extend([sym for coord in surf_coordinate for sym in coord])

            elif tag == 2:
                ads.append(atom.symbol)
                ads_coordinate = [f'{np.round(i, round_range):.6f}' for i in pos]
                ads.extend([sym for coord in ads_coordinate for sym in coord])

        return ' '.join(lattice + bulk + surf + ads)

    else:
        for atom, pos in zip(atoms, atoms.get_scaled_positions(wrap=True)):
            surf.append(atom.symbol)
            surf.extend([f'{np.round(i, round_range):.6f}' for i in pos])
        return ' '.join(lattice + surf)


def atom_to_pmg(cif_path):
    parser = Structure.from_file(cif_path, site_tolerance = 1e-4, occupancy_tolerance=1.0)

    structure = parser.get_reduced_structure()

    canonical_structure = Structure(
        lattice=Lattice.from_parameters(*structure.lattice.parameters),
        species=structure.species,
        coords=structure.frac_coords,
        coords_are_cartesian=False,
    )

    lattice_parameters = canonical_structure.lattice.parameters

    lengths = lattice_parameters[:3]
    angles = lattice_parameters[3:]

    matrix = lattice_params_to_matrix(*lengths, *angles)

    assert np.allclose(matrix, canonical_structure.lattice.matrix,
                       atol=1e-6), "An error occurred. Please check the source data!"

    return canonical_structure


def abs_cap(val, max_abs_val=1):
    """
    Returns the value with its absolute value capped at max_abs_val.
    Particularly useful in passing values to trignometric functions where
    numerical errors may result in an argument > 1 being passed in.
    https://github.com/materialsproject/pymatgen/blob/b789d74639aa851d7e5ee427a765d9fd5a8d1079/pymatgen/util/num.py#L15
    Args:
        val (float): Input value.
        max_abs_val (float): The maximum absolute value for val. Defaults to 1.
    Returns:
        val if abs(val) < 1 else sign of val * max_abs_val.
    """
    return max(min(val, max_abs_val), -max_abs_val)

def lattice_params_to_matrix(a, b, c, alpha, beta, gamma):
    """Converts lattice from abc, angles to matrix.
    https://github.com/materialsproject/pymatgen/blob/b789d74639aa851d7e5ee427a765d9fd5a8d1079/pymatgen/core/lattice.py#L311
    """
    angles_r = np.radians([alpha, beta, gamma])
    cos_alpha, cos_beta, cos_gamma = np.cos(angles_r)
    sin_alpha, sin_beta, sin_gamma = np.sin(angles_r)

    val = (cos_alpha * cos_beta - cos_gamma) / (sin_alpha * sin_beta)
    # Sometimes rounding errors result in values slightly > 1.
    val = abs_cap(val)
    gamma_star = np.arccos(val)

    vector_a = [a * sin_beta, 0.0, a * cos_beta]
    vector_b = [
        -b * sin_alpha * np.cos(gamma_star),
        b * sin_alpha * np.sin(gamma_star),
        b * cos_alpha,
    ]
    vector_c = [0.0, 0.0, float(c)]

    return np.array([vector_a, vector_b, vector_c])