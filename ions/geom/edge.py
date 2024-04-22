import itertools
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
from ase.neighborlist import neighbor_list
from ase import Atoms
from ase.io import read
from ase.cell import Cell
from ase.build import make_supercell
from ase.spacegroup import get_spacegroup
from spglib import standardize_cell
from spglib import get_symmetry_dataset
from ions.utils import lineseg_dists
from ase.build.tools import sort
from .box import Box


class Edge:

    def __init__(self, atoms, source, target, offset, verbose = False):

        """
        supposed that any offset component lies within [-1, 1] range
        i.e. edge is collected within 3x3x3 supercell

        """
        self.verbose = verbose
        self.atoms = atoms.copy()
        self.source = source
        self.target = target
        self.offset = np.array(offset)
        self.cell = self.atoms.cell
        self.p1 = self.atoms.positions[self.source]
        self.p2 = np.dot(self.offset, self.cell) + self.atoms.positions[self.target]
        self.wrapped_target = self._wrapped_target()

    @property
    def length(self):
        return np.linalg.norm(self.p1 - self.p2)
    
    #@property
    def _wrapped_target(self):
        p2_scaled = self.cell.scaled_positions(self.p2).round(10) # rounding for a safer wrapping
        p2_scaled[:]%=1.0
        p2_wrapped = self.cell.cartesian_positions(p2_scaled)
        target_specie = self.atoms.numbers[self.target]
        specie_ids = np.argwhere(self.atoms.numbers == target_specie).ravel()
        atoms = self.atoms.copy()
        atoms.wrap()
        tree = cKDTree(atoms.positions[specie_ids])
        distance, index = tree.query(p2_wrapped)
        target = int(specie_ids[index])
        if distance > 1e-8:
            print('target was not properly wrapped', distance)
            raise
        assert self.atoms.numbers[self.target] == self.atoms.numbers[target] # not needed
        return target

    #@property
    def nn(self, r_cut = None):
        if r_cut == None:
            r_cut = 2 * self.length # this is heuristics but should work well
        edge = self.superedge(r_cut, center = True)
        frame_ids = np.array([i for i in range(len(edge.atoms)) if i not in [edge.source, edge.wrapped_target]])
        p = edge.atoms.positions[frame_ids]
        a = edge.atoms.positions[edge.source]
        b = edge.atoms.positions[edge.wrapped_target]
        dd = lineseg_dists(p, a, b)
        nn_id = frame_ids[np.where(dd == dd.min())[0][0]]
        return {'min_dist': dd.min(), 'nn': nn_id, 'nnp': self._project_point_on_edge(p[nn_id])}
    

    def _project_point_on_edge(self, p):
        ap = p-self.p1
        ab = self.p2 - self.p1
        result = self.p1 + np.dot(ap,ab)/np.dot(ab,ab) * ab
        return result
    

    def superedge(self, r_cut, center = True):
        scale = np.ceil(r_cut/abs(np.diag(self.cell)))
        box = Box(self.atoms)
        supercell = box._super_box(scale)
        if center:
            p1 = supercell.positions[self.source]
            p2 = supercell.positions[self.target] + np.dot(self.offset, self.cell)
            d = np.linalg.norm(p1 - p2)
            assert abs(d - self.length) < 1e-10
            centroid = (p1 + p2) / 2
            box_center = np.dot([0.5, 0.5, 0.5], supercell.cell)
            shift = box_center - centroid
            supercell.positions += shift
        offset = self.offset / scale
        return Edge(supercell, self.source, self.target, offset)
    

    @property
    def centroid(self):
        return (self.p1 +  self.p2) / 2
    
    def interpolate(self, spacing = 0.75, n_images = None, n_max = 11, n_min = 5, sort_atoms = True):
        # interpolate between self.p1 and self.p2
        if not n_images:
            n_images = max(n_min, min(int(np.ceil(self.length/spacing) + (1 + np.ceil(self.length/spacing))%2), n_max))
        traj = np.linspace(self.p1, self.p2, n_images)
        images = []
        wrapped_target = self.wrapped_target
        source_mask = np.array([i != self.source for i in range(len(self.atoms))])
        atoms = self.atoms.copy()
        atoms.set_array('freezed', source_mask)

        for p in traj:
            base = atoms.copy()
            base.positions[self.source] = p
            # remove wrapped target
            base = base[[i for i in range(len(base)) if i != wrapped_target]]
            if sort_atoms:
                base = sort(base, base.numbers)
                base = sort(base, base.get_array('freezed'))

            images.append(base)
        
        # check for errors
        source = np.argwhere(images[0].get_array('freezed') == False).ravel()[0]
        for i in range(1, n_images):
            pos_frame = images[0].positions[[i for i in range(len(images)) if i != source]]
            pos_frame_i = images[i].positions[[i for i in range(len(images)) if i != source]]
            assert abs(pos_frame_i - pos_frame).min() < 1e-10
        return images
    



    def __repr__(self):
        #return f"Edge(source={self.source}, target={self.target}, offset={self.offset}), atoms={self.atoms}"
        return f"Edge({self.source},{self.target},{self.offset}, d={round(self.length, 2)}, wrapped_target={self.wrapped_target}')"


    
    