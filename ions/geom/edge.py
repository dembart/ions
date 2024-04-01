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


class Edge:


    def __init__(self, atoms, source, target, offset, verbose = False):

        """
        supposed that any offset component lies within [-1, 1] range
        i.e. edge is collected within 3x3x3 supercell

        """
        if target == source:
            if sum(abs(offset)) == 0:
                print('edge with 0 norm')
                raise  
        self.verbose = verbose
        self.atoms = atoms.copy()
        self.source = source
        self.target = target
        self.offset = np.array(offset)
        self.shift = np.where(self.offset < 0, 1, 0)
        self.shifted_offset = self.offset + self.shift
        self.box = self.atoms.cell


    @property
    def distance(self):
        p1 = self.atoms.positions[self.source]
        p2 = np.dot(self.offset, self.box) + self.atoms.positions[self.target]
        d = np.linalg.norm(p1 - p2)
        return d
    


    @property
    def nn(self):
        supercell, translation_to_id = self._super_box([5, 5, 5]) # !test if 3x3x3 works ok
        # pick up source correposning to the translation == self.shift and target with translation == self.shifted_offset
        source_cell_id, target_cell_id = translation_to_id[tuple(self.shift)], translation_to_id[tuple(self.shifted_offset)]
        source = self.source + len(self.atoms) * source_cell_id
        target = self.target + len(self.atoms) * target_cell_id
        #print(source, target)
        frame_ids = np.array([i for i in range(len(supercell)) if i not in [source, target]])
        p = supercell.positions[frame_ids]
        a = supercell.positions[source]
        b = supercell.positions[target]
        dd = self._lineseg_dists(p, a, b)
        nn_id = frame_ids[np.where(dd == dd.min())[0][0]]
        return {'min_dist': dd.min(), 'nn': nn_id}



    def _lineseg_dists(self, p, a, b):
        
        # source:    https://stackoverflow.com/questions/54442057/
        # calculate-the-euclidian-distance-between-an-array-of-points-to-a-line-segment-in/
        # 54442561#54442561 
        
        if np.all(a == b):
            return np.linalg.norm(p - a, axis=1)
        d = np.divide(b - a, np.linalg.norm(b - a))
        s = np.dot(a - p, d)
        t = np.dot(p - b, d)
        h = np.maximum.reduce([s, t, np.zeros(len(p))])
        c = np.cross(p - a, d)
        return np.hypot(h, np.linalg.norm(c, axis = 1))



    def _scale_cell(self, cell, r_cut):

        """ Scaling of the unit cell for the search of neighbors

        Parameters
        ----------

        r_cut: float
            cutoff distance for interaction between tracer ion and framework
            
        cell: ase.atoms.cell
            Unit cell parameters

        Returns
        ----------
        scale: np.array (3, 1)
            scales for the unit cell transformation
        
        """
        # a, b, c, angle(b,c), angle(a,c), angle(a,b)
        a, b, c, alpha, beta, gamma = cell.cellpar(radians = True) 
        scale_a = np.ceil(r_cut/min(a*np.sin(gamma), a*np.sin(beta)))
        scale_b = np.ceil(r_cut/min(b*np.sin(gamma), b*np.sin(beta)))
        scale_c = np.ceil(r_cut/min(c*np.sin(beta), c*np.sin(beta)))
        scale = np.array([scale_a, scale_b, scale_c])
        return scale



    def _super_box(self, scale):
        scales = []
        for s in scale:
            #if -1 in self.offset:
            #    scales.append(np.arange(-s + 1, s-1))
            #else:
            scales.append(np.arange(s))
        translations = tuple(list(itertools.product(
                                scales[0],
                                scales[1],
                                scales[2])))   
        super_p = []
        positions = self.atoms.positions
        self._translations = translations
        arrays = {}
        #translation_ids = []
        transaltion_to_id = dict(zip(translations, np.arange(len(translations))))
        for i, translation in enumerate(translations):
            positions_tr = positions + np.dot(translation, self.box)
            super_p.append(positions_tr)
            #translation_ids.append(i * np.ones(len(positions), dtype = int))
            for key in self.atoms.arrays.keys():
                if key == 'positions':
                    continue
                arr = self.atoms.arrays[key]
                try:
                    arrays[key] = np.hstack([arrays[key], arr])
                except:
                    arrays[key] = arr
        #arrays.update({'translation_ids': np.hstack(translation_ids)})
        super_p = np.vstack(super_p)
        cellpar = self.box.cellpar()
        cellpar[:3] = cellpar[:3] * [scale[0], scale[1], scale[2]]
        supercell = Atoms(numbers = arrays['numbers'], positions = super_p, cell = Cell.fromcellpar(cellpar), pbc = True)
        for key in arrays.keys():
            if key == 'numbers':
                continue
            supercell.set_array(key, arrays[key])
        return supercell, transaltion_to_id
    


    def unwrap(self, cutoff = 10.0):

        """by unwrapping it's meant  that the edge is not crossing the box boundary,
           i.e. we construct the """

        scale = self._scale_cell(self.box, cutoff) # calculate scale from cutoff  
        scale_final = np.where(abs(self.offset) + 1 > scale, abs(self.offset) + 1, scale) # be sure that boundary is not crossed
        if self.verbose:
            print('scale', scale_final)
        supercell, translation_to_id = self._super_box(scale_final) # make _super_box
        # pick up source correposning to the translation == self.shift and target with translation == self.shifted_offset
        source_cell_id, target_cell_id = translation_to_id[tuple(self.shift)], translation_to_id[tuple(self.shifted_offset)]
        source = self.source + len(self.atoms) * source_cell_id
        target = self.target + len(self.atoms) * target_cell_id
        return supercell, source, target



    def interpolate(self, cutoff = 8.0, spacing = 0.75, n_images = None, n_max = 11):

        assert cutoff > 0.0
        scale = self._scale_cell(self.box, cutoff)
        if self.source == self.target:
            scale = np.where(abs(self.offset) + 1 > scale, abs(self.offset) + 1, scale)
        if self.verbose:
            print(scale)
        supercell, translation_to_id = self._super_box(scale) # make _super_box
        p1 = supercell.positions[self.source]
        p2 = supercell.positions[self.target] + np.dot(self.offset, self.box)
        p1p2 = np.linalg.norm(p1 - p2)
        if n_images:
            pass
        else:
            n_images = min(int(np.ceil(p1p2/spacing) + (1 + np.ceil(p1p2/spacing))%2), n_max)
        if self.verbose:
            print(f'spacing {p1p2/n_images}, n_images {n_images}')
        try:
            source_cell_id, target_cell_id = translation_to_id[tuple(self.shift)], translation_to_id[tuple(self.shifted_offset)]
            source = self.source + len(self.atoms) * source_cell_id
            target = self.target + len(self.atoms) * target_cell_id
            assert abs(p1p2 - np.linalg.norm(supercell.positions[source] - supercell.positions[target])) < 1e-9
        except:
            p2_scaled = supercell.cell.scaled_positions( p2)# .round(10) # rounding for a safer wrapping
            p2_scaled[:]%=1.0
            p2_wrapped = supercell.cell.cartesian_positions(p2_scaled)
            target_replicas = [self.target + len(self.atoms) * i for i in range(int(scale[0] * scale[1] * scale[2]))]
            tree = cKDTree(supercell.positions[target_replicas])
            distance, index = tree.query(p2_wrapped)
            assert distance < 1e-10
            source = self.source
            target = int(target_replicas[index])
        images = []
        freezed = np.array([i != source for i in range(len(supercell))])
        supercell.set_array('freezed', freezed)
        base = supercell.copy()
        traj = np.linspace(p1, p2, n_images)
        for p in traj:
            image = base.copy()
            image.positions[source] = p
            image = image[[i for i in range(len(image)) if i != target]]
            images.append(image)
        return images
    
    def __repr__(self):
        return f"Edge(source={self.source}, target={self.target}, offset={self.offset}), atoms={self.atoms}"