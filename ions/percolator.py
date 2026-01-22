import itertools
import numpy as np
from spglib import get_symmetry_dataset
from ase.io import read
from ase.neighborlist import neighbor_list

from .geometry import lineseg_dists
from .periodic_graph import PeriodicGraph



class Percolator:

    def __init__(self, atoms, mobile_specie=None, upper_bound=8.0, symprec=1e-3):

        """ 
        Parameters
        ----------

        atoms: ase's Atoms object
            Atomic structure. Should contain a mobile specie of interest.

        mobile_specie: str
            Atomic symbol of a mobile specie, e.g. "Li"

        upper_bound: float, 10.0 by default
            maximum jump distance between equilibrium sites of a mobile specie
            
        symprec: float, 1e-3 by default
            precision for a space group analysis
        """

        self.atoms = atoms.copy()
        self.mobile_specie = mobile_specie 
        if mobile_specie not in atoms.symbols:
            raise ValueError(f'No provided mobile_specie ({mobile_specie}) in a structure')
        self.symprec = symprec
        self.upper_bound = upper_bound
        self.edges = None           
        self.edge_lengths = None
        self.edge_min_dists = None

    @classmethod
    def from_file(cls, file, mobile_specie=None, upper_bound=8.0, symprec=1e-3):
        atoms = read(file)
        return cls(atoms, mobile_specie=mobile_specie, upper_bound=upper_bound, symprec=symprec)
    


    def _collect_edges(self):
        
        atoms = self.atoms.copy()
        atoms.set_array('index', np.arange(0, len(atoms)))        
        mobile_atoms = atoms[atoms.symbols == self.mobile_specie].copy()
        sources, targets, offsets, lengths =  neighbor_list('ijSd', mobile_atoms, self.upper_bound)
        # renumerate in accordance with the initial indices in the unitcell
        sources = np.take(mobile_atoms.arrays['index'], sources)
        targets = np.take(mobile_atoms.arrays['index'], targets)
        self.edges = np.column_stack((sources, targets, offsets))
        self.edge_lengths = lengths
        if len(self.edges) == 0:
            print(f'No edges found for the given cutoff={self.upper_bound} and mobile_specie={self.mobile_specie}.')
            raise



    def _annotate_edges(self):

        atoms = self.atoms.copy()
        min_distances = []
        for _, edge in enumerate(self.edges):

            source, target, offset = edge[0], edge[1], edge[2:]
            shift = [0, 0, 0]
            p1 = atoms.positions[source] + np.dot(shift, atoms.cell)
            p2 = atoms.positions[target] + np.dot(offset + shift, atoms.cell)
            base = atoms[[i for i in range(len(atoms)) if i not in [source, target]]]
            translations = np.array(list(itertools.product(
                                            [0, 1, -1], # should be [0, 1, -1, 2, -2] ideally
                                            [0, 1, -1],
                                            [0, 1, -1])))
            #assert abs(np.linalg.norm(p2 - p1) - d_) < 1e-10
            coords = []
            idx = []
            for tr in translations:
                coords.append(base.positions + np.dot(tr, base.cell))
                idx.append(np.arange(0, len(base)))
            idx = np.hstack(idx) # to track nearest neighbor if needed
            p = np.vstack(coords)
            dd = lineseg_dists(p, p1, p2)
            d_min = min(dd)
            min_distances.append(d_min)
        self.edge_min_dists = np.array(min_distances)



    def _filter_edges(self, cutoff, bottleneck_radius):
        mask = (self.edge_lengths <= cutoff) & (self.edge_min_dists >= bottleneck_radius)
        return self.edges[mask], mask

    

    def cutoff_search(self, dim, bottleneck_radius):
        
        """
        Calculates minimum value of a jump distance between sites in a mobile sublattice
        required to form a 1-3D percolating network.
        
        
        Parameters
        ----------

        dim: int
            percolation dimensionality, can be 1, 2, 3

        bottleneck_radius: float
            percolation threshold for an edge, i.e. minium distance between edge (line segment)
            between two sites in the mobile sublattice and the framework sublattice
        """

        if self.edges is None:
            self._collect_edges()

        if self.edge_min_dists is None:
            self._annotate_edges()
        
        rmin = 0.0
        rmax = self.upper_bound
        cutoff = -1.0
        while (rmax - rmin) > 0.01:
            probe = (rmin + rmax) / 2
            edges, _ = self._filter_edges(probe, bottleneck_radius)
            if len(edges) > 0:
                try:
                    pg = PeriodicGraph(edges[:, 0], edges[:,1], edges[:, 2:])
                    current_dim = pg.find_periodicity()[0]
                    if current_dim >= dim:
                        rmax = probe
                        cutoff = rmax
                    else:
                        rmin = probe
                except:
                    rmin = probe
            else:
                rmin = probe
        return cutoff
    


    def mincut_maxdim(self, bottleneck_radius):

        """Finds minimum cutoff for jump distances and maximum percolation dimensionality"""
        max_dim = 0
        dim_cutoff = 0.0
        for dim in (1, 2, 3):
            cutoff = self.cutoff_search(dim, bottleneck_radius)
            if cutoff > 0.0:
                max_dim = dim
                dim_cutoff = cutoff
        return  dim_cutoff, max_dim


                
    def percolation_threshold(self, dim, cutoff = 10.0):

        """
        Calculates maximum distance between percolating edges
        and the framework sublattice. 
        
        Parameters
        ----------

        dim: int
            percolation dimensionality, can be 1, 2, 3

        cutoff: float, 10.0 by default
            maximum allowed jump distance between sites in the mobile sublattice
        """

        if self.edges is None:
            self._collect_edges()

        if self.edge_min_dists is None:
            self._annotate_edges()
        
        rmin = 0
        rmax = cutoff
        bottleneck_radius = 0
        while (rmax - rmin) > 0.01:
            probe = (rmin + rmax) / 2
            edges, _ = self._filter_edges(cutoff, probe)
            if len(edges) > 0:
                try:
                    pg = PeriodicGraph(edges[:, 0], edges[:,1], edges[:, 2:])
                    current_dim = pg.find_periodicity()[0]
                    if current_dim >= dim:
                        rmin = probe
                        bottleneck_radius = round(rmin,4)
                    else:
                        rmax = probe
                except:
                    rmax = probe
            else:
                rmax = probe
        return bottleneck_radius
    


    def _inequivalent_edges(self, edges):
        
        decimals = int(np.ceil(-np.log10(self.symprec))) # precision
        frac_pos = self.atoms.get_scaled_positions()
        cell = self.atoms.cell

        if self.symprec:
            dataset = get_symmetry_dataset((cell, frac_pos, self.atoms.numbers),
                                            symprec=self.symprec)
            rotations = dataset['rotations']
            translations = dataset['translations']
        else:
            rotations = [np.eye(3)]
            translations = [np.zeros(3)]

        keys = []
        for _, edge in enumerate(edges):

            i, j = int(edge[0]), int(edge[1])
            offset = edge[2:]

            p_i = frac_pos[i]
            p_j = frac_pos[j] + offset
            d_ij = p_j - p_i

            orbit_keys = []
            for R, t in zip(rotations, translations):
                p_sym = (R @ p_i + t) % 1.0
                d_sym = R @ d_ij

                k1 = np.concatenate([p_sym, d_sym]).round(decimals)
                k2 = np.concatenate([(p_sym + d_sym) % 1.0, -d_sym]).round(decimals)
                
                orbit_keys.append(min(tuple(k1), tuple(k2)))
            edge_key = min(orbit_keys)
            keys.append(edge_key)
        self.edge_keys = np.array(keys)
        _, index, inverse = np.unique(keys, axis = 0, return_index=True, return_inverse=True)
        return edges[index], index, inverse
    


    def _inequivalent_edges_naive(self, edges, lengths, min_dists):

        decimals = int(np.ceil(-np.log10(self.symprec))) # precision
        atoms = self.atoms.copy()
        spglib_cell = (atoms.cell,
                       atoms.get_scaled_positions(),
                       atoms.numbers)
        equivalent_sites = get_symmetry_dataset(spglib_cell,
                                                self.symprec)['equivalent_atoms']
        
        ij = np.column_stack([
            np.take(equivalent_sites, edges[:, 0]),
            np.take(equivalent_sites, edges[:, 1]),
        ])


        ij.sort(axis = 1)
        features = np.column_stack([ij, lengths, min_dists]).round(decimals)
        _, index, inverse = np.unique(features, axis=0, return_index=True, return_inverse=True)
        return edges[index], index, inverse



    def unique_edges(self, cutoff, bottleneck_radius, method = 'naive'):

        """Find inequivalent jumps [source, target, offset_x, offset_y, offset_z]
        
        Parameters
        ----------

        cutoff: float
            maximum distance of the jump, i.e. edge length

        bottleneck_radius: float
            Minium allowed distance between the edge and the framework sublattice

        method: str, can be "naive" (default) or "symop"
            method to detect symmetrically equivalent edges.
            - "naive" uses wyckoff positions, edge length, 
              and a minimum distance between the edge and the framework 
              to find unique edges
            
            - "symop" uses symmetry operations to generate all possible edges
              and select inequivalent ones

            "naive" method is faster and yields smaller number of edges,
            but stronger depends on numerical accuracy.

            "symop" is safer but may output equivalent edges which may results
            in a non-efficient use of compute

        
        Returns
        ----------
        inequivalent jumps - array of [source, target, offset_x, offset_y, offset_z]

        """
        
        if self.edges is None:
            self._collect_edges()

        if self.edge_min_dists is None:
            self._annotate_edges()

        edges, mask = self._filter_edges(cutoff, bottleneck_radius)
        if method == 'naive':
            lengths = self.edge_lengths[mask]
            min_dists = self.edge_min_dists[mask]
            unique_edges, index, inverse = self._inequivalent_edges_naive(edges,lengths, min_dists)
        elif method == 'symop':
            unique_edges, index, inverse = self._inequivalent_edges(edges)
        else:
            raise ValueError(f'Wrong method ({method}). Can be "naive" or "symop"')
        return unique_edges
    

    def find_percolation_barriers(self, cutoff, bottleneck_radius, emins_unique, emaxs_unique, method):

        """
        Compute 1D, 2D, and 3D percolation energy barriers using precomputed minimum
        and maximum energies for unique ionic jumps forming a percolation network at
        a given jump distance cutoff and bottleneck radius.

        Parameters
        ----------

        cutoff: float
            maximum distance of the jump, i.e. edge length

        bottleneck_radius: float
            Minium allowed distance between the edge and the framework sublattice


        emins_unique: list of N x float
            minimum energies in the energy profile for unique edges
            obtained using self.unique_edges

        emaxs_unique: list of N x float
            maximum energies in the energy profile for unqiue edges
            obtained using self.unique_edges

        method: str, can be "naive" or "symop"
            method to detect symmetrically equivalent edges.
            must be consistent with the method used in self.unique_edges

        Returns
        -------
        dict with "e1d", e2d", and "e3d" percolation barriers
        
        If percolation is not possible for a given dimension, the value is `None`.

        """

        barriers = {}
        for dim in [1, 2, 3]:
            try:
                result = self._propagate_barriers(cutoff, bottleneck_radius, emins_unique, emaxs_unique, dim, method)
                barriers.update({f'e{dim}d': result[f'e{dim}d']})
            except:
                barriers.update({f'e{dim}d': None})
        return barriers
                


    def _propagate_barriers(self, cutoff, bottleneck_radius, emins_unique, emaxs_unique, dim,
                           method):
        
        """
        Compute percolation energy barrier for a provided dimensionality using precomputed minimum
        and maximum energies for unique ionic jumps forming a percolation network at
        a given jump distance cutoff and bottleneck radius.

        Parameters
        ----------

        cutoff: float
            maximum distance of the jump, i.e. edge length

        bottleneck_radius: float
            Minium allowed distance between the edge and the framework sublattice


        emins_unique: list of N x float
            minimum energies in the energy profile for unique edges
            obtained using self.unique_edges

        emaxs_unique: list of N x float
            maximum energies in the energy profile for unqiue edges
            obtained using self.unique_edges

        dim: int
            percolation dimensionality, can be 1, 2, 3

        method: str, can be "naive" or "symop"
            method to detect symmetrically equivalent edges.
            must be consistent with the method used in self.unique_edges

        """
        
        if method == 'naive':
            edges, mask = self._filter_edges(cutoff, bottleneck_radius)
            lengths = self.edge_lengths[mask]
            min_dists = self.edge_min_dists[mask]
            _, _, inverse = self._inequivalent_edges_naive(edges,
                                                               lengths,
                                                               min_dists)
        elif method == 'symop':
            edges, _ = self._filter_edges(cutoff, bottleneck_radius)
            _, _, inverse = self._inequivalent_edges(edges)
        else:
            raise ValueError(f'Wrong method ({method}). Can be "naive" or "symop"')


        emins = np.asarray(emins_unique)[inverse]
        emaxs = np.asarray(emaxs_unique)[inverse]

        upper_barrier = None
        for E in np.unique(emaxs):
            mask = emaxs <= E
            current_edges = edges[mask]

            if len(current_edges) == 0:
                continue

            pg = PeriodicGraph(current_edges[:, 0], current_edges[:,1], current_edges[:, 2:])
            current_dim = pg.find_periodicity()[0]
            if current_dim >= dim:
                upper_barrier = E
                break

        if upper_barrier is None:
            raise RuntimeError("No percolating network found")


        lower_barrier = None

        mask_upper = emaxs <= upper_barrier
        edges_upper = edges[mask_upper]
        emins_upper = emins[mask_upper]

        for E in np.unique(emins_upper):
            mask = emins_upper >= E
            current_edges = edges_upper[mask]

            if len(current_edges) == 0:
                continue

            pg = PeriodicGraph(current_edges[:, 0], current_edges[:,1], current_edges[:, 2:])
            current_dim = pg.find_periodicity()[0]
            if current_dim >= dim:
                lower_barrier = E
                break

        if lower_barrier is None:
            raise RuntimeError("Lower barrier not found")
        
        result = {
            f'e{dim}d': upper_barrier - lower_barrier,
            'emax': upper_barrier,
            'emin': lower_barrier,
            'edges': current_edges,
        }
        
        return result