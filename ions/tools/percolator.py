import itertools
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
from ase import Atoms
from ase.io import write, read
#from ase.neb import NEB
from ase.spacegroup import get_spacegroup
from ase.neighborlist import neighbor_list
from spglib import get_symmetry_dataset
from ions.geom import Edge, Box
from ions.utils import lineseg_dists



class Percolator:

    def __init__(self, atoms, specie: int, upper_bound: float, symprec = 1e-3, oxi_states = False, check_distances = False):
        """ 
    
        Parameters
        ----------

        atoms: ase's Atoms object
            Should contain a mobile specie of interest

        specie: int
            atomic number of a mobile specie, e.g. 11 for Na

        upper_bound: float, 10.0 by default
            maximum jump distance between equilibrium sites of a mobile specie
            
        symprec: float, 1e-3 by default
            precision for a space group analysis

        oxi_states: boolean, False by default
            whether atoms has 'oxi_states' attribute
            
        """
        
        self.atoms = atoms.copy()
        self._set_symmetry_labels(self.atoms, symprec)
        self.specie = specie
        self.upper_bound = min(atoms.cell.cellpar()[:3].max() + 0.1, upper_bound)
        #self._symprec = symprec
        self._check_distances = check_distances
        self._mobile_atoms = self.atoms[self.atoms.numbers == self.specie]
        #self.oxi_states = oxi_states
        
        self._annotate_edges()
        

    def _set_symmetry_labels(self, atoms, symprec):
        
        spglib_cell = (atoms.cell,
                       atoms.get_scaled_positions(),
                       atoms.numbers)
        equivalent_sites = get_symmetry_dataset(spglib_cell, symprec)['equivalent_atoms']
        atoms.set_array('sym_label', np.array(equivalent_sites))
        
        
        
    def _recalc_dim_for_algo(self, dim):
        dim_for_algo = {1: 2, 2: 4, 3: 8}
        return dim_for_algo[dim]
    

    

    def _collect_edges_within_supercell(self):
        
        #atoms = self.atoms.copy()
        box = Box(self.atoms)

        supercell = box._super_box([2, 2, 2])
        mobile_sc = supercell[supercell.numbers == self.specie]
        mobile_sc.pbc = False

        ii, jj, dd = neighbor_list('ijd', mobile_sc, self.upper_bound, self_interaction=False)
        ij = np.vstack([ii, jj]).T
        if len(ij) == 0:
            print(f'no edges found for a given cutoff={self.upper_bound}')
            raise
        ij.sort(axis = 1) # (source, target)  == (target, source)

        #check 1
        if self._check_distances:
            for (i,j), d in zip(ij, dd):
                d1 = np.linalg.norm(mobile_sc.positions[i] - mobile_sc.positions[j])
                d2 = d
                if abs(d1 - d2) > 1e-10:
                        print(i,j,d,d2)

        #check 2
        if self._check_distances:
            for (i,j), d in zip(ij, dd):
                id1 = mobile_sc.get_array('index')[i]
                id2 = mobile_sc.get_array('index')[j]
                o1 = mobile_sc.get_array('offset')[i]
                o2 = mobile_sc.get_array('offset')[j]
                p1 = self.atoms.positions[id1] 
                p2 = self.atoms.positions[id2] + np.dot(o2 - o1, self.atoms.cell)
                d1 = np.linalg.norm(p1-p2)
                d2 = d
                if abs(d1 - d2) > 1e-10:
                        print(i,j,d1,d2)
                        raise
                    
        pairs, idx = np.unique(ij, axis = 0, return_index=True) # remove duplicates
        edge_lengths = dd[idx]
        translations = np.take(mobile_sc.get_array('offset'), pairs, axis = 0)
        offsets = translations[:, 1] - translations[:, 0]
        u = np.hstack([pairs, offsets])
        #w = np.take(mobile_sc.get_array('index'), pairs)
        wrapped_pairs = np.take(mobile_sc.get_array('index'), pairs, axis = 0)
        #wrapped_pairs.sort(axis = 1)
        w = np.hstack([wrapped_pairs, offsets])
        #unique_wrapped_edges, idx = np.unique(w, axis = 0, return_index=True) # remove duplicates

        # check 3
        if self._check_distances:
            for edge, d in zip(w, edge_lengths):
                source = edge[0]
                target = edge[1]
                offset = edge[2:]
                p1 = self.atoms.positions[source] 
                p2 = self.atoms.positions[target] + np.dot(offset, self.atoms.cell)
                d1 = np.linalg.norm(p1-p2)
                d2 = d
                if abs(d1 - d2) > 1e-10:
                        print(i,j,d1,d2)
                        raise
        return u, w, edge_lengths
    



    def _annotate_edges(self):
        
        u, w, jump_distances = self._collect_edges_within_supercell()
        unique_edges, ue_idx, inverse = np.unique(w, axis = 0, return_index = True, return_inverse = True)
        distances = []
        for edge, d_  in zip(w[ue_idx], jump_distances[ue_idx]):
            offset = edge[2:]
            source = edge[0]
            target = edge[1]
            #shift = np.where(offset < 0, 1, 0)
            shift = [0, 0, 0]
            p1 = self.atoms.positions[source] + np.dot(shift, self.atoms.cell)
            p2 = self.atoms.positions[target] + np.dot(offset + shift, self.atoms.cell)
            base = self.atoms[[i for i in range(len(self.atoms)) if i not in [source, target]]]
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
            ii = np.hstack(idx)
            p = np.vstack(coords)
            dd = lineseg_dists(p, p1, p2)
            d_min = min(dd)
            # if self.oxi_states:
            #     d_min = dd.min() - max(base.get_array('r_i')[ii[dd == dd.min()]])
            distances.append(d_min)
        self.distances = np.take(distances, inverse)
        self.jump_distances = jump_distances
        self.u = u
        self.w = w
        return u, w, self.distances, self.jump_distances
    

    def _filter_edges(self, tr = 0.75, cutoff = 100.0):

        accept = []
        for i, (u, jump, dist) in enumerate(zip(self.u, self.jump_distances, self.distances)):
            if (jump > cutoff) or (dist < tr):
                continue
            else:
                accept.append(i)
        if len(accept) > 1:
            accept = np.array(accept)
        else:
            accept = []
        return accept
    

        
    def _percolation_dimensionality(self, edgelist):
        
        n_species = len(self._mobile_atoms)       
        G = nx.from_edgelist(edgelist)
        _, idx = np.unique(self._mobile_atoms.get_array('sym_label'),
                                   return_index = True) # index should be from self.mobile_atoms
        perco_dims_per_site = {}
        sym_uniq_sites = np.arange(0, n_species)[idx]
        for i in sym_uniq_sites:
            dim = 0
            for j in range(0, 2 ** 3):
                i_next_cell = i + j * n_species
                
                try:
                    path_idx = nx.algorithms.shortest_path(G, i, i_next_cell)
                    dim += 1
                    perco_dims_per_site.update({i:dim})
                except:
                    continue
        return perco_dims_per_site
        
        
                
    def percolation_threshold(self, dim, cutoff = 10.0):

        """
        Calculates maximum distance between percolating edges
        and the framework sublattice. 
        If self.oxi_states = True the percolation ionic radius is calculated.
        
        
        Parameters
        ----------

        dim: int
            percolation dimensionality, can be 1, 2, 3

        cutoff: float, 10.0 by default
            maximum allowed jump distance between sites in the mobile sublattice
        """
    
    #    self._annotate_edges()
        
        emin = 0 # legacy naming
        emax = 10.0
        tr = 0
        dim = self._recalc_dim_for_algo(dim)
        while (emax - emin) > 0.01:
            probe = (emin + emax) / 2
            mask = self._filter_edges(tr = probe)
            edges = self.u[mask, :2]
            if len(edges) > 0:
                try:
                    data = self._percolation_dimensionality(edges)
                    if max(list(data.values())) >= dim:
                        emin = probe
                        tr = round(emin,4)
                    else:
                        emax = probe
                except:
                    emax = probe
            else:
                emax = probe
        return tr

    
    
    def cutoff_search(self, dim, tr):
        
        """
        Calculates minimum value of a jump distance between sites in a mobile sublattice
        required to form a 1-3D percolating network.
        
        
        Parameters
        ----------

        dim: int
            percolation dimensionality, can be 1, 2, 3

        tr: float
            percolation threshold for an edge, i.e. minium distance between edge (line segment)
            between two sites in the mobile sublattice and the framework sublattie below which 
            edge is rejected
        """
        
        #self._annotate_edges()
        dim = self._recalc_dim_for_algo(dim)
        emin = 0.0 # legacy naming
        emax = self.upper_bound
        cutoff = -1.0
        while (emax - emin) > 0.01:
            probe = (emin + emax) / 2
            mask = self._filter_edges(tr = tr, cutoff = probe)
            edges = self.u[mask, :2]
            if len(edges) > 0:
                try:
                    data = self._percolation_dimensionality(edges)
                    if max(list(data.values())) >= dim:
                        emax = probe
                        cutoff = emax
                    else:
                        emin = probe
                except:
                    emin = probe
            else:
                emin = probe
        return cutoff
    


    def mincut_maxdim(self, tr):
        """finds minimum cutoff for jump distances and maximum percolation dimensionality"""
        max_dim = 0
        dim_cutoff = 0.0
        for dim in (1, 2, 3):
            cutoff = self.cutoff_search(dim, tr)
            if cutoff > 0.0:
                max_dim = dim
                dim_cutoff = cutoff
        return  dim_cutoff, max_dim
    

    def unique_edges(self, cutoff, tr):

        """returns inequivalent jumps [source, target, offset_x, offset_y, offset_z]
        
        Parameters
        ----------

        cutoff: float
            maximum distance of the jump, i.e. edge length

        tr: float
            percolation threshold for an edge, i.e. minium distance between edge (line segment)
            between two sites in the mobile sublattice and the framework sublattie below which 
            edge is rejected

        
        Returns
        ----------
        inequivalent jumps - array of [source, target, offset_x, offset_y, offset_z]

        """
        mask = self._filter_edges(tr = tr, cutoff = cutoff)
        #self.accepted_mask = mask
        s = np.vstack(self._mobile_atoms.get_array('sym_label')[self.w[:, :2] - self.w[:, :2].min()][mask]) # ?
        s.sort(axis = 1)
        d = self.distances[mask].round(3)
        j = self.jump_distances[mask].round(3)
        unique_pairs, idx, inverse = np.unique(np.column_stack((s, d, j)), axis = 0, return_index = True, return_inverse = True)
        edges = []
        for e in self.w[mask][idx]:
            edge = Edge(self.atoms, e[0], e[1], e[2:])
            edges.append(edge)

        #return self.w[mask][idx], unique_pairs
        return edges, unique_pairs
    

    def propagate_barriers(self, cutoff, tr, emins, emaxs, dim):

        """
        Used for postprocessing. Defines percolation barriers for the 
        passed percolating edges and associated with them min and max 
        energies in the energy profile.
        
        
        Parameters
        ----------

        dim: int
            percolation dimensionality, can be 1, 2, 3

        percoedges: list of N x [source, target], where N is the number of edges
            unwrapped percolating edges

        emins: list of N x float
            minimum energies in the energy profile for each edge

        emaxs: list of N x float
            maximum energies in the energy profile for each edge

        """

        #self._annotate_edges()
        mask = self._filter_edges(tr = tr, cutoff = cutoff)
        percoedges = self.u[mask][:, :2]
        s = np.vstack(self._mobile_atoms.get_array('sym_label')[self.w[:, :2] - self.w[:, :2].min()][mask])
        s.sort(axis = 1)
        d = self.distances[mask].round(3)
        j = self.jump_distances[mask].round(3)
        _, idx, inverse = np.unique(np.column_stack((s, d, j)), axis = 0, return_index = True, return_inverse = True)
        emins = np.array(emins)[inverse]
        emaxs = np.array(emaxs)[inverse]
        emin = min(emins)
        emax = max(emaxs)
        tr = False
        unique_energies = np.unique(emaxs)
        unique_energies.sort()
        dim = self._recalc_dim_for_algo(dim)
        for e in unique_energies[::-1]:
            probe = e
            mask = emaxs <= probe
            edges = percoedges[mask]
            if len(edges) > 0:
                try:
                    data = self._percolation_dimensionality(edges)
                    if max(list(data.values())) >= dim:
                        emin = probe
                        tr = emin
                    else:
                        emax = probe
                except:
                    emax = probe
            else:
                emax = probe

        if tr:
            tr2 = False
            unique_energies = np.unique(emins)
            unique_energies.sort()
            mask = emaxs <= tr
            percoedges = percoedges[mask]
            emax = tr
            for e in unique_energies:
                probe = e
                mask2 = emins[mask] >= probe
                edges = percoedges[mask2]
                if len(edges) > 0:
                    try:
                        data = self._percolation_dimensionality(edges)
                        if max(list(data.values())) >= dim:
                            emin = probe
                            tr2 = emin
                        else:
                            emin = probe
                    except:
                        emin = probe
                else:
                    emin = probe
        else:
            print('Unexpected behavior')
            raise
        
        return abs(tr2 - tr), tr2, tr



