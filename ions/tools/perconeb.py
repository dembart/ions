import itertools
import os
import json
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
from ase import Atoms
from ase.io import write, read
from ase.neighborlist import NeighborList
from ase.neb import NEB
from ase.spacegroup import get_spacegroup
from ase.neighborlist import neighbor_list
from ase.build import make_supercell
from ase.data import covalent_radii
from spglib import get_symmetry_dataset
from spglib import standardize_cell
from ions.tools import SaddleFinder



__version__ = "0.1"

        
class PathFinder:
    
    """ 
    Perconeb object.
        
    The class can be used to find the percolating pathways of a mobile specie in a framework. 
    The functionality includes: 
    
    - calculating 1-3D percolation radius for a given mobile specie in a structure
    - searching for a minimum cutoff for maximum jump distance of a mobile specie required for 1-3D percolation
    - finding percolation pathway and its inequivalent parts (tests are required)

    For more details read the docs. 
    
    """ 
    def __init__(self, atoms, specie: int, upper_bound: float, symprec = 1e-3, oxi_states = False):
        
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
        
        
        self.specie = specie
        self.symprec = symprec
        self.upper_bound = min(atoms.cell.cellpar()[:3].max(), upper_bound) + 0.1
        self.oxi_states = oxi_states
        self._set_symmetry_labels(atoms)
        if self.oxi_states:
            self._set_ionic_radii(atoms)
        self.atoms = atoms.copy()
        self.mobile_atoms = self.atoms[self.atoms.numbers == specie]
        self.freezed_atoms = self.atoms[self.atoms.numbers != specie]
        self.mobile_atoms.set_array('unitcell_idx', np.argwhere(atoms.numbers == specie).ravel())
    
    
    
    # def _set_ionic_radii(self, atoms):

    #     symbols = atoms.symbols
    #     charges = atoms.get_array('oxi_states')
    #     r_i = np.array([ionic_radii[s][o] for (s, o) in zip(symbols, charges)])
    #     atoms.set_array('r_i', r_i)
        
        
        
    def _set_symmetry_labels(self, atoms):
        
        spglib_cell = (atoms.cell,
                       atoms.get_scaled_positions(),
                       atoms.numbers)
        equivalent_sites = get_symmetry_dataset(spglib_cell, symprec=self.symprec)['equivalent_atoms']
        atoms.set_array('sym_label', np.array(equivalent_sites))
        
    
    
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
        
        
        
    def _collect_edges_within_supercell(self):
        
        mobile_atoms = self.mobile_atoms.copy()
        n_vertices = len(mobile_atoms)
        scale = [
            [2, 0, 0],
            [0, 2, 0],
            [0, 0, 2]
        ]
        
        supercell = make_supercell(mobile_atoms.copy(), scale) 
        supercell.pbc = False # we are interested in the peroclation within the supercell
        shifts = np.where((supercell.get_scaled_positions() * 2.0).round(4) >= 1.0, 1, 0)
        supercell.set_array('shift', shifts)
        self.mobile_supercell = supercell
        i, j, d = neighbor_list('ijd', supercell, self.upper_bound)
        ij = np.array(list(zip(i, j)))
        if len(ij) == 0:
            raise
        ij.sort(axis = 1) # (source, target)  == (target, source)
        pairs, idx = np.unique(ij, axis = 0, return_index=True) # remove duplicates
        offsets = np.squeeze(np.diff(supercell.get_array('shift')[pairs], axis = 1))
        self.pairs = pairs
        self.offsets = offsets
        unwrapped_edges = np.hstack([pairs, offsets])
        wrapped_edges = np.hstack([pairs - n_vertices * (pairs // n_vertices), offsets])

        return unwrapped_edges, wrapped_edges, d[idx]
    


    def _annotate_edges(self):

        u, w, jump_distances = self._collect_edges_within_supercell()
        unique_edges, ue_idx, inverse = np.unique(w, axis = 0, return_index = True, return_inverse = True)
        unitcell_idx = self.mobile_atoms.get_array('unitcell_idx')
        distances = []
        for edge  in w[ue_idx]:
            offset = edge[2:]
            source = unitcell_idx[int(edge[0])]
            #source = edge[0]
            #target = edge[1]
            target = unitcell_idx[int(edge[1])]
            shift = np.where(offset < 0, 1, 0)
            p1 = self.atoms.positions[source] + np.dot(shift, self.atoms.cell)
            p2 = self.atoms.positions[target] + np.dot(offset + shift, self.atoms.cell)
            base = self.atoms[[i for i in range(len(self.atoms)) if i not in [source, target]]]
            translations = np.array(list(itertools.product(
                                            [0, 1, -1], # should be [0, 1, -1, 2, -2] ideally
                                            [0, 1, -1],
                                            [0, 1, -1])))
            coords = []
            idx = []
            for tr in translations:
                coords.append(base.positions + np.dot(tr, base.cell))
                idx.append(np.arange(0, len(base)))
            ii = np.hstack(idx)
            p = np.vstack(coords)
            dd = self._lineseg_dists(p, p1, p2)
            d_min = min(dd)
            if self.oxi_states:
                d_min = dd.min() - max(base.get_array('r_i')[ii[dd == dd.min()]])
            distances.append(d_min)
        self.distances = np.take(distances, inverse)
        self.jump_distances = jump_distances
        self.u = u
        self.w = w


    def _filter_edges(self, tr = 0.5, cutoff = 1e2):

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
        
        n_species = len(self.mobile_atoms)       
        G = nx.from_edgelist(edgelist)
        _, idx = np.unique(self.mobile_atoms.get_array('sym_label'),
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

        dim: int, 2 by default
            percolation dimensionality 2 -> 1D, 4 -> 2D, 8 -> 3D

        cutoff: float, 10.0 by default
            maximum allowed jump distance between sites in the mobile sublattice
        """
    
        self._annotate_edges()
        
        emin = 0 # legacy naming
        emax = 10.0
        tr = 0
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

    
    
    def cutoff_search(self, dim = 2, tr = 0.75):
        
        """
        Calculates minimum value of a jump distance between sites in a mobile sublattice
        required to form a 1-3D percolating network.
        
        
        Parameters
        ----------

        dim: int, 2 by default
            percolation dimensionality 2 -> 1D, 4 -> 2D, 8 -> 3D

        tr: float, 0.75 by default
            percolation threshold for an edge, i.e. minium distance between edge (line segment)
            between two sites in the mobile sublattice and the framework sublattie below which 
            edge is rejected

        upper_bound: float, 10.0 by default
            starting cutoff value for the search algorithm
        """
        
        self._annotate_edges()

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



    def unique_edges(self, tr = 0.75, dim = 2, cutoff = False):
        
        if cutoff:
            mask = self._filter_edges(tr = tr, cutoff = cutoff)
        else:
            cutoff = self.cutoff_search(dim = dim, tr = tr)
            mask = self._filter_edges(tr = tr, cutoff = cutoff)
        self.accepted_mask = mask
        s = np.vstack(self.mobile_atoms.get_array('sym_label')[self.w[:, :2]][mask])
        s.sort(axis = 1)
        d = self.distances[mask].round(3)
        j = self.jump_distances[mask].round(3)
        unique_pairs, idx, inverse = np.unique(np.column_stack((s, d, j)), axis = 0,
                                              return_index = True, return_inverse = True)
        return self.w[mask][idx], idx, inverse
    


    def percolation_dimensionality(self, tr = 0.75):
        max_dim = 0
        dim_cutoff = 0.0
        for dim in [2, 4, 8]:
            cutoff = self.cutoff_search(dim = dim, tr = tr)
            if cutoff > 0.0:
                max_dim = dim
                dim_cutoff = cutoff
        return max_dim, dim_cutoff

    


    def propagate_barriers(self, dim, percoedges, emins, emaxs):

        """
        Used for postprocessing. Defines percolation barriers for the 
        passed percolating edges and associated with them min and max 
        energies in the energy profile.
        
        
        Parameters
        ----------

        dim: int, 2 by default
            percolation dimensionality 2 -> 1D, 4 -> 2D, 8 -> 3D

        percoedges: list of N x [source, target], where N is the number of edges
            unwrapped percolating edges

        emins: list of N x float
            minimum energies in the energy profile for each edge

        emaxs: list of N x float
            maximum energies in the energy profile for each edge

        """

        #self._annotate_edges()
        emin = min(emins)
        emax = max(emaxs)
        tr = False
        unique_energies = np.unique(emaxs)
        unique_energies.sort()
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




class Perconeb(PathFinder):

    def __init__(self, atoms, specie, upper_bound, self_interaction = True):

        self.sf = SaddleFinder(self_interaction=self_interaction)
        super().__init__(atoms, specie, upper_bound)



    def mincut_maxdim(self, tr = 0.75):
        dim, cutoff = self.percolation_dimensionality(tr = tr)
        return dim, cutoff
    

    def percolating_jumps(self, tr = 0.75, min_sep_dist = 10.0, spacing = 0.5):

        dim, cutoff = self.mincut_maxdim(tr = tr)
        ue, ids, inverse = self.unique_edges(cutoff = cutoff, dim = dim, tr = tr)
        edges = self.w[self.accepted_mask]

        traj = []
        for i, edge in enumerate(ue):
            idx = self.mobile_atoms.get_array('unitcell_idx')
            source, target = idx[edge[:2]]
            offset = edge[2:]
            images = self.sf.interpolate(self.atoms, source, target, offset,
                                          spacing = spacing,
                                          min_sep_dist = min_sep_dist
                                        )
            traj.append(images)
        return traj
    


    def create_neb(self, images, k = 5.0, method = 'default', gm = False, **kwargs):
        """Wrapper for ase's NEB object. Sets BVSECalculator for each image.

        Parameters
        ----------

        images: list of Ase's atoms object
            structure should contain mobile species of interest
            and array named "freezed" defined as
            freezed = np.array([i != source for i in range(len(supercell))])
            atoms.sett_array('freezed', freezed)
            see .interpolate method for the details

        k: float, 5.0 by default
            string force costant, eV
        
        default: boolean, whether to use default method or not, True by default
            if False, will use **kwargs passed for ase's NEB object

        gm: boolean, True by default
            shift moving ion in the center image into its geometric median within supercell
            was not properly tested
        
        """
        _, neb = self.sf.bvse_neb(images, k = k, method = method, gm = gm, **kwargs)
        return neb
    