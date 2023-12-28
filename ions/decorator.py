import yaml
import json
import os 
import sys
import re
import collections
import operator
import functools
import numpy as np
from ase.neighborlist import neighbor_list
from spglib import get_symmetry_dataset


__version__ = '0.1.3'

class Decorator:
    
    """
    The class is used to decorate ase's Atoms with oxidation states.
    This is a reimplemented method from pymatgen's BVAnalyzer which
    operates with pymatgen's Structure object.
    source: https://github.com/materialsproject/pymatgen/blob/master/\
        pymatgen/analysis/bond_valence.py
    
    According to the pymatgen's documentaion it works as follows:
    'This class implements a maximum a posteriori (MAP) estimation method to
    determine oxidation states in a structure. The algorithm is as follows:
    1) The bond valence sum of all symmetrically distinct sites in a structure
    is calculated using the element-based parameters in M. O'Keefe, & N. Brese,
    JACS, 1991, 113(9), 3226-3229. doi:10.1021/ja00009a002.
    2) The posterior probabilities of all oxidation states is then calculated
    using: P(oxi_state/BV) = K * P(BV/oxi_state) * P(oxi_state), where K is
    a constant factor for each element. P(BV/oxi_state) is calculated as a
    Gaussian with mean and std deviation determined from an analysis of
    the ICSD. The posterior P(oxi_state) is determined from a frequency
    analysis of the ICSD.
    3) The oxidation states are then ranked in order of decreasing probability
    and the oxidation state combination that result in a charge neutral cell
    is selected.'
    
    """

    def __init__(self, r_cut = 3.5, forbidden_species = ['O1-'], max_permutations = 1e4, symprec = 1e-3):
        """
        
        Initialization
        
        Parameters
        ----------

        r_cut: float, 3.5 by default
            cutoff radius for neighbors search

        forbidden_species: list of str, ['O1-'] by default
            list of forbidden oxidation states

        max_permutations: int, 1e4 by default
            maximum number of oxidation sate combinations to check

        symprec: float, 1e-3 by default
            symmetry precision for space group analysis
        
        """
        self.r_cut = r_cut
        self.symprec = symprec
        self.max_permutations = max_permutations
        self.forbidden_species = forbidden_species
        self._load_data()

    def _load_data(self):

        self.data_pth = _resource_path('data')
        self.bv_params_file = f'{self.data_pth}/bvparam_1991.yaml'
        self.icsd_data_file = f'{self.data_pth}/icsd_bv.yaml'
        self.elneg_file = f'{self.data_pth}/elneg_data.json'
        self.elneg_elements = "H B C Si N P As Sb O S Se Te F Cl Br I".split()
        with open(self.bv_params_file, 'r') as f:
            self.bv_data = yaml.safe_load(f)
        with open(self.elneg_file, 'r') as f:
            self.elneg_data = json.load(f)
        with open(self.icsd_data_file, 'r') as f:
            data = yaml.safe_load(f)
            self.icsd_bv_data = {}
            for key in data['bvsum'].keys():
                el, oxi = self._decompose(key)
                self.icsd_bv_data.update({(el, oxi): data['bvsum'][key]})
            self.prior_prob = {}
            for key in data['occurrence'].keys():
                el, oxi = self._decompose(key)
                self.prior_prob.update({(el, oxi): data['occurrence'][key]})

        if self.forbidden_species:
            forbidden_species = [self._decompose(s) for s in self.forbidden_species]
            self.icsd_bv_data = {specie: data for specie, data in self.icsd_bv_data.items() if specie not in forbidden_species}



    def _decompose(self, mobile_ion):
        element = re.sub('\d', '', mobile_ion).replace("+","").replace("-","")
        oxi_state = re.sub('\D', '', mobile_ion)
        if '-' in mobile_ion:
            sign = -1
        else:
            sign = 1
        if len(oxi_state) > 0:
            if sign > 0:
                oxi_state = int(oxi_state)
            else:
                oxi_state = -int(oxi_state)
        else:
            oxi_state = sign
        return element, oxi_state



    def _set_symmetry_labels(self, atoms):
        spglib_cell = (atoms.cell,
                       atoms.get_scaled_positions(),
                       atoms.numbers)
        equivalent_sites = get_symmetry_dataset(spglib_cell, symprec=self.symprec)['equivalent_atoms']
        atoms.set_array('sym_label', np.array(equivalent_sites))



    def _collect_neighbors(self, atoms):
        """pymatgen's code with minor changes"""
        ii, jj, dd = neighbor_list('ijd', atoms, self.r_cut)
        symbols = atoms.symbols
        bvs = []
        for i in np.unique(ii):
            el1 = symbols[i]
            bv_sum = 0
            for n, d in zip(jj[ii == i], dd[ii == i]):
                el2 = symbols[n]
                if (el1 in self.elneg_elements  or el2 in self.elneg_elements) and el1 != el2:
                    r1 = self.bv_data[el1]["r"]
                    r2 = self.bv_data[el2]["r"]
                    c1 = self.bv_data[el1]["c"]
                    c2 = self.bv_data[el2]["c"]
                    R = r1 + r2 - r1 * r2 * (np.sqrt(c1) - np.sqrt(c2)) ** 2 / (c1 * r1 + c2 * r2)
                    vij = np.exp((R - d) / 0.37)
                    bv_sum += vij * (1 if self.elneg_data[el1] < self.elneg_data[el2] else -1)
            bvs.append(bv_sum)
        atoms.set_array('bvs', np.array(bvs))



    def _calc_site_probabilities(self, atoms, site):
        """pymatgen's code with minor changes"""
        el = atoms.symbols[site]
        bv_sum = atoms.get_array('bvs')[site]
        prob = {}
        for sp, data in self.icsd_bv_data.items():
            if sp[0] == el and sp[1] != 0 and data["std"] > 0:
                u = data["mean"]
                sigma = data["std"]
                # Calculate posterior probability. Note that constant
                # factors are ignored. They have no effect on the results.
                prob[sp[1]] = np.exp(-((bv_sum - u) ** 2) / 2 / (sigma**2)) / sigma * self.prior_prob[sp]
        # Normalize the probabilities
        try:
            prob = {k: v / sum(prob.values()) for k, v in prob.items()}
        except ZeroDivisionError:
            prob = {key: 0 for key in prob}
        return prob
    



    def _selection_loop(self, atoms):
        """pymatgen's code with minor changes"""
        uniq_sites = np.unique(atoms.get_array('sym_label'))
        symbols = atoms.symbols[uniq_sites]
        check_params =set(symbols) - set(self.bv_data)
        if check_params:
            err = f"Structure contains elements not in set of BV parameters: {check_params}"
            raise ValueError(err)  
        elneg = np.array([self.elneg_data[s] for s in symbols])
        elneg_idx = elneg.argsort()
        sorted_elneg = elneg[elneg_idx[::-1]]
        sorted_sites = uniq_sites[elneg_idx[::-1]]
        sorted_symbols = symbols[elneg_idx[::-1]]
        valences, all_prob = [], []
        for i in sorted_sites:
            prob = self._calc_site_probabilities(atoms, i)
            all_prob.append(prob)
            val = list(prob)
            val = sorted(val, key=lambda v: -prob[v])
            valences.append(list(filter(lambda v: prob[v] > 0.01 * prob[val[0]], val)))

        _, n_sites = np.unique(atoms.get_array('sym_label'), return_counts=True)
        n_sites = n_sites[elneg_idx[::-1]]
        valence_min = np.array(list(map(min, valences)))
        valence_max = np.array(list(map(max, valences)))
        self._n = 0
        self._best_score = 0
        self._best_vset = None

        def evaluate_assignment(v_set):
            el_oxi = collections.defaultdict(list)
            for i, site in enumerate(sorted_sites):
                el_oxi[sorted_symbols[i]].append(v_set[i])
            max_diff = max(max(v) - min(v) for v in el_oxi.values())
            if max_diff > 1:
                return
            score = functools.reduce(operator.mul, [all_prob[i][v] for i, v in enumerate(v_set)])
            if score > self._best_score:
                self._best_vset = v_set
                self._best_score = score

        def _recurse(assigned=None):
            # recurses to find permutations of valences based on whether a
            # charge balanced assignment can still be found
            if self._n > self.max_permutations:
                return
            if assigned is None:
                assigned = []

            i = len(assigned)
            highest = valence_max.copy()
            highest[:i] = assigned
            highest *= n_sites
            highest = np.sum(highest)

            lowest = valence_min.copy()
            lowest[:i] = assigned
            lowest *= n_sites
            lowest = np.sum(lowest)

            if highest < 0 or lowest > 0:
                self._n += 1
                return

            if i == len(valences):
                evaluate_assignment(assigned)
                self._n += 1
                return
            for v in valences[i]:
                new_assigned = list(assigned)
                _recurse([*new_assigned, v])
            return
        _recurse()
        if self._best_score:
            assigned = {}
            oxi_states = []
            valences_ = dict(zip(sorted_sites, self._best_vset))
            for l in atoms.get_array('sym_label'):
                oxi_states.append(valences_[l])
            atoms.set_array('oxi_states', np.array(oxi_states))
        else:
            raise ValueError("Valences cannot be assigned!")


    def decorate(self, atoms):
        """
        Decorate ase's Atoms with oxidation states.
        
        Parameters
        ----------

        atoms: ase's Atoms object
            supposed to be an ionic compound
        
        Returns
        ----------
        
        ase's Atoms object with oxidation states
            to see oxidation states, use:
            oxi_states = atoms.get_array('oxi_states')
        
        """
        self._collect_neighbors(atoms)
        self._set_symmetry_labels(atoms)
        self._selection_loop(atoms)
        return atoms

def _resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base_path, relative_path)
    return path