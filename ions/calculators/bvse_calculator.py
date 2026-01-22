import numpy as np
from ase.data import covalent_radii
from ase.stress import full_3x3_to_voigt_6_stress
from ase.neighborlist import NeighborList
from ase.calculators.calculator import Calculator, all_changes

from ..data import principle_number
from .potential import BVFF



class BVSECalculator(Calculator):
    """
    BVSE calculator based on ASE's Calculator. It is used for NEB calculations
    of a mobile ion migration barrier.
    """

    implemented_properties = ["energy", "forces", "stressses", "stress"]
    default_parameters = {
        'rc': 10.0,
    }
    nolabel = True # check

    def __init__(self, sites = None, nl = None, f = 0.74, **kwargs):
        """
        Parameters
        ----------
        sites: list of int
            indices of sites of interest in ase.Atoms
            Note: Considering source, taret notation it is the source
        
        nl: ase NeighborList, optional
            object should have method get_neighbors(site)
        
        """
        super().__init__(**kwargs)
        self.sites = sites
        self.nl = nl
        self.f = f

    def calculate(
        self,
        atoms = None,
        properties = None,
        system_changes = None,
    ):
        """
        Parameters
        ----------

        atoms: ASE's Atoms object
            atomic structure

        properties: list
            list of properties to calculate
        
        system_changes: list
            monitor which properties of atoms were
        
        changed for new calculation. If not, the previous calculation
                results will be loaded.
        Returns:

        """
        properties = properties or ["energy", "forces", "stress"]
        system_changes = system_changes or all_changes
        super().calculate(atoms=atoms, properties=properties, system_changes=system_changes)
        
        natoms = len(self.atoms)
        rc = self.parameters.rc

        if self.nl is None or 'numbers' in system_changes:
            self.nl = NeighborList(
                [rc / 2] * natoms, self_interaction=False, bothways=True, skin = 0.2
            )

        self.nl.update(self.atoms)
        positions = self.atoms.positions
        cell = self.atoms.cell

        energies = np.zeros(natoms)
        forces = np.zeros((natoms, 3))
        stresses = np.zeros((natoms, 3, 3))
        atoms = self.atoms.copy()
        for ii in self.sites:
            
            neighbors, offsets = self.nl.get_neighbors(ii)
            cells = np.dot(offsets, cell)
            distance_vectors = positions[neighbors] + cells - positions[ii]
            r2 = np.linalg.norm(distance_vectors, axis = 1)
            q1 = atoms.get_array('oxi_states')[ii]
            z1 = atoms.numbers[ii]
            n1 = principle_number[z1]
            rc1 = covalent_radii[z1]
            q2 = np.take(atoms.get_array('oxi_states'), neighbors)
            q2 = np.where(q2 * q1 < 0.0, 0, q2)
            n2 = np.take(atoms.get_array('n2'), neighbors)
            rc2 = np.take(atoms.get_array('rc2'), neighbors)
            alpha =  np.take(atoms.get_array('alpha'), neighbors)
            r_min =  np.take(atoms.get_array('r_min'), neighbors)
            d0 =  np.take(atoms.get_array('d0'), neighbors)
            mask = np.take(atoms.get_array('mask'), neighbors)
            e1 = BVFF.Morse(r2, r_min, d0, alpha).sum() 
            e2 = (mask * BVFF.Coulomb(q1, q2, r2, rc1, rc2, n1, n2, f = self.f)).sum()
            e = e1 + e2
            
            pairwise_energies = e
            # alpha * d0 (np.exp (alpha * (r_min - rij)) - 1) * np.exp (alpha * (r_min - rij)), where rij = r2 
            pairwise_forces = BVFF.Morse_pairwise_force(alpha, d0, r_min, r2) #
            # q1, q2, rij, f, rc1, rc2, n1, n2
            pairwise_forces += BVFF.Coulomb_pairwise_force(q1, q2, r2, self.f, rc1, rc2, n1, n2)
            pairwise_forces = pairwise_forces[:, np.newaxis] * -distance_vectors
            energies[ii] += pairwise_energies.sum()
            forces[ii] += pairwise_forces.sum(axis=0)
            stresses[ii] += 0.5 * np.dot(
                pairwise_forces.T, distance_vectors
            )  # equivalent to outer product

        # no lattice, no stress
        if self.atoms.cell.rank == 3:
            stresses = full_3x3_to_voigt_6_stress(stresses)
            self.results['stress'] = stresses.sum(
                axis=0) / self.atoms.get_volume()
            self.results['stresses'] = stresses / self.atoms.get_volume()
        else:
            print('stresses cannot be computed')
        energy = energies.sum()
        self.results['energy'] = energy
        self.results['energies'] = energies
        self.results['free_energy'] = energy
        self.results['forces'] = forces