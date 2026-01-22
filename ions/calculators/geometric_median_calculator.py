import numpy as np
from ase.data import covalent_radii
from ase.neighborlist import NeighborList
from ase.calculators.calculator import Calculator, all_changes



class GMCalculator(Calculator):
    """
    Geometric median calculator
    """

    implemented_properties = ["energy", "forces", "min_distance"]
    default_parameters = {
        'rc': 10.0,
    }
    nolabel = True # check

    def __init__(self, site = None, nl = None, covalent = False, **kwargs):
        """

        Parameters
        ----------
        site: int
            indices of sites of interest in ase.Atoms
            Note: Considering source, taret notation it is the source
        
        nl: ase NeighborList, optional
            object should have method get_neighbors(site)
        
        """
        super().__init__(**kwargs)
        self.site = site
        self.nl = nl
        self.covalent = covalent

    def calculate(
        self,
        atoms = None,
        properties = None,
        system_changes = None,
    ):
        """
        Args:
            atoms (ase.Atoms): ase Atoms object
            properties (list): list of properties to calculate
            system_changes (list): monitor which properties of atoms were
                changed for new calculation. If not, the previous calculation
                results will be loaded.
        Returns:

        """
        properties = properties or ["energy"]
        system_changes = system_changes or all_changes
        super().calculate(atoms=atoms, properties=properties, system_changes=system_changes)
        
        natoms = len(self.atoms)
        rc = self.parameters.rc

        if self.covalent:
            self.rc = [covalent_radii[n] for n in self.atoms.numbers]

        if self.nl is None or 'numbers' in system_changes:
            self.nl = NeighborList(
                [rc / 2] * natoms, self_interaction=False, bothways=True, skin = 0.3
            )

        self.nl.update(self.atoms)

        energies = np.zeros(natoms)
        forces = np.zeros((natoms, 3))
        #positions = self.atoms.positions
        #cell = self.atoms.cell
        neighbors, offsets = self.nl.get_neighbors(self.site)

        r_vec = self.atoms.positions[self.site] - (atoms.positions[neighbors] + np.dot(offsets, atoms.cell))
        r_norm = np.linalg.norm(r_vec, axis = 1)
        if self.covalent:
            r_norm -= np.take(self.rc, neighbors)

        f = (1/r_norm).sum()
        energies[self.site] = f
        energy = energies.sum()
        x, y, z = r_vec[:, 0], r_vec[:, 1], r_vec[:, 2]
        dfdx = (-x/(r_norm ** 3)).sum()
        dfdy = (-y/(r_norm ** 3)).sum()
        dfdz = (-z/(r_norm ** 3)).sum()
        grad = np.array([dfdx, dfdy, dfdz])
        forces[self.site] = -grad
        self.results['energy'] = energy
        self.results['energies'] = energies
        self.results['free_energy'] = energy
        self.results['forces'] = forces
        self.results['min_distance'] = r_norm.min()