import numpy as np
from scipy.special import erfc
from ase.data import covalent_radii, atomic_numbers
from ase.calculators.calculator import Calculator, all_changes
from ase.neighborlist import NeighborList
from ions.data import principle_number
from ions.potential import BVFF
import numpy as np
from scipy.spatial.distance import cdist, euclidean


class BVSECalculator(Calculator):
    """
    BVSE calculator based on ase Calculator. It is used for NEB calculations
    of a mobile ion migration activation energy. 
    """

    implemented_properties = ["energy", "forces"]
    default_parameters = {
        'rc': 10.0,
    }
    nolabel = True # check

    def __init__(self, site = None, nl = None, **kwargs):
        """

        Args:
            site: index of site in ase.Atoms for which NEB, int
                Note: Considering source, taret notation it is the source
            nl: ase NeighborList, optional
                object should have method get_neighbors(site)
            **kwargs:
        """
        super().__init__(**kwargs)
        self.site = site
        self.nl = nl

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

        if self.nl is None or 'numbers' in system_changes:
            self.nl = NeighborList(
                [rc / 2] * natoms, self_interaction=False, bothways=True, skin = 0.3
            )

        self.nl.update(self.atoms)

        energies = np.zeros(natoms)
        forces = np.zeros((natoms, 3))
        positions = self.atoms.positions
        cell = self.atoms.cell
        neighbors, offsets = self.nl.get_neighbors(self.site)
        cells = np.dot(offsets, cell)
        r = self.atoms.positions[self.site] - (atoms.positions[neighbors] + np.dot(atoms.cell.T, offsets.T).T)
        r2 = np.linalg.norm(r, axis = 1)

        q1 = self.atoms.get_array('oxi_states')[self.site]
        s1 = self.atoms.symbols[self.site]
        z1 = self.atoms.numbers[self.site]
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
        e2 = (mask * BVFF.Coulomb(q1, q2, r2, rc1, rc2, n1, n2)).sum()
        e = e1 + e2
        
        energies[self.site] = e 
        energy = energies.sum()

        x, y, z = r[:, 0], r[:, 1], r[:, 2]
        dfdx1 = BVFF.dMorse_dX(x, alpha, r_min, d0, r2).sum()
        dfdy1 = BVFF.dMorse_dX(y, alpha, r_min, d0, r2).sum()
        dfdz1 = BVFF.dMorse_dX(z, alpha, r_min, d0, r2).sum()

        dfdx2 = (mask * BVFF.dCoulomb_dX(x, q1, q2, rc1, rc2, n1, n2, r2)).sum()
        dfdy2 = (mask * BVFF.dCoulomb_dX(y, q1, q2, rc1, rc2, n1, n2, r2)).sum()
        dfdz2 = (mask * BVFF.dCoulomb_dX(z, q1, q2, rc1, rc2, n1, n2, r2)).sum()

        grad = np.array([dfdx1 + dfdx2, dfdy1 + dfdy2, dfdz1 + dfdz2])
        forces[self.site] = -grad
        self.results['energy'] = energy
        self.results['energies'] = energies
        self.results['free_energy'] = energy
        self.results['forces'] = forces



    
class BVFFCalculator(Calculator):
    """
    Not tested.
    """

    implemented_properties = ["energy", "forces"]
    default_parameters = {
        'rc': 10.0,
    }
    nolabel = True # check

    def __init__(self, V, site = None, nl = None, **kwargs):
        """

        Args:
            site: index of site in ase.Atoms for which NEB, int
                Note: Considering source, taret notation it is the source
            nl: ase NeighborList, optional
                object should have method get_neighbors(site)
            **kwargs:
        """
        super().__init__(**kwargs)
        self.V = V
        self.size = np.array(V.shape)
        self.mesh = np.argwhere(V) / (self.size - 1)
        self.site = site
        self.nl = nl

    def locate(self, p):

        X, Y, Z = self.mesh[:, 0], self.mesh[:, 1], self.mesh[:, 2]
        d, idx = self.closest_pythagoras(X, Y, Z, p)
        if d > 0.2:
            raise
        return idx


    def closest_pythagoras(self, X, Y, Z, P):
        """
            X: x-axis series
            Y: y-axis series
            Z: z-axis series
            P: centre point to measure distance from
            
            Returns: tuple(closest point, distance between P and closest point)
        """ 
        # Compute the distances between each of the points and the desired point using Pythagoras' theorem
        distances = np.sqrt((X - P[0]) ** 2 + (Y - P[1]) ** 2 + (Z - P[2]) ** 2)
        # Get the index of the smallest value in the array to determine which point is closest and return it
        idx_of_min = np.argmin(distances)
        #(np.array([X[idx_of_min], Y[idx_of_min], Z[idx_of_min]])
        return distances[idx_of_min], idx_of_min



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
        natoms = len(atoms)
        energies = np.zeros(natoms)
        forces = np.zeros((natoms, 3))

        p = atoms.get_scaled_positions()[self.site]
        idx = self.locate(p)
        e = self.V.reshape(-1)[idx]

        energies[self.site] = e 
        energy = energies.sum()


        dr = self.atoms.cell.cellpar()[:3] / self.size
        gradV = np.gradient(self.V, edge_order=2)  # how to treat pbc?
        dfdx = - gradV[0].reshape(-1)[idx]/dr[0]
        dfdy = - gradV[1].reshape(-1)[idx]/dr[1]
        dfdz = - gradV[1].reshape(-1)[idx]/dr[2]

        forces[self.site] = np.array([dfdx, dfdy, dfdz])
        self.results['energy'] = energy
        self.results['energies'] = energies
        self.results['free_energy'] = energy
        self.results['forces'] = forces
