import numpy as np
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.bond_valence import BVAnalyzer
from ase.data import atomic_numbers, covalent_radii

from .data import bvse_data, principle_number



def assign_oxidation_states(atoms,
                   max_radius=4.0,
                   symm_tol = 1e-3,
                   max_permutations=10000,
                   forbidden_species = ['O1-'],
                   distance_scale_factor=1.05,
                   charge_neutrality_tolerance=1e-5,
                   ):
    
    """
    Decorate ase.Atoms with oxidation states using
    pymatgen's BVAnalyzer.
    
    Parameters 
    ----------

    atoms: ASE's Atoms object
        atomic structure
    
    see BVAnalyzer class in https://pymatgen.org/pymatgen.analysis.html
    for the details regarding other parameters

    Returns
    -------
    atoms with oxidation states stored in atoms.arrays['oxi_states']
    
    """
            
    bva = BVAnalyzer(forbidden_species=forbidden_species,
                    symm_tol=symm_tol,
                    max_radius=max_radius,
                    max_permutations=max_permutations,
                    distance_scale_factor=distance_scale_factor,
                    charge_neutrality_tolerance=charge_neutrality_tolerance
                    )
    st = AseAtomsAdaptor().get_structure(atoms)
    st = bva.get_oxi_state_decorated_structure(st)
    oxi_states = AseAtomsAdaptor.get_atoms(st).arrays['oxi_states']
    atoms.set_array('oxi_states', oxi_states)
    return atoms



def assign_bvse_params(atoms, symbol, charge, self_interaction = True):

    """
    Collect BVSE parameters for a given structure and specific ion
    
    Parameters 
    ----------

    atoms: ASE's Atoms object
        atomic structure
    
    symbol: str
        atomic symbol of the ion of intertest

    charge: int (can be negative)
        formal charge of the ion of interest

    self_interaction: boolean, True by default
        consider interaction between ions of interest (e.g. Li-Li repulsion)
    
    """
    
    s1 = symbol
    z1 = atomic_numbers[symbol]
    q1 = charge
    r0, r_min, alpha, d0, n2, rc2, mask = [], [], [], [], [], [], []
    for _, (s2, q2, z2) in enumerate(zip(atoms.symbols, atoms.get_array('oxi_states'), atoms.numbers)):
        if q2 < 0:
            params = bvse_data[s1][q1][s2][q2]
            r0.append(params['r0']) # dont need it, actually
            r_min.append(params['r_min'])
            alpha.append(params['alpha'])
            d0.append(params['d0'])
            n2.append(principle_number[z2]); rc2.append(covalent_radii[z2])
            mask.append(1.0)
        else:
            r0.append(0); r_min.append(0); alpha.append(0); d0.append(0)
            n2.append(principle_number[z2])
            rc2.append(covalent_radii[z2])
            if z1 == z2:
                if self_interaction:
                    mask.append(1.0)
                else:
                    mask.append(0.0)
            else:
                mask.append(1.0)
    coulomb_params = {'n2': n2, 'rc2': rc2, 'mask': mask}
    morse_params = {'r0': r0, 'd0': d0, 'alpha': alpha, 'r_min': r_min}
    for key in morse_params:
        atoms.set_array(key, np.array(morse_params[key]))
    for key in coulomb_params:
        atoms.set_array(key, np.array(coulomb_params[key]))
    return atoms
