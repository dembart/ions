import numpy as np
from ase.neb import NEB
from ions.utils import collect_bvse_params
from ions.calculators import BVSECalculator
from ions.utils import geometric_median_with_bounds
from ase.build import make_supercell
from scipy.spatial import cKDTree



class SaddleFinder:
    """  
    Nudged Elastic band implementation for bond valence force field.
    """
    def __init__(self, self_interaction = True):

        """  
        Initialization

        Parameters
        ----------

        self_interaction: boolean, True by default
            whether to consider interaction between mobile species

        """
        self.self_interaction = self_interaction

    
    def interpolate(self, atoms, source, target, offset, min_sep_dist = 10.0, spacing = 0.5):
        """
        Linearly interpolates trajectory between source and target ions. 
        Uses min_sep_dist to create supercell. 

        Parameters
        ----------

        atoms: Ase's atoms object
            structure should contain mobile species of interest

        source: int
            index of the source positions

        target: int
            index of the target positions

        offset: list of int
            unitcell boundary crossing offset vector between source and target, e.g. [1, 0, 0]
        
        min_sep_dist: float, 10.0 by default
            minimum separation distance between moving ion and its periodic replica
        
        spacing: float, 0.5 by default
            length of the linear segments for the linear interpolation
            Note: if the number of images in the iterpolated trajectory is > 9,
                  spacing will be automatically recalculated for 9 images.

        Returns
        ----------
        images, list of atoms objects
            linearly interpolated trajectory
        """
        symbol = atoms.symbols[source]
        charge = int(atoms.get_array('oxi_states')[source])
        collect_bvse_params(atoms, symbol, charge, self_interaction = self.self_interaction)

        scale = np.ceil(min_sep_dist/atoms.cell.cellpar()[:3]).astype(int)
        p1 = atoms.positions[source]
        p2 = atoms.positions[target] + np.dot(offset, atoms.cell)
        
        d = np.linalg.norm(p1 - p2)
        n_images = int(d // spacing)
        if n_images % 2 == 0:
            n_images += 1
        if n_images > 9:
            n_images = 9

        P = [
            [scale[0], 0, 0],
            [0, scale[1], 0],
            [0, 0, scale[2]]
        ]
        supercell = make_supercell(atoms.copy(), P)
        scaled_edge = supercell.cell.scaled_positions([p1, p2]).round(10) # rounding for a safer wrapping
        scaled_edge[:]%=1.0 #wrapping
        wrapped_edge = supercell.cell.cartesian_positions(scaled_edge)
        if np.linalg.norm(wrapped_edge[0] - wrapped_edge[1]) < 0.1:
            print('source == target', source, target)
            print('min_sep_dist is too small for this jump', source, target, offset)
            raise
        tree = cKDTree(supercell.positions)
        dd, ii = tree.query(wrapped_edge)
        if dd.max() > 1e-6:
            print(dd.max())
            raise

        source_new = ii[0]
        target_new = ii[1]
        assert supercell.symbols[source_new] == symbol
        assert supercell.symbols[target_new] == symbol
        assert source_new != target_new
        traj = np.linspace(p1, p2, n_images)
        images = []
        freezed = np.array([i != source_new for i in range(len(supercell))])
        supercell.set_array('freezed', freezed)
        for p in traj:
            image = supercell.copy()
            image.positions[source_new] = p
            image = image[[i for i in range(len(image)) if i != target_new]]
            image = image[image.numbers.argsort()]
            #image.wrap()
            images.append(image)
            assert len(image) == len(supercell) - 1
        return images



    def bvse_neb(self, images, k = 5.0, default = True, gm = True, **kwargs):

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
        
        """

        site = np.argwhere(images[0].get_array('freezed') == False).ravel()[0]
        for image in images:
            image.calc =  BVSECalculator(site = site)

        if gm:
            center = images[len(images) // 2].copy()
            left = images[len(images) // 2 - 1].copy()
            right = images[len(images) // 2 + 1].copy()
            gm, y0 = geometric_median_with_bounds(left, center, right, site)
            images[len(images) // 2].positions[site] = np.array(gm)

        if default:
            neb = NEB(
                    images,
                    climb=False,
                    k = k,
                    method = 'improvedtangent',
                    )
        else:
            neb = NEB(images, k = k, **kwargs)
        return neb
    


    def get_barrier(self, images):
        """returns barrier for the passed images"""
        e = []
        for image in images:
            e.append(image.get_potential_energy())
        barrier = max(e) - min(e)
        return barrier
    

    def get_profile(self, images):
        """returns energy profile for the passed images"""
        profile = []
        for image in images:
            profile.append(image.get_potential_energy())
        return np.array(profile)
    

    def get_forces(self, images):
        """returns forces acting on moving ion for the passed images"""
        forces = []
        for image in images:
            site = np.argwhere(image.get_array('freezed') == False).ravel()[0]
            forces.append(image.get_forces()[site])
        return np.array(forces)