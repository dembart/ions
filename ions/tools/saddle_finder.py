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


    def bvse_neb(self, images, k = 2.0, default = True, gm = False, **kwargs):

        """Wrapper for ase's NEB object. Sets BVSECalculator for each image.

        Parameters
        ----------

        images: list of Ase's atoms object
            structure should contain mobile species of interest
            and array named "freezed" defined as
            freezed = np.array([i != source for i in range(len(supercell))])
            atoms.sett_array('freezed', freezed)
            see .interpolate method for the details

        k: float, 2.0 by default
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