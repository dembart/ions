import numpy as np
from ase.mep import NEB
from ions.utils import collect_bvse_params
from ions.calculators import BVSECalculator, GMCalculator#, BVSECalculator2, BVSECalculator3, 
from ions.utils import geometric_median_with_bounds
from ase.build import make_supercell
from scipy.spatial import cKDTree
from ase.optimize import FIRE


class Trajectory(NEB):
    def __init__(self, images, **kwargs):
        super().__init__()


class SaddleFinder:
    """  
    Nudged Elastic band implementation for bond valence force field.
    """
    def __init__(self):

        """  
        Initialization

        Parameters
        ----------

        """


    def bvse_neb_old(self, images, k = 2.0, default = True, gm = False, distort = True, **kwargs):

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
            if distort:
                rng = np.random.default_rng(42)
                noise = rng.normal(0, 0.1, size=(len(image), 3))
                noise = 0.05 * noise / abs(noise).max()
                noise[site] = 0.0 * noise[site]
                image.positions +=  noise
                image.info['perturbation'] = noise

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
    

    def bvse_neb(self, images, k = 2.0, default = True, distort = True, gm = False, **kwargs):

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

        sites = np.argwhere(images[0].get_array('freezed') == False).ravel()
        for image in images:
            image.calc =  BVSECalculator(sites = sites)
            if distort:
                rng = np.random.default_rng(42)
                noise = rng.normal(0, 1, size=(len(image), 3))
                noise = 0.2 * noise / abs(noise).max()
                noise[sites] = 0.0 * noise[sites]
                image.positions +=  noise
                image.info['perturbation'] = noise

        if gm:
            center = images[len(images) // 2].copy()
            left = images[len(images) // 2 - 1].copy()
            right = images[len(images) // 2 + 1].copy()
            gm, y0 = geometric_median_with_bounds(left, center, right, sites[0])
            images[len(images) // 2].positions[sites[0]] = np.array(gm)


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
    

    def run(self, neb, steps = 100, fmax = 0.1, relax_endpoints = True, **kwargs):
        images = neb.images
        if relax_endpoints:
            optim = FIRE(images[0], **kwargs)
            optim.run(fmax = fmax, steps = steps)
            optim = FIRE(images[-1], **kwargs)
            optim.run(fmax = fmax, steps = steps)
        
        optim = FIRE(neb, **kwargs)
        optim.run(fmax = fmax, steps = steps)
        
        if 'perturbation' in images[0].info.keys():
            for image in images:
                image.positions -= image.info['perturbation']
                image.info['perturbation'] = 0.0 * image.info['perturbation'] 

        optim = FIRE(neb, **kwargs)
        optim.run(fmax = fmax, steps = steps)
        return neb, optim



    def bvse_cooperative_neb(self, images, sites, k = 2.0, default = True, **kwargs):

        """Wrapper for ase's NEB object. Sets BVSECalculator for each image.

        Parameters
        ----------

        images: list of Ase's atoms object
            structure should contain mobile species of interest
            and array named "freezed" defined as
            freezed = np.array([i != source for i in range(len(supercell))])
            atoms.sett_array('freezed', freezed)
            see .interpolate method for the details

        sites: idx of the sites to move

        k: float, 2.0 by default
            string force costant, eV
        
        default: boolean, whether to use default method or not, True by default
            if False, will use **kwargs passed for ase's NEB object

        
        """

        for image in images:
            image.calc =  BVSECalculator(sites = sites)

        if default:
            neb = NEB(
                    images,
                    climb=False,
                    k = k,
                    method = 'aseneb',
                    )
        else:
            neb = NEB(images, k = k, **kwargs)
        return neb
    
    


    def gm_neb(self, images, k = 2.0, default = True, **kwargs):

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
        
        """

        site = np.argwhere(images[0].get_array('freezed') == False).ravel()[0]
        for image in images:
            image.calc =  GMCalculator(site = site)

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