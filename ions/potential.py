import numpy as np
from scipy.special import erfc

class BVFF:

    def Morse(r, r_min, d0, alpha):
        
        """ Calculate Morse-type interaction energy.
            Note: interaction is calculated between ions 
                    of opposite sign compared to mobile ion


        Parameters
        ----------

        r: np.array of floats
            distance between mobile ion and framework
            
        r_min: np.array of floats
            minimum energy distance between mobile and framework ions
            
        d0: np.array of floats
            bond breaking parameter
            
        alpha: np.array of floats
            inverse BVS tabulated parameter b (alpha = 1 / b)
        
        Returns
        ----------
        
        np.array
            Morse-type interaction energies
        """
        energy = d0 * ((np.exp( alpha * (r_min - r) ) - 1) ** 2 - 1)
        return energy / 2



    def Coulomb(q1, q2, r, rc1, rc2, n1, n2, f = 0.74):
        
        
        """ Calculate Coulombic interaction energy.
            Note: interaction is calculate between ions 
                    of same sign as mobile ion

        Parameters
        ----------

        q1: float.
            formal charge of mobile speice
        
        q2: np.array of floats
            formal charges of framework ions
            
        R: np.array of floats
            distance between mobile ion and framework
            
        rc1: float
            covalent radii of mobile specie

        rc2: np.array of floats
            covalent radii of framework ions
            
        n1: int
            principle quantum numbers of mobile ion
            
        n2: np.array of integers
            principle quantum numbers of framework ions
            
        f: float, 0.74 by default
            screening factor
        
        Returns
        ----------
        
        np.array
            Coulombic interaction energies
        """

        energy = 14.4 * (q1 * q2 / (n2 * n1) ** (1/2)) * (1 / (r)) * erfc(r / (f * (rc2 + rc1)))
        return energy

    
    def dMorse_dX(x, alpha, r_min, d0, r):
        """ Calculate partial derivative of Morse-type pair potential.
            Note: interaction is calculated between ions 
                    of opposite sign compared to mobile ion


        Parameters
        ----------
        x: np.array of floats
            x (or y or z) coordinate of distance vectors between mobile ion and framework

        alpha: np.array of floats
            inverse BVS tabulated parameter b (alpha = 1 / b)

        r_min: np.array of floats
            minimum energy distance between mobile and framework ions
  
        d0: np.array of floats
            bond breaking parameter

        r: np.array of floats
            distance between mobile ion and framework
            
        Returns
        ----------
        
        np.array
            partial derivative of the pair potential
        """
        dfdx = -2 * alpha * d0 * x * (np.exp(alpha * (r_min - r)) - 1) * np.exp(alpha * (r_min - r)) * (1/r)
        return dfdx/2


    def dCoulomb_dX(x, q1, q2, rc1, rc2, n1, n2, r, f = 0.74):

        """ Calculate partial derivative of Coulomb-type pair potential.
            Note: interaction is calculated between ions 
                    of same sign compared to mobile ion


        Parameters
        ----------
        x: np.array of floats
            x (or y or z) coordinate of distance vectors between mobile ion and framework

        q1: float.
            formal charge of mobile speice
        
        q2: np.array of floats
            formal charges of framework ions
            
        R: np.array of floats
            distance between mobile ion and framework
            
        rc1: float
            covalent radii of mobile specie

        rc2: np.array of floats
            covalent radii of framework ions

        n1: int
            principle quantum numbers of mobile ion
            
        n2: np.array of integers
            principle quantum numbers of framework ions
            
        f: float, 0.74 by default
            screening factor
            
        Returns
        ----------
        
        np.array
            partial derivative of the pair potential
        """


        frc = f * (rc1 + rc2)
        A = 14.4 * (q1 * q2 / ((n2 * n1) ** (1/2)))
        z = r/frc
        term1 = - x * erfc(r / frc) / (r**3) # (x^2 + y^2 + z^2) == r^2
        term2 = - x * 2 * np.exp(-z**2) / (r**2 * np.sqrt(np.pi) * frc)
        dfdx =  A * (term1 + term2)
        return dfdx