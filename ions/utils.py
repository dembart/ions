import itertools
import numpy as np
from scipy.spatial.distance import cdist, euclidean
from scipy.optimize import minimize
from ase.data import atomic_numbers, covalent_radii
from ions.data import bvse_data, principle_number

def collect_bvse_params(atoms, symbol, charge, self_interaction = True):
    
    s1 = symbol
    z1 = atomic_numbers[symbol]
    q1 = charge
    n1 = principle_number[z1]
    rc1 = covalent_radii[z1]
    r0, r_min, alpha, d0, n2, rc2, mask = [], [], [], [], [], [], []
    for i, (s2, q2, z2) in enumerate(zip(atoms.symbols, atoms.get_array('oxi_states'), atoms.numbers)):
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



def geometric_median(X, y0, eps=1e-5, max_iter = 200):
    """
    Finds geometric median of the X points using y0 as the initial guess
    """
    #y = np.mean(X, 0)
    y = y0
    counter = 0
    while True:
        counter += 1
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros/r
            y1 = max(0, 1-rinv)*T + min(1, rinv)*y

        if euclidean(y, y1) < eps:
            return y1
        
        
        if counter > max_iter:
            return y
        y = y1


def geometric_median_string(left, center, right, source):

    base = np.zeros((len(left)+1, 3))
    base[:len(left), :] = left.positions
    base[-1, :] = right.positions[source]
    translations = np.array(list(itertools.product(
                                    [0, 1, -1], # should be [0, 1, -1, 2, -2] ideally
                                    [0, 1, -1],
                                    [0, 1, -1])))
    coords = []
    for tr in translations:
        coords.append(base + np.dot(tr, left.cell))
    X = np.vstack(coords)
    y0 = center.positions[source]
    gm = geometric_median(X, y0)
    return gm, y0





def geometric_median_with_bounds(left, center, right, source):

    """bounds are hard coded"""

    base = np.zeros((len(left)+1, 3))
    base[:len(left), :] = left.copy().positions
    base[-1, :] = right.copy().positions[source]
    translations = np.array(list(itertools.product(
                                    [0, 1, -1], # should be [0, 1, -1, 2, -2] ideally
                                    [0, 1, -1],
                                    [0, 1, -1])))
    coords = []
    for tr in translations:
        coords.append(base + np.dot(tr, left.copy().cell))
    X = np.vstack(coords)

    x = X[:, 0]
    y = X[:, 1]
    z = X[:, 2]
    x0 = center.copy().positions[source]
    def dist_func(x0):
        return sum(((np.full(len(x),x0[0])-x)**2+(np.full(len(x),x0[1])-y)**2+(np.full(len(x),x0[2])-z)**2)**(1/2))
    
    bounds = ((x0[0] - 0.25, x0[0] + 0.25),(x0[1] - 0.25, x0[1] + 0.25), (x0[2] - 0.25, x0[2] + 0.25))
    res = minimize(dist_func, x0, method='nelder-mead', options={'xtol': 1e-8, 'disp': False}, bounds=bounds)
    return res.x, x0