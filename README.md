![ions_logo](https://raw.githubusercontent.com/dembart/ions/main/ions_logo.png)


#### About

ions is a python library made for studying percolation in ionic crystals

Functionality includes:

* calculating 1-3D percolation radius of mobile species in crystals

* finding percolation pathway and its inequivalent parts (tests are required)

* calculating activation barrier using nudged elastic band method implementing bond valence force field calculator

* searching for a minimum jump distance of a mobile specie required for 1-3D percolation

* handling data associated with ionic crystals

* decoration of ase's Atoms objects with oxidation states (pymatgen's reimplementaion)


Note:
The library is under active development. Errors are expected. Most of the features are not well documented for now.


#### Installation

```pip install ions```

#### Maximum percolation dimensionality and minimum required jump distance


```python
from ase.io import read, write
from ions.tools import PathFinder

file = '/Users/artemdembitskiy/Downloads/LiFePO4.cif'
atoms = read(file)  

specie = 3
pf = PathFinder(atoms, specie, 10.0)

tr = 0.5 # Minimum allowed distance between the edge and the framework
max_dim, cutoff = pf.mincut_maxdim(tr)

print(f'Maximum percolation dimensionality: {max_dim}')
print(f'Jump distance cutoff: {cutoff} angstrom')
```

    Maximum percolation dimensionality: 3
    Jump distance cutoff: 5.7421875 angstrom


#### Save percolating network


```python
traj = pf.create_percotraj(cutoff, tr)
write('perolating_sublattice.cif', traj) # jumps between nearest neighbors are linearly interpolated within 2x2x2 supercell
```

#### Inequivalent ionic hops forming percolating network


```python
edges = pf.unique_edges(cutoff, tr) # list of (source, target, offset_x, offset_y, offset_z)
edges
```




    array([[0, 1, 0, 0, 0],
           [0, 3, 1, 0, 0],
           [0, 3, 0, 0, 0],
           [0, 0, 1, 0, 0]])



#### Find saddle point for each ionic hop using bond valence force field potential


```python
import numpy as np
from ase.io import read, write
from ase.optimize import FIRE

from ions.tools import SaddleFinder
from ions.decorator import Decorator


Decorator().decorate(atoms) # oxidation states are required for this method

sf = SaddleFinder(self_interaction=True) # do not omit Li-Li interaction
traj = []
for i, edge in enumerate(edges):
    source, target = edge[:2]
    offset = edge[2:]
    images = sf.interpolate(atoms, source, target, offset, min_sep_dist = 10.0, spacing = 0.5)
    traj.append(images)
    neb = sf.bvse_neb(images, k = 5.0) # Note that images are linked to the neb object and will be changed after optimization
    optimizer = FIRE(neb, logfile = 'log')
    optimizer.run(fmax =.1, steps = 100)
    print(f'Unique jump #{i}: Fmax {neb.get_forces().max().round(2)} eV/angstrom |',
          f'Activation barrier {sf.get_barrier(images).round(2)} eV')
```

    Unique jump #0: Fmax 0.08 eV/angstrom | Activation barrier 3.24 eV
    Unique jump #1: Fmax 0.05 eV/angstrom | Activation barrier 3.56 eV
    Unique jump #2: Fmax 0.08 eV/angstrom | Activation barrier 0.35 eV
    Unique jump #3: Fmax 0.06 eV/angstrom | Activation barrier 3.29 eV


#### Plot profile


```python
import matplotlib.pyplot as plt
from scipy.interpolate import pchip_interpolate

plt.rcParams.update({'font.size': 8})
plt.rcParams.update({'font.family': 'Arial'})



fig, axes = plt.subplots(dpi = 600, figsize = (9, 2.5), ncols = len(traj), sharey  = True)
for ax, images in zip(axes, traj):
    profile = sf.get_profile(images)
    x = np.arange(0, len(images))
    x_fit = np.linspace(0, len(images), 100)
    y_fit = pchip_interpolate(x, profile, x_fit)
    ax.plot(x, profile, 'o', label = 'calculated')
    ax.plot(x_fit, y_fit, zorder = 1, label = 'pchip fit', color = 'darkred')
    ax.set_xlabel('Reaction coordinate')
    ax.set_xlim(x.min(), x.max())
    ax.legend(frameon = False)
axes[0].set_ylabel('Energy, eV')
plt.tight_layout()
```


    
![png](example_files/example_12_0.png)
    


#### Percolation dimensionality study


```python
emins = []
emaxs = []
for images in traj:
    emins.append(sf.get_profile(images).min())
    emaxs.append(sf.get_profile(images).max())

for i, dim in enumerate(np.arange(1, max_dim + 1)):
    e_a, tr_min, tr_max = pf.propagate_barriers(cutoff, tr, emins, emaxs, dim)
    print(f'Activation barrier of {i + 1}D percolation: {round(e_a, 2)} eV')
```

    Activation barrier of 1D percolation: 0.35 eV
    Activation barrier of 2D percolation: 3.24 eV
    Activation barrier of 3D percolation: 3.29 eV


#### Put all together


```python
from ase.io import read, write
from ions.tools import PathFinder, SaddleFinder

file = '/Users/artemdembitskiy/Downloads/LiFePO4.cif'
atoms = read(file)  
Decorator().decorate(atoms) # oxidation states are required for this method

specie = 3
pf = PathFinder(atoms, specie, 10.0)

tr = 0.5 # Minimum allowed distance between the edge and the framework
max_dim, cutoff = pf.mincut_maxdim(tr)
edges = pf.unique_edges(cutoff, tr) # list of (source, target, offset_x, offset_y, offset_z)

sf = SaddleFinder(self_interaction=True) # do not omit Li-Li interaction
traj = []
emins = []
emaxs = []
for edge in edges:
    source, target = edge[:2]
    offset = edge[2:]
    images = sf.interpolate(atoms, source, target, offset, min_sep_dist = 10.0, spacing = 0.5)
    traj.append(images)
    neb = sf.bvse_neb(images, k = 5.0, gm = True) # Note that images are linked to the neb object and will be changed after optimization
    optimizer = FIRE(neb, logfile = 'log')
    optimizer.run(fmax =.1, steps = 100)
    emins.append(sf.get_profile(images).min())
    emaxs.append(sf.get_profile(images).max())


for dim in np.arange(1, max_dim + 1):
    e_a, tr_min, tr_max = pf.propagate_barriers(cutoff, tr, emins, emaxs, dim)
    print(f'Activation barrier of {dim}D percolation: {round(e_a, 2)} eV')

```

    Activation barrier of 1D percolation: 0.35 eV
    Activation barrier of 2D percolation: 3.24 eV
    Activation barrier of 3D percolation: 3.29 eV


#### Compare with BVSE meshgrid approach (i.e. empty lattice)


```python
from bvlain import Lain

calc = Lain(verbose = False)
atoms = calc.read_file(file)
_ = calc.bvse_distribution(mobile_ion = 'Li1+') # Li-Li interaction is omitted
calc.percolation_barriers()
```




    {'E_1D': 0.4395, 'E_2D': 3.3301, 'E_3D': 3.3594}



One may see that method implemented in bvlain library is much faster and more concise. It can be used for fast prediction of the percolation barriers, while the PathFinder and SaddleFinder can be used as a tool for interpolating local migration trajectory for further DFT-NEB calculations.

#### Available data

* bv_data - bond valence parameters [1]

* bvse_data - bond valence site energy parameters[2]

* ionic_radii - Shannon ionic radii [3, 4]

* crystal_radii - Shannon crystal radii [3, 4]

* elneg_pauling - Pauling's elenctronegativities [5]



##### References

[1]. https://www.iucr.org/resources/data/datasets/bond-valence-parameters (bvparam2020.cif)

[2]. He, B., Chi, S., Ye, A. et al. High-throughput screening platform for solid electrolytes combining hierarchical ion-transport prediction algorithms. Sci Data 7, 151 (2020). https://doi.org/10.1038/s41597-020-0474-y

[3] http://abulafia.mt.ic.ac.uk/shannon/ptable.php

[4] https://github.com/prtkm/ionic-radii

[5] https://mendeleev.readthedocs.io/en/stable/



### How to handle data


```python
from ions.data import ionic_radii, crystal_radii, bv_data, bvse_data

#ionic radius
symbol, valence = 'V', 4
r_ionic = ionic_radii[symbol][valence]  


#crystal radius
symbol, valence = 'F', -1
r_crystal = crystal_radii[symbol][valence]


# bond valence parameters
source, source_valence = 'Li', 1
target, target_valence = 'O', -2
params = bv_data[source][source_valence][target][target_valence]
r0, b = params['r0'], params['b']


# bond valence site energy parameters
source, source_valence = 'Li', 1
target, target_valence = 'O', -2
params = bvse_data[source][source_valence][target][target_valence]
r0, r_min, alpha, d0  = params['r0'], params['r_min'], params['alpha'], params['d0']
```

### How to decorate ase's Atoms


```python
import numpy as np
from ase.io import read
from ions import Decorator


file = '/Users/artemdembitskiy/Downloads/LiFePO4.cif'
atoms = read(file)
calc = Decorator()
atoms = calc.decorate(atoms)
oxi_states = atoms.get_array('oxi_states')
np.unique(list(zip(atoms.symbols, oxi_states)), axis = 0)

```




    array([['Fe', '2'],
           ['Li', '1'],
           ['O', '-2'],
           ['P', '5']], dtype='<U21')



### Example


```python
import numpy as np
from ions import Decorator
from ase.io import read
from ase.neighborlist import neighbor_list
from ions.data import bv_data

file = '/Users/artemdembitskiy/Downloads/LiFePO4.cif'
atoms = read(file)
calc = Decorator()
atoms = calc.decorate(atoms)
ii, jj, dd = neighbor_list('ijd', atoms, 5.0)  

symbols = atoms.symbols
valences = atoms.get_array('oxi_states')
for i in np.unique(ii):
    source = symbols[i]
    source_valence = valences[i]
    neighbors = jj[ii == i]
    distances = dd[ii == i]
    if source_valence > 0:
        bvs = 0
        for n, d in zip(neighbors, distances):
            target = symbols[n]
            target_valence = valences[n]
            if source_valence * target_valence < 0:
                params = bv_data[source][source_valence][target][target_valence]
                r0, b = params['r0'], params['b']
                bvs += np.exp((r0 - d) / b)
        print(f'Bond valence sum for {source} is {round(bvs, 4)}')

```

    Bond valence sum for Li is 1.0775
    Bond valence sum for Li is 1.0775
    Bond valence sum for Li is 1.0775
    Bond valence sum for Li is 1.0775
    Bond valence sum for Fe is 1.8394
    Bond valence sum for Fe is 1.8394
    Bond valence sum for Fe is 1.8394
    Bond valence sum for Fe is 1.8394
    Bond valence sum for P is 4.6745
    Bond valence sum for P is 4.6745
    Bond valence sum for P is 4.6745
    Bond valence sum for P is 4.6745

