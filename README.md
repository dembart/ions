![ions_logo](https://raw.githubusercontent.com/dembart/ions/main/ions_logo.png)


### About

ions is a python library made for studying crystalline ionic conductors

### Installation

```bash
pip install ions
```

or 

```bash
git clone https://github.com/dembart/ions
cd ions
pip install .
```

### Functionality examples:

- [Percolation radius and dimensionality](#percolation-radius-and-dimensionality)

- [Inequivalent ionic hops forming percolating network](#inequivalent-ionic-hops-forming-percolating-network)

- [Nudged elastic band (NEB) calculation of a mobile ion activation barrier employing bond valence force field calculator and Atomic Simulation Environment optimizers](#nudged-elastic-band-calculations)
- [Compare NEB with bond valence energy landscape (BVEL) method](#compare-with-bvse-meshgrid-approach-ie-empty-lattice)

- [Decorate Atoms with oxidation states](#how-to-decorate-ases-atoms)

- [Available data](#available-data)


Note:
The library is under active development. Errors are expected. Most of the features are not well documented for now.


#### Percolation radius and dimensionality


```python
from ase.io import read, write
from ions.tools import Percolator

file = '/Users/artemdembitskiy/Downloads/LiFePO4.cif'
atoms = read(file)  

specie = 3 
pr = Percolator(
                atoms, 
                specie, # atomic number
                10.0,   # upper bound for Li-Li hops search 
                )

tr = 0.5 # Minimum allowed distance between the Li-Li edge and the framework
cutoff, dim = pr.mincut_maxdim(tr)

print(f'Maximum percolation dimensionality: {dim}')
print(f'Jump distance cutoff: {cutoff} angstrom', '\n')

for i in range(1, 4):
    percolation_radius = pr.percolation_threshold(i)
    print(f'{i}D percolation radius: {round(percolation_radius, 2)} angstrom')
```

    Maximum percolation dimensionality: 3
    Jump distance cutoff: 5.7421875 angstrom 
    
    1D percolation radius: 1.58 angstrom
    2D percolation radius: 1.46 angstrom
    3D percolation radius: 0.97 angstrom


#### Inequivalent ionic hops forming percolating network


```python
edges, _ = pr.unique_edges(cutoff, tr) # list of (source, target, offset_x, offset_y, offset_z)
edges
```




    [Edge(0,1,[0 0 0], d=5.74, wrapped_target=1, info = {'multiplicity': 36, 'index': 0}'),
     Edge(0,3,[1 0 0], d=5.64, wrapped_target=3, info = {'multiplicity': 24, 'index': 4}'),
     Edge(0,3,[0 0 0], d=3.05, wrapped_target=3, info = {'multiplicity': 24, 'index': 1}'),
     Edge(0,0,[1 0 0], d=4.75, wrapped_target=0, info = {'multiplicity': 16, 'index': 3}')]



#### Nudged elastic band calculations


```python
from ase.io import read
from ions import Decorator
from ase.optimize import FIRE
from ions.tools import Percolator, SaddleFinder
from ions.utils import collect_bvse_params


def optimize(neb, fmax = 0.1, steps = 100, logfile = 'log'):
        images = neb.images
        
        # relax source
        optim = FIRE(images[0], logfile = logfile)
        optim.run(fmax = fmax, steps = steps)

        # relax target
        optim = FIRE(images[-1], logfile = logfile)
        optim.run(fmax = fmax, steps = steps)
        
        # relax band
        optim = FIRE(neb, logfile = logfile)
        optim.run(fmax = fmax, steps = steps)
        
        # we perturb the structure before the calculations 
        if 'perturbation' in images[0].info.keys():
            for image in images:
                image.positions -= image.info['perturbation']
                image.info['perturbation'] = 0.0 * image.info['perturbation'] 

        optim = FIRE(neb, logfile = logfile)
        optim.run(fmax = fmax, steps = steps)
```


```python
file = '/Users/artemdembitskiy/Downloads/LiFePO4.cif'
atoms = read(file)
Decorator().decorate(atoms)
collect_bvse_params(atoms, 'Li', 1, self_interaction=True)
pl = Percolator(atoms, 3, 10.0)
tr = 0.5
cutoff, dim = pl.mincut_maxdim(tr = tr)
edges, _ = pl.unique_edges(cutoff, tr = 0.5)

relaxed_images = []
for edge in edges:
    images = edge.superedge(8.0).interpolate(spacing = .75) # we create a supercell with 8.0 Angstrom size
    sf = SaddleFinder()
    neb = sf.bvse_neb(images, distort = True, gm = False)
    optimize(neb)
    barrier = sf.get_barrier(images)
    relaxed_images.append(images)
    print(edge)
    print(f'Ea: {round(barrier, 2)} eV', '\n')
```

    Edge(0,1,[0 0 0], d=5.74, wrapped_target=1, info = {'multiplicity': 36, 'index': 0}')
    Ea: 3.27 eV 
    
    Edge(0,3,[1 0 0], d=5.64, wrapped_target=3, info = {'multiplicity': 24, 'index': 4}')
    Ea: 3.56 eV 
    
    Edge(0,3,[0 0 0], d=3.05, wrapped_target=3, info = {'multiplicity': 24, 'index': 1}')
    Ea: 0.33 eV 
    
    Edge(0,0,[1 0 0], d=4.75, wrapped_target=0, info = {'multiplicity': 16, 'index': 3}')
    Ea: 3.3 eV 
    


#### Compare with BVSE meshgrid approach (i.e. empty lattice)
 - For more details see [BVlain](https://github.com/dembart/BVlain) library


```python
from bvlain import Lain

calc = Lain(verbose = False)
atoms = calc.read_file(file)
_ = calc.bvse_distribution(mobile_ion = 'Li1+') # Li-Li interaction is omitted
calc.percolation_barriers()
```




    {'E_1D': 0.4395, 'E_2D': 3.3301, 'E_3D': 3.3594}



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



#### Available data

- bv_data - bond valence parameters [1]

- bvse_data - bond valence site energy parameters[2]

- ionic_radii - Shannon ionic radii [3, 4]

- crystal_radii - Shannon crystal radii [3, 4]

- elneg_pauling - Pauling's elenctronegativities [5]



[1]. https://www.iucr.org/resources/data/datasets/bond-valence-parameters (bvparam2020.cif)

[2]. He, B., Chi, S., Ye, A. et al. High-throughput screening platform for solid electrolytes combining hierarchical ion-transport prediction algorithms. Sci Data 7, 151 (2020). https://doi.org/10.1038/s41597-020-0474-y

[3] http://abulafia.mt.ic.ac.uk/shannon/ptable.php

[4] https://github.com/prtkm/ionic-radii

[5] https://mendeleev.readthedocs.io/en/stable/




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

### Bond valence sum calculation


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


