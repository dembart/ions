![ions_logo](https://raw.githubusercontent.com/dembart/ions/main/ions_logo.png)



## Contents
- [About](#about)
- [Installation](#installation)
- [Minimal usage example](#minimal-usage-example)
- [Notebooks](#notebooks)
- [How to cite](#how-to-cite)


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

### Minimal usage examples:


> Note: The last library update was not systematically tested. Errors are expected. Be careful.

##### Symmetrically inequivalent ionic hops (edges) forming percolating network


#### Inequivalent ionic jumps (edges) forming a percolating network


```python
from ase.io import read
from ions import Percolator

file = '/Users/artemdembitskiy/Downloads/LiFePO4.cif'
atoms = read(file)  

pl = Percolator(atoms, mobile_specie='Li', upper_bound=5.0)

bottleneck_radius = 0.5 
mincut, maxdim = pl.mincut_maxdim(bottleneck_radius)
print(f'Maximum percolation dimensionality: {maxdim}')
print(f'Jump distance cutoff: {mincut} angstrom', '\n')

edges = pl.unique_edges(mincut, bottleneck_radius, method = 'naive')
print(f'Inequivalent edges forming {maxdim}D percolating network:')
print(edges)
```
    Maximum percolation dimensionality: 2
    Jump distance cutoff: 4.755859375 angstrom 
    
    Inequivalent edges forming 2D percolating network:
    [[ 0  3  0 -1  0]
     [ 0  0 -1  0  0]]

#### Notebooks

- [Percolations barriers from nudged elastic band (NEB) calculations using universal machine learning interatomic potentials (UMLIPS)](notebooks/neb_umlip.ipynb)
- [Available data](notebooks/available_data.ipynb)
