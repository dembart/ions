import numpy as np
from scipy.spatial import Voronoi
from ase.data import covalent_radii
from ase.spacegroup import get_spacegroup
from spglib import get_symmetry_dataset

from ions.tools import Percolator
from ions.geom import Box
from ions.geom.utils import _solid_angle, _vol_tetra, _effective_CN, _find_vneighbors
from ions.data import ionic_radii, elneg_pauling


class StructureFeaturizer:

    def __init__(self, atoms, specie, r_cut = 10.0, oxi = False, symprec = 0.1):

        self._set_symmetry_labels(atoms, symprec)
        self.specie = specie
        self.cell = atoms.cell
        self.r_cut = r_cut
        self.atoms = atoms.copy()
        self.oxi = oxi

        
        elneg = []
        for s in atoms.symbols:
            elneg.append(elneg_pauling[s])
        atoms.set_array('elneg', np.array(elneg))
        self.atoms.set_array('elneg', np.array(elneg))        

        if self.oxi:
            ri = []
            for s, o in zip(self.atoms.symbols, self.atoms.arrays['oxi_states']):
                ri.append(ionic_radii[s][o])
            self.atoms.set_array('ri', np.array(ri))  
        self.sg = get_spacegroup(self.atoms)



    def _set_symmetry_labels(self, atoms, symprec):
        
        spglib_cell = (atoms.cell,
                       atoms.get_scaled_positions(),
                       atoms.numbers)
        equivalent_sites = get_symmetry_dataset(spglib_cell, symprec)['equivalent_atoms']
        atoms.set_array('sym_label', np.array(equivalent_sites))
        
            

    def _create_supercell(self, atoms):
        scale_a = np.ceil(self.r_cut/(self.cell.volume/np.linalg.norm(np.cross(self.cell[2, :], self.cell[1, :]))))
        scale_b = np.ceil(self.r_cut/(self.cell.volume/np.linalg.norm(np.cross(self.cell[0, :], self.cell[2, :]))))
        scale_c = np.ceil(self.r_cut/(self.cell.volume/np.linalg.norm(np.cross(self.cell[0, :], self.cell[1, :]))))
        scale = np.array([scale_a, scale_b, scale_c])
        scale = np.where(scale < 3, 3, scale)        
        #mobile_atoms = self.atoms[self.atoms.numbers == self.specie]
        supercell = Box(atoms)._super_box(scale, center = True)
        return supercell



    def _voro_data(self, supercell, central_points_ids):
        """
        Modified pymatgen's version
        """
        
        # if mobile_sublattice:
        #     central_points_ids = central_points_ids[atoms.numbers == self.specie]


        voro = Voronoi(supercell.positions)
        all_vertices = voro.vertices
        positions = supercell.positions
        # Get the coordinates of the central site

        poly_data = {}
        all_vertices = voro.vertices
        positions = voro.points
        unitcell_ids = supercell.get_array('index')
        for site_id in central_points_ids:
            center_coords = positions[site_id]
            data = {
                'volume': [],
                'area': [],
                'face_dist': [],
                'solid_angle': [],
                'weight': [],
                'nn_id': [],
                'nn_id_unitcell': [],
            }
            if self.oxi:
                data.update({'ionic_dist': []})

            #other_sites = []
            for nn, vind in voro.ridge_dict.items():
                # Get only those that include the site in question
                if site_id in nn:
                    other_site = nn[0] if nn[1] == site_id else nn[1]
                    if -1 in vind:
                        raise RuntimeError("This structure is pathological, infinite vertex in the Voronoi construction")
                    # Get the solid angle of the face
                    
                    facets = [all_vertices[i] for i in vind]
                    angle = _solid_angle(center_coords, facets)
                    # Compute the volume of associated with this face
                    volume = 0
                    # qvoronoi returns vertices in CCW order, so I can break
                    # the face up in to segments (0,1,2), (0,2,3), ... to compute
                    # its area where each number is a vertex size
                    for j, k in zip(vind[1:], vind[2:]):
                        volume += _vol_tetra(
                            center_coords,
                            all_vertices[vind[0]],
                            all_vertices[j],
                            all_vertices[k],
                        )
                    # Compute the distance of the site to the face
                    face_dist = np.linalg.norm(center_coords - positions[other_site]) / 2
                    # Compute the area of the face (knowing V=Ad/3)
                    face_area = 3 * volume / face_dist
                    # Compute the normal of the facet
                    normal = np.subtract(positions[other_site], center_coords)
                    normal /= np.linalg.norm(normal)

                    data['nn_id'].append(other_site)
                    data['nn_id_unitcell'].append(unitcell_ids[other_site])
                    data['volume'].append(volume)
                    data['area'].append(face_area)
                    data['solid_angle'].append(angle)
                    data['face_dist'].append(face_dist)   
                    if self.oxi: 
                        ri = supercell.arrays['ri']
                        data['ionic_dist'].append(2 * face_dist - ri[[site_id, other_site]].sum())
                    
            for key in data:
                data[key] = np.array(data[key])
            data['effective_CN'] = _effective_CN(data['area'])
            data['weight'] = data['solid_angle']/data['solid_angle'].sum()
            data['elneg'] = np.take(supercell.arrays['elneg'], data['nn_id'])
            data['bond_ionicity'] = abs(supercell.arrays['elneg'][site_id] - supercell.arrays['elneg'][data['nn_id']])
            data['weighted_elneg'] = data['elneg'] * data['weight']
            if self.oxi:
                data['oxi_state'] = np.take(supercell.arrays['oxi_states'], data['nn_id'])
                data['weighted_oxi_state'] = data['oxi_state'] * data['weight']
            poly_data.update({site_id: data})

        return poly_data



    def _percolation_radii(self):
        
        pl = Percolator(self.atoms, self.specie, self.r_cut)
        self.pl = pl
        radii = {
            'r1d': pl.percolation_threshold(1), 
            'r2d': pl.percolation_threshold(2),
            'r3d': pl.percolation_threshold(3)
            }
        return radii
    
    
    def _covalent_free_space(self):
        
        framework = self.atoms[self.atoms.numbers != self.specie].copy()
        v_cell = framework.cell.volume
        v_atoms = (4/3 * np.pi * (covalent_radii[framework.numbers]) ** 3).sum()
        v_free = (v_cell - v_atoms) / len(framework)
        
        return {'framework_covalent_free_space_fraction': v_free}
    
    
    
    def _covalent_packing_fraction(self):
        
        v = self.atoms.cell.volume
        atoms_volume = (4/3 * np.pi * np.array([covalent_radii[n] for n in self.atoms.numbers]) ** 3).sum()
        fraction = (v - atoms_volume) / v
        return {'covalent_packing_fraction_crystal': fraction}
    
    
    
    def _covalent_packing_fraction_framework(self):
        
        v = self.atoms.cell.volume
        rc = np.array([covalent_radii[n] for n in self.atoms.numbers if n != self.specie])
        atoms_volume = (4/3 * np.pi * (rc ** 3)).sum()
        fraction = (v - atoms_volume) / v
        return {'covalent_packing_fraction_framework': fraction}
    
    
    
    
    def _volume_per_atom(self):
        
        vpa = self.atoms.cell.volume / len(self.atoms)
        return {'vpa': vpa}
    

    def _volume_per_atom_framework(self):

        framework = self.atoms[self.atoms.numbers != self.specie].copy()
        vpa = framework.cell.volume / len(framework)
        return {'vpa_framework': vpa}
    

    
    def _free_volume_per_atom_framework(self):
        
        v = self.atoms.cell.volume
        rc = np.array([covalent_radii[n] for n in self.atoms.numbers if n != self.specie])
        atoms_volume = (4/3 * np.pi * (rc ** 3)).sum()
        fvpa = (v - atoms_volume) / len(rc)
        return {'free_framework_volume_per_atom': fvpa}
    
    
    def _calc_stat(self, arr, stat):

        if stat == 'min':
            return arr.min()
        if stat =='max':
            return arr.max()
        if stat == 'range':
            return arr.max() - arr.min()
        if stat == 'std':
            return arr.std()
        if stat == 'mean':
            return arr.mean()   
        
    def _process_site_poly_data(self, site_poly_data, stats = ['min', 'mean']):
        features = {}
        for key in site_poly_data.keys():
            if key in ['nn_id_unitcell', 'nn_id', 'weight']:
                continue
            if key == 'effective_CN':
                features.update({key: site_poly_data[key]})
                continue
            if key in ('volume', 'area'):
                features.update({key: site_poly_data[key].sum()})
                continue
            if key in ('solid_angle'):
                features.update({f'max_{key}': site_poly_data[key].max()})
                continue
            for stat in stats:
                features.update({f'{stat}_{key}': self._calc_stat(site_poly_data[key], stat)})
        return features


    def _mobile_ion_hops_centroids(self):
        mobile_atoms = self.atoms[self.atoms.numbers == self.specie]
        supercell = self._create_supercell(mobile_atoms)
        central_points_ids = np.arange(len(mobile_atoms) * supercell.info['translation_id'][(0,0,0)],
                                      (len(mobile_atoms) * (supercell.info['translation_id'][(0,0,0)] + 1)))
    

        nn = _find_vneighbors(supercell.positions, central_points_ids)
        edges = []
        for source in nn.keys():
            for target in nn[source]:
                if source > target:
                    edge = (target, source)
                else:
                    edge = (source, target)
                if edge not in edges:
                    edges.append(edge)
        edges = np.array(edges)
        centroids = (supercell.positions[edges]).sum(axis = 1)/2
        return centroids
        
    
    def featurize(self, stats = ['mean', 'min']):

        features = {}
        features.update(self._percolation_radii())
        features.update(self._covalent_free_space())
        features.update(self._covalent_packing_fraction())
        features.update(self._covalent_packing_fraction_framework())
        features.update(self._volume_per_atom())
        features.update(self._volume_per_atom_framework())
        features.update(self._free_volume_per_atom_framework())

        centroids = self._mobile_ion_hops_centroids()


        label = 'mobile_ions_framework'
        atoms = self.atoms.copy()
        supercell = self._create_supercell(atoms)
        central_points_ids = np.arange(len(atoms) * supercell.info['translation_id'][(0,0,0)],
                                      (len(atoms) * (supercell.info['translation_id'][(0,0,0)] + 1))) 
        central_points_ids = central_points_ids[atoms.numbers == self.specie]

        poly_data = self._voro_data(supercell, central_points_ids)
        site_features = {}
        for site in poly_data:
            site_poly_data = poly_data[site]
            data = self._process_site_poly_data(site_poly_data, ['min', 'mean'])
            for key in data.keys():
                if key in site_features.keys():
                    site_features[key].append(data[key])
                else:
                    site_features.update({key: [data[key]]})
        for stat in stats:
            for key in site_features.keys():
                features.update({f'{label}_{stat}_{key}': self._calc_stat(np.array(site_features[key]), stat)})


        label = 'mobile_sublattice'
        atoms = self.atoms[self.atoms.numbers == self.specie].copy()
        supercell = self._create_supercell(atoms)
        central_points_ids = np.arange(len(atoms) * supercell.info['translation_id'][(0,0,0)],
                                      (len(atoms) * (supercell.info['translation_id'][(0,0,0)] + 1))) 
        poly_data = self._voro_data(supercell, central_points_ids)
        site_features = {}
        for site in poly_data:
            site_poly_data = poly_data[site]
            data = self._process_site_poly_data(site_poly_data, ['min', 'mean'])
            for key in data.keys():
                if key in site_features.keys():
                    site_features[key].append(data[key])
                else:
                    site_features.update({key: [data[key]]})
        for stat in stats:
            for key in site_features.keys():
                features.update({f'{label}_{stat}_{key}': self._calc_stat(np.array(site_features[key]), stat)})



        label = 'centroid'
        
        atoms = self.atoms[self.atoms.numbers != self.specie].copy()
        supercell = self._create_supercell(atoms)
        central_points_ids = np.arange(len(supercell), len(supercell) + len(centroids))
        for p in centroids:
            supercell.append(self.specie)
            supercell.positions[-1] = p
        poly_data = self._voro_data(supercell, central_points_ids)
        site_features = {}
        for site in poly_data:
            site_poly_data = poly_data[site]
            data = self._process_site_poly_data(site_poly_data, ['min', 'mean'])
            for key in data.keys():
                if key in site_features.keys():
                    site_features[key].append(data[key])
                else:
                    site_features.update({key: [data[key]]})
        for stat in stats:
            for key in site_features.keys():
                features.update({f'{label}_{stat}_{key}': self._calc_stat(np.array(site_features[key]), stat)})



        if self.oxi:
            label = 'anions'
            atoms = self.atoms[self.atoms.arrays['oxi_states'] < 0].copy()
            supercell = self._create_supercell(atoms)
            central_points_ids = np.arange(len(atoms) * supercell.info['translation_id'][(0,0,0)],
                                        (len(atoms) * (supercell.info['translation_id'][(0,0,0)] + 1))) 
            poly_data = self._voro_data(supercell, central_points_ids)
            site_features = {}
            for site in poly_data:
                site_poly_data = poly_data[site]
                data = self._process_site_poly_data(site_poly_data, ['min', 'mean'])
                for key in data.keys():
                    if key in site_features.keys():
                        site_features[key].append(data[key])
                    else:
                        site_features.update({key: [data[key]]})
            for stat in stats:
                for key in site_features.keys():
                    features.update({f'{label}_{stat}_{key}': self._calc_stat(np.array(site_features[key]), stat)})


            label = 'cations'
            atoms = self.atoms[self.atoms.arrays['oxi_states'] > 0].copy()
            supercell = self._create_supercell(atoms)
            central_points_ids = np.arange(len(atoms) * supercell.info['translation_id'][(0,0,0)],
                                        (len(atoms) * (supercell.info['translation_id'][(0,0,0)] + 1))) 
            poly_data = self._voro_data(supercell, central_points_ids)
            site_features = {}
            for site in poly_data:
                site_poly_data = poly_data[site]
                data = self._process_site_poly_data(site_poly_data, ['min', 'mean'])
                for key in data.keys():
                    if key in site_features.keys():
                        site_features[key].append(data[key])
                    else:
                        site_features.update({key: [data[key]]})
            for stat in stats:
                for key in site_features.keys():
                    features.update({f'{label}_{stat}_{key}': self._calc_stat(np.array(site_features[key]), stat)})

            label = 'centroid_anions'
            #centroids = self._mobile_ion_hops_centroids()
            atoms = self.atoms[self.atoms.arrays['oxi_states'] < 0].copy()
            supercell = self._create_supercell(atoms)
            central_points_ids = np.arange(len(supercell), len(supercell) + len(centroids))
            for p in centroids:
                supercell.append(self.specie)
                supercell.positions[-1] = p
            poly_data = self._voro_data(supercell, central_points_ids)
            site_features = {}
            for site in poly_data:
                site_poly_data = poly_data[site]
                data = self._process_site_poly_data(site_poly_data, ['min', 'mean'])
                for key in data.keys():
                    if key in site_features.keys():
                        site_features[key].append(data[key])
                    else:
                        site_features.update({key: [data[key]]})
            for stat in stats:
                for key in site_features.keys():
                    features.update({f'{label}_{stat}_{key}': self._calc_stat(np.array(site_features[key]), stat)})
            label = 'centroid_cations'
            #centroids = self._mobile_ion_hops_centroids()
            atoms = self.atoms[self.atoms.arrays['oxi_states'] > 0].copy()
            supercell = self._create_supercell(atoms)
            central_points_ids = np.arange(len(supercell), len(supercell) + len(centroids))
            for p in centroids:
                supercell.append(self.specie)
                supercell.positions[-1] = p
            poly_data = self._voro_data(supercell, central_points_ids)
            site_features = {}
            for site in poly_data:
                site_poly_data = poly_data[site]
                data = self._process_site_poly_data(site_poly_data, ['min', 'mean'])
                for key in data.keys():
                    if key in site_features.keys():
                        site_features[key].append(data[key])
                    else:
                        site_features.update({key: [data[key]]})
            for stat in stats:
                for key in site_features.keys():
                    features.update({f'{label}_{stat}_{key}': self._calc_stat(np.array(site_features[key]), stat)})
        return features