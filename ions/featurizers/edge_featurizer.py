import itertools
import numpy as np
from scipy.spatial import cKDTree

from ase.data import covalent_radii
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.local_env import VoronoiNN

from ..edge import Edge 
from ..geometry import VoroSite, _geometric_median, lineseg_dists, project_point_on_edge
from ..data import ionic_radii, elneg_pauling




class EdgeFeaturizer(Edge):

    """
    Featurizes edge connecting source and target 
    positions of a mobile atom 

    Most of the features are derived from the Voronoi partitioning
    Works similarly to the pymatgen' VoronoiNN 
    but should be faster as the smaller r_cut is needed

    We use solid_angle/max(solid_angle) as the weights for each neighbor

    The following descriptors are calculated for source, target, gm, and nnp positions
    of a mobile ion, and statistics of some of these:

    CN
    volume
    area
    face_dist
    ionic_dist
    elneg
    weighted_elneg
    oxi_state
    weighted_oxi_state
    length
    cylinder_volume
    cylinder_volume^1/3
    cylinder_ionic_volume^1/3
    has_Element
    """


    def __init__(self, edge, oxi = False, use_VoronoiNN = False, bvse = False):

        """
        Parameters
        ----------

        edge: ions.geom.Edge object
            edge connecting source and target positions of a mobile atom
            
        oxi: boolean, False by defalut
            whether collect features associated with oxidation states
            Note: edge.atoms should contain 'oxi_states' array
        
        bvse: boolean, False by defalut
            whether collect features associated with BVSE parameters (alpha, d0, r_min)
            Note: edge.atoms should contain 'oxi_states' array
            Not implemented yet

        use_VoronoiNN: boolean, False by defalut
            use pymatgen's VoronoiNN instead of the custom ions.geom.utils.VoroSite
            VornoiNN is supposed to be more accurate but takes more compute time

        """

        super().__init__(edge.atoms.copy(), edge.source, edge.target, edge.offset)
        self.st = AseAtomsAdaptor.get_structure(self.atoms.copy())
        self.use_VoronoiNN = use_VoronoiNN
        self._set_covalent_radii()
        self._set_elneg()
        self.oxi = oxi
        if self.oxi:
            self._set_ionic_radii()
        # if self.bvse:
        #     collect_bvse_params(self.atoms, self.atoms.symbols[self.target], self.atoms.arrays['oxi_states'][self.source])


    def __repr__(self):
        return f"EdgeFeaturizer(source={self.source}, target={self.target}, offset={self.offset}, atoms={self.atoms})"


    def _set_ionic_radii(self):
        ri = np.array([ionic_radii[s][q] for (s, q) in zip(self.atoms.symbols, self.atoms.arrays['oxi_states'])])
        self.atoms.set_array('ri', ri)

    def _set_covalent_radii(self):
        rc = np.array([covalent_radii[n] for n in self.atoms.numbers])
        self.atoms.set_array('rc', rc)

    def _set_elneg(self):
        elneg = np.array([elneg_pauling[s] for s in self.atoms.symbols])
        self.atoms.set_array('elneg', elneg)



    def _vcutoff(self):
        """from pymatgen.analysis.local_env"""
        corners = [[1, 1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, -1]]
        d_corners = [np.linalg.norm(self.atoms.cell.cartesian_positions(c)) for c in corners]
        max_cutoff = max(d_corners) + 0.01
        return max_cutoff

    def _length(self):
        return super().length
    
    def min_distance_to_edge(self):
        rest_idx = [i for i in range(len(self.atoms)) if i not in (self.source, self.target)]
        positions = self.atoms.positions[rest_idx]
        translations = np.array(list(itertools.product(
                                        [0, 1, -1], # should be [0, 1, -1, 2, -2] ideally
                                        [0, 1, -1],
                                        [0, 1, -1])))
        coords = []
        idx = []
        for tr in translations:
            coords.append(positions + np.dot(tr, self.cell))
            idx.append(rest_idx)
        ii = np.hstack(idx)
        p = np.vstack(coords)
        #self._coords = p
        dd  = lineseg_dists(p, self.p1, self.p2)
        d_min = min(dd)
        nn_id = ii[dd == dd.min()].ravel()[0]
        nearest_point = p[dd == dd.min()]
        edge_point = project_point_on_edge(nearest_point, self.p1 , self.p2)
        return d_min, nn_id, edge_point

    def _effective_CN(self, areas):
        CN = (areas.sum()**2) / ((areas**2).sum())
        return CN
    

    
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
    
    def _cylinder_volume_of_max_percolation_radius(self, min_dist):
        return np.pi * (min_dist**2) * self.length


    
    def _get_voro_data(self, atoms, site_id):
        if self.use_VoronoiNN:
            # shift = np.dot([0.5, 0.5, 0.5], atoms.cell) - atoms.positions[site_id]
            # atoms.positions += shift
            # atoms.wrap()
            st = AseAtomsAdaptor.get_structure(atoms)
            calc = VoronoiNN(cutoff = self._vcutoff(), weight='solid_angle')
            data_voro = calc.get_nn_info(st, site_id)
            #self.data_voro = data_voro
            data = {
                'volume': np.array([dat['poly_info']['volume'] for dat in data_voro]),
                'area': np.array([dat['poly_info']['area'] for dat in data_voro]),
                'face_dist': np.array([dat['poly_info']['face_dist'] for dat in data_voro]),
                'solid_angle': np.array([dat['poly_info']['solid_angle'] for dat in data_voro]),
                'nn_id_unitcell': np.array([dat['site_index'] for dat in data_voro]),
                'weight': np.array([dat['weight'] for dat in data_voro])
            }
        else:
            calc = VoroSite(atoms)
            data, supercell = calc.get_poly_data(site_id)
            data['weight'] = data['solid_angle'] / data['solid_angle'].max()
            
        data.update({'rc': atoms.arrays['rc'][data['nn_id_unitcell']]})
        data.update({'elneg': atoms.arrays['elneg'][data['nn_id_unitcell']]})
        data.update({'elneg_weighted': data['weight'] * atoms.arrays['elneg'][data['nn_id_unitcell']]})
        if self.oxi:
            data.update({'ri': atoms.arrays['ri'][data['nn_id_unitcell']]})
            data.update({'oxi_state': atoms.arrays['oxi_states'][data['nn_id_unitcell']]})
            data.update({'oxi_state_weighted': data['weight'] * atoms.arrays['oxi_states'][data['nn_id_unitcell']]})
            data.update({'ionic_dist': 2 * data['face_dist'] - data['ri']})
        return data
    
    def _get_voro_features(self):

        #source
        data_source = self._get_voro_data(self.atoms, self.source)
        #target
        data_target = self._get_voro_data(self.atoms, self.source)

        # nnp
        d_min, nn_id, pos = self.min_distance_to_edge()
        atoms = self.atoms.copy()
        site_id = min(self.source, self.target)
        atoms.positions[site_id] = pos
        atoms = atoms[[i for i in range(len(atoms)) if i != max(self.source, self.target)]]
        data_nnp = self._get_voro_data(atoms, site_id)

        # gm
        CN = len(data_nnp['nn_id_unitcell'])
        atoms = self.atoms.copy()
        site_id = min(self.source, self.target)
        rest_idx = [i for i in range(len(atoms)) if i not in (self.source, self.target)]
        atoms = atoms[[i for i in range(len(atoms)) if i != max(self.source, self.target)]]
        positions = self.atoms.positions
        translations = np.array(list(itertools.product(
                                        [0, 1, -1], # should be [0, 1, -1, 2, -2] ideally
                                        [0, 1, -1],
                                        [0, 1, -1])))
        coords = []
        idx = []
        for tr in translations:
            coords.append(positions[rest_idx] + np.dot(tr, self.cell))
            idx.append(rest_idx)

        ii = np.hstack(idx)
        p = np.vstack(coords)
        tree = cKDTree(p)
        dd, ii = tree.query(pos, distance_upper_bound=10.0, k = CN) 
        gm = _geometric_median(p[ii], pos) # the initial guess is from nnp
        atoms.positions[site_id] = gm
        data_gm = self._get_voro_data(atoms, site_id)
        return data_source, data_target, data_nnp, data_gm
     

    def _is_element_in_structure(self, possible_elements = ['O', 'F', 'P', 'S', 'B', 'I', 'F', 'Cl', 'N']):
        data = {}
        for element in possible_elements:
            data.update({f'has_{element}': int(element in self.atoms.symbols)})
        return data

    

    def featurize(self, stats = ['min', 'max', 'mean', 'range']):

        """
        Featurize edge

        Parameters
        ----------

        stats: list of str, ['min', 'max', 'mean', 'range'] by default
            which feature statistics to calculate
            possible stats are 'min', 'max', 'mean', 'range', 'std'
        
        Returns
        ----------
        features: dict, {feature_label: feature, ...}
            calculated features
        """

        features = {}
        features.update({'length': self._length()})
        d_min, nn_id, pos = self.min_distance_to_edge()
        features.update({'min_dist_to_edge': d_min})

        data_source, data_target, data_nnp, data_gm = self._get_voro_features()

        percolation_radii = 2 * min(
                                data_source['face_dist'].min(),
                                data_target['face_dist'].min(),
                                data_nnp['face_dist'].min(),
                                data_gm['face_dist'].min()
                                )
        features.update({'cylinder_volume': self._cylinder_volume_of_max_percolation_radius(percolation_radii)})
        features.update({'cylinder_volume^1/3': features['cylinder_volume'] ** 1/3}) 
        if self.oxi:
            percolation_ionic_radii =  min(
                                data_source['face_dist'].min(),
                                data_target['face_dist'].min(),
                                data_nnp['face_dist'].min(),
                                data_gm['face_dist'].min()
                                )
        if self.oxi:
            features.update({'cylinder_ionic_volume': self._cylinder_volume_of_max_percolation_radius(percolation_ionic_radii)})
            features.update({'cylinder_ionic_volume^1/3': features['cylinder_ionic_volume'] ** 1/3}) 

        CN_source = self._effective_CN(data_source['area'])
        CN_target = self._effective_CN(data_target['area'])
        CN_nnp = self._effective_CN(data_nnp['area'])
        CN_gm = self._effective_CN(data_gm['area'])
        arr = np.array([CN_source, CN_target, CN_nnp, CN_gm])
        for stat in stats:
            features.update({f'{stat}_CN': self._calc_stat(arr, stat = stat)})


        total_volume_source = data_source['volume'].sum()
        total_volume_target = data_target['volume'].sum()
        total_volume_nnp = data_nnp['volume'].sum()
        total_volume_gm = data_gm['volume'].sum()
        arr = np.array([total_volume_source, total_volume_target, total_volume_nnp, total_volume_gm])
        for stat in stats:
            features.update({f'{stat}_volume': self._calc_stat(arr, stat = stat)})



        total_area_source = data_source['area'].sum()
        total_area_target = data_target['area'].sum()
        total_area_nnp = data_nnp['area'].sum()
        total_area_gm = data_gm['area'].sum()
        arr = np.array([total_area_source, total_area_target, total_area_nnp, total_area_gm])
        for stat in stats:
            features.update({f'{stat}_area': self._calc_stat(arr, stat = stat)})



        for property in data_source.keys():
            if property in ['weight', 'nn_id', 'nn_id_unitcell']:
                continue
            for stat1 in stats:
                property_source = self._calc_stat(data_source[property], stat = stat1)
                property_target = self._calc_stat(data_target[property], stat = stat1)
                property_nnp = self._calc_stat(data_nnp[property], stat = stat1)
                property_gm = self._calc_stat(data_gm[property], stat = stat1)
                arr = np.array([property_source, property_target, property_nnp, property_gm])
                for stat2 in stats:
                    features.update({f'{stat2}_{stat1}_{property}': self._calc_stat(arr, stat = stat2)})
        features.update(self._is_element_in_structure())
        return features