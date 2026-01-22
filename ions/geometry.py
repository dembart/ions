import numpy as np 
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist, euclidean
from ase.neighborlist import NeighborList



def repeated_supercell(atoms, scale, center = False, tol=1e-10):

    atoms = atoms.copy()
    atoms.set_array('index', np.arange(len(atoms)))
    supercell = atoms.repeat([int(s) for s in scale])

    if center:
        shift = np.dot([0.5, 0.5, 0.5], supercell.cell)
        supercell.translate(shift)

    frac = np.linalg.solve(atoms.cell.T, supercell.positions.T).T

    # Separate integer and fractional parts
    offsets = np.floor(frac + tol).astype(int)
    # intra_frac = frac - offsets # for debug
    supercell.set_array('offset', offsets)
    return supercell



def lineseg_dists(p, a, b):
    
    """Distance from p to the linear sigment (a, b)"""
    # source:    https://stackoverflow.com/questions/54442057/
    # calculate-the-euclidian-distance-between-an-array-of-points-to-a-line-segment-in/
    # 54442561#54442561 
    
    if np.all(a == b):
        return np.linalg.norm(p - a, axis=1)
    d = np.divide(b - a, np.linalg.norm(b - a))
    s = np.dot(a - p, d)
    t = np.dot(p - b, d)
    h = np.maximum.reduce([s, t, np.zeros(len(p))])
    c = np.cross(p - a, d)
    return np.hypot(h, np.linalg.norm(c, axis = 1))



def project_point_on_edge(p, p1, p2):
    """Orthogonal projection of point p onto the linear segment (p1,p2)"""
    ap = p-p1
    ab = p2 - p1
    result = p1 + np.dot(ap,ab)/np.dot(ab,ab) * ab
    return result


"""some geometry utils copied from pymatgen"""

def _effective_CN(areas):
    CN = (areas.sum()**2) / ((areas**2).sum())
    return CN


def _find_vneighbors(points, central_points_ids, key=1, min_dist=-float("inf"), max_dist=float("inf")):
    """soure: https://github.com/mcs-cice/IonExplorer2/blob/main/geometry.py
     Parameter mod can takes values 1, 2, or 3 that correspond to the
    search for domains adjacent by vertices, edges or faces.
    """

    neighbors = {i: None for i in central_points_ids}
    vor = Voronoi(points)
    for i in central_points_ids:
        cp = points[i]
        region = vor.regions[vor.point_region[i]]
        if -1 in region:
            raise ValueError("The domain for \"" + str(i) + "\" point is not closed!")
        local_neighbors = []
        for j in range(len(points)):
            numb_common_vertices = len(np.intersect1d(region, vor.regions[vor.point_region[j]]))
            if i != j and numb_common_vertices >= key and min_dist < np.linalg.norm(cp - points[j]) < max_dist:
                local_neighbors.append(j)
        neighbors[i] =  local_neighbors
    return neighbors



def _vol_tetra(vt1, vt2, vt3, vt4):
    """
    Copied from pymatgen's repo
    Calculate the volume of a tetrahedron, given the four vertices of vt1,
    vt2, vt3 and vt4.

    Args:
        vt1 (array-like): coordinates of vertex 1.
        vt2 (array-like): coordinates of vertex 2.
        vt3 (array-like): coordinates of vertex 3.
        vt4 (array-like): coordinates of vertex 4.

    Returns:
        (float): volume of the tetrahedron.

    """
    vol_tetra = np.abs(np.dot((vt1 - vt4), np.cross((vt2 - vt4), (vt3 - vt4)))) / 6    
    return vol_tetra



def _solid_angle(center, coords):
    """
    Helper method to calculate the solid angle of a set of coords from the
    center.

    Args:
        center (3x1 array): Center to measure solid angle from.
        coords (Nx3 array): List of coords to determine solid angle.

    Returns:
        The solid angle.
    """
    # Compute the displacement from the center
    r = [np.subtract(c, center) for c in coords]

    # Compute the magnitude of each vector
    r_norm = [np.linalg.norm(i) for i in r]

    # Compute the solid angle for each tetrahedron that makes up the facet
    #  Following: https://en.wikipedia.org/wiki/Solid_angle#Tetrahedron
    angle = 0
    for i in range(1, len(r) - 1):
        j = i + 1
        tp = np.abs(np.dot(r[0], np.cross(r[i], r[j])))
        de = (
            r_norm[0] * r_norm[i] * r_norm[j]
            + r_norm[j] * np.dot(r[0], r[i])
            + r_norm[i] * np.dot(r[0], r[j])
            + r_norm[0] * np.dot(r[i], r[j])
        )
        my_angle = (0.5 * np.pi if tp > 0 else -0.5 * np.pi) if de == 0 else np.arctan(tp / de)
        angle += (my_angle if my_angle > 0 else my_angle + np.pi) * 2
    return angle



def _geometric_median(X, y0, eps=1e-5, max_iter = 200):
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



class VoroSite:

    def __init__(self, atoms, upper_bound = 10.0):
        self.atoms = atoms.copy()
        self.cell = atoms.cell
        self.r_cut = min(upper_bound, self._vcutoff())

    def _vcutoff(self):
        """from pymatgen.analysis.local_env"""
        corners = [[1, 1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, -1]]
        d_corners = [np.linalg.norm(self.atoms.cell.cartesian_positions(c)) for c in corners]
        max_cutoff = max(d_corners) + 0.01
        return max_cutoff
    
    def _vneighbors(self, site_id):
        scale = np.ceil(self.r_cut/abs(np.diag(self.cell)))
        #supercell = Box(self.atoms.copy())._super_box(scale)
        supercell = repeated_supercell(self.atoms, scale)
        shift = np.dot([0.5, 0.5, 0.5], supercell.cell) - supercell.positions[site_id]
        supercell.positions += shift
        supercell.wrap()
        voro = Voronoi(supercell.positions)
        unitcell_ids = supercell.get_array('index')
        return voro, supercell
    
    def _vneighbors2(self, site_id):

        nl = NeighborList([self.r_cut] * len(self.atoms), self_interaction=False)
        nl.update(self.atoms)
        indices, offsets = nl.get_neighbors(site_id)
        coords = self.atoms.positions[indices] + np.dot(offsets, self.cell)
        voro = Voronoi(coords)
        return voro, indices
    
    def get_poly_data(self, site_id):
        voro, supercell = self._vneighbors(site_id)
        all_vertices = voro.vertices
        positions = voro.points
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
        unitcell_ids = supercell.get_array('index')
        other_sites = []
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

        for key in data.keys():
            data[key] = np.array(data[key])
        return data, supercell    


