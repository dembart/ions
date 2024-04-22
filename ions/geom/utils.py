import numpy as np 

"""some geometry utils copied from pymatgen"""

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