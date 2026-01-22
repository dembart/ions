import numpy as np
from .geometry import lineseg_dists, repeated_supercell



class Edge:

    def __init__(self, atoms, source, target, offset):

        """
        Helper class for handling ionic jumps (edges) under PBC.

        Parameters
        ----------
        atoms : ase.Atoms
            Structure containing both endpoints.

        source : int
            Index of the moving atom.

        target : int
            Index of the target site atom.

        offset : array-like of float, shape (3,)
            lattice translation connecting source to target.

        """

        self.atoms = atoms.copy()
        self.source = int(source)
        self.target = int(target)
        self.offset = np.array(offset)
        self.cell = self.atoms.cell

    @property
    def p1(self):
        return self.atoms.positions[self.source]

    @property
    def p2(self):
        return self.atoms.positions[self.target] + self.offset @ self.cell

    @property
    def length(self):
        return np.linalg.norm(self.p2 - self.p1)

    @property
    def centroid(self):
        return 0.5 * (self.p1 + self.p2)


    @property
    def wrapped_target(self):
        """
        Find the index of the atom in the unit cell corresponding
        to the periodic image of `target + offset`.
        """

        atoms = self.atoms
        target = self.target
        offset = self.offset

        cell = atoms.cell

        frac_target = cell.scaled_positions(atoms.positions[target]) + offset
        frac_target -= np.floor(frac_target)  # wrap to [0,1)

        frac_atoms = cell.scaled_positions(atoms.positions)
        frac_atoms -= np.floor(frac_atoms)

        ids = np.where(atoms.numbers == atoms.numbers[target])[0]

        dfrac = frac_atoms[ids] - frac_target
        dfrac -= np.round(dfrac)

        disp = dfrac @ cell
        dist = np.linalg.norm(disp, axis=1)

        i = np.argmin(dist)
        if dist[i] > 1e-6:
            raise RuntimeError(
                f"Wrapped target not found (min dist = {dist[i]:.3e} Angst.)"
            )

        return int(ids[i])



    def project_point(self, p):
        """
        Orthogonal projection of point p onto the edge segment.
        """
        ab = self.p2 - self.p1
        t = np.dot(p - self.p1, ab) / np.dot(ab, ab)
        return self.p1 + t * ab



    def nearest_atom_to_edge(self, exclude_endpoints=True):
        """
        Find nearest atom to the edge.

        Returns
        -------
        idx: int
            Atom index
        
        dist: float
            Distance to edge
        
        proj: ndarray (3,)
            Projection point on edge

        """
        pos = self.atoms.positions
        mask = np.ones(len(pos), dtype=bool)

        if exclude_endpoints:
            mask[[self.source, self.wrapped_target]] = False

        candidates = np.where(mask)[0]
        dists = lineseg_dists(pos[candidates], self.p1, self.p2)

        i = np.argmin(dists)
        idx = candidates[i]

        return idx, dists[i], self.project_point(pos[idx])



    def superedge(self, r_cut):
        """
        Create an edge within a supercell

        Parameters
        ----------

        r_cut: float
            supercell minimum size
        
        Returns
        -------
        edge for the supercell

        """
        cell = self.atoms.cell
        h1 = cell.volume/np.linalg.norm(np.cross(cell[2, :], cell[1, :]))
        h2 = cell.volume/np.linalg.norm(np.cross(cell[0, :], cell[2, :]))
        h3 = cell.volume/np.linalg.norm(np.cross(cell[0, :], cell[1, :]))
        scale = np.ceil(r_cut/np.array([h1, h2, h3]))
        supercell = repeated_supercell(self.atoms, scale)
        return Edge(supercell, self.source, self.target, self.offset/scale)
        


    def interpolate(self, n_images=5, center=True):
        
        """
        Generate interpolated images along the edge.

        Parameters
        ----------
        spacing : float
            Approximate spacing between images (Å)
        center : bool
            Center path in the simulation cell

        Returns
        -------
        images : list of ase.Atoms
        """

        traj = np.linspace(self.p1, self.p2, n_images)
        wrapped_target = self.wrapped_target

        images = []
        for p in traj:

            atoms = self.atoms.copy()
            atoms.positions[self.source] = p
    
            moving = np.zeros(len(atoms), dtype=bool)
            moving[self.source] = True
            atoms.set_array("moving", moving)
            
            # remove wrapped target
            keep = [i for i in range(len(atoms)) if i != wrapped_target]
            atoms = atoms[keep]

            images.append(atoms)

        if center:
            shift = 0.5 * self.cell.sum(axis=0) - self.centroid
            for img in images:
                img.positions += shift
                img.wrap()

        return images



    def __repr__(self):
        return (
            f"Edge(source={self.source}, "
            f"target={self.target}, "
            f"offset={tuple(self.offset)}, "
            f"length={self.length:.2f})"
        )
