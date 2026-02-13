import numpy as np
from collections import deque



class PeriodicGraph:

    def __init__(self, sources, targets, offsets):
        self.sources = sources
        self.targets = targets
        self.offsets = offsets

    def find_periodicity(self):
        """
        Returns:
        -------
        periodicity: int
            Percolation dimensionality (0, 1, 2, or 3)
        
        translations: list
            independent translation vectors
        """

        # visited[atom_index] = lattice_offset at first encounter
        visited = {}
        translations = []

        all_nodes = np.unique(self.sources) # need to consider sym_labels in future
        
        for start in all_nodes:
            if start in visited:
                continue

            # BFS queue: (atom_index, lattice_offset)
            queue = deque()
            start = int(np.min(self.sources))
            queue.append((start, np.zeros(3, dtype=int)))
            visited[start] = np.zeros(3, dtype=int)

            while queue:
                i, offset_i = queue.popleft()
                # find neighbors of i
                mask = self.sources == i
                for j, off_ij in zip(self.targets[mask], self.offsets[mask]):
                    offset_j = offset_i + off_ij

                    if j not in visited:
                        visited[j] = offset_j
                        queue.append((j, offset_j))
                    else:
                        # found a translational equivalence
                        delta = offset_j - visited[j]
                        if not np.all(delta == 0):
                            translations.append(delta)

        independent = self._independent_vectors(translations)
        return len(independent), independent

    @staticmethod
    def _independent_vectors(vectors, tol=1e-8):
        """
        Returns a list of linearly independent integer translation vectors.
        """
        if not vectors:
            return []

        basis = []
        for v in vectors:
            v = np.array(v, dtype=float)
            if not basis:
                basis.append(v)
                continue

            mat = np.vstack(basis + [v])
            if np.linalg.matrix_rank(mat, tol=tol) > len(basis):
                basis.append(v)

            if len(basis) == 3:
                break

        # convert back to integer vectors
        return [np.round(v).astype(int) for v in basis]