import numpy as np
import itertools 
from ase import Atoms
from ase.cell import Cell

class Box:

    def __init__(self, atoms, **kwargs):
        self.atoms = atoms.copy()
        self._set_index(self.atoms)
        self.cell = atoms.cell


    def _set_index(self, atoms):
        index = np.arange(len(atoms))
        atoms.set_array('index', index)


    def _super_box(self, scale):
        scales = []
        for s in scale:
            # if -1 in self.offset:
            #    scales.append(np.arange(-s + 1, s-1))
            #else:
            scales.append(np.arange(s))
        translations = tuple(list(itertools.product(
                                scales[0],
                                scales[1],
                                scales[2])))   
        super_p = []
        positions = self.atoms.positions
        self._translations = translations
        #self._translation_id = dict(zip(translations, np.arange(len(translations))))
        arrays = {}
        translation_array = []
        #transaltion_to_id = dict(zip(translations, np.arange(len(translations))))
        for i, translation in enumerate(translations):
            positions_tr = positions + np.dot(translation, self.cell)
            super_p.append(positions_tr)
            translation_array.append(len(positions) * [tuple(translation)])
            for key in self.atoms.arrays.keys():
                if key == 'positions':
                    continue
                arr = self.atoms.arrays[key]
                try:
                    arrays[key] = np.hstack([arrays[key], arr])
                except:
                    arrays[key] = arr

        arrays.update({'offset': np.vstack(translation_array)})
        super_p = np.vstack(super_p)
        cellpar = self.cell.cellpar()
        cellpar[:3] = cellpar[:3] * [scale[0], scale[1], scale[2]]
        supercell = Atoms(numbers = arrays['numbers'], positions = super_p, cell = Cell.fromcellpar(cellpar), pbc = True)
        for key in arrays.keys():
            if key == 'numbers':
                continue
            supercell.set_array(key, arrays[key])
        supercell.info.update({'translation_id': dict(zip(translations, np.arange(len(translations))))})
        assert len(supercell) == len(self.atoms) * scale[0] * scale[1] * scale[2]
        return supercell