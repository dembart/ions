
import json
import pickle
from importlib.resources import files

__version__ = "0.2"


_DATA_DIR = files(__package__)


def _load_pickle(filename):
    with (_DATA_DIR / filename).open("rb") as f:
        return pickle.load(f)


def _load_json(filename):
    with (_DATA_DIR / filename).open("r") as f:
        return json.load(f)



_LOADERS = {
    "shannon_data": lambda: _load_pickle("shannon-radii.pickle"),
    "ionic_radii": lambda: _load_pickle("shannon-data-ionic.pickle"),
    "crystal_radii": lambda: _load_pickle("shannon-data-crystal.pickle"),
    "bv_data": lambda: _load_pickle("bvparams2020-average.pickle"),
    "bvse_data": lambda: _load_pickle("bvse_data.pickle"),
    "principle_number": lambda: _load_pickle("quantum_n.pkl"),
    "elneg_pauling": lambda: _load_json("elneg_data.json"),
}


__all__ = list(_LOADERS.keys())


def __getattr__(name):
    if name not in _LOADERS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = _LOADERS[name]()
    globals()[name] = value  # cache for future access
    return value