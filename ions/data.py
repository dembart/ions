import sys
import os 
import pickle
import json

__version__ = '0.1.3'


def _resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base_path, relative_path)
    return path

data_path = _resource_path('data')



radii_file = os.path.join(data_path, 'shannon-radii.pickle')
with open(radii_file, 'rb') as f:
  shannon_data = pickle.load(f)



radii_file_ionic = os.path.join(data_path, 'shannon-data-ionic.pickle')
with open(radii_file_ionic, 'rb') as f:
  ionic_radii = pickle.load(f)



radii_file_crystal = os.path.join(data_path, 'shannon-data-crystal.pickle')
with open(radii_file_ionic, 'rb') as f:
  crystal_radii = pickle.load(f)



bv_file = os.path.join(data_path, 'bvparams2020-average.pickle')
with open(bv_file, 'rb') as f:
  bv_data = pickle.load(f)



bvse_file = os.path.join(data_path, 'bvse_data.pickle')
with open(bvse_file, 'rb') as f:
  bvse_data = pickle.load(f)



quantum_n_file = os.path.join(data_path, 'quantum_n.pkl')
with open(quantum_n_file,'rb') as f:
     principle_number = pickle.load(f)


elneg_file = os.path.join(data_path, 'elneg_data.json')
with open(elneg_file,'r') as f:
     elneg_pauling = json.load(f)