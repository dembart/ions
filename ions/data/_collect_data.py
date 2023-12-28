# collect data
import json
import pandas as pd
import os
import pickle

df = pd.read_csv('/Users/artemdembitskiy/Desktop/crystaldata/src/ions/data/bvparm2020_edit.txt' , comment = '#', sep = '\s+')
df_clean = df[df['note'] == '?']
df_merged = df_clean.groupby(['source', 'v1', 'target', 'v2']).mean().round(4).reset_index()
df_merged.to_csv('/Users/artemdembitskiy/Desktop/crystaldata/src/ions/data/bvparams_averaged2020.csv', sep = '\t', index = False)
df_ = df_merged[['source', 'v1', 'target', 'v2', 'r0', 'b']]

data = {}
for k, f in df_.groupby('source'):
    data.update({k: {}})
    for k2, f2 in f.groupby('v1'):
        data[k].update({int(k2): {}})
        for k3, f3 in f2.groupby('target'):
            data[k][k2].update({k3:{}})
            for k4, f4 in f3.groupby('v2'):
                
                params = {'r0': f4['r0'].values[0], 'b': f4['b'].values[0]}
                data[k][k2][k3].update({int(k4): params })

with open('/Users/artemdembitskiy/Desktop/crystaldata/src/ions/data/bvparams2020-average.json', 'w') as outfile:
    json.dump(data, outfile)

with open('/Users/artemdembitskiy/Desktop/crystaldata/src/ions/data/bvparams2020-average.pickle', 'wb') as handle:
    pickle.dump(data, handle)

data_path = '/Users/artemdembitskiy/Desktop/crystaldata/src/ions/data/'
radii_file = os.path.join(data_path, 'shannon-radii.json')
with open(radii_file) as f:
  out = f.read()
shannon_data = json.loads(out)

shannon_data = json.loads(out)
shannon_data_ionic = {}
shannon_data_crystal = {}

for s in shannon_data.keys():
  shannon_data_ionic.update({s: {}})
  shannon_data_crystal.update({s: {}})
  for q in shannon_data[s].keys():
    r_ionic, r_crystal, n = 0.0, 0.0, 0
    for CN in shannon_data[s][q].keys():
      n += 1 
      r_ionic += shannon_data[s][q][CN]['r_ionic']
      r_crystal += shannon_data[s][q][CN]['r_crystal']
    r_ionic /= n
    r_crystal /= n
    shannon_data_ionic[s].update({int(q): r_ionic})
    shannon_data_crystal[s].update ({int(q): r_crystal})

with open('/Users/artemdembitskiy/Desktop/crystaldata/src/ions/data/shannon-data-ionic.json', 'w') as outfile:
    json.dump(shannon_data_ionic, outfile)

with open('/Users/artemdembitskiy/Desktop/crystaldata/src/ions/data/shannon-data-ionic.pickle', 'wb') as handle:
    pickle.dump(shannon_data_ionic, handle)

with open('/Users/artemdembitskiy/Desktop/crystaldata/src/ions/data/shannon-data-crystal.json', 'w') as outfile:
    json.dump(shannon_data_crystal, outfile)

with open('/Users/artemdembitskiy/Desktop/crystaldata/src/ions/data/shannon-data-crystal.pickle', 'wb') as handle:
    b = pickle.dump(shannon_data_crystal, handle)


import pandas as pd 
import json
df = pd.read_csv('/Users/artemdembitskiy/Desktop/crystaldata/src/ions/data/bvse.dat', sep = '\t', comment = '#')
df_ = df[['cation', 'cation_valence', 'anion', 'anion_valence', 'R0', 'D0', 'Rmin', 'alpha']]
bvse_data = {}
for k, f in df_.groupby('cation'):
    bvse_data.update({k: {}})
    for k2, f2 in f.groupby('cation_valence'):
        bvse_data[k].update({k2: {}})
        for k3, f3 in f2.groupby('anion'):
            bvse_data[k][k2].update({k3: {}})
            for k4, f4 in f3.groupby('anion_valence'):
                r0 = f4['R0'].values[0]
                d0 = f4['D0'].values[0]
                alpha = f4['alpha'].values[0]
                r_min = f4['Rmin'].values[0]
                params = {'r0': r0, 'r_min': r_min, 'alpha': alpha, 'd0': d0}
                bvse_data[k][k2][k3].update({k4: params})

with open('/Users/artemdembitskiy/Desktop/crystaldata/src/ions/data/bvse_data.json', 'w') as outfile:
    json.dump(bvse_data, outfile)

with open('/Users/artemdembitskiy/Desktop/crystaldata/src/ions/data/bvse_data.pickle', 'wb') as handle:
    b = pickle.dump(bvse_data, handle)




radii_file = os.path.join(data_path, 'shannon-radii.json')
with open(radii_file, 'r') as f:
  out = f.read()
shannon_data = json.loads(out)



radii_file = os.path.join(data_path, 'shannon-radii.pickle')
with open(radii_file, 'wb') as f:
    pickle.dump(shannon_data, f)
