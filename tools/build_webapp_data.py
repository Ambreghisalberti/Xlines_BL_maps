"""Builds the data of the web app (webapp/data/) from the shared cache of X-lines (xline_cache/).

usage (from the repo root): python tools/build_webapp_data.py [--maps DIR]...
  --maps DIR: a personal cache folder (with maps/*.npz) where to look for the Delta Vz maps of the shared X-lines.
              The maps are optional: X-lines without a map are shown without it.

Writes, for each dataset of xline_cache/datasets.json (folder named by the first 12 characters of its fingerprint):
  webapp/data/index.json             datasets, magnetopause grid (X at each (Y, Z), for the 3D view)
  webapp/data/<fp>/lines.json        all the X-lines of this dataset (rounded to 1e-3 Re: for display only)
  webapp/data/<fp>/maps/<key>.bin.gz Delta Vz maps smoothed for display as in plot_xline (sigma 3), int16 in 0.1 km/s,
                                     -32768 for NaN, gzip
"""
import argparse, glob, gzip, json, os, sys
import numpy as np

sys.path.insert(0, os.path.abspath('.'))
from blxline.maps import gaussian_filter_nan_datas
from blxline.xlines import mp_grid, SHARED_CACHE_DIR

ap = argparse.ArgumentParser()
ap.add_argument('--maps', action='append', default=[])
ap.add_argument('--out', default='webapp/data')
a = ap.parse_args()

datasets = json.load(open(os.path.join(SHARED_CACHE_DIR, 'datasets.json')))
lines = {fp[:12]: [] for fp in datasets}
n_maps = 0
for f in sorted(glob.glob(os.path.join(SHARED_CACHE_DIR, 'lines', '*.json'))):
    r = json.load(open(f))
    fp = r['dataset_fingerprint'][:12]
    if fp not in lines:
        continue
    key = r['key']
    rnd = lambda v: None if v is None else round(v, 3)
    entry = {'key': key, 'conditions': r['conditions'],
             'parameters': {k: r['parameters'][k] for k in ['k_neighbors', 'max_distance_barycenter', 'median_distance',
                                                            'N_grid', 'sigma_map', 'sigma_line']},
             'n_points': r['n_points'], 'y': [rnd(v) for v in r['Yxline']], 'z': [rnd(v) for v in r['Zxline']],
             'has_map': False}
    for d in a.maps:
        p = os.path.join(d, 'maps', f'{key}.npz')
        if os.path.exists(p):
            m = gaussian_filter_nan_datas(np.load(p)['DeltaVz'].copy(), 3)
            q = np.where(np.isfinite(m), np.clip(np.round(m * 10), -32767, 32767), -32768).astype('<i2')
            os.makedirs(os.path.join(a.out, fp, 'maps'), exist_ok=True)
            with gzip.open(os.path.join(a.out, fp, 'maps', f'{key}.bin.gz'), 'wb', compresslevel=9) as g:
                g.write(q.tobytes())
            entry['has_map'] = True
            n_maps += 1
            break
    lines[fp].append(entry)

os.makedirs(a.out, exist_ok=True)
for fp, ls in lines.items():
    os.makedirs(os.path.join(a.out, fp), exist_ok=True)
    json.dump(ls, open(os.path.join(a.out, fp, 'lines.json'), 'w'), separators=(',', ':'))

Xmp, Ymp, Zmp = mp_grid(100)
index = {'datasets': [{'id': fp[:12], 'fingerprint': fp, **info, 'n_lines': len(lines[fp[:12]])}
                      for fp, info in datasets.items()],
         'grid': {'N_grid': 100, 'y': [-15, 15], 'z': [-15, 15],
                  'X': [[None if x < 0.01 else round(float(x), 3) for x in row] for row in Xmp]}}
json.dump(index, open(os.path.join(a.out, 'index.json'), 'w'), separators=(',', ':'))
print({fp: len(ls) for fp, ls in lines.items()}, 'X-lines,', n_maps, 'maps')
