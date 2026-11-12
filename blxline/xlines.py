"""Delta Vz maps and X-lines for any set of conditions, with a cache of the results.

NEW code (not copied from the original notebooks). It only chains the functions used for the paper
(blxline.maps: prepare_data, get_weights_gaussian, knn_weighted_feature_map, make_mp_grid,
find_stagnation_line, gaussian_filter_nan_1d), exactly as in FinalMapsPaper cells 628/629, so that with the
paper parameters it gives the same maps and X-lines.

Cache. Every result has a key made from
  - the fingerprint of the dataset file (sha256 of its content),
  - all the parameters of the computation (conditions, number of neighbours, distances, smoothing, ...),
  - XLINE_CODE_VERSION (increase it when the computation changes, so that old results are not reused).
Two caches are searched, in this order:
  - the personal cache (`cache_dir`, on your computer): everything you compute is saved there;
  - the shared cache (`shared_cache_dir`, the folder `xline_cache/` of the repo, versioned on GitHub):
    results computed by others, read only. `XlineCalculator.share()` copies personal results into it.
The X-line itself is small (a json file of a few kB, saved in both caches). The Delta Vz map (about 160 kB)
is only saved in the personal cache, in `maps/`: when a map is asked for and is not saved, it is computed again,
which needs the dataset.
"""
import functools
import glob
import hashlib
import json
import os
import shutil
import time

import numpy as np
import pandas as pd

from blxline.maps import (prepare_data, get_weights_gaussian, knn_weighted_feature_map, make_mp_grid,
                          find_stagnation_line, gaussian_filter_nan_1d)

XLINE_CODE_VERSION = 2

# Shared cache shipped with the repo
SHARED_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'xline_cache')

# Columns stored in radians in the map dataset: their centres and widths are given in degrees by the user.
ANGLE_COLUMNS = ('omni_CLA', 'omni_COA', 'tilt')

# Parameters of the computation, with the values used for the paper (FinalMapsPaper cell 628: CLA and tilt maps;
# cell 629: CLA-only maps use k_neighbors=2000).
#   sigma_map: smoothing (in grid cells) of the Delta Vz map before looking for its Delta Vz = 0 contour
#              (argument `sigma` of find_stagnation_line).
#   sigma_line: smoothing (in points) of the Z coordinate along the line found.
COMPUTATION_PARAMETERS = ('k_neighbors', 'max_distance_barycenter', 'median_distance', 'N_grid', 'sigma_map', 'sigma_line')
PAPER_PARAMETERS = dict(k_neighbors=1000, max_distance_barycenter=1.5, median_distance=3, N_grid=100,
                        sigma_map=7, sigma_line=1, lower_depth=0.83, symetry_cla=True, symetry_coa=True)


# ----------------------------------------------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------------------------------------------
def dataset_fingerprint(path, cache_dir):
    """sha256 of the dataset file. Computed once per (path, size, modification time) and remembered in cache_dir."""
    os.makedirs(cache_dir, exist_ok=True)
    st = os.stat(path)
    key = f'{os.path.abspath(path)}|{st.st_size}|{st.st_mtime_ns}'
    known_file = os.path.join(cache_dir, 'dataset_fingerprints.json')
    known = json.load(open(known_file)) if os.path.exists(known_file) else {}
    if key not in known:
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for block in iter(lambda: f.read(1 << 24), b''):
                h.update(block)
        known[key] = h.hexdigest()
        json.dump(known, open(known_file, 'w'), indent=1)
    return known[key]


def load_map_dataset(path, lower_depth=0.83, symetry_cla=True, symetry_coa=True):
    """Read the map dataset and prepare it as for the paper (prepare_data + depth selection, cell 628)."""
    df = prepare_data(pd.read_pickle(path), lower_depth=lower_depth, symetry_cla=symetry_cla, symetry_coa=symetry_coa)
    df = df[df.uniform_depthonly_spectro.values > lower_depth]
    return df


# ----------------------------------------------------------------------------------------------------------------
# Computation
# ----------------------------------------------------------------------------------------------------------------
def _to_internal(conditions):
    """[(column, centre, width)] with angles in degrees -> list for get_weights_gaussian (angles in radians)."""
    out = []
    for col, centre, width in conditions:
        if col in ANGLE_COLUMNS:
            out.append((col, centre * np.pi / 180, width * np.pi / 180))
        else:
            out.append((col, centre, width))
    return out


@functools.lru_cache(maxsize=4)
def mp_grid(N_grid=100):
    """Grid of the maps (make_mp_grid, cartesian): X of the magnetopause at each (Y, Z).
    make_mp_grid takes about 30 s: its result is saved in the shared cache (mp_grid_N{N_grid}.npz) and reused."""
    path = os.path.join(SHARED_CACHE_DIR, f'mp_grid_N{N_grid}.npz')
    if os.path.exists(path):
        g = np.load(path)
        return g['Xmp'], g['Ymp'], g['Zmp']
    Xmp, Ymp, Zmp = make_mp_grid(N_grid=N_grid, coord='cartesian')
    try:
        _write_atomic(path, lambda p: np.savez_compressed(open(p, 'wb'), Xmp=Xmp, Ymp=Ymp, Zmp=Zmp))
    except OSError:
        pass
    return Xmp, Ymp, Zmp


def yz_grid(N_grid=100):
    """Y and Z of the grid of the maps (same as make_mp_grid(coord='cartesian'), without computing X)."""
    return np.meshgrid(np.linspace(-15, 15, 2 * N_grid), np.linspace(-15, 15, N_grid))


def compute_xline(df, conditions, k_neighbors=1000, max_distance_barycenter=1.5, median_distance=3, N_grid=100,
                  sigma_map=7, sigma_line=1):
    """Delta Vz map and X-line for the given conditions (same steps as FinalMapsPaper cells 628/629).

    conditions: list of (column, centre, width). Angles (omni_CLA, omni_COA, tilt) in degrees.
        Example: [('omni_CLA', 150, 30), ('tilt', 0, 10)].
        Points are weighted by a gaussian of the distance to the centre, and removed beyond `width`
        (see blxline.maps.get_weights_gaussian).
    Returns a dict with DeltaVz (N_grid x 2N_grid map), Yxline, Zxline and the number of points used.
    """
    Xmp, Ymp, Zmp = mp_grid(N_grid)
    df2 = get_weights_gaussian(df, _to_internal(conditions))
    df2['weights'] = df2.weights.values / np.nansum(df2.weights.values)
    temp = df2[df2.weights.values > 0]
    mapped_feature, valid = knn_weighted_feature_map(temp, 'dVz', 'weights',
                                                     np.array([Xmp.flatten(), Ymp.flatten(), Zmp.flatten()]).T,
                                                     k_neighbors=k_neighbors,
                                                     max_distance_barycenter=max_distance_barycenter,
                                                     median_distance=median_distance)
    mapped_feature[valid == 0] = np.nan
    mapped_feature[Xmp.flatten() < 0.01] = np.nan

    y_measure, z_measure = find_stagnation_line(mapped_feature.reshape(Xmp.shape), Y_mp=Ymp, Z_mp=Zmp,
                                                sigma=sigma_map)
    z_measure = gaussian_filter_nan_1d(np.asarray(z_measure, dtype=float), sigma_line)
    return {'DeltaVz': mapped_feature.reshape(Xmp.shape), 'Yxline': np.asarray(y_measure, dtype=float),
            'Zxline': np.asarray(z_measure, dtype=float), 'n_points': int(len(temp))}


# ----------------------------------------------------------------------------------------------------------------
# Cache
# ----------------------------------------------------------------------------------------------------------------
def _clean_conditions(conditions):
    # rounded so that slider values such as 9.999999999999998 are the same request as 10
    return [[str(c), round(float(centre), 6), round(float(width), 6)] for c, centre, width in conditions]


def _canonical_parameters(conditions, **params):
    conds = sorted(_clean_conditions(conditions))
    p = {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool) else v)
         for k, v in params.items()}
    return {'conditions': conds, **dict(sorted(p.items()))}


def cache_key(fingerprint, conditions, **params):
    payload = {'dataset': fingerprint, 'code_version': XLINE_CODE_VERSION,
               'parameters': _canonical_parameters(conditions, **params)}
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _to_json_list(a):
    return [None if not np.isfinite(v) else float(v) for v in np.asarray(a, dtype=float)]


def _from_json_list(a):
    return np.array([np.nan if v is None else v for v in a], dtype=float)


def _write_atomic(path, write):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write(path + '.tmp')
    os.replace(path + '.tmp', path)  # never leave a half-written result in the cache


class XlineCalculator:
    """Computes Delta Vz maps and X-lines on one dataset, and reuses the saved results.

    calc = XlineCalculator(path_to_map_dataset, personal_cache_dir)
    res = calc.get([('omni_CLA', 150, 30), ('tilt', 0, 10)], k_neighbors=1000)            # X-line only
    res = calc.get([('omni_CLA', 150, 30), ('tilt', 0, 10)], k_neighbors=1000, with_map=True)  # + Delta Vz map
    res['Yxline'], res['Zxline'], res['DeltaVz'], res['from_cache']

    Without the dataset file, only saved results can be read: give `fingerprint` (or let it be taken from
    the shared cache, `xline_cache/datasets.json`, when the file name is known there).
    """

    def __init__(self, dataset_path, cache_dir, shared_cache_dir=SHARED_CACHE_DIR, fingerprint=None,
                 lower_depth=0.83, symetry_cla=True, symetry_coa=True):
        self.dataset_path = dataset_path
        self.cache_dir = cache_dir
        self.shared_cache_dir = shared_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.data_options = dict(lower_depth=lower_depth, symetry_cla=symetry_cla, symetry_coa=symetry_coa)
        self.has_dataset = dataset_path is not None and os.path.exists(dataset_path)
        if self.has_dataset:
            self.fingerprint = dataset_fingerprint(dataset_path, cache_dir)
            if fingerprint is not None and fingerprint != self.fingerprint:
                raise ValueError(f'{dataset_path} does not have the fingerprint given ({fingerprint[:12]}...).')
        else:
            self.fingerprint = fingerprint or self._fingerprint_from_shared_cache()
            if self.fingerprint is None:
                raise FileNotFoundError(f'{dataset_path} not found, and its fingerprint is unknown: '
                                        'give fingerprint=... to read the saved results without the dataset.')
        self._df = None  # the dataset is only read when a result has to be computed

    def _fingerprint_from_shared_cache(self):
        f = os.path.join(self.shared_cache_dir or '', 'datasets.json')
        if self.dataset_path is None or not os.path.exists(f):
            return None
        for fp, info in json.load(open(f)).items():
            if info.get('file') == os.path.basename(self.dataset_path):
                return fp
        return None

    @property
    def df(self):
        if self._df is None:
            if not self.has_dataset:
                raise FileNotFoundError(f'This result is not saved and has to be computed, but the dataset '
                                        f'{self.dataset_path} is not available here.')
            self._df = load_map_dataset(self.dataset_path, **self.data_options)
        return self._df

    # --- where the results are -------------------------------------------------------------------------------
    def _caches(self):
        return [c for c in [self.cache_dir, self.shared_cache_dir] if c]

    def _line_path(self, key, cache=None):
        if cache is not None:
            return os.path.join(cache, 'lines', f'{key}.json')
        for c in self._caches():
            p = os.path.join(c, 'lines', f'{key}.json')
            if os.path.exists(p):
                return p
        return None

    def _map_path(self, key):
        for c in self._caches():
            p = os.path.join(c, 'maps', f'{key}.npz')
            if os.path.exists(p):
                return p
        return None

    def key(self, conditions, **params):
        return cache_key(self.fingerprint, conditions, **self._parameters(**params))

    def _parameters(self, k_neighbors=1000, max_distance_barycenter=1.5, median_distance=3, N_grid=100,
                    sigma_map=7, sigma_line=1):
        return dict(k_neighbors=k_neighbors, max_distance_barycenter=max_distance_barycenter,
                    median_distance=median_distance, N_grid=N_grid, sigma_map=sigma_map,
                    sigma_line=sigma_line, **self.data_options)

    # --- main function ---------------------------------------------------------------------------------------
    def get(self, conditions, k_neighbors=1000, max_distance_barycenter=1.5, median_distance=3, N_grid=100,
            sigma_map=7, sigma_line=1, with_map=False, recompute=False):
        """X-line (and Delta Vz map if with_map) for these conditions: from the caches if saved, else computed and
        saved in the personal cache. res['from_cache'] tells which."""
        params = self._parameters(k_neighbors=k_neighbors, max_distance_barycenter=max_distance_barycenter,
                                  median_distance=median_distance, N_grid=N_grid, sigma_map=sigma_map,
                                  sigma_line=sigma_line)
        key = cache_key(self.fingerprint, conditions, **params)
        if not recompute and self._line_path(key) and (not with_map or self._map_path(key)):
            res = self.load(key, with_map=with_map)
            res['from_cache'] = True
            return res
        t0 = time.time()
        res = compute_xline(self.df, _clean_conditions(conditions), **{k: params[k] for k in COMPUTATION_PARAMETERS})
        res.update({'key': key, 'conditions': _clean_conditions(conditions),
                    'parameters': params, 'dataset': os.path.basename(self.dataset_path),
                    'dataset_fingerprint': self.fingerprint, 'code_version': XLINE_CODE_VERSION,
                    'computed_on': pd.Timestamp.now().isoformat(timespec='seconds'),
                    'computing_time_s': round(time.time() - t0, 1)})
        self._save(key, res)
        res['Ymp'], res['Zmp'] = yz_grid(N_grid)
        res['from_cache'] = False
        if not with_map:
            res.pop('DeltaVz')
        return res

    def _save(self, key, res):
        line = {k: v for k, v in res.items() if k != 'DeltaVz'}
        line['Yxline'], line['Zxline'] = _to_json_list(res['Yxline']), _to_json_list(res['Zxline'])
        _write_atomic(self._line_path(key, self.cache_dir),
                      lambda p: json.dump(line, open(p, 'w'), indent=1))
        _write_atomic(os.path.join(self.cache_dir, 'maps', f'{key}.npz'),
                      lambda p: np.savez_compressed(open(p, 'wb'), DeltaVz=res['DeltaVz']))

    def load(self, key, with_map=False):
        """Saved result of this key (raises an error if it is not saved)."""
        path = self._line_path(key)
        if path is None:
            raise KeyError(f'No saved result with key {key}.')
        res = json.load(open(path))
        res['Yxline'], res['Zxline'] = _from_json_list(res['Yxline']), _from_json_list(res['Zxline'])
        res['Ymp'], res['Zmp'] = yz_grid(int(res['parameters']['N_grid']))
        res['saved_in'] = 'personal' if os.path.abspath(path).startswith(os.path.abspath(self.cache_dir) + os.sep) else 'shared'
        if with_map:
            mp = self._map_path(key)
            if mp is None:
                raise KeyError(f'The X-line {key} is saved, but not its map: use get(..., with_map=True).')
            res['DeltaVz'] = np.load(mp)['DeltaVz']
        return res

    def saved_results(self, this_dataset_only=True):
        """Table of the saved results (one row per result), in the personal and the shared caches."""
        rows = {}
        for cache, where in [(self.shared_cache_dir, 'shared'), (self.cache_dir, 'personal')]:
            for f in sorted(glob.glob(os.path.join(cache or '', 'lines', '*.json'))):
                r = json.load(open(f))
                if r['key'] in rows:  # in both caches
                    rows[r['key']]['saved_in'] = 'personal+shared'
                    continue
                rows[r['key']] = {'key': r['key'], 'saved_in': where, 'conditions': describe_conditions(r['conditions']),
                                  **{k: r['parameters'][k] for k in COMPUTATION_PARAMETERS},
                                  'n_points': r['n_points'], 'map_saved': os.path.exists(
                                      os.path.join(self.cache_dir, 'maps', f"{r['key']}.npz")),
                                  'dataset': r['dataset'], 'dataset_fingerprint': r['dataset_fingerprint'][:12],
                                  'computed_on': r['computed_on']}
        cols = ['key', 'saved_in', 'conditions', *COMPUTATION_PARAMETERS, 'n_points', 'map_saved', 'dataset',
                'dataset_fingerprint', 'computed_on']
        t = pd.DataFrame(list(rows.values()), columns=cols)
        if this_dataset_only:
            t = t[t.dataset_fingerprint == self.fingerprint[:12]]
        return t.sort_values('computed_on').reset_index(drop=True)

    def share(self, keys=None, shared_cache_dir=None):
        """Copy X-lines of the personal cache (all, or the given keys) into the shared cache (the X-lines only, not
        the maps). Then commit the new files of `xline_cache/lines/` to make them available to everyone."""
        dest = shared_cache_dir or self.shared_cache_dir
        os.makedirs(os.path.join(dest, 'lines'), exist_ok=True)
        files = glob.glob(os.path.join(self.cache_dir, 'lines', '*.json'))
        if keys is not None:
            files = [f for f in files if os.path.basename(f)[:-5] in set(keys)]
        n = 0
        for f in files:
            target = os.path.join(dest, 'lines', os.path.basename(f))
            if not os.path.exists(target):
                shutil.copy2(f, target)
                n += 1
        if self.has_dataset:
            f = os.path.join(dest, 'datasets.json')
            known = json.load(open(f)) if os.path.exists(f) else {}
            if self.fingerprint not in known:
                known[self.fingerprint] = {'file': os.path.basename(self.dataset_path)}
                json.dump(known, open(f, 'w'), indent=1)
        print(f'{n} new X-line(s) copied into {dest}/lines')


# ----------------------------------------------------------------------------------------------------------------
# keep_main_part: copied verbatim from FinalMapsPaper.ipynb cell 632 (also used in notebooks 05 and 06)
# ----------------------------------------------------------------------------------------------------------------
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def label_pieces(y, z, max_gap=2.):
    """Label of the piece each point belongs to (-1 for NaN points)."""
    y, z = np.asarray(y, dtype=float), np.asarray(z, dtype=float)
    labels = np.full(y.shape, -1)
    valid = np.isfinite(y) & np.isfinite(z)
    pts = np.array([y[valid], z[valid]]).T
    if len(pts) == 0:
        return labels
    pairs = cKDTree(pts).query_pairs(max_gap, output_type='ndarray')
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(pts), len(pts)))
    labels[valid] = connected_components(graph, directed=False)[1]
    return labels


def piece_extent(y, z):
    """Size of a piece: diagonal of its bounding box."""
    return np.hypot(np.ptp(y), np.ptp(z))


def keep_main_part(y, z, max_gap=2., min_fraction=None, return_mask=False):
    """Remove the pieces of the line (y, z) that are far from its main part.
    y, z : positions of the points of the line (the order is kept)
    max_gap : points closer than max_gap are in the same piece. Should be larger than the
        gaps you accept in a line, and smaller than the distance of the pieces to remove.
    min_fraction : if given, also keep the pieces whose extent is at least min_fraction times
        the extent of the main piece.
    Returns the cleaned y, z (and the boolean mask of the kept points if return_mask).
    """
    y, z = np.asarray(y, dtype=float), np.asarray(z, dtype=float)
    labels = label_pieces(y, z, max_gap=max_gap)
    pieces = np.unique(labels[labels >= 0])
    if len(pieces) == 0:
        mask = np.zeros(y.shape, dtype=bool)
    else:
        extents = np.array([piece_extent(y[labels == p], z[labels == p]) for p in pieces])
        if min_fraction is None:
            kept = pieces[[np.argmax(extents)]]
        else:
            kept = pieces[extents >= min_fraction * extents.max()]
        mask = np.isin(labels, kept)
    if return_mask:
        return y[mask], z[mask], mask
    return y[mask], z[mask]


# ----------------------------------------------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------------------------------------------
def line_to_plot(res, main_part_only=True, max_gap=2.):
    """Y, Z of the X-line to show: the line saved, or only its main part (keep_main_part, as in the paper,
    max_gap=2 Re). The saved line is never modified: this is only applied for display."""
    y, z = np.asarray(res['Yxline'], float), np.asarray(res['Zxline'], float)
    if main_part_only and len(y):
        y, z = keep_main_part(y, z, max_gap=max_gap)
    return y, z


def plot_xline(res, ax=None, smoothing=3, main_part_only=True, max_gap=2., vmax=None, colorbar=True,
               line_color='purple', label=None, show_map=True, lw=3):
    """Plot the X-line of a result of XlineCalculator.get, on its Delta Vz map if the result contains it
    (smoothed for display as in the paper figures). main_part_only: keep only the main piece of the line."""
    import matplotlib.pyplot as plt
    from blxline.maps import gaussian_filter_nan_datas, plot_CLA_sector
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    if show_map and 'DeltaVz' in res:
        m = gaussian_filter_nan_datas(res['DeltaVz'].copy(), smoothing)
        vmax = vmax or np.nanmax(abs(m))
        im = ax.pcolormesh(res['Ymp'], res['Zmp'], m, cmap='seismic', vmin=-vmax, vmax=vmax)
        if colorbar:
            plt.colorbar(im, ax=ax, label=r'$\Delta V_z$ (km/s)')
    if show_map:
        th = np.linspace(0, 2 * np.pi, 200)
        r_circle = 15.2
        ax.plot(r_circle * np.cos(th), r_circle * np.sin(th), color='k', lw=1)
        ax.axhline(0, color='k', ls='--', lw=1)
        ax.axvline(0, color='k', ls='--', lw=1)
        for col, centre, width in res['conditions']:
            if col == 'omni_CLA':
                plot_CLA_sector(13, 13, 2.5, (centre - width) * np.pi / 180, (centre + width) * np.pi / 180, ax)
    y, z = line_to_plot(res, main_part_only, max_gap)
    # the line is made of several contours put end to end: do not join them with straight segments
    breaks = np.where(np.hypot(np.diff(y), np.diff(z)) > 1)[0] + 1
    y, z = np.insert(y, breaks, np.nan), np.insert(z, breaks, np.nan)
    ax.plot(y, z, color=line_color, lw=lw, label=label)
    ax.set_xlim(-16, 16)
    ax.set_ylim(-16, 16)
    ax.set_aspect('equal')
    ax.set_xlabel('Y (Re)')
    ax.set_ylabel('Z (Re)')
    return ax


def plot_xlines_3d(results, colors=None, labels=None, main_part_only=True, max_gap=2., map_of=None, smoothing=3):
    """3D view (plotly) of X-lines on the magnetopause surface of the maps (Shue model, X > 0).
    results: list of results of XlineCalculator.get. map_of: one result with its DeltaVz, shown in colour on the
    surface (else the surface is grey)."""
    import plotly.graph_objects as go
    from scipy.interpolate import RegularGridInterpolator
    from blxline.maps import gaussian_filter_nan_datas
    N_grid = int(results[0]['parameters']['N_grid']) if results else 100
    Xmp, Ymp, Zmp = mp_grid(N_grid)
    Xs = np.where(Xmp < 0.01, np.nan, Xmp)
    fig = go.Figure()
    fig.add_trace(go.Surface(x=Xs, y=Ymp, z=Zmp, surfacecolor=np.zeros_like(Xs), colorscale=[[0, '#d0d0d0'], [1, '#d0d0d0']],
                             showscale=False, opacity=0.6, name='magnetopause', hoverinfo='skip'))
    if map_of is not None and 'DeltaVz' in map_of:
        m = gaussian_filter_nan_datas(map_of['DeltaVz'].copy(), smoothing)
        vmax = np.nanmax(abs(m))
        hide = ~np.isfinite(m)  # where there is no map, only the grey magnetopause is shown
        fig.add_trace(go.Surface(x=np.where(hide, np.nan, Xs + 0.02), y=np.where(hide, np.nan, Ymp),
                                 z=np.where(hide, np.nan, Zmp), surfacecolor=m, colorscale='RdBu_r', cmin=-vmax, cmax=vmax,
                                 colorbar=dict(title='ΔVz (km/s)', len=0.6), name='ΔVz'))
    u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
    fig.add_trace(go.Surface(x=np.cos(u) * np.sin(v), y=np.sin(u) * np.sin(v), z=np.cos(v), showscale=False,
                             colorscale=[[0, '#3060c0'], [1, '#3060c0']], name='Earth'))
    interp = RegularGridInterpolator((Zmp[:, 0], Ymp[0]), Xmp, bounds_error=False)
    colors = colors or ['purple'] * len(results)
    for i, res in enumerate(results):
        y, z = line_to_plot(res, main_part_only, max_gap)
        if not len(y):
            continue
        # the points of the line are in the order of the contours: split where consecutive points are far apart
        x = interp(np.array([z, y]).T) + 0.2
        breaks = np.where(np.hypot(np.diff(y), np.diff(z)) > 1)[0] + 1
        for j, idx in enumerate(np.split(np.arange(len(y)), breaks)):
            fig.add_trace(go.Scatter3d(x=x[idx], y=y[idx], z=z[idx], mode='lines', line=dict(color=colors[i], width=7),
                                       name=(labels[i] if labels else describe(res)), legendgroup=str(i),
                                       showlegend=(j == 0)))
    fig.update_layout(scene=dict(xaxis_title='X (Re)', yaxis_title='Y (Re)', zaxis_title='Z (Re)', aspectmode='data',
                                 camera=dict(eye=dict(x=1.8, y=0, z=0.3))),
                      height=750, margin=dict(l=0, r=0, t=10, b=0),
                      legend=dict(font=dict(size=10), orientation='h', yanchor='top', y=0, xanchor='left', x=0))
    return fig


def describe_conditions(conditions):
    return ', '.join(f'{c}={centre:g}±{width:g}' + ('°' if c in ANGLE_COLUMNS else '') for c, centre, width in conditions)


def describe(res):
    """One-line description of the conditions of a result."""
    p = res['parameters']
    return (f"{describe_conditions(res['conditions'])} | K={p['k_neighbors']:g}, "
            f"dist. barycenter={p['max_distance_barycenter']:g}, median dist.={p['median_distance']:g}, "
            f"sigma map={p['sigma_map']:g}, sigma line={p['sigma_line']:g}")
