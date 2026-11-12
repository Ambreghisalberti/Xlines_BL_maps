"""Writes notebooks/07_interactive_xlines.ipynb (run from the repo root)."""
import json

cells = []
def md(s): cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': s.strip('\n')})
def code(s): cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': s.strip('\n')})

md(r"""
# Interactive Delta Vz maps and X-lines

Build your own X-lines ($\Delta V_z = V_z - V_{z,MSH} = 0$ in the boundary layer, with diverging flows), and the
$\Delta V_z$ maps they come from, for the conditions you choose: IMF clock angle, dipole tilt, or any other column of
the dataset.

The computation is the one used for the paper (`blxline/xlines.py`, same steps as `05_maps_and_xline.ipynb`).

**Saved results.** Two caches are searched before computing anything:
- the **shared cache** (`xline_cache/` in the repo, on GitHub): X-lines already computed, among which the whole grid
  of the paper (CLA 120° to 180°, tilt -30° to 30°);
- your **personal cache** (`PERSONAL_CACHE` below, on your computer): everything you compute is saved there.

If the same dataset and the same parameters were already used, the X-line is loaded instantly instead of computed
(about 15 s, plus about 30 s to read the dataset the first time). Only the X-lines are shared (a few kB each). The
$\Delta V_z$ maps are larger and only saved in your personal cache: showing a map that is not saved there computes it
again, which needs the dataset.

How to use it:
1. Run the setup cells.
2. Section 2: choose the conditions and click **Compute / load**. **Keep this line** keeps it on the plot, to compare
   it with the next ones.
3. Section 3 lists the saved results and overlays any of them.
4. Section 4 shows how to do the same without widgets, in a script or a loop.
5. Section 6: add your own X-lines to the shared cache.
""")
code(r"""
import sys, os
sys.path.insert(0, os.path.abspath('..'))  # makes the blxline package importable
from blxline.paths import OUT, MODEL_LINES
""")
md(r"""
## 1. Dataset and caches

`DATASET`: the map dataset of the paper, `BL_dataset_multisat_limited.pkl` (1.1 GB), published on Zenodo
([10.5281/zenodo.20312678](https://doi.org/10.5281/zenodo.20312678)). Set `DOWNLOAD = True` to download it into `OUT`
(the folder `data/` of the repo outside kaa). Without it, you can still look at all the saved X-lines, but not compute
new ones. You can also use another map dataset (e.g. the output of `04_map_dataset.ipynb`): the saved results of each
dataset are kept apart.
""")
code(r"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from blxline.xlines import XlineCalculator, plot_xline, plot_xlines_3d, describe, PAPER_PARAMETERS
from blxline.data import download

DOWNLOAD = False  # True: download the dataset from Zenodo if it is not in OUT (1.1 GB)
DATASET = f'{OUT}/BL_dataset_multisat_limited.pkl'
if DOWNLOAD:
    DATASET = download('BL_dataset_multisat_limited.pkl', OUT)
PERSONAL_CACHE = f'{OUT}/xline_cache_personal'

calc = XlineCalculator(DATASET, PERSONAL_CACHE)  # the dataset itself is only read when something must be computed
print('dataset fingerprint:', calc.fingerprint[:12],
      '| dataset available here:', calc.has_dataset,
      '| saved X-lines for this dataset:', len(calc.saved_results()))
""")
md(r"""
## 2. Interactive X-lines

**Conditions**
- **CLA**: IMF clock angle (deg) and the width of the selection. Points are weighted by a gaussian of their distance
  to the chosen value and removed beyond the width (`get_weights_gaussian`).
- **tilt**: dipole tilt (deg), same principle. Untick it to use all tilts.
- **extra condition**: any other column of the dataset (e.g. `omni_COA`), with its centre and width (in the column's
  unit, degrees for angles). Leave the column empty to skip it.

**Computation**
- **K**: number of neighbours used at each grid point. **max dist. barycenter** and **median dist.** (Re): a grid
  point is kept only if the median distance to its neighbours, and the distance to their barycenter, are below these
  values.
- **sigma map**: smoothing (in grid cells) of the $\Delta V_z$ map before looking for its $\Delta V_z = 0$ contour.
  **sigma line**: smoothing (in points) of the line found.

**Display** (does not change what is computed and saved)
- **only the main part of the line**: removes the pieces of the line that are further than **max gap** (Re) from
  its main part (`keep_main_part`, as in the paper figures, max gap = 2 Re). With a small sigma map, the contour
  often has small pieces at high latitude: lower max gap to remove more of them.
- **show the Delta Vz map** (off by default: computes the map if it is not saved in your personal cache).
- **3D view**: the lines on the magnetopause (interactive, plotly).

Paper values: CLA width 30°, tilt width 10°, K = 1000 (2000 for CLA-only maps), 1.5 Re, 3 Re, sigma map 7,
sigma line 1.
""")
code(r"""
import ipywidgets as w
from IPython.display import display

style = {'description_width': '160px'}; lay = w.Layout(width='400px')
def slider(cls, value, mn, mx, step, desc):
    return cls(value=value, min=mn, max=mx, step=step, description=desc, style=style, layout=lay)

W = dict(
    cla=slider(w.FloatSlider, 150, -180, 180, 5, 'CLA (deg)'),
    cla_width=slider(w.FloatSlider, 30, 5, 90, 5, 'CLA width (deg)'),
    use_tilt=w.Checkbox(value=True, description='select on tilt'),
    tilt=slider(w.FloatSlider, 0, -35, 35, 5, 'tilt (deg)'),
    tilt_width=slider(w.FloatSlider, 10, 1, 30, 1, 'tilt width (deg)'),
    extra_col=w.Text(value='', description='extra condition: column', style=style, layout=lay),
    extra_centre=w.FloatText(value=0, description='centre', style=style, layout=lay),
    extra_width=w.FloatText(value=1, description='width', style=style, layout=lay),
    k=slider(w.IntSlider, 1000, 100, 5000, 100, 'K neighbours'),
    max_bary=slider(w.FloatSlider, 1.5, 0.5, 5, 0.1, 'max dist. barycenter (Re)'),
    median=slider(w.FloatSlider, 3, 0.5, 6, 0.1, 'median dist. (Re)'),
    sigma_map=slider(w.IntSlider, 7, 0, 15, 1, 'sigma map'),
    sigma_line=slider(w.IntSlider, 1, 0, 5, 1, 'sigma line'),
    main_part=w.Checkbox(value=True, description='only the main part of the line'),
    max_gap=slider(w.FloatSlider, 2, 0.5, 6, 0.5, 'max gap (Re)'),
    show_map=w.Checkbox(value=False, description='show the Delta Vz map'),
    view3d=w.Checkbox(value=False, description='3D view'),
)
compute_button = w.Button(description='Compute / load', button_style='primary', icon='play')
keep_button = w.Button(description='Keep this line', icon='thumb-tack')
clear_button = w.Button(description='Clear kept lines', icon='trash')
status = w.HTML()
out = w.Output()

state = {'current': None, 'kept': []}  # results shown: the current one and the kept ones
COLORS = plt.cm.tab10.colors

def conditions_from_widgets():
    conds = [('omni_CLA', W['cla'].value, W['cla_width'].value)]
    if W['use_tilt'].value:
        conds.append(('tilt', W['tilt'].value, W['tilt_width'].value))
    if W['extra_col'].value.strip():
        conds.append((W['extra_col'].value.strip(), W['extra_centre'].value, W['extra_width'].value))
    return conds

def parameters_from_widgets():
    return dict(k_neighbors=W['k'].value, max_distance_barycenter=W['max_bary'].value, median_distance=W['median'].value,
                sigma_map=W['sigma_map'].value, sigma_line=W['sigma_line'].value)

def draw():
    current, kept = state['current'], state['kept']
    # the current line first (purple), then the kept lines
    shown, colors, labels = [], [], []
    if current is not None and all(current['key'] != k['key'] for k in kept):
        shown, colors, labels = [current], ['purple'], [f'current: {describe(current)}']
    for i, r in enumerate(kept):
        shown.append(r); colors.append(COLORS[i % len(COLORS)]); labels.append(f'kept {i + 1}: {describe(r)}')
    main_part, max_gap = W['main_part'].value, W['max_gap'].value
    with out:
        out.clear_output(wait=True)
        if not shown:
            return
        if W['view3d'].value:
            hex_colors = ['#%02x%02x%02x' % tuple(int(255 * v) for v in c[:3]) if not isinstance(c, str) else c for c in colors]
            map_of = current if (W['show_map'].value and current is not None) else None
            plot_xlines_3d(shown, colors=hex_colors, labels=labels, main_part_only=main_part, max_gap=max_gap,
                           map_of=map_of).show()
        else:
            fig, ax = plt.subplots(figsize=(9, 5.5))
            for i, (r, c, l) in enumerate(zip(shown, colors, labels)):
                if i == 0 and not W['show_map'].value:
                    r = {k: v for k, v in r.items() if k != 'DeltaVz'}  # background (circle, axes) without the map
                plot_xline(r, ax=ax, main_part_only=main_part, max_gap=max_gap, line_color=c, label=l,
                           show_map=(i == 0), lw=3 if c == 'purple' else 2)
            ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.25 if W['show_map'].value else 1.02, 1))
            plt.show()

def on_compute(_=None):
    conds = conditions_from_widgets()
    with_map = W['show_map'].value
    status.value = '<b>Working...</b> (about 15 s if this has never been computed, more the first time the dataset is read)'
    try:
        res = calc.get(conds, with_map=with_map, **parameters_from_widgets())
    except Exception as e:
        status.value = f'<b style="color:red">Error:</b> {type(e).__name__}: {e}'
        return
    state['current'] = res
    where = (f"loaded from the {res['saved_in']} cache" if res['from_cache']
             else f"computed in {res['computing_time_s']} s and saved in your personal cache")
    status.value = f"{where} &mdash; {res['n_points']} points (after symmetries) &mdash; {describe(res)}"
    draw()

def on_keep(_):
    if state['current'] is not None and all(state['current']['key'] != k['key'] for k in state['kept']):
        state['kept'].append(state['current'])
    draw()

def on_clear(_):
    state['kept'] = []
    draw()

compute_button.on_click(on_compute)
keep_button.on_click(on_keep)
clear_button.on_click(on_clear)
for name in ['main_part', 'max_gap', 'view3d']:
    W[name].observe(lambda change: draw(), names='value')
W['show_map'].observe(lambda change: on_compute() if change['new'] and state['current'] is not None else draw(),
                      names='value')

box = lambda title, names: w.VBox([w.HTML(f'<b>{title}</b>')] + [W[n] for n in names])
display(w.HBox([box('Conditions', ['cla', 'cla_width', 'use_tilt', 'tilt', 'tilt_width', 'extra_col', 'extra_centre', 'extra_width']),
                box('Computation', ['k', 'max_bary', 'median', 'sigma_map', 'sigma_line']),
                box('Display', ['main_part', 'max_gap', 'show_map', 'view3d'])]),
        w.HBox([compute_button, keep_button, clear_button]), status, out)
""")
md(r"""
The results shown are in `state['current']` and `state['kept']` (dicts with `Yxline`, `Zxline`, the conditions, the
parameters, and `DeltaVz` if the map was asked for).

## 3. Saved results, and several X-lines on the same plot
""")
code(r"""
saved = calc.saved_results()
saved
""")
code(r"""
# Choose the rows of the table above to overlay. Here: CLA 150° for all the tilts of the paper grid.
rows = saved[saved.conditions.str.startswith('omni_CLA=150±30°, tilt=')]
if len(rows) == 0:
    print('No saved result selected.')
else:
    results = [calc.load(key) for key in rows.key]
    results.sort(key=lambda r: [c[1] for c in r['conditions'] if c[0] == 'tilt'] or [0])
    fig, ax = plt.subplots(figsize=(7, 7))
    for res, color in zip(results, plt.cm.coolwarm(np.linspace(0, 1, len(results)))):
        plot_xline(res, ax=ax, show_map=False, line_color=color, label=describe(res).split(' |')[0], lw=2)
    ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()
""")
md(r"""
## 4. Without widgets (scripts, loops)

`calc.get(conditions, **parameters)` returns the result, from the caches if it is saved. Conditions are
`(column, centre, width)`, angles in degrees. Add `with_map=True` to get the $\Delta V_z$ map too.
Below: X-lines for one tilt and several clock angles, and their 3D view.
""")
code(r"""
fig, ax = plt.subplots(figsize=(7, 7))
clas = [120, 150, 180]
results = []
for cla, color in zip(clas, plt.cm.plasma(np.linspace(0, 0.8, len(clas)))):
    res = calc.get([('omni_CLA', cla, 30), ('tilt', 0, 10)], k_neighbors=1000, max_distance_barycenter=1.5,
                   median_distance=3, sigma_map=7, sigma_line=1)
    results.append(res)
    plot_xline(res, ax=ax, show_map=False, line_color=color,
               label=f'CLA {cla}° ' + (f"({res['saved_in']} cache)" if res['from_cache'] else '(computed)'))
ax.legend()
plt.show()
""")
code(r"""
plot_xlines_3d(results, colors=['#0d0887', '#cc4778', '#f0a830'], labels=[f'CLA {c}°' for c in clas]).show()
""")
md(r"""
## 5. Check: same results as the paper

With the paper parameters, the dataset of Zenodo gives exactly the X-lines and maps of the paper
(`DeltaVz_cla*_tilt*.pkl`, `05_maps_and_xline.ipynb`; checked for the 182 X-lines of the paper grid). This cell
recomputes one of them; it only runs where the paper files are available (kaa).
""")
code(r"""
reference = f'{MODEL_LINES}/DeltaVz_cla150_tilt0.pkl'
if os.path.exists(reference) and calc.has_dataset:
    p = {k: PAPER_PARAMETERS[k] for k in ['k_neighbors', 'max_distance_barycenter', 'median_distance', 'N_grid', 'sigma_map', 'sigma_line']}
    res = calc.get([('omni_CLA', 150, 30), ('tilt', 0, 10)], with_map=True, recompute=True, **p)
    ref = pd.read_pickle(reference)
    print('same X-line as the paper:', np.array_equal(res['Yxline'], np.asarray(ref['Yxline'], float)) and
          np.array_equal(res['Zxline'], np.asarray(ref['Zxline'], float)))
    print('max |difference| of the Delta Vz map:', np.nanmax(abs(res['DeltaVz'] - ref['DeltaVz'])),
          '| same NaN pattern:', np.array_equal(np.isnan(res['DeltaVz']), np.isnan(ref['DeltaVz'])))
else:
    print('Paper files or dataset not available here: check skipped.')
""")
md(r"""
## 6. Add your X-lines to the shared cache

`calc.share()` copies the X-lines of your personal cache (or only `calc.share(keys=[...])`) into `xline_cache/lines/`
of the repo. Commit these new files (and open a pull request) to make them available to everyone. Maps are not copied.
""")
code(r"""
# calc.share()
""")

nb = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'space/python3', 'language': 'python', 'name': 'conda-env-space-py'},
                                   'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 5}
for i, c in enumerate(nb['cells']):
    c['id'] = f'cell-{i}'
json.dump(nb, open('notebooks/07_interactive_xlines.ipynb', 'w'), indent=1)
print('wrote notebooks/07_interactive_xlines.ipynb', len(cells), 'cells')
