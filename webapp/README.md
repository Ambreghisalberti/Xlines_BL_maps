# Web app

Static viewer of the X-lines saved in `xline_cache/` (and of their Delta Vz maps), published with GitHub Pages
(`.github/workflows/pages.yml`): https://ambreghisalberti.github.io/Xlines_BL_maps/

Nothing is computed in the browser: the app shows the X-lines already computed with `blxline/xlines.py`
(only the main part of the line and the 3D position on the magnetopause are computed for display).
To compute new X-lines, use `notebooks/07_interactive_xlines.ipynb`.

| File | Purpose |
|------|---------|
| `index.html`, `style.css` | layout, controls (Plotly.js from cdn.plot.ly) |
| `app.js` | loading of the data, main part of the line, 2D and 3D plots, controls |
| `data/` | built by `tools/build_webapp_data.py` from `xline_cache/` (do not edit by hand) |

## Updating the data

After adding X-lines to `xline_cache/` (`calc.share()` in notebook 07), from the repo root:

```bash
python tools/build_webapp_data.py --maps OUT/xline_cache_personal   # the maps are taken from personal caches
```

then commit `xline_cache/` and `webapp/data/`. Pushing to `main` redeploys the app.

## Running locally

```bash
cd webapp && python -m http.server 8000   # open http://localhost:8000
```
