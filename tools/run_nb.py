"""Execute the code cells of a notebook in one namespace (headless), like a Jupyter kernel would.
usage: python run_nb.py NOTEBOOK [--skip TAG]... [--only-from TAG] [--sub OLD=>NEW]...
TAG matches the '# [from ..., cell N]' first line of a cell (substring). --sub is applied to every cell (verification only)."""
import json, sys, os, time, argparse, traceback
import matplotlib; matplotlib.use('Agg')
# Running the container's python through /proc/<pid>/root (ssh, outside JupyterHub): realpath() turns that prefix
# into the container path /opt/conda/..., which does not exist on the host. Keep font paths as they are.
import matplotlib.font_manager as _fm
if hasattr(_fm, '_cached_realpath') and matplotlib.__file__.startswith('/proc/'):
    _fm._cached_realpath = lambda path: os.path.abspath(path)
import matplotlib.pyplot as plt
ap = argparse.ArgumentParser(); ap.add_argument('nb'); ap.add_argument('--skip', action='append', default=[])
ap.add_argument('--sub', action='append', default=[]); ap.add_argument('--dump', default=None); ap.add_argument('--track', default=None); ap.add_argument('--only-from', default=None)
a = ap.parse_args()
nb = json.load(open(a.nb)); os.chdir(os.path.dirname(os.path.abspath(a.nb)))
ns = {'__name__': '__main__'}; started = a.only_from is None
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code': continue
    src = c['source'] if isinstance(c['source'], str) else ''.join(c['source'])
    first = src.splitlines()[0] if src.strip() else ''
    rel = os.path.relpath(os.path.abspath(a.nb), os.path.expanduser('~'))
    tag = first if first.startswith('# [from') else f'# [from {rel}, cell {i}]'
    is_setup = i == 1
    if not started and not is_setup:
        if a.only_from in tag: started = True
        else: print(f'--- skip (before start) {tag}', flush=True); continue
    if any(t in tag for t in a.skip): print(f'--- skip {tag}', flush=True); continue
    for s in a.sub:
        old, new = s.split('=>', 1)
        if old in src: src = src.replace(old, new); print(f'    [sub applied] {old[:60]!r}', flush=True)
    t0 = time.time(); print(f'=== run {tag}', flush=True)
    try:
        exec(compile(src, f'{os.path.basename(a.nb)}[{i}]', 'exec'), ns)
    except Exception:
        traceback.print_exc(); print(f'!!! FAILED at {tag}', flush=True); sys.exit(1)
    plt.close('all'); print(f'    done in {time.time()-t0:.0f}s', flush=True)
    if a.track:
        import copy
        tracked = ns.setdefault('__tracked__', {})
        for v in a.track.split(','):
            if v in ns and (v not in ns.setdefault('__last__', {}) or ns['__last__'][v] is not ns[v]):
                tracked[(tag, v)] = copy.deepcopy(ns[v]); ns['__last__'][v] = ns[v]
if a.dump:
    import pickle
    path, names = a.dump.split(':')
    keep = {k: ns[k] for k in names.split(',') if k in ns}
    keep['__tracked__'] = ns.get('__tracked__', {})
    pickle.dump(keep, open(path, 'wb')); print('dumped', sorted(keep), 'to', path)
print('ALL CELLS DONE', flush=True)
