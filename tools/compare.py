"""Compare two pickled objects (DataFrame / dict of arrays / arrays). usage: compare.py NEW ORIGINAL [--cols c1,c2]"""
import sys, numpy as np, pandas as pd
new, ref = pd.read_pickle(sys.argv[1]), pd.read_pickle(sys.argv[2])
cols = sys.argv[sys.argv.index('--cols')+1].split(',') if '--cols' in sys.argv else None
def cmp_arr(name, a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape: return f'{name}: SHAPE {a.shape} vs {b.shape}'
    if a.dtype.kind in 'fc' or b.dtype.kind in 'fc':
        a, b = a.astype(float), b.astype(float)
        nan_ok = np.array_equal(np.isnan(a), np.isnan(b)); m = ~np.isnan(a) & ~np.isnan(b)
        d = np.abs(a[m]-b[m]); nd = int((d > 0).sum())
        return f"{name}: {'IDENTICAL' if nan_ok and nd==0 else 'DIFF'} (nan pattern {'same' if nan_ok else 'DIFFERENT: %d vs %d' % (np.isnan(a).sum(), np.isnan(b).sum())}, {nd} values differ, max|diff|={d.max() if d.size else 0:.3g})"
    same = (a == b) | (pd.isna(a) & pd.isna(b))   # NaN/NaT count as equal
    return f"{name}: {'IDENTICAL' if same.all() else 'DIFF (%d differ)' % int((~same).sum())}"
if isinstance(new, pd.DataFrame):
    print(f'shape new {new.shape} / original {ref.shape}')
    cn, cr = set(new.columns), set(ref.columns)
    if cn != cr: print('columns only in new:', sorted(cn-cr)[:30], '| only in original:', sorted(cr-cn)[:30])
    same_index = new.index.equals(ref.index)
    print('index identical:', same_index)
    if not same_index:
        print(f'  rows only in new: {(~new.index.isin(ref.index)).sum()}, only in original: {(~ref.index.isin(new.index)).sum()}')
        if 'sat' in new.columns and 'sat' in ref.columns:
            for s in sorted(set(ref.sat)):
                print(f'  {s}: new {int((new.sat==s).sum())} rows, original {int((ref.sat==s).sum())} rows')
    else:
        for c in (cols or sorted(cn & cr)): print('  ' + cmp_arr(c, new[c].values, ref[c].values))
elif isinstance(new, dict):
    print('keys same:', set(new) == set(ref))
    for k in sorted(set(new) & set(ref), key=str): print('  ' + cmp_arr(str(k), new[k], ref[k]))
else:
    print(cmp_arr('object', new, ref))
