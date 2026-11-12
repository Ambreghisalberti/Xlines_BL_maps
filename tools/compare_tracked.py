"""Compare variables recorded cell by cell (run_nb.py --track) in two runs. usage: compare_tracked.py NEW.pkl ORIGINAL.pkl"""
import pickle, sys, numpy as np
new, ref = [pickle.load(open(p, 'rb'))['__tracked__'] for p in sys.argv[1:3]]
def flat(x):
    try: return np.asarray(x, dtype=float).ravel()
    except Exception: return np.concatenate([flat(e) for e in x]) if len(x) else np.array([])
n_same = 0
for key in sorted(set(new) | set(ref)):
    if key not in new or key not in ref:
        print('ONLY IN', 'new' if key in new else 'original', key); continue
    a, b = flat(new[key]), flat(ref[key])
    if a.shape == b.shape and np.array_equal(a, b, equal_nan=True): n_same += 1
    else: print('DIFF', key, a.shape, b.shape, 'max|diff|', np.nanmax(np.abs(a-b)) if a.shape == b.shape else '-')
print(f'{n_same} / {len(set(new) & set(ref))} recorded (cell, variable) pairs identical')
