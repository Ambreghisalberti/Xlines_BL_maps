"""Build minimal notebooks from the ORIGINAL notebooks (read-only) on kaa.
Spec: list of notebooks; each has 'title', 'intro' (markdown) and 'cells': list of
  {'src': path rel. to ~, 'cell': index, 'repl': [(old, new, expected_count or None)], 'why': str}
or {'md': text}. Every replacement must match (count checked) -> otherwise abort.
Outputs the .ipynb files + PROVENANCE.md under the repo dir given as argv[2]."""
import json, os, sys, hashlib
HOME = os.path.expanduser('~')
spec = json.load(open(sys.argv[1])); repo = sys.argv[2]
SETUP = ("import sys, os\n"
         "sys.path.insert(0, os.path.abspath('..'))  # makes the blxline package importable\n"
         "from blxline.paths import OUT, MODEL_LINES  # all outputs of this repo are written under OUT (never over the original files)")
def load(p):
    return json.loads(open(os.path.join(HOME, p), encoding='utf-8', errors='replace').read(), strict=False)
cache = {}
prov = ["# Provenance of every notebook cell\n",
        "Each code cell starts with a `# [from <notebook>, cell N]` line. Cells are copied verbatim, except for the replacements listed here.\n"]
for nb in spec:
    cells = [{'cell_type': 'markdown', 'metadata': {}, 'source': f"# {nb['title']}\n\n{nb['intro']}"},
             {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': SETUP}]
    prov.append(f"\n## {nb['file']}\n")
    for c in nb['cells']:
        if 'md' in c:
            cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': c['md']}); continue
        if 'code' in c:
            cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
                          'source': '# [RECONSTRUCTED: ' + c['why'] + ']\n' + c['code']})
            prov.append(f"- **reconstructed cell** — {c['why']}\n"); continue
        if c['src'] not in cache: cache[c['src']] = load(c['src'])
        orig = cache[c['src']]['cells'][c['cell']]
        s = ''.join(orig['source'])
        if orig['cell_type'] == 'markdown':
            cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': s})
            prov.append(f"- `{c['src']}` cell {c['cell']} (markdown)\n"); continue
        md5 = hashlib.md5(s.encode()).hexdigest()[:8]
        changes = []
        for old, new, n in c.get('repl', []):
            k = s.count(old)
            if n == 'opt' and k == 0: continue
            if k == 0 or (n not in (None, 'opt') and k != n):
                sys.exit(f"ABORT {nb['file']}: '{old[:60]}' found {k} times in {c['src']} cell {c['cell']} (expected {n})")
            s = s.replace(old, new); changes.append(f"`{old.strip()[:90]}` → `{new.strip()[:90]}` (×{k})")
        head = f"# [from {c['src']}, cell {c['cell']}]"
        cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': head + '\n' + s})
        prov.append(f"- `{c['src']}` cell {c['cell']} (source md5 {md5}){' — ' + c['why'] if c.get('why') else ''}\n")
        for ch in changes: prov.append(f"    - {ch}\n")
    out = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'space', 'language': 'python', 'name': 'space'}},
           'nbformat': 4, 'nbformat_minor': 5}
    path = os.path.join(repo, 'notebooks', nb['file'])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(out, open(path, 'w'), indent=1, ensure_ascii=False)
    print('wrote', path, len(cells), 'cells')
open(os.path.join(repo, 'PROVENANCE.md'), 'w').write(''.join(prov))
