// Viewer of the X-lines saved in the shared cache of the repo (xline_cache/), converted by tools/build_webapp_data.py.
// Nothing is computed here except the display: main part of the line (same algorithm as keep_main_part in
// blxline/xlines.py) and the position of the line on the magnetopause for the 3D view.

const DATA = (window.XLINES_DATA_BASE || './data').replace(/\/$/, '');
const PAPER = { k_neighbors: 1000, max_distance_barycenter: 1.5, median_distance: 3, N_grid: 100, sigma_map: 7, sigma_line: 1 };
const TAB10 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'];
const CURRENT_COLOR = '#b77cff';
const SEISMIC = [[0, '#00004c'], [0.25, '#0000ff'], [0.5, '#ffffff'], [0.75, '#ff0000'], [1, '#800000']];

const $ = (id) => document.getElementById(id);
const state = { index: null, dataset: null, lines: [], current: null, kept: [], maps: new Map() };

// ------------------------------------------------------------------------------------------------------------
// Data
// ------------------------------------------------------------------------------------------------------------
async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}

async function getMap(line) {
  if (!line.has_map) return null;
  if (state.maps.has(line.key)) return state.maps.get(line.key);
  const r = await fetch(`${DATA}/${state.dataset.id}/maps/${line.key}.bin.gz`);
  if (!r.ok) throw new Error(`map ${line.key}: ${r.status}`);
  const stream = r.body.pipeThrough(new DecompressionStream('gzip'));
  const buf = await new Response(stream).arrayBuffer();
  const q = new Int16Array(buf);
  const N = state.index.grid.N_grid, M = 2 * N;
  const z = [];
  for (let i = 0; i < N; i++) {
    const row = new Array(M);
    for (let j = 0; j < M; j++) { const v = q[i * M + j]; row[j] = v === -32768 ? null : v / 10; }
    z.push(row);
  }
  state.maps.set(line.key, z);
  return z;
}

const condOf = (line, col) => { const c = line.conditions.find((c) => c[0] === col); return c ? c : null; };
const isPaper = (line) => Object.keys(PAPER).every((k) =>
  k === 'k_neighbors' ? true : Math.abs(line.parameters[k] - PAPER[k]) < 1e-9);

function gridLine(cla, tilt) {
  // tilt === null: selection on the clock angle only (paper: K = 2000); else CLA and tilt (paper: K = 1000)
  return state.lines.find((l) => {
    const c = condOf(l, 'omni_CLA'), t = condOf(l, 'tilt');
    if (!c || c[1] !== cla || c[2] !== 30 || !isPaper(l) || l.conditions.length !== (tilt === null ? 1 : 2)) return false;
    if (tilt === null) return l.parameters.k_neighbors === 2000;
    return t && t[1] === tilt && t[2] === 10 && l.parameters.k_neighbors === 1000;
  });
}

function describe(line) {
  const conds = line.conditions.map(([c, m, w]) =>
    `${c}=${m}±${w}${['omni_CLA', 'omni_COA', 'tilt'].includes(c) ? '°' : ''}`).join(', ');
  const p = line.parameters;
  return `${conds} | K=${p.k_neighbors}, dist. barycenter=${p.max_distance_barycenter}, median dist.=${p.median_distance}, ` +
         `sigma map=${p.sigma_map}, sigma line=${p.sigma_line}`;
}
const shortLabel = (line) => line.conditions.map(([c, m]) => `${c === 'omni_CLA' ? 'CLA' : c} ${m}°`).join(', ');

// ------------------------------------------------------------------------------------------------------------
// Main part of the line (same as keep_main_part: pieces = connected components of the points closer than maxGap,
// keep the piece with the largest bounding-box diagonal) and breaks between the contours
// ------------------------------------------------------------------------------------------------------------
function keepMainPart(y, z, maxGap) {
  const idx = [];
  for (let i = 0; i < y.length; i++) if (y[i] !== null && z[i] !== null) idx.push(i);
  if (!idx.length) return [[], []];
  const parent = idx.map((_, k) => k);
  const find = (a) => { while (parent[a] !== a) { parent[a] = parent[parent[a]]; a = parent[a]; } return a; };
  const cells = new Map();
  const cellOf = (k) => [Math.floor(y[idx[k]] / maxGap), Math.floor(z[idx[k]] / maxGap)];
  idx.forEach((_, k) => {
    const [cy, cz] = cellOf(k), key = `${cy},${cz}`;
    if (!cells.has(key)) cells.set(key, []);
    cells.get(key).push(k);
  });
  const g2 = maxGap * maxGap;
  idx.forEach((_, k) => {
    const [cy, cz] = cellOf(k);
    for (let dy = -1; dy <= 1; dy++) for (let dz = -1; dz <= 1; dz++) {
      const list = cells.get(`${cy + dy},${cz + dz}`);
      if (!list) continue;
      for (const m of list) {
        if (m <= k) continue;
        const ddy = y[idx[k]] - y[idx[m]], ddz = z[idx[k]] - z[idx[m]];
        if (ddy * ddy + ddz * ddz <= g2) { const a = find(k), b = find(m); if (a !== b) parent[a] = b; }
      }
    }
  });
  const box = new Map();
  idx.forEach((i, k) => {
    const r = find(k);
    const b = box.get(r) || [Infinity, -Infinity, Infinity, -Infinity];
    box.set(r, [Math.min(b[0], y[i]), Math.max(b[1], y[i]), Math.min(b[2], z[i]), Math.max(b[3], z[i])]);
  });
  let best = null, bestExt = -1;
  for (const [r, b] of box) { const e = Math.hypot(b[1] - b[0], b[3] - b[2]); if (e > bestExt) { bestExt = e; best = r; } }
  const yo = [], zo = [];
  idx.forEach((i, k) => { if (find(k) === best) { yo.push(y[i]); zo.push(z[i]); } });
  return [yo, zo];
}

function lineToPlot(line) {
  let y = line.y, z = line.z;
  if ($('main-part').checked) [y, z] = keepMainPart(y, z, +$('max-gap').value);
  else { const ok = y.map((v, i) => v !== null && z[i] !== null); y = y.filter((_, i) => ok[i]); z = z.filter((_, i) => ok[i]); }
  // the line is made of several contours put end to end: do not join them
  const yo = [], zo = [];
  for (let i = 0; i < y.length; i++) {
    if (i > 0 && Math.hypot(y[i] - y[i - 1], z[i] - z[i - 1]) > 1) { yo.push(null); zo.push(null); }
    yo.push(y[i]); zo.push(z[i]);
  }
  return [yo, zo];
}

// X of the magnetopause at (y, z), bilinear interpolation on the grid of the maps
function xOnMP(y, z) {
  const g = state.index.grid, N = g.N_grid, M = 2 * N;
  const fj = (y - g.y[0]) / (g.y[1] - g.y[0]) * (M - 1), fi = (z - g.z[0]) / (g.z[1] - g.z[0]) * (N - 1);
  const i0 = Math.max(0, Math.min(N - 2, Math.floor(fi))), j0 = Math.max(0, Math.min(M - 2, Math.floor(fj)));
  const di = fi - i0, dj = fj - j0;
  const v = [[i0, j0, (1 - di) * (1 - dj)], [i0 + 1, j0, di * (1 - dj)], [i0, j0 + 1, (1 - di) * dj], [i0 + 1, j0 + 1, di * dj]];
  let s = 0, w = 0;
  for (const [i, j, c] of v) { const x = g.X[i][j]; if (x !== null && c > 0) { s += c * x; w += c; } }
  return w > 0 ? s / w : null;
}

// ------------------------------------------------------------------------------------------------------------
// Plots
// ------------------------------------------------------------------------------------------------------------
const linspace = (a, b, n) => Array.from({ length: n }, (_, i) => a + (b - a) * i / (n - 1));
const darkLayout = {
  paper_bgcolor: '#161b22', plot_bgcolor: '#161b22', font: { color: '#e6edf3', size: 12 },
  margin: { l: 50, r: 10, t: 10, b: 40 }, showlegend: true,
  legend: { orientation: 'h', y: -0.12, font: { size: 11 } },
};

function shownLines() {
  const lines = [], colors = [], labels = [];
  const cur = state.current;
  if (cur && !state.kept.some((k) => k.key === cur.key)) { lines.push(cur); colors.push(CURRENT_COLOR); labels.push(`${shortLabel(cur)} (current)`); }
  const n = state.kept.length;
  state.kept.forEach((l, i) => {
    lines.push(l); labels.push(shortLabel(l));
    colors.push(n <= TAB10.length ? TAB10[i] : `hsl(${Math.round(220 - 220 * i / (n - 1))}, 75%, 55%)`);  // many lines: blue to red
  });
  return { lines, colors, labels };
}

async function draw() {
  const { lines, colors, labels } = shownLines();
  const N = state.index.grid.N_grid;
  const Y = linspace(-15, 15, 2 * N), Z = linspace(-15, 15, N);
  let map = null;
  $('map-status').textContent = '';
  if ($('show-map').checked && state.current) {
    if (state.current.has_map) {
      try { map = await getMap(state.current); } catch (e) { $('map-status').textContent = `(${e.message})`; }
    } else $('map-status').textContent = '(no map saved for this X-line)';
  }
  let vmax = 1;
  if (map) for (const row of map) for (const v of row) if (v !== null) vmax = Math.max(vmax, Math.abs(v));

  // --- 2D ---
  const t2 = [];
  if (map) t2.push({ type: 'heatmap', x: Y, y: Z, z: map, colorscale: SEISMIC, zmin: -vmax, zmax: vmax, showlegend: false,
                     colorbar: { title: { text: 'ΔVz (km/s)', side: 'right' }, len: 0.8 }, hovertemplate: 'Y %{x:.1f}, Z %{y:.1f}: %{z:.0f} km/s<extra></extra>' });
  const th = linspace(0, 2 * Math.PI, 200);
  t2.push({ type: 'scatter', x: th.map((t) => 15.2 * Math.cos(t)), y: th.map((t) => 15.2 * Math.sin(t)), mode: 'lines',
            line: { color: '#9aa4ad', width: 1 }, hoverinfo: 'skip', showlegend: false });
  lines.forEach((l, i) => {
    const [y, z] = lineToPlot(l);
    t2.push({ type: 'scatter', x: y, y: z, mode: 'lines', name: labels[i], line: { color: colors[i], width: colors[i] === CURRENT_COLOR ? 4 : 2.5 },
              hovertemplate: `${labels[i]}<br>Y %{x:.2f}, Z %{y:.2f}<extra></extra>` });
  });
  const ax = { range: [-16, 16], zeroline: true, zerolinecolor: '#5a6470', gridcolor: '#232a33', constrain: 'domain' };
  Plotly.react('plot-2d', t2, { ...darkLayout, margin: { l: 50, r: 10, t: 10, b: 10 },
                                legend: { orientation: 'h', y: -0.16, yanchor: 'top', font: { size: 11 } },
                                xaxis: { ...ax, title: 'Y (R<sub>E</sub>)' },
                                yaxis: { ...ax, title: 'Z (R<sub>E</sub>)', scaleanchor: 'x' } }, { responsive: true });

  // --- 3D ---
  const X = state.index.grid.X;
  const Y2 = Z.map(() => Y), Z2 = Z.map((z) => Y.map(() => z));  // plotly needs x, y, z all 2D here
  const t3 = [{ type: 'surface', x: X, y: Y2, z: Z2, surfacecolor: X.map((r) => r.map(() => 0)),
                colorscale: [[0, '#5a6470'], [1, '#5a6470']], showscale: false, opacity: 0.55, hoverinfo: 'skip', name: 'magnetopause' }];
  if (map) {
    const Xm = X.map((r, i) => r.map((x, j) => (x === null || map[i][j] === null ? null : x + 0.02)));
    t3.push({ type: 'surface', x: Xm, y: Y2, z: Z2, surfacecolor: map, colorscale: SEISMIC,
              cmin: -vmax, cmax: vmax, showscale: false, hoverinfo: 'skip', name: 'ΔVz' });
  }
  const u = linspace(0, 2 * Math.PI, 30), v = linspace(0, Math.PI, 15);
  t3.push({ type: 'surface', x: v.map((b) => u.map((a) => Math.cos(a) * Math.sin(b))), y: v.map((b) => u.map((a) => Math.sin(a) * Math.sin(b))),
            z: v.map((b) => u.map(() => Math.cos(b))), colorscale: [[0, '#3060c0'], [1, '#3060c0']], showscale: false, hoverinfo: 'skip' });
  lines.forEach((l, i) => {
    const [y, z] = lineToPlot(l);
    const x = y.map((yy, k) => (yy === null ? null : (xOnMP(yy, z[k]) ?? null)));
    t3.push({ type: 'scatter3d', x: x.map((xx) => (xx === null ? null : xx + 0.2)), y, z, mode: 'lines', name: labels[i],
              line: { color: colors[i], width: colors[i] === CURRENT_COLOR ? 8 : 6 }, showlegend: false, hoverinfo: 'skip' });
  });
  const a3 = { gridcolor: '#2a3038', backgroundcolor: '#161b22', showbackground: true };
  Plotly.react('plot-3d', t3, { ...darkLayout, showlegend: false, margin: { l: 0, r: 0, t: 0, b: 0 },
    scene: { xaxis: { ...a3, title: 'X (RE)' }, yaxis: { ...a3, title: 'Y (RE)' }, zaxis: { ...a3, title: 'Z (RE)' },
             aspectmode: 'data', camera: { eye: { x: 1.7, y: 0.25, z: 0.35 } } },
    uirevision: 'keep-camera' }, { responsive: true });

  $('description').innerHTML = state.current
    ? `<b>${describe(state.current)}</b> — ${state.current.n_points.toLocaleString()} points (after symmetries)` +
      (state.kept.length ? ` — ${state.kept.length} kept line(s)` : '')
    : (lines.length ? `${lines.length} kept line(s)` : 'No saved X-line for these conditions.');
}

// ------------------------------------------------------------------------------------------------------------
// Controls
// ------------------------------------------------------------------------------------------------------------
function setupSlider(id, values, initial) {
  const s = $(id);
  s.min = 0; s.max = values.length - 1; s.step = 1;
  s.value = Math.max(0, values.indexOf(initial));
  s.dataset.values = JSON.stringify(values);
  const show = () => { $(`${id}-value`).textContent = values[+s.value]; };
  show();
  s.addEventListener('input', () => { show(); selectFromSliders(); });
}
const sliderValue = (id) => JSON.parse($(id).dataset.values)[+$(id).value];

function selectFromSliders() {
  $('other').value = '';
  $('tilt').disabled = $('all-tilts').checked;
  state.current = gridLine(sliderValue('cla'), $('all-tilts').checked ? null : sliderValue('tilt')) || null;
  draw();
}

async function loadDataset(id) {
  state.dataset = state.index.datasets.find((d) => d.id === id);
  state.lines = await getJSON(`${DATA}/${id}/lines.json`);
  state.kept = []; state.maps.clear();
  const clas = [...new Set(state.lines.map((l) => condOf(l, 'omni_CLA')?.[1]).filter((v) => v !== undefined))].sort((a, b) => a - b);
  const tilts = [...new Set(state.lines.map((l) => condOf(l, 'tilt')?.[1]).filter((v) => v !== undefined))].sort((a, b) => a - b);
  setupSlider('cla', clas.length ? clas : [150], 150);
  setupSlider('tilt', tilts.length ? tilts : [0], 0);
  const other = $('other');
  other.innerHTML = '<option value="">—</option>';
  state.lines.filter((l) => !(isPaper(l) && (gridLine(condOf(l, 'omni_CLA')?.[1], condOf(l, 'tilt')?.[1] ?? null) === l)))
    .forEach((l) => { const o = document.createElement('option'); o.value = l.key; o.textContent = describe(l); other.appendChild(o); });
  other.disabled = other.options.length === 1;
  $('dataset-info').innerHTML = `${state.dataset.n_lines} saved X-lines` +
    (state.dataset.url ? ` · <a href="${state.dataset.url}" style="color:var(--accent)">data</a>` : '');
  selectFromSliders();
}

async function main() {
  try {
    state.index = await getJSON(`${DATA}/index.json`);
    const sel = $('dataset');
    state.index.datasets.forEach((d) => { const o = document.createElement('option'); o.value = d.id; o.textContent = d.label || d.file; sel.appendChild(o); });
    sel.addEventListener('change', () => loadDataset(sel.value));
    $('all-tilts').addEventListener('change', selectFromSliders);
    $('other').addEventListener('change', () => { const k = $('other').value; if (k) { state.current = state.lines.find((l) => l.key === k); draw(); } else selectFromSliders(); });
    $('main-part').addEventListener('change', draw);
    $('max-gap').addEventListener('input', () => { $('max-gap-value').textContent = $('max-gap').value; draw(); });
    $('show-map').addEventListener('change', draw);
    $('keep').addEventListener('click', () => { if (state.current && !state.kept.some((k) => k.key === state.current.key)) state.kept.push(state.current); draw(); });
    $('clear').addEventListener('click', () => { state.kept = []; draw(); });
    $('all-tilts-at-cla').addEventListener('click', () => {
      const cla = sliderValue('cla');
      state.kept = JSON.parse($('tilt').dataset.values).map((t) => gridLine(cla, t)).filter(Boolean);
      draw();
    });
    $('all-clas-at-tilt').addEventListener('click', () => {
      const tilt = $('all-tilts').checked ? null : sliderValue('tilt');
      state.kept = JSON.parse($('cla').dataset.values).map((c) => gridLine(c, tilt)).filter(Boolean);
      draw();
    });
    await loadDataset(sel.value);
    $('status').textContent = 'ready';
  } catch (e) {
    $('status').innerHTML = `<span class="error">${e.message}</span>`;
    throw e;
  }
}
main();
