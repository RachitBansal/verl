"""Build the interactive (batch size, LR) grid: slide the target accuracy, see which LR is fastest at
each batch and whether that optimum is bracketed by a tested LR above and below.

Reads the JSON written by pull_val_curves.py (full validation curve per n = 16 run, KL 1e-3 and 1e-2)
and embeds it in a self-contained page. For the chosen target the browser computes, per (KL, batch, LR)
configuration, the fastest run's interpolated crossing; per (KL, batch) column the best LR and its
bracket status:
  complete     a tested LR above AND below the best, each either slower or conclusively not reaching
               the target (finished, and ran at least as long as the best crossing)
  inconclusive a neighbour exists but its only runs are still running / too short to judge
  open         no tested LR on that side
A strip under the chart shows, for every target, how many columns are complete, so the reliability of
the LR-scaling picture can be read off as a function of target accuracy.

Usage: python build_lr_grid_interactive.py csv/val_curves_n16.json html/lr_grid_interactive.html
"""
import json, sys, os

src, out = sys.argv[1], sys.argv[2]
runs = json.load(open(src))
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

TEMPLATE = r"""<title>LR Grid Brackets</title>
<style>
:root {
  color-scheme: light;
  --surface: #fcfcfb; --plane: #f4f4f1; --ink: #0b0b0b; --ink2: #52514e; --muted: #898781;
  --grid: #e6e5e1; --axis: #c3c2b7; --border: rgba(11,11,11,.10);
  --blue: #2a78d6; --ramp0: #cde2fb; --ramp1: #86b6ef; --ramp2: #3987e5; --ramp3: #184f95; --ramp4: #0d366b;
  --kl2: #52514e; --focus: #2a78d6; --good: #008300; --warn: #b8860b; --bad: #c62828;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --surface: #1a1a19; --plane: #121211; --ink: #ffffff; --ink2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,.10);
    --blue: #3987e5; --ramp0: #184f95; --ramp1: #256abf; --ramp2: #3987e5; --ramp3: #86b6ef; --ramp4: #cde2fb;
    --kl2: #c3c2b7; --focus: #86b6ef; --good: #4cc04c; --warn: #e0b040; --bad: #ef6b6b;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface: #1a1a19; --plane: #121211; --ink: #ffffff; --ink2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,.10);
  --blue: #3987e5; --ramp0: #184f95; --ramp1: #256abf; --ramp2: #3987e5; --ramp3: #86b6ef; --ramp4: #cde2fb;
  --kl2: #c3c2b7; --focus: #86b6ef; --good: #4cc04c; --warn: #e0b040; --bad: #ef6b6b;
}
body { background: var(--plane); color: var(--ink); font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif;
       padding-block: 20px 32px; padding-inline: 16px; }
.wrap { max-width: 1120px; margin: 0 auto; display: grid; gap: 14px; }
h1 { font-size: 20px; font-weight: 600; margin: 0; letter-spacing: -.01em; text-wrap: balance; }
.sub { color: var(--ink2); margin: 0; max-width: 80ch; }
.panel { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; }
.controls { display: flex; flex-wrap: wrap; gap: 14px 28px; align-items: end; padding: 12px 16px; }
.ctl { display: grid; gap: 4px; }
.ctl .lbl { font-size: 11px; letter-spacing: .04em; text-transform: uppercase; color: var(--ink2); }
.target { display: flex; align-items: center; gap: 10px; }
.target input[type=range] { width: min(360px, 60vw); accent-color: var(--blue); }
.target output { font-variant-numeric: tabular-nums; font-size: 22px; font-weight: 600; min-width: 4.2ch; }
.checks { display: flex; gap: 14px; align-items: center; }
.checks label { display: inline-flex; gap: 6px; align-items: center; font-size: 13px; color: var(--ink); cursor: pointer; }
.checks input { accent-color: var(--blue); }
.chart { position: relative; padding: 6px 8px 0; }
svg { width: 100%; height: auto; display: block; }
.axis-lbl { fill: var(--ink2); font-size: 11px; }
.tick { fill: var(--ink2); font-size: 10.5px; font-variant-numeric: tabular-nums; }
.gridline { stroke: var(--grid); stroke-width: 1; }
.head3 { stroke: var(--ink); stroke-width: 1.4; fill: none; stroke-dasharray: 6 4; }
.head2 { stroke: var(--kl2); stroke-width: 1.4; fill: none; stroke-dasharray: 2 4; }
.dot { stroke: var(--surface); stroke-width: 1; }
.miss { fill: none; stroke: var(--muted); stroke-width: 1.4; }
.inc { stroke: var(--muted); stroke-width: 1.4; }
.ring { fill: none; stroke-width: 2.2; }
.lab { fill: var(--ink); font-size: 10.5px; font-variant-numeric: tabular-nums; }
.fx { fill: var(--ink); }
.tip { position: absolute; pointer-events: none; background: var(--ink); color: var(--surface); font-size: 12px; padding: 7px 9px;
       border-radius: 4px; max-width: 320px; opacity: 0; transition: opacity .08s; white-space: nowrap; z-index: 2; }
.legend { display: flex; flex-wrap: wrap; gap: 8px 18px; padding: 6px 16px 12px; font-size: 12.5px; color: var(--ink2); }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend svg { width: 26px; height: 14px; }
.strip { padding: 10px 16px 4px; }
.strip h2 { font-size: 13px; font-weight: 600; margin: 0 0 4px; color: var(--ink); }
.foot { color: var(--ink2); font-size: 12.5px; max-width: 90ch; margin: 0; }
.foot b { color: var(--ink); font-weight: 600; }
.tblwrap { overflow-x: auto; padding: 0 16px 12px; }
table { border-collapse: collapse; font-size: 12.5px; min-width: 100%; }
th, td { text-align: left; padding: 5px 10px; border-bottom: 1px solid var(--grid); white-space: nowrap; }
th { color: var(--ink2); font-weight: 500; font-size: 11px; letter-spacing: .04em; text-transform: uppercase; }
td.n { font-variant-numeric: tabular-nums; }
.st-complete { color: var(--good); } .st-inconclusive { color: var(--warn); } .st-open { color: var(--bad); }
summary { cursor: pointer; color: var(--ink2); font-size: 13px; padding: 10px 16px 6px; }
</style>
<div class="wrap">
  <h1>Which learning rate is fastest at each batch size, and is it bracketed?</h1>
  <p class="sub">GRPO on-policy, n = 16 rollouts, AIME 1983–2024 validation. Slide the target accuracy. Each marker is a
  (batch, LR) configuration; the ringed marker in a column is the fastest LR to the target, and the ring colour says whether a
  tested LR above <em>and</em> below it is slower or conclusively fails. Columns without a complete bracket are the places where the
  LR-scaling plots are not yet reliable at that target.</p>
  <div class="panel">
    <div class="controls">
      <div class="ctl"><span class="lbl">Target accuracy</span>
        <div class="target"><input type="range" id="target" min="0.30" max="0.60" step="0.005" value="0.50"><output id="tval">50.0%</output></div></div>
      <div class="ctl"><span class="lbl">Show</span>
        <div class="checks">
          <label><input type="checkbox" id="kl3" checked> KL 1e-3 (circles)</label>
          <label><input type="checkbox" id="kl2" checked> KL 1e-2 (diamonds)</label>
          <label><input type="checkbox" id="labels" checked> step labels</label>
        </div></div>
    </div>
    <div class="chart"><svg id="chart" viewBox="0 0 1000 560" role="img" aria-label="LR versus batch size grid"></svg><div class="tip" id="tip"></div></div>
    <div class="legend">
      <span><svg viewBox="0 0 26 14"><circle cx="13" cy="7" r="4.5" fill="var(--ramp2)" class="dot"/></svg>reached the target, coloured by steps (dark = fewer)</span>
      <span><svg viewBox="0 0 26 14"><circle cx="13" cy="7" r="4.5" class="miss"/></svg>conclusively did not reach it</span>
      <span><svg viewBox="0 0 26 14"><path d="M9 3 L17 11 M17 3 L9 11" class="inc"/></svg>inconclusive: running or too short to judge</span>
      <span><svg viewBox="0 0 26 14"><circle cx="13" cy="7" r="6" class="ring" stroke="var(--good)"/></svg>best LR, bracket complete</span>
      <span><svg viewBox="0 0 26 14"><circle cx="13" cy="7" r="6" class="ring" stroke="var(--warn)"/></svg>best LR, a neighbour is inconclusive</span>
      <span><svg viewBox="0 0 26 14"><circle cx="13" cy="7" r="6" class="ring" stroke="var(--bad)"/></svg>best LR at the edge of the tested grid</span>
      <span><svg viewBox="0 0 26 14"><circle cx="13" cy="7" r="4.5" fill="var(--ramp2)" class="dot"/><circle cx="13" cy="7" r="1.6" class="fx"/></svg>centre dot = fixed loss scaling</span>
    </div>
    <div class="strip"><h2>Columns with a complete bracket, by target accuracy</h2>
      <svg id="strip" viewBox="0 0 1000 120" role="img" aria-label="complete brackets versus target"></svg></div>
    <p class="foot" id="foot" style="padding: 6px 16px 12px"></p>
    <details open><summary>Bracket status per column at this target</summary>
      <div class="tblwrap"><table id="tbl"></table></div></details>
  </div>
</div>
<script>
const RUNS = __RUNS__;
const $ = id => document.getElementById(id);
const svg = $("chart"), tip = $("tip"), strip = $("strip");
const W = 1000, H = 560, M = {l: 70, r: 20, t: 14, b: 50};
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const RAMP = ["--ramp0", "--ramp1", "--ramp2", "--ramp3", "--ramp4"];
const KLS = [0.001, 0.01];
const SHORT = 200;

function crossing(run, t) {
  const {steps, vals} = run;
  for (let i = 0; i < vals.length; i++) if (vals[i] >= t) {
    if (i === 0) return steps[0];
    const s0 = steps[i-1], v0 = vals[i-1], s1 = steps[i], v1 = vals[i];
    return v1 === v0 ? s1 : s0 + (s1 - s0) * (t - v0) / (v1 - v0);
  }
  return null;
}
function hexToRgb(h) { const n = parseInt(h.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
function rampColor(u) {
  const cols = RAMP.map(v => hexToRgb(css(v)));
  const x = Math.max(0, Math.min(1, u)) * (cols.length - 1), i = Math.min(cols.length - 2, Math.floor(x)), f = x - i;
  const c = cols[i].map((a, k) => Math.round(a + (cols[i+1][k] - a) * f));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}
const fmtLR = lr => { const e = Math.floor(Math.log10(lr) + 1e-9); const m = lr / 10 ** e; return (Math.round(m * 10) / 10).toString().replace(/\.0$/, "") + "e" + e; };
const fmtSteps = s => s >= 100 ? Math.round(s).toString() : (Math.round(s * 10) / 10).toString();
const el = (tag, attrs, parent) => { const e = document.createElementNS("http://www.w3.org/2000/svg", tag); for (const k in attrs) e.setAttribute(k, attrs[k]); (parent || svg).appendChild(e); return e; };

// ---- core: per (kl, bsz, lr) configuration and per (kl, bsz) column, at target t
function analyse(t) {
  const cfg = new Map();                 // key -> {kl,bsz,lr,steps,longest,running,fixed,run}
  for (const r of RUNS) {
    if (!KLS.some(k => Math.abs(r.kl - k) < 1e-12)) continue;
    const s = crossing(r, t);
    const key = `${r.kl}|${r.bsz}|${r.lr}`;
    const last = r.steps[r.steps.length - 1];
    const c = cfg.get(key) || {kl: r.kl, bsz: r.bsz, lr: r.lr, steps: null, longest: 0, running: false, fixed: false, run: null, runs: 0};
    c.runs++;
    if (s !== null && (c.steps === null || s < c.steps)) { c.steps = s; c.run = r.name; c.fixed = r.fixed; }
    if (last > c.longest) { c.longest = last; if (c.steps === null) { c.run = r.name; c.fixed = r.fixed; } }
    if (r.state === "running") c.running = true;
    cfg.set(key, c);
  }
  const cols = new Map();                // `${kl}|${bsz}` -> {kl,bsz,items:[cfg sorted by lr], best, lower, upper, status}
  for (const c of cfg.values()) {
    const k = `${c.kl}|${c.bsz}`; if (!cols.has(k)) cols.set(k, {kl: c.kl, bsz: c.bsz, items: []});
    cols.get(k).items.push(c);
  }
  for (const col of cols.values()) {
    col.items.sort((a, b) => a.lr - b.lr);
    const hits = col.items.filter(c => c.steps !== null);
    col.best = hits.length ? hits.reduce((a, b) => a.steps <= b.steps ? a : b) : null;
    const ref = col.best ? col.best.steps : SHORT;
    // classify every non-crossing config: conclusive "never" needs a finished run that lasted >= the best crossing
    for (const c of col.items) c.verdict = c.steps !== null ? "hit" : (!c.running && c.longest >= Math.min(ref, SHORT) ? "never" : "inconclusive");
    if (col.best) {
      const i = col.items.indexOf(col.best);
      const side = j => (j < 0 || j >= col.items.length) ? "open" : (col.items[j].verdict === "inconclusive" ? "inconclusive" : "complete");
      col.lower = side(i - 1); col.upper = side(i + 1);
      col.status = (col.lower === "open" || col.upper === "open") ? "open" : (col.lower === "inconclusive" || col.upper === "inconclusive") ? "inconclusive" : "complete";
    } else { col.lower = col.upper = "n/a"; col.status = "no crossing"; }
  }
  return {cfg: [...cfg.values()], cols: [...cols.values()].sort((a, b) => a.kl - b.kl || a.bsz - b.bsz)};
}

// ---- main chart
function draw() {
  const t = parseFloat($("target").value); $("tval").textContent = (t * 100).toFixed(1) + "%";
  const show = {0.001: $("kl3").checked, 0.01: $("kl2").checked}, labels = $("labels").checked;
  const {cfg, cols} = analyse(t);
  svg.innerHTML = "";
  const vis = cfg.filter(c => show[c.kl]);
  const bszs = [...new Set(cfg.map(c => c.bsz))].sort((a, b) => a - b);
  const lrs = cfg.map(c => c.lr);
  const x = b => M.l + (Math.log2(b) - Math.log2(bszs[0]) + 0.6) / (Math.log2(bszs[bszs.length-1]) - Math.log2(bszs[0]) + 1.2) * (W - M.l - M.r);
  const lo = Math.log10(Math.min(...lrs)) - 0.15, hi = Math.log10(Math.max(...lrs)) + 0.15;
  const y = lr => H - M.b - (Math.log10(lr) - lo) / (hi - lo) * (H - M.t - M.b);
  const hits = vis.filter(c => c.steps !== null).map(c => Math.log10(c.steps));
  const smin = Math.min(...hits), smax = Math.max(...hits);
  const color = s => rampColor(1 - (Math.log10(s) - smin) / Math.max(1e-9, smax - smin));   // dark = fewer steps
  // grid + axes
  for (const b of bszs) { el("line", {x1: x(b), x2: x(b), y1: M.t, y2: H - M.b, class: "gridline"}); el("text", {x: x(b), y: H - M.b + 16, class: "tick", "text-anchor": "middle"}).textContent = b; }
  for (let e = Math.ceil(lo); e <= Math.floor(hi); e++) { const yy = y(10 ** e); el("line", {x1: M.l, x2: W - M.r, y1: yy, y2: yy, class: "gridline"}); el("text", {x: M.l - 8, y: yy + 4, class: "tick", "text-anchor": "end"}).textContent = `1e${e}`; }
  el("text", {x: (M.l + W - M.r) / 2, y: H - 10, class: "axis-lbl", "text-anchor": "middle"}).textContent = "batch size (prompts per step)";
  el("text", {x: 16, y: (M.t + H - M.b) / 2, class: "axis-lbl", "text-anchor": "middle", transform: `rotate(-90 16 ${(M.t + H - M.b) / 2})`}).textContent = "learning rate";
  const nudge = kl => (show[0.001] && show[0.01]) ? (kl === 0.001 ? -7 : 7) : 0;
  const marker = (c, cx, cy, r, attrs, cls) => c.kl === 0.001
    ? el("circle", Object.assign({cx, cy, r, class: cls}, attrs))
    : el("path", Object.assign({d: `M${cx} ${cy - r * 1.25} L${cx + r * 1.25} ${cy} L${cx} ${cy + r * 1.25} L${cx - r * 1.25} ${cy} Z`, class: cls}, attrs));
  // best-LR lines per KL
  for (const kl of KLS) if (show[kl]) {
    const pts = cols.filter(c => c.kl === kl && c.best).map(c => `${x(c.bsz) + nudge(kl)},${y(c.best.lr)}`);
    if (pts.length > 1) el("polyline", {points: pts.join(" "), class: kl === 0.001 ? "head3" : "head2"});
  }
  // configurations
  for (const c of vis) {
    const cx = x(c.bsz) + nudge(c.kl), cy = y(c.lr);
    let g;
    if (c.verdict === "hit") { g = marker(c, cx, cy, 6, {fill: color(c.steps)}, "dot"); if (c.fixed) el("circle", {cx, cy, r: 1.8, class: "fx"}); }
    else if (c.verdict === "never") g = marker(c, cx, cy, 5.5, {}, "miss");
    else g = el("path", {d: `M${cx-4} ${cy-4} L${cx+4} ${cy+4} M${cx+4} ${cy-4} L${cx-4} ${cy+4}`, class: "inc"});
    const hit = el("circle", {cx, cy, r: 11, fill: "transparent"});
    attachTip(hit, c, t);
  }
  // rings on the best LR, coloured by bracket status
  for (const col of cols) if (show[col.kl] && col.best) {
    const cx = x(col.bsz) + nudge(col.kl), cy = y(col.best.lr);
    const stroke = col.status === "complete" ? css("--good") : col.status === "inconclusive" ? css("--warn") : css("--bad");
    marker(col, cx, cy, 10.5, {stroke}, "ring");
    if (labels) el("text", {x: cx + (col.kl === 0.001 ? -13 : 13), y: cy + (col.kl === 0.001 ? -9 : 15), class: "lab", "text-anchor": col.kl === 0.001 ? "end" : "start"}).textContent = fmtSteps(col.best.steps);
  }
  // table + foot
  const shownCols = cols.filter(c => show[c.kl]);
  const nComplete = shownCols.filter(c => c.status === "complete").length;
  $("foot").innerHTML = `At <b>${(t * 100).toFixed(1)}%</b>: <b>${nComplete}</b> of <b>${shownCols.length}</b> (KL, batch) columns have a complete bracket; ` +
    `<b>${shownCols.filter(c => c.status === "open").length}</b> have the best LR at the edge of the tested grid and <b>${shownCols.filter(c => c.status === "inconclusive").length}</b> have an inconclusive neighbour.`;
  const rows = shownCols.map(c => `<tr><td>${c.kl}</td><td class="n">${c.bsz}</td><td class="n">${c.best ? fmtLR(c.best.lr) : "—"}</td><td class="n">${c.best ? fmtSteps(c.best.steps) : "—"}</td>` +
    `<td class="st-${c.lower}">${c.lower}</td><td class="st-${c.upper}">${c.upper}</td><td class="st-${c.status}">${c.status}</td><td class="n">${c.items.map(i => fmtLR(i.lr) + (i.verdict === "hit" ? "" : i.verdict === "never" ? "×" : "?")).join(", ")}</td></tr>`);
  $("tbl").innerHTML = `<thead><tr><th>KL</th><th>batch</th><th>best LR</th><th>steps</th><th>lower</th><th>upper</th><th>bracket</th><th>LRs tested (× = never, ? = inconclusive)</th></tr></thead><tbody>${rows.join("")}</tbody>`;
  drawStrip(t, show);
}

// ---- strip: complete-bracket count per target
let STRIP_CACHE = null;
function drawStrip(t, show) {
  strip.innerHTML = "";
  if (!STRIP_CACHE) {
    STRIP_CACHE = [];
    for (let tt = 0.30; tt <= 0.6001; tt += 0.005) {
      const {cols} = analyse(tt);
      const row = {t: tt};
      for (const kl of KLS) { const cs = cols.filter(c => c.kl === kl); row[kl] = {complete: cs.filter(c => c.status === "complete").length, total: cs.filter(c => c.best).length}; }
      STRIP_CACHE.push(row);
    }
  }
  const SW = 1000, SH = 120, m = {l: 70, r: 20, t: 8, b: 26};
  const sx = tt => m.l + (tt - 0.30) / 0.30 * (SW - m.l - m.r);
  const maxN = Math.max(...STRIP_CACHE.map(r => Math.max(r[0.001].total, r[0.01].total)));
  const sy = n => SH - m.b - n / maxN * (SH - m.t - m.b);
  const e = (tag, attrs) => { const q = document.createElementNS("http://www.w3.org/2000/svg", tag); for (const k in attrs) q.setAttribute(k, attrs[k]); strip.appendChild(q); return q; };
  for (const tt of [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]) { e("line", {x1: sx(tt), x2: sx(tt), y1: m.t, y2: SH - m.b, class: "gridline"}); e("text", {x: sx(tt), y: SH - m.b + 14, class: "tick", "text-anchor": "middle"}).textContent = (tt * 100).toFixed(0) + "%"; }
  for (const n of [0, Math.round(maxN / 2), maxN]) { e("line", {x1: m.l, x2: SW - m.r, y1: sy(n), y2: sy(n), class: "gridline"}); e("text", {x: m.l - 8, y: sy(n) + 4, class: "tick", "text-anchor": "end"}).textContent = n; }
  e("text", {x: 14, y: (m.t + SH - m.b) / 2, class: "axis-lbl", "text-anchor": "middle", transform: `rotate(-90 14 ${(m.t + SH - m.b) / 2})`}).textContent = "complete";
  for (const kl of KLS) if (show[kl]) {
    e("polyline", {points: STRIP_CACHE.map(r => `${sx(r.t)},${sy(r[kl].complete)}`).join(" "), class: kl === 0.001 ? "head3" : "head2", "stroke-dasharray": "none"});
    e("polyline", {points: STRIP_CACHE.map(r => `${sx(r.t)},${sy(r[kl].total)}`).join(" "), class: kl === 0.001 ? "head3" : "head2", opacity: 0.35});
  }
  e("line", {x1: sx(t), x2: sx(t), y1: m.t, y2: SH - m.b, stroke: css("--blue"), "stroke-width": 1.5});
  e("text", {x: SW - m.r, y: m.t + 10, class: "tick", "text-anchor": "end"}).textContent = "solid = complete brackets, faint = columns with any crossing (KL 1e-3 dashed / KL 1e-2 dotted)";
}

function attachTip(g, c, t) {
  g.addEventListener("mousemove", ev => {
    const r = svg.getBoundingClientRect();
    tip.style.left = (ev.clientX - r.left + 12) + "px"; tip.style.top = (ev.clientY - r.top + 12) + "px"; tip.style.opacity = 1;
    const what = c.verdict === "hit" ? `<b>${fmtSteps(c.steps)} steps</b> to ${(t * 100).toFixed(1)}%` : c.verdict === "never" ? `did not reach ${(t * 100).toFixed(1)}% in ${c.longest} steps` : `inconclusive: ${c.running ? "still running" : "only " + c.longest + " steps logged"}`;
    tip.innerHTML = `KL ${c.kl} · batch ${c.bsz} · LR ${fmtLR(c.lr)}<br>${what}<br><span style="opacity:.75">${c.run}${c.runs > 1 ? ` (+${c.runs - 1} more run${c.runs > 2 ? "s" : ""})` : ""}</span>`;
  });
  g.addEventListener("mouseleave", () => tip.style.opacity = 0);
}
for (const id of ["target", "kl3", "kl2", "labels"]) $(id).addEventListener("input", draw);
new MutationObserver(draw).observe(document.documentElement, {attributes: true, attributeFilter: ["data-theme"]});
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", draw);
draw();
</script>
"""

slim = [dict(name=r["name"], state=r.get("state", "finished"), bsz=r["bsz"], lr=r["lr"], kl=r["kl"], fixed=bool(r.get("fixed")),
             steps=r["steps"], vals=r["vals"]) for r in runs if not r["name"].startswith("downsample")]
html = TEMPLATE.replace("__RUNS__", json.dumps(slim, separators=(",", ":")))
open(out, "w").write(html)
print(f"wrote {out}  ({len(slim)} runs, {len(html) // 1024} KB)")
