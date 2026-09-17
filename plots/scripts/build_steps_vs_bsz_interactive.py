"""Build the interactive steps-vs-batch page: pick a target AIME accuracy, see steps to reach it.

Reads the JSON written by pull_val_curves.py (full validation curve per run) and embeds it in a
self-contained HTML page. Everything else -- steps-to-target for the chosen threshold, the
fastest-LR-per-batch headline per KL coefficient, the perfect-scaling reference -- is computed
in the browser, so the threshold is a live slider.

Usage: python build_steps_vs_bsz_interactive.py csv/val_curves_n16.json html/steps_vs_bsz_interactive.html
"""
import json, sys, os

src, out = sys.argv[1], sys.argv[2]
runs = json.load(open(src))
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

TEMPLATE = r"""<title>Steps to Target</title>
<style>
:root {
  color-scheme: light;
  --surface: #fcfcfb; --plane: #f4f4f1; --ink: #0b0b0b; --ink2: #52514e; --muted: #898781;
  --grid: #e6e5e1; --axis: #c3c2b7; --border: rgba(11,11,11,.10);
  --blue: #2a78d6; --ramp0: #cde2fb; --ramp1: #86b6ef; --ramp2: #3987e5; --ramp3: #184f95; --ramp4: #0d366b;
  --kl2: #52514e; --focus: #2a78d6;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --surface: #1a1a19; --plane: #121211; --ink: #ffffff; --ink2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,.10);
    --blue: #3987e5; --ramp0: #184f95; --ramp1: #256abf; --ramp2: #3987e5; --ramp3: #86b6ef; --ramp4: #cde2fb;
    --kl2: #c3c2b7; --focus: #86b6ef;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface: #1a1a19; --plane: #121211; --ink: #ffffff; --ink2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,.10);
  --blue: #3987e5; --ramp0: #184f95; --ramp1: #256abf; --ramp2: #3987e5; --ramp3: #86b6ef; --ramp4: #cde2fb;
  --kl2: #c3c2b7; --focus: #86b6ef;
}
body { background: var(--plane); color: var(--ink); font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif;
       padding-block: 20px 32px; padding-inline: 16px; }
.wrap { max-width: 1080px; margin: 0 auto; display: grid; gap: 14px; }
h1 { font-size: 20px; font-weight: 600; margin: 0; letter-spacing: -.01em; text-wrap: balance; }
.sub { color: var(--ink2); margin: 0; max-width: 72ch; }
.panel { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; }
.controls { display: flex; flex-wrap: wrap; gap: 14px 28px; align-items: end; padding: 12px 16px; }
.ctl { display: grid; gap: 4px; }
.ctl label, .ctl .lbl { font-size: 11px; letter-spacing: .04em; text-transform: uppercase; color: var(--ink2); }
.target { display: flex; align-items: center; gap: 10px; }
.target input[type=range] { width: min(320px, 60vw); accent-color: var(--blue); }
.target output { font-variant-numeric: tabular-nums; font-size: 22px; font-weight: 600; min-width: 4.2ch; }
.seg { display: inline-flex; border: 1px solid var(--border); border-radius: 5px; overflow: hidden; }
.seg label { padding: 5px 10px; cursor: pointer; color: var(--ink2); font-size: 13px; text-transform: none; letter-spacing: 0; }
.seg input { position: absolute; opacity: 0; pointer-events: none; }
.seg input:checked + label { background: var(--ink); color: var(--surface); }
.seg input:focus-visible + label { outline: 2px solid var(--focus); outline-offset: -2px; }
.checks { display: flex; gap: 14px; align-items: center; }
.checks label { display: inline-flex; gap: 6px; align-items: center; font-size: 13px; color: var(--ink); text-transform: none; letter-spacing: 0; cursor: pointer; }
.checks input { accent-color: var(--blue); }
.chart { padding: 10px 6px 4px; position: relative; }
svg { width: 100%; height: auto; display: block; }
.axis-lbl { fill: var(--ink2); font-size: 11px; }
.tick { fill: var(--ink2); font-size: 10.5px; font-variant-numeric: tabular-nums; }
.gridline { stroke: var(--grid); stroke-width: 1; }
.ax { stroke: var(--axis); stroke-width: 1; }
.perf { stroke: var(--muted); stroke-width: 1.2; stroke-dasharray: 2 4; fill: none; }
.perf-lbl { fill: var(--ink2); font-size: 10.5px; }
.head3 { stroke: var(--ink); stroke-width: 2; fill: none; }
.head2 { stroke: var(--kl2); stroke-width: 2; fill: none; stroke-dasharray: 7 5; }
.mk3 { fill: var(--surface); stroke: var(--ink); stroke-width: 2; }
.mk2 { fill: var(--surface); stroke: var(--kl2); stroke-width: 2; }
.lab { fill: var(--ink); font-size: 10.5px; font-variant-numeric: tabular-nums; }
.lab2 { fill: var(--kl2); }
.dot { stroke: var(--surface); stroke-width: 1; }
.fx { fill: var(--ink); }
.hit { cursor: default; }
.hit:hover, .hit:focus { outline: none; }
.tip { position: absolute; pointer-events: none; background: var(--ink); color: var(--surface); font-size: 12px; padding: 7px 9px;
       border-radius: 4px; max-width: 320px; line-height: 1.35; opacity: 0; transition: opacity .08s; font-variant-numeric: tabular-nums; }
@media (prefers-reduced-motion: reduce) { .tip { transition: none; } }
.legend { display: flex; flex-wrap: wrap; gap: 8px 18px; padding: 6px 16px 12px; font-size: 12.5px; color: var(--ink2); }
.legend span { display: inline-flex; align-items: center; gap: 7px; }
.legend svg { width: 26px; height: 14px; }
.foot { color: var(--ink2); font-size: 12.5px; max-width: 80ch; margin: 0; }
.foot b { color: var(--ink); font-weight: 600; }
details { padding: 0 16px 12px; }
summary { cursor: pointer; color: var(--ink2); font-size: 13px; padding: 10px 0 6px; }
.tblwrap { overflow-x: auto; }
table { border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; width: 100%; min-width: 560px; }
th, td { text-align: left; padding: 5px 10px; border-bottom: 1px solid var(--grid); white-space: nowrap; }
th { color: var(--ink2); font-weight: 500; font-size: 11px; letter-spacing: .04em; text-transform: uppercase; }
td.num, th.num { text-align: right; }
td.run { color: var(--ink2); font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 12px; }
</style>

<div class="wrap">
  <h1>Training steps to reach a target AIME accuracy, by batch size</h1>
  <p class="sub">GRPO on-policy, n&nbsp;=&nbsp;16 rollouts per prompt, no downsampling. For each batch size and KL coefficient the
  headline is the fastest learning rate tried; every (batch, LR) configuration that reached the target is a small point coloured by LR.</p>

  <div class="panel">
    <div class="controls">
      <div class="ctl">
        <label for="target">Target accuracy (AIME 1983-2024, mean@1)</label>
        <div class="target">
          <input type="range" id="target" min="0.26" max="0.66" step="0.005" value="0.50">
          <output id="target-out" for="target">50.0%</output>
        </div>
      </div>
      <div class="ctl">
        <span class="lbl">Crossing</span>
        <div class="seg" role="radiogroup" aria-label="Crossing method">
          <input type="radio" name="method" id="m-interp" value="interp" checked><label for="m-interp">interpolated</label>
          <input type="radio" name="method" id="m-first" value="first"><label for="m-first">first validation ≥ target</label>
        </div>
      </div>
      <div class="ctl">
        <span class="lbl">Show</span>
        <div class="checks">
          <label><input type="checkbox" id="kl3" checked> KL 1e-3</label>
          <label><input type="checkbox" id="kl2" checked> KL 1e-2</label>
          <label><input type="checkbox" id="allruns" checked> every run</label>
          <label><input type="checkbox" id="perf" checked> 1/batch line</label>
        </div>
      </div>
    </div>
    <div class="chart">
      <svg id="chart" viewBox="0 0 960 560" role="img" aria-labelledby="chart-title"><title id="chart-title">Steps to target accuracy versus batch size</title></svg>
      <div class="tip" id="tip"></div>
    </div>
    <div class="legend">
      <span><svg viewBox="0 0 26 14"><line x1="0" y1="7" x2="26" y2="7" class="head3"/><circle cx="13" cy="7" r="4.5" class="mk3"/></svg>KL 1e-3, fastest LR per batch</span>
      <span><svg viewBox="0 0 26 14"><line x1="0" y1="7" x2="26" y2="7" class="head2"/><rect x="9" y="3" width="8" height="8" transform="rotate(45 13 7)" class="mk2"/></svg>KL 1e-2, fastest LR per batch</span>
      <span><svg viewBox="0 0 26 14"><circle cx="13" cy="7" r="4" fill="var(--ramp2)" class="dot"/></svg>run that reached the target, coloured by LR (light → dark = small → large)</span>
      <span><svg viewBox="0 0 26 14"><circle cx="13" cy="7" r="4" fill="var(--ramp2)" class="dot"/><circle cx="13" cy="7" r="1.6" class="fx"/></svg>centre dot = fixed loss scaling (runs after 2026-09-11)</span>
      <span><svg viewBox="0 0 26 14"><line x1="0" y1="7" x2="26" y2="7" class="perf"/></svg>perfect scaling, steps ∝ 1/batch, through the smallest-batch KL 1e-3 point</span>
    </div>
    <details>
      <summary>Headline points as a table</summary>
      <div class="tblwrap"><table id="tbl"></table></div>
    </details>
  </div>

  <p class="foot" id="foot"></p>
  <p class="foot">Validation runs every 25 training steps. <b>Interpolated</b> places the crossing linearly between the two validation
  readings that bracket the target; <b>first validation ≥ target</b> reports the checkpoint itself, which is how the static figures were
  made and overstates fast runs by up to one interval. Pre-fix runs (no centre dot) stepped at roughly 0.7–0.9× their nominal LR
  because of the loss-scaling bug; the LR shown is nominal.</p>
</div>

<script>
const RUNS = __RUNS__;
const $ = id => document.getElementById(id);
const svg = $("chart"), tip = $("tip");
const W = 960, H = 560, M = {l: 66, r: 24, t: 14, b: 52};
const RAMP = ["--ramp0", "--ramp1", "--ramp2", "--ramp3", "--ramp4"];
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

function crossing(run, t, method) {
  const {steps, vals} = run;
  for (let i = 0; i < vals.length; i++) {
    if (vals[i] >= t) {
      if (method === "first" || i === 0) return steps[i];
      const s0 = steps[i-1], v0 = vals[i-1], s1 = steps[i], v1 = vals[i];
      return v1 === v0 ? s1 : s0 + (s1 - s0) * (t - v0) / (v1 - v0);
    }
  }
  return null;
}
function hexToRgb(h) { const n = parseInt(h.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
function rampColor(u) {   // u in [0,1] -> interpolate the 5-step ramp
  const cols = RAMP.map(v => hexToRgb(css(v)));
  const x = Math.max(0, Math.min(1, u)) * (cols.length - 1), i = Math.min(cols.length - 2, Math.floor(x)), f = x - i;
  const c = cols[i].map((a, k) => Math.round(a + (cols[i+1][k] - a) * f));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}
const fmtLR = lr => { const e = Math.floor(Math.log10(lr)); const m = lr / 10 ** e; return (Math.round(m * 10) / 10).toString().replace(/\.0$/, "") + "e" + e; };
const fmtSteps = s => s >= 100 ? Math.round(s).toString() : (Math.round(s * 10) / 10).toString();
const el = (tag, attrs, parent) => { const e = document.createElementNS("http://www.w3.org/2000/svg", tag); for (const k in attrs) e.setAttribute(k, attrs[k]); (parent || svg).appendChild(e); return e; };

function compute() {
  const t = parseFloat($("target").value);
  const method = document.querySelector("input[name=method]:checked").value;
  const show = {0.001: $("kl3").checked, 0.01: $("kl2").checked};
  // per (kl, bsz, lr): fastest run to the target (nulls last, then longest run)
  const groups = new Map();
  for (const r of RUNS) {
    if (!show[r.kl]) continue;
    const s = crossing(r, t, method);
    const key = `${r.kl}|${r.bsz}|${r.lr}`;
    const cand = {run: r, steps: s, maxVal: Math.max(...r.vals), lastStep: r.steps[r.steps.length - 1]};
    const cur = groups.get(key);
    const better = !cur || (s !== null && (cur.steps === null || s < cur.steps)) || (s === null && cur.steps === null && cand.lastStep > cur.lastStep);
    if (better) groups.set(key, cand);
  }
  const cfgs = [...groups.values()];
  const hits = cfgs.filter(c => c.steps !== null);
  const never = cfgs.filter(c => c.steps === null);
  const best = {};   // kl -> [{bsz, ...}] fastest per bsz
  for (const kl of [0.001, 0.01]) {
    const byB = new Map();
    for (const c of hits.filter(c => c.run.kl === kl)) { const b = byB.get(c.run.bsz); if (!b || c.steps < b.steps) byB.set(c.run.bsz, c); }
    best[kl] = [...byB.values()].sort((a, b) => a.run.bsz - b.run.bsz);
  }
  return {t, method, hits, never, best};
}

function draw() {
  const {t, method, hits, never, best} = compute();
  $("target-out").textContent = (t * 100).toFixed(1) + "%";
  while (svg.childNodes.length > 1) svg.removeChild(svg.lastChild);
  const all = hits;
  if (!all.length) { el("text", {x: W/2, y: H/2, "text-anchor": "middle", class: "axis-lbl"}).textContent = "No run reaches this target."; $("tbl").innerHTML = ""; $("foot").textContent = ""; return; }
  const bszs = [...new Set(RUNS.map(r => r.bsz))].sort((a, b) => a - b);
  const x = b => M.l + (Math.log2(b) - Math.log2(bszs[0])) / (Math.log2(bszs[bszs.length-1]) - Math.log2(bszs[0])) * (W - M.l - M.r) * 0.92 + (W - M.l - M.r) * 0.04;
  const sMin = Math.min(...all.map(c => c.steps)), sMax = Math.max(...all.map(c => c.steps));
  const yLo = Math.pow(10, Math.floor(Math.log10(Math.max(1, sMin / 1.6)))), yHi = Math.pow(10, Math.ceil(Math.log10(sMax * 1.4)));
  const y = s => M.t + (Math.log10(yHi) - Math.log10(s)) / (Math.log10(yHi) - Math.log10(yLo)) * (H - M.t - M.b);
  const lrs = all.map(c => c.run.lr), lrMin = Math.min(...lrs), lrMax = Math.max(...lrs);
  const lrU = lr => lrMax === lrMin ? 0.5 : (Math.log10(lr) - Math.log10(lrMin)) / (Math.log10(lrMax) - Math.log10(lrMin));

  // grid + axes
  for (let d = yLo; d <= yHi; d *= 10) for (const m of [1, 2, 5]) { const v = d * m; if (v < yLo || v > yHi) continue;
    el("line", {x1: M.l, x2: W - M.r, y1: y(v), y2: y(v), class: "gridline"});
    if (m === 1) el("text", {x: M.l - 8, y: y(v) + 3.5, "text-anchor": "end", class: "tick"}).textContent = v.toLocaleString(); }
  for (const b of bszs) { el("line", {x1: x(b), x2: x(b), y1: M.t, y2: H - M.b, class: "gridline"});
    el("text", {x: x(b), y: H - M.b + 16, "text-anchor": "middle", class: "tick"}).textContent = b.toLocaleString(); }
  el("line", {x1: M.l, x2: W - M.r, y1: H - M.b, y2: H - M.b, class: "ax"});
  el("line", {x1: M.l, x2: M.l, y1: M.t, y2: H - M.b, class: "ax"});
  el("text", {x: (M.l + W - M.r) / 2, y: H - 10, "text-anchor": "middle", class: "axis-lbl"}).textContent = "batch size (prompts per step)";
  const yl = el("text", {x: 14, y: (M.t + H - M.b) / 2, "text-anchor": "middle", class: "axis-lbl", transform: `rotate(-90 14 ${(M.t + H - M.b) / 2})`});
  yl.textContent = `training steps to reach ${(t * 100).toFixed(1)}% AIME`;

  // perfect scaling from the smallest-batch KL 1e-3 headline point (fall back to KL 1e-2)
  const anchorSeries = best[0.001].length ? best[0.001] : best[0.01];
  if ($("perf").checked && anchorSeries.length) {
    const a = anchorSeries[0], b0 = a.run.bsz, s0 = a.steps;
    const pts = [bszs[0] / 1.3, bszs[bszs.length - 1] * 1.3].map(b => [b, s0 * b0 / b]).filter(([, s]) => s >= yLo && s <= yHi);
    const xb = b => M.l + (Math.log2(b) - Math.log2(bszs[0])) / (Math.log2(bszs[bszs.length-1]) - Math.log2(bszs[0])) * (W - M.l - M.r) * 0.92 + (W - M.l - M.r) * 0.04;
    // clip to the y-range analytically
    const bLo = Math.max(bszs[0] / 1.3, s0 * b0 / yHi), bHi = Math.min(bszs[bszs.length - 1] * 1.3, s0 * b0 / yLo);
    if (bHi > bLo) { el("line", {x1: xb(bLo), y1: y(s0 * b0 / bLo), x2: xb(bHi), y2: y(s0 * b0 / bHi), class: "perf"});
      el("text", {x: xb(bHi) - 4, y: y(s0 * b0 / bHi) - 6, "text-anchor": "end", class: "perf-lbl"}).textContent = "steps ∝ 1/batch"; }
  }

  // every run that reached the target
  if ($("allruns").checked) for (const c of all) {
    const g = el("g", {class: "hit", tabindex: 0});
    el("circle", {cx: x(c.run.bsz), cy: y(c.steps), r: 4.5, fill: rampColor(lrU(c.run.lr)), class: "dot"}, g);
    if (c.run.fixed) el("circle", {cx: x(c.run.bsz), cy: y(c.steps), r: 1.7, class: "fx"}, g);
    attachTip(g, c);
  }
  // headlines
  for (const kl of [0.01, 0.001]) {
    const series = best[kl]; if (series.length < 1) continue;
    const cls = kl === 0.001 ? "head3" : "head2", mk = kl === 0.001 ? "mk3" : "mk2";
    if (series.length > 1) el("path", {d: series.map((c, i) => (i ? "L" : "M") + x(c.run.bsz) + " " + y(c.steps)).join(" "), class: cls});
    for (const c of series) {
      const g = el("g", {class: "hit", tabindex: 0}), cx = x(c.run.bsz), cy = y(c.steps);
      if (kl === 0.001) el("circle", {cx, cy, r: 7, class: mk}, g);
      else el("rect", {x: cx - 6, y: cy - 6, width: 12, height: 12, transform: `rotate(45 ${cx} ${cy})`, class: mk}, g);
      if (c.run.fixed) el("circle", {cx, cy, r: 1.8, class: "fx"}, g);
      const lab = el("text", {x: cx + (kl === 0.001 ? -10 : 10), y: cy + (kl === 0.001 ? 16 : -10), "text-anchor": kl === 0.001 ? "end" : "start", class: "lab" + (kl === 0.01 ? " lab2" : "")}, g);
      lab.textContent = fmtSteps(c.steps);
      attachTip(g, c);
    }
  }
  // table + footnote
  const rows = [];
  for (const kl of [0.001, 0.01]) for (const c of best[kl]) rows.push(`<tr><td>${kl === 0.001 ? "1e-3" : "1e-2"}</td><td class="num">${c.run.bsz.toLocaleString()}</td><td class="num">${fmtLR(c.run.lr)}</td><td class="num">${fmtSteps(c.steps)}</td><td class="num">${(c.maxVal * 100).toFixed(1)}%</td><td class="num">${c.lastStep}</td><td>${c.run.fixed ? "fixed" : "pre-fix"}</td><td class="run">${c.run.name}</td></tr>`);
  $("tbl").innerHTML = `<thead><tr><th>KL</th><th class="num">batch</th><th class="num">LR</th><th class="num">steps to ${(t*100).toFixed(1)}%</th><th class="num">max val</th><th class="num">last val step</th><th>scaling</th><th>run</th></tr></thead><tbody>${rows.join("")}</tbody>`;
  $("foot").innerHTML = `<b>${all.length}</b> configurations reach ${(t * 100).toFixed(1)}%; <b>${never.length}</b> never do within their logged steps and are not drawn. Crossing: <b>${method === "interp" ? "interpolated" : "first validation ≥ target"}</b>.`;
}

function attachTip(g, c) {
  const show = ev => {
    const r = c.run;
    tip.innerHTML = `<b>${r.name}</b><br>batch ${r.bsz.toLocaleString()} · lr ${fmtLR(r.lr)} · KL ${r.kl === 0.001 ? "1e-3" : "1e-2"} · ${r.fixed ? "fixed scaling" : "pre-fix scaling"}<br>` +
      `steps to target: <b>${fmtSteps(c.steps)}</b> · max val ${(c.maxVal * 100).toFixed(1)}% · validated to step ${c.lastStep}`;
    tip.style.opacity = 1;
    const box = svg.parentElement.getBoundingClientRect(), p = ev.touches ? ev.touches[0] : ev;
    const px = (p ? p.clientX : box.left + box.width / 2) - box.left, py = (p ? p.clientY : box.top + 40) - box.top;
    tip.style.left = Math.min(px + 12, box.width - tip.offsetWidth - 8) + "px"; tip.style.top = Math.max(4, py - tip.offsetHeight - 10) + "px";
  };
  const hide = () => { tip.style.opacity = 0; };
  g.addEventListener("mousemove", show); g.addEventListener("mouseleave", hide);
  g.addEventListener("focus", show); g.addEventListener("blur", hide);
}

for (const id of ["target", "kl3", "kl2", "allruns", "perf"]) $(id).addEventListener("input", draw);
for (const r of document.querySelectorAll("input[name=method]")) r.addEventListener("change", draw);
try { const s = localStorage.getItem("svb-target"); if (s) $("target").value = s; } catch (e) {}
$("target").addEventListener("change", () => { try { localStorage.setItem("svb-target", $("target").value); } catch (e) {} });
if (window.matchMedia) window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", draw);
new MutationObserver(draw).observe(document.documentElement, {attributes: true, attributeFilter: ["data-theme"]});
draw();
</script>
"""

html = TEMPLATE.replace("__RUNS__", json.dumps(runs, separators=(",", ":")))
with open(out, "w") as f:
    f.write(html)
print(f"wrote {out}  ({len(runs)} runs, {os.path.getsize(out)/1024:.0f} KB)")
