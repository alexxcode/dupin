"use strict";

// ── Estado ──────────────────────────────────────────────────────────────────
const records = [];           // {tx, score, decisionApi, reasons, latency, isFraud, el}
let feed = [], idx = 0;
let running = true, speedMs = 220;
let tReview = 0.5, tBlock = 0.9999, deployedReview = null;
let timer = null;
const latencies = [], scoreTimes = [];
let selected = null;

const $ = (id) => document.getElementById(id);
const money = new Intl.NumberFormat("es", { maximumFractionDigits: 0 });
const COLORS = { approve: "#2ea043", review: "#d29922", block: "#f85149" };

// ── Init ─────────────────────────────────────────────────────────────────────
async function init() {
  try {
    const v = await (await fetch("/version")).json();
    $("modelV").textContent = v.model_version;
    $("featV").textContent = v.feature_version;
    if (v.thresholds) {
      tBlock = v.thresholds.block ?? tBlock;
      deployedReview = v.thresholds.review ?? null;
    }
  } catch (e) { setStatus(false); }

  try {
    feed = await (await fetch("/v1/demo-feed?limit=4000")).json();
  } catch (e) { setStatus(false); }

  updateThrLabel();
  start();
  setInterval(refreshThroughput, 500);
}

function setStatus(ok) {
  $("statusDot").classList.toggle("paused", !ok);
}

// ── Loop de scoring ───────────────────────────────────────────────────────────
function start() {
  clearInterval(timer);
  timer = setInterval(tick, speedMs);
}

async function tick() {
  if (!running || feed.length === 0) return;
  const tx = feed[idx % feed.length];
  idx++;
  try {
    const t0 = performance.now();
    const res = await fetch("/v1/score", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        step: tx.step, type: tx.type, amount: tx.amount,
        nameOrig: tx.nameOrig, nameDest: tx.nameDest,
      }),
    });
    if (!res.ok) { setStatus(false); return; }
    const r = await res.json();
    setStatus(true);
    const rec = {
      tx, score: r.score, decisionApi: r.decision, reasons: r.reasons,
      latency: r.latency_ms, isFraud: tx.isFraud === 1, el: null,
    };
    records.push(rec);
    latencies.push(r.latency_ms);
    scoreTimes.push(performance.now());
    renderRow(rec);
    updateKPIs();
    updateDonut();
  } catch (e) { setStatus(false); }
}

// ── Decisión cliente (umbral interactivo) ─────────────────────────────────────
function decisionOf(score) {
  if (score >= tBlock) return "block";
  if (score >= tReview) return "review";
  return "approve";
}

// ── Render del stream ─────────────────────────────────────────────────────────
function renderRow(rec) {
  const d = decisionOf(rec.score);
  const el = document.createElement("div");
  el.className = `row ${d}`;
  el.innerHTML = `
    <div class="step">t${rec.tx.step}</div>
    <div class="type">${rec.tx.type}</div>
    <div class="amt">$${money.format(rec.tx.amount)}</div>
    <div class="dest">→ ${rec.tx.nameDest}${rec.isFraud ? '<span class="gt fraud">⚑ fraude</span>' : ''}</div>
    <div class="meter"><i></i><span class="thrmark"></span></div>
    <div class="chip ${d}">${d}</div>`;
  rec.el = el;
  styleRow(rec);
  el.addEventListener("click", () => selectRow(rec));
  const stream = $("stream");
  stream.prepend(el);
  while (stream.children.length > 60) stream.lastChild.remove();
}

function styleRow(rec) {
  if (!rec.el) return;
  const d = decisionOf(rec.score);
  rec.el.className = `row ${d}${rec === selected ? " sel" : ""}`;
  const fill = rec.el.querySelector(".meter > i");
  fill.style.width = `${Math.max(2, rec.score * 100)}%`;
  fill.style.background = COLORS[d];
  rec.el.querySelector(".meter > .thrmark").style.left = `${tReview * 100}%`;
  const chip = rec.el.querySelector(".chip");
  chip.className = `chip ${d}`;
  chip.textContent = d;
}

function selectRow(rec) {
  selected = rec;
  records.forEach((r) => r.el && r.el.classList.toggle("sel", r === rec));
  renderReasons(rec);
}

function renderReasons(rec) {
  const gt = rec.isFraud
    ? '<span style="color:#f85149">fraude real</span>'
    : '<span style="color:#2ea043">legítimo</span>';
  const maxc = Math.max(1e-9, ...rec.reasons.map((x) => Math.abs(x.contribution ?? 0)));
  const items = rec.reasons.map((x) => {
    const c = x.contribution ?? 0;
    const w = Math.round((Math.abs(c) / maxc) * 100);
    const bar = x.contribution == null ? "" :
      `<div class="bar"><i class="${c < 0 ? "neg" : ""}" style="width:${w}%"></i></div>`;
    return `<div class="reason"><div class="msg">${x.message}</div>${bar}</div>`;
  }).join("");
  $("reasons").innerHTML =
    `<div class="tx-line">t${rec.tx.step} · ${rec.tx.type} · $${money.format(rec.tx.amount)}
       → ${rec.tx.nameDest}<br>score <b>${rec.score.toFixed(4)}</b> · ${gt}</div>${items}`;
}

// ── KPIs ──────────────────────────────────────────────────────────────────────
function updateKPIs() {
  $("count").textContent = records.length;
  let frauds = 0, flagged = 0, flaggedFraud = 0;
  for (const r of records) {
    const f = r.score >= tReview;
    if (r.isFraud) frauds++;
    if (f) flagged++;
    if (f && r.isFraud) flaggedFraud++;
  }
  $("kRecall").textContent = frauds ? `${(100 * flaggedFraud / frauds).toFixed(0)}%` : "—";
  $("kReview").textContent = records.length ? `${(100 * flagged / records.length).toFixed(1)}%` : "—";
  $("kPrecision").textContent = flagged ? `${(100 * flaggedFraud / flagged).toFixed(0)}%` : "—";
  const p = pcts(latencies);
  $("kP50").textContent = p.p50 != null ? p.p50.toFixed(1) : "—";
  $("kP99").textContent = p.p99 != null ? `p50 / p99 · ${p.p99.toFixed(1)} ms` : "p50 / p99 ms";
}

function refreshThroughput() {
  const now = performance.now();
  while (scoreTimes.length && now - scoreTimes[0] > 1000) scoreTimes.shift();
  $("kThroughput").textContent = scoreTimes.length;
}

function pcts(arr) {
  if (!arr.length) return { p50: null, p99: null };
  const s = [...arr].sort((a, b) => a - b);
  const at = (q) => s[Math.min(s.length - 1, Math.floor(q * s.length))];
  return { p50: at(0.5), p99: at(0.99) };
}

// ── Donut de decisiones ───────────────────────────────────────────────────────
function updateDonut() {
  const c = { approve: 0, review: 0, block: 0 };
  for (const r of records) c[decisionOf(r.score)]++;
  const total = records.length || 1;
  const R = 52, C = 2 * Math.PI * R;
  let off = 0;
  const arcs = ["approve", "review", "block"].map((k) => {
    const frac = c[k] / total;
    const seg = `<circle r="${R}" cx="70" cy="70" fill="none" stroke="${COLORS[k]}"
      stroke-width="18" stroke-dasharray="${frac * C} ${C}" stroke-dashoffset="${-off}"
      transform="rotate(-90 70 70)"/>`;
    off += frac * C;
    return seg;
  }).join("");
  $("donut").innerHTML = `<svg width="140" height="140" viewBox="0 0 140 140">
    <circle r="${R}" cx="70" cy="70" fill="none" stroke="#0a0e15" stroke-width="18"/>
    ${arcs}
    <text x="70" y="66" text-anchor="middle" fill="#e6edf3" font-size="22" font-family="monospace">${records.length}</text>
    <text x="70" y="84" text-anchor="middle" fill="#8b98a9" font-size="11">scored</text></svg>`;
  $("donutLegend").innerHTML = ["approve", "review", "block"].map((k) =>
    `<span><i style="background:${COLORS[k]}"></i>${k} ${c[k]}</span>`).join("");
}

// ── Controles ──────────────────────────────────────────────────────────────────
$("toggle").addEventListener("click", () => {
  running = !running;
  $("toggle").textContent = running ? "❚❚ Pausar" : "▶ Reanudar";
  $("statusDot").classList.toggle("paused", !running);
});
$("speed").addEventListener("input", (e) => {
  speedMs = 640 - Number(e.target.value);  // slider alto = más rápido
  start();
});
$("thr").addEventListener("input", (e) => {
  tReview = Number(e.target.value);
  updateThrLabel();
  records.forEach(styleRow);
  updateKPIs();
  updateDonut();
});

function updateThrLabel() {
  $("thrVal").textContent = tReview.toFixed(3);
  const dep = deployedReview != null ? ` · desplegado ${deployedReview.toFixed(3)}` : "";
  $("tradeoff").innerHTML =
    `block ≥ <b>${tBlock.toFixed(3)}</b>${dep}`;
}

init();
