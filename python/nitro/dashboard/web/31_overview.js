// The Overview tab: the value warm-up, the progress of a copy of replays, the training charts (31_charts.js),
// and the tools beside them.

// The value warm-up of a bot started from replays: how far it is, and its number to change.
function warmupHTML(d) {
  const w = d.warmup;
  if (!w) return "";
  const item = allSections(d.settings || []).flatMap(s => s.items).find(i => i.path === w.path);
  const n = x => Number(x || 0).toLocaleString("en-US");
  const share = w.steps ? Math.min(1, w.done / w.steps) : 1, done = w.done >= w.steps;
  const sps = Number(d.steps_per_second || 0);
  const left = !done && sps > 0 && d.status === "training" ? ` About ${duration((w.steps - w.done) / sps)} left at the current speed.` : "";
  const doing = done ? "Done: the policy learns too now." : `Only the value network learns; the policy is held still as the copy of the replays left it, so the value network learns what situations are worth before it steers the policy.${left}`;
  return `<section class="card pad" style="display:flex;flex-direction:column;gap:12px;margin-bottom:14px;" aria-label="Value warm-up">
    <div class="row"><h3 class="grow">Value warm-up</h3><span class="pill ${done ? "" : "training"}">${done ? "Done" : "Policy held still"}</span></div>
    <div><div class="row" style="justify-content:space-between;font-size:13px;margin-bottom:6px;"><span>${n(w.done)} of ${n(w.steps)} steps</span><span class="mono">${(100 * share).toFixed(1)}%</span></div>
      <div class="bar" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${Math.round(100 * share)}"><span style="width:${(100 * share).toFixed(1)}%"></span></div>
      <div class="muted" style="font-size:13px;margin-top:6px;">${esc(doing)}</div></div>
    ${item ? settingHTML(item) : ""}
  </section>`;
}

function drawCopyCharts() {
  const cc = copyCharts(S.data.copy);
  document.querySelectorAll("canvas[data-cpchart]").forEach(cv => { if (cc[+cv.dataset.cpchart]) drawChart(cv, cc[+cv.dataset.cpchart], +cv.dataset.cpchart); });
}
// The tab's frame: the warm-up and copy blocks and the charts (31_charts.js) on the left, the tools on the
// right (below them in a narrow window). Each part is written again only when it changed.
function renderOverview(box) {
  box.innerHTML = `<div class="overview"><div class="ov-main"><div id="ov-blocks"></div><section id="telemetry" aria-label="Training charts"></section></div><aside id="ov-tools" aria-label="Tools"></aside></div>`;
  fillOverview();
  loadCharts();
}
function fillOverview() {
  if (!$("#telemetry") || !S.data) return;
  let blocks = false;
  keepFocus(() => { blocks = put($("#ov-blocks"), warmupHTML(S.data) + copyProgressHTML(S.data)); });
  fillTelemetry();
  put($("#ov-tools"), toolsHTML());
  if (blocks) requestAnimationFrame(drawCopyCharts);
}

// ---------- learning from replays: progress and reports ----------
function copyCharts(c) {
  if (!c) return [];
  const rounds = (c.rounds || []).filter(r => r.held_agreement != null);
  const versions = ((c.skill || {}).versions || []);
  return [
    { title: "Chooses as the players did", unit: "%", minSpan: 5, points: rounds.map(r => [r.replays_in_all, 100 * r.held_agreement]), help: "On replays it never learns from: how often its first choice is the player's." },
    { title: "Skill against its earlier versions", unit: "", minSpan: 4, points: versions.map(v => [v.replays, v.mu]), help: "A rating from goals against where it started (25) and its last check's version: higher is better at the game." },
  ].filter(ch => ch.points.length >= 1).map(ch => ch.points.length === 1 ? Object.assign(ch, { points: [ch.points[0], ch.points[0]] }) : ch);
}
function copyProgressHTML(d) {
  const c = d.copy;
  if (!c && !d.copy_first) return "";
  const n = x => Number(x || 0).toLocaleString("en-US");
  if (!c) return `<section class="card pad" style="display:flex;flex-direction:column;gap:8px;margin-bottom:14px;"><h3>Learning from replays</h3><div class="muted" style="font-size:13px;">Not started yet. <b>Copy replays</b> (above) converts the replays and copies them into the bot, then saves and stops. Its settings are on the Settings tab (Learning from replays); the progress shows here.</div></section>`;
  const copying = d.status === "copying", done = !!c.finished;
  const share = c.limit ? Math.min(1, (c.learned || 0) / c.limit) : 0;
  const pill = done ? "Done" : copying ? "Copying now" : "Stopped";
  const left = c.seconds_left != null && copying ? (c.seconds_left < 60 ? "under a minute" : "about " + duration(c.seconds_left)) : "–";
  const doneAt = c.seconds_left != null && copying ? "Done about " + new Date(Date.now() + c.seconds_left * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : "";
  const rounds = c.rounds || [], last = [...rounds].reverse().find(r => r.held_agreement != null);
  const skill = c.skill || {}, versions = skill.versions || [], newest = versions[versions.length - 1];
  const saved = d.copy_so_far && !done ? `Saved part-way at ${n(d.copy_so_far.learned)} replays: Watch it play, Make RLBot bot and Try these rewards (once it is stopped) use that save.${c.save_every ? ` It saves again every ${n(c.save_every)} replays.` : ""}`
    : c.save_every && !done ? `The copy so far is saved as the bot's save every ${n(c.save_every)} replays, so it can be watched or have rewards tried on it part-way.` : "";
  const doing = done ? `Done: ${n(c.learned)} replays copied${c.learned < c.limit ? ` (all the folders had of the ${c.games || "chosen"} size)` : ""}. Try rewards on it (Rewards tab), then Start training from the copy.`
    : copying ? `Now ${esc(c.doing || "working")}. Stopping it (Stop and save) or turning the PC off is safe: Copy replays carries on from its last finished round.`
    : `Stopped after ${n(c.learned)} of ${n(c.limit)} replays. Copy replays carries on from its last finished round.`;
  const stat = (k, v, title, sub) => `<div class="stat"${title ? ` title="${esc(title)}"` : ""}><div class="k">${k}</div><div class="v display">${v}</div>${sub ? `<div class="muted" style="font-size:12px;">${esc(sub)}</div>` : ""}</div>`;
  const cc = copyCharts(c);
  const matches = (skill.matches || []).slice(-6).reverse().map(m => {
    const goals = m.won + m.lost;
    const res = goals ? `took ${Math.round(100 * m.won / goals)}% of the ${goals} goal${goals === 1 ? "" : "s"}${m.low != null ? ` (sure range ${Math.round(100 * m.low)}–${Math.round(100 * m.high)}%)` : ""}` : "nobody scored";
    return `<li>After round ${m.round}, against ${versions.length && m.against === versions[0].round ? "where it started" : "round " + m.against}: ${res} in ${n(m.points)} points.</li>`;
  }).join("");
  return `<section class="card pad" style="display:flex;flex-direction:column;gap:12px;margin-bottom:14px;" aria-label="Learning from replays">
    <div class="row"><h3 class="grow">Learning from replays</h3><span class="pill ${copying ? "copying" : done ? "training" : ""}">${pill}</span></div>
    <div><div class="row" style="justify-content:space-between;font-size:13px;margin-bottom:6px;"><span>${n(c.learned)} of ${n(c.limit)} ${esc(c.games || "")} replays learned</span><span class="mono">${(100 * share).toFixed(1)}%</span></div>
      <div class="bar" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${Math.round(100 * share)}"><span style="width:${(100 * share).toFixed(1)}%"></span></div>
      <div class="muted" style="font-size:13px;margin-top:6px;">${doing}</div>${saved ? `<div class="muted" style="font-size:13px;margin-top:4px;">${esc(saved)}</div>` : ""}</div>
    <div class="cp-stats">
      ${stat("Time left", left, "", doneAt)}
      ${stat("Replays a minute", copying && c.per_minute ? num(c.per_minute, 0) : "–", "Replays converted and made ready a minute this start")}
      ${stat("Rounds", n(rounds.length), "Learned a batch of replays at a time")}
      ${stat("Chooses as the players did", last ? (100 * last.held_agreement).toFixed(1) + "%" : "–", `On ${n(c.held_back)} replays it never learns from: how often its first choice is the player's`)}
      ${stat("Skill rating", newest && versions.length > 1 ? newest.mu.toFixed(1) : "–", "Against its earlier versions; where the copy started is 25")}
      ${c.failed ? stat("Could not be used", n(c.failed), "Replays the converter could not read") : ""}
    </div>
    ${cc.length ? `<div class="cp-charts">${cc.map((ch, i) => `<div><div class="row" style="justify-content:space-between;font-size:13px;"><b>${esc(ch.title)}</b><span class="mono">${fmt(ch.points[ch.points.length - 1][1])}${ch.unit}</span></div><canvas data-cpchart="${i}" role="img" aria-label="${esc(ch.title)} over replays learned"></canvas><div class="muted" style="font-size:12px;display:flex;justify-content:space-between;"><span>${n(ch.points[0][0])}</span><span>replays learned</span><span>${n(ch.points[ch.points.length - 1][0])}</span></div><div class="muted" style="font-size:12px;">${esc(ch.help)}</div></div>`).join("")}</div>` : ""}
    ${matches ? `<div><b style="font-size:13px;">Skill checks</b><ul class="muted" style="font-size:13px;margin:6px 0 0;padding-left:18px;">${matches}</ul></div>` : c.skill_every ? `<div class="muted" style="font-size:13px;">A skill check plays every ${c.skill_every} rounds; the first shows here after round ${c.skill_every}.</div>` : ""}
  </section>`;
}

// A chart's line takes one of these colours of the theme in turn (the two team colours are chart colours too).
const CHART_COLOURS = ["--text", "--blue", "--good", "--orange"];
// Draws `chart` ({points: [[x, y], ...], minSpan}) on the canvas `cv`: a flat line on three rules, a square
// on its newest point. `index` picks its colour; `mark` is the point pointed at, if any.
function drawChart(cv, chart, index = 0, mark = null) {
  const dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight;
  cv.width = w * dpr; cv.height = h * dpr;
  const g = cv.getContext("2d"); g.scale(dpr, dpr);
  const css = getComputedStyle(document.documentElement);
  const pts = chart.points;
  let lo = Infinity, hi = -Infinity, x0 = pts[0][0], x1 = pts[pts.length - 1][0];
  for (const [, y] of pts) { lo = Math.min(lo, y); hi = Math.max(hi, y); }
  if (hi - lo < 1e-12) { hi += 1; lo -= 1; }
  if (chart.minSpan && hi - lo < chart.minSpan) { const mid = (hi + lo) / 2; hi = mid + chart.minSpan / 2; lo = mid - chart.minSpan / 2; }
  const pad = (hi - lo) * 0.12; lo -= pad; hi += pad;
  const X = x => 4 + (x1 > x0 ? (x - x0) / (x1 - x0) : 0.5) * (w - 8);
  const Y = y => h - 6 - (y - lo) / (hi - lo) * (h - 12);
  g.strokeStyle = css.getPropertyValue("--rule").trim(); g.lineWidth = 1;
  for (let k = 1; k < 4; k++) { const y = Math.round(h * k / 4) + 0.5; g.beginPath(); g.moveTo(0, y); g.lineTo(w, y); g.stroke(); }
  const colour = css.getPropertyValue(CHART_COLOURS[index % CHART_COLOURS.length]).trim();
  const marked = mark != null && pts[mark] ? pts[mark] : null;
  if (marked) {
    const x = Math.round(X(marked[0])) + 0.5;
    g.strokeStyle = css.getPropertyValue("--dim").trim();
    g.beginPath(); g.moveTo(x, 0); g.lineTo(x, h); g.stroke();
  }
  g.beginPath();
  pts.forEach(([x, y], i) => { if (i === 0) g.moveTo(X(x), Y(y)); else g.lineTo(X(x), Y(y)); });
  g.strokeStyle = colour; g.lineWidth = 2.2; g.lineJoin = "bevel"; g.lineCap = "butt"; g.stroke();
  const [lx, ly] = pts[pts.length - 1];
  g.fillStyle = colour; g.fillRect(X(lx) - 3.5, Y(ly) - 3.5, 7, 7);
  if (marked) {
    g.fillStyle = css.getPropertyValue("--bg").trim(); g.fillRect(X(marked[0]) - 6.5, Y(marked[1]) - 6.5, 13, 13);
    g.fillStyle = colour; g.fillRect(X(marked[0]) - 4.5, Y(marked[1]) - 4.5, 9, 9);
  }
}
TAB_VIEWS.overview = renderOverview;
// Each part is written again only when it changed, and never under an open popup.
TAB_REFRESH.overview = () => {
  if ($("#modal").hidden === false) return;
  fillOverview();
  loadCharts();
};
