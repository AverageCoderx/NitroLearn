// The Overview's training charts: every number the bot's training records (`bot_charts` sorts them into
// folders), "Your charts" pinned on top, folders that open and close, the Add charts list, each chart's menu,
// the ranges (all / last hour / last 10 minutes) and smoothing. Only the charts on show are asked for and
// drawn, and only while the Overview is open. The choices are kept in the dashboard's preferences
// (`charts`: pinned, open folders, smoothing), the same for every bot.

Object.assign(ICON, {
  pin: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M9 4h6l-1 6 4 4H6l4-4z"/><path d="M12 14v7"/></svg>',
  dots: '<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><circle cx="5" cy="12" r="2"/><circle cx="12" cy="12" r="2"/><circle cx="19" cy="12" r="2"/></svg>',
});
const SPANS = { hour: 3600, minutes: 600 };  // the ranges besides all of the training, in seconds of training
const YOURS = "yours";  // the pinned folder's id
// S.ch: { data: the last bot_charts answer, bot, span, menu: the chart id whose menu is open, query, seq }.
function chState() {
  if (!S.ch || S.ch.bot !== S.bot) S.ch = { bot: S.bot, data: null, menu: null, query: "", seq: 0, active: {} };
  return S.ch;
}
// The choices this window made until the dashboard has kept them: a Garage refresh asked for before that would
// bring the old ones back. Then the dashboard's, which another window may change too.
function chPrefs() {
  const own = S.chOwn;
  if (own && !(S.homeAsked > own.kept)) return own.prefs;
  const prefs = S.home && S.home.prefs && S.home.prefs.charts;
  return prefs && typeof prefs === "object" ? prefs : {};
}
function chSave(patch) {
  const prefs = Object.assign({}, chPrefs(), patch), own = { prefs, kept: Infinity };
  S.chOwn = own;
  call("set_pref", "charts", prefs).then(() => { own.kept = performance.now(); });
}
function chData() { return chState().data; }
function chById(id) { const d = chData(); return d ? d.charts.find(c => c.id === id) : null; }
function chFolder(id) { const d = chData(); return d ? d.folders.find(f => f.id === id) : null; }
// Your charts: the ones chosen (or the starting set), those this bot records, in their order.
function chPinned() {
  const d = chData(), chosen = chPrefs().pinned;
  const ids = Array.isArray(chosen) ? chosen : d ? d.default : [];
  return d ? ids.filter((id, i) => chById(id) && ids.indexOf(id) === i) : [];
}
function chIsPinned(id) {
  const chosen = chPrefs().pinned;
  return (Array.isArray(chosen) ? chosen : (chData() || {}).default || []).includes(id);
}
function chSetPinned(ids) { chSave({ pinned: ids }); }
function chTogglePin(id) {
  const chosen = chPrefs().pinned, ids = (Array.isArray(chosen) ? chosen : (chData() || {}).default || []).slice();
  const at = ids.indexOf(id);
  if (at >= 0) ids.splice(at, 1); else ids.push(id);
  chSetPinned(ids);
}
function chOpen() { const open = chPrefs().open; return new Set(Array.isArray(open) ? open : [YOURS]); }
function chSmooth() { return !!chPrefs().smooth; }
// The chart ids on show: Your charts (if open) and the open folders' charts.
function chShownIds() {
  const d = chData(), open = chOpen();
  if (!d) return [];
  const ids = open.has(YOURS) ? chPinned() : [];
  d.charts.forEach(c => { if (open.has(c.folder) && !ids.includes(c.id)) ids.push(c.id); });
  return ids;
}
function spanLoaded() { const d = chData(); return S.span !== "all" && d && d.span === S.span; }

// Asks for the charts on show (the first time for the list of charts only, then for their points).
async function loadCharts() {
  const st = chState(), bot = S.bot, span = S.span, seq = ++st.seq;
  if (!st.data) {
    const first = await call("bot_charts", bot, span, []);
    if (S.bot !== bot || first.error || seq !== st.seq) return;
    st.data = first;
  }
  const out = await call("bot_charts", bot, span, chShownIds());
  if (S.bot !== bot || S.span !== span || out.error || seq !== st.seq) return;
  st.data = out;
  fillTelemetry();
  fillPicker();
}

// ---------- drawing the page ----------
function chColour(folder) { const f = chFolder(folder); return `var(--c${f ? f.colour : 0})`; }
// A number in its chart's unit: "42%", "88 km/h", "3.1/min".
function chValue(v, unit) {
  if (v == null || !isFinite(v)) return "–";
  const text = fmt(v);
  if (!unit) return text;
  return unit === "%" || unit.startsWith("/") ? text + unit : text + " " + unit;
}
function chPoints(id) { const d = chData(); return (d && d.series && d.series[id]) || null; }
function chartCardHTML(c, place) {
  const pinned = chIsPinned(c.id), inYours = place === YOURS, f = chFolder(c.folder) || { title: "" };
  const key = `${place}-${c.id}`, open = chState().menu === key;
  return `<section class="chart" data-chart-id="${esc(c.id)}" data-place="${esc(place)}" style="--fc:${chColour(c.folder)}">
    <div class="head">
      <div class="who">${inYours ? `<span class="fchip">${esc(f.title)}</span>` : ""}<h3 title="${esc(c.about)}">${esc(c.title)}</h3></div>
      <div class="ch-tools">
        ${inYours ? "" : `<button type="button" class="iconbtn pin${pinned ? " on" : ""}" id="ch-pin-${esc(key)}" data-chpin="${esc(c.id)}" aria-pressed="${pinned}" title="${pinned ? "In Your charts: click to take it out" : "Add to Your charts"}" aria-label="${pinned ? "Remove " + esc(c.title) + " from Your charts" : "Add " + esc(c.title) + " to Your charts"}">${ICON.pin}</button>`}
        <button type="button" class="iconbtn" id="ch-menu-${esc(key)}" data-chmenu="${esc(key)}" aria-haspopup="menu" aria-expanded="${open}" aria-label="Chart menu: ${esc(c.title)}" title="More">${ICON.dots}</button>
        ${open ? chMenuHTML(c, place, key) : ""}
      </div>
    </div>
    <div class="now"><span class="last num" data-last>–</span><span class="note">${esc(c.about)}</span></div>
    <canvas data-chart="${esc(c.id)}" tabindex="0" role="img" aria-label="${esc(c.title)}"></canvas>
    <div class="axis"><span data-x0></span><span class="read" data-read aria-live="polite">agent steps</span><span data-x1></span></div>
  </section>`;
}
function chMenuHTML(c, place, key) {
  const item = (attrs, label, off) => `<button type="button" role="menuitem" ${attrs} ${off ? "disabled" : ""}>${label}</button>`;
  const pins = chPinned(), at = pins.indexOf(c.id), pinned = chIsPinned(c.id);
  return `<div class="ch-menu" role="menu" aria-label="${esc(c.title)}">
    ${item(`data-chpin="${esc(c.id)}"`, pinned ? "Remove from Your charts" : "Add to Your charts")}
    ${place === YOURS ? item(`data-chmove="${esc(c.id)}" data-by="-1"`, "Move earlier", at <= 0) + item(`data-chmove="${esc(c.id)}" data-by="1"`, "Move later", at < 0 || at >= pins.length - 1) + item(`data-chmove="${esc(c.id)}" data-by="-999"`, "Move to the start", at <= 0) : ""}
    ${item(`data-chabout="${esc(c.id)}"`, "Numbers and details…")}
  </div>`;
}
function folderHTML(id, title, about, charts, colour, place) {
  const open = chOpen().has(id), n = charts.length;
  const add = id === YOURS ? `<button type="button" class="btn small" data-act="ch-add">+ Add chart</button>` : "";
  const body = !open ? "" : n ? `<div class="charts">${charts.map(c => chartCardHTML(c, place)).join("")}</div>`
    : `<div class="ch-empty">No charts here yet. <button type="button" class="btn link" data-act="ch-add">Add charts</button> from any folder, or with the pin on a chart.</div>`;
  return `<section class="ch-folder${open ? "" : " closed"}" data-folder="${esc(id)}" style="--fc:${colour}">
    <div class="ch-head"><button type="button" class="ch-fold" id="ch-fold-${esc(id)}" data-chfold="${esc(id)}" aria-expanded="${open}">${ICON.chev}<span class="swatch" aria-hidden="true"></span><span class="name">${esc(title)}</span><span class="tag">${n} chart${n === 1 ? "" : "s"}</span></button><span class="about">${esc(about)}</span>${add}</div>
    ${body}
  </section>`;
}
function telemetryHTML() {
  const d = chData(), b = S.data;
  if (!d) return `<div class="heading"><h2 class="display">Telemetry</h2></div><div class="empty">Loading the charts…</div>`;
  if (!d.rows) {
    const waits = b.status === "training" || b.status === "starting";
    return `<div class="heading"><h2 class="display">Telemetry</h2></div><div class="empty"><b>${waits ? "The first numbers are on their way" : "Nothing to chart yet"}</b><br>${waits ? "The charts appear here a few minutes into training." : b.copy || b.copy_first ? "The training charts appear here once the bot trains." : `Once ${esc(b.name)} trains, its rewards, its speed, its play and how it learns are drawn here.`}</div>`;
  }
  const short = spanLoaded() && d.seconds < SPANS[S.span] * 0.98;
  const note = S.span === "all" ? "" : short ? `Only ${duration(d.seconds)} of training so far` : "Training time: a pause does not count";
  const seg = [["all", "All"], ["hour", "Last hour"], ["minutes", "Last 10 minutes"]].map(([v, label]) => `<button type="button" data-v="${v}" aria-pressed="${S.span === v}">${label}</button>`).join("");
  const smooth = [["0", "Raw"], ["1", "Smoothed"]].map(([v, label]) => `<button type="button" data-v="${v}" aria-pressed="${String(chSmooth()) === String(v === "1")}">${label}</button>`).join("");
  const yours = folderHTML(YOURS, "Your charts", "Pinned here from any folder", chPinned().map(chById), "var(--c0)", YOURS);
  const folders = d.folders.map(f => [f, d.charts.filter(c => c.folder === f.id)]).filter(([, cs]) => cs.length)
    .map(([f, cs]) => folderHTML(f.id, f.title, f.about, cs, chColour(f.id), f.id)).join("");
  return `<div class="heading"><h2 class="display">Telemetry</h2><div class="row ch-controls"><span class="muted" style="font-size:12.5px;">${note}</span>
      <div class="seg" data-seg="chsmooth" aria-label="Smoothing">${smooth}</div>
      <div class="seg" data-seg="span" aria-label="How much of the training the charts show">${seg}</div></div></div>
    <div class="ch-folders">${yours}${folders}</div>`;
}
// Writes the charts' frame when it changed, then the numbers and lines of every chart on show.
function fillTelemetry() {
  const box = $("#telemetry");
  if (!box || !S.data) return;
  let fresh = false;
  keepFocus(() => { fresh = put(box, telemetryHTML()); });
  box.querySelectorAll("section.chart").forEach(card => {
    const c = chById(card.dataset.chartId), pts = chPoints(card.dataset.chartId) || [];
    if (!c) return;
    const last = pts[pts.length - 1];
    card.querySelector("[data-last]").textContent = last ? chValue(last[1], c.unit) : "–";
    card.querySelector("[data-x0]").textContent = pts.length ? steps(pts[0][0]) : "";
    card.querySelector("[data-x1]").textContent = pts.length ? steps(last[0]) : "";
    const cv = card.querySelector("canvas");
    cv.setAttribute("aria-label", pts.length ? `${c.title} over agent steps, now ${chValue(last[1], c.unit)}. Arrow keys read earlier values.` : `${c.title}: no numbers yet`);
    const key = JSON.stringify([pts.length, pts[0], last, chSmooth(), S.theme]);
    if (fresh || cv.__key !== key) {
      cv.__key = key; drawSeries(cv, c);
      // The point read stays on its step: the new numbers may hold it at another place.
      const hover = S.chHover;
      if (hover && hover.cv === cv && pts.length) chMark(cv, chNearest(pts, hover.step));
    }
  });
}

// ---------- the lines ----------
// The line's points, smoothed when asked: an average of the points around each, wider for longer series.
function chLine(pts) {
  if (!chSmooth() || pts.length < 5) return pts;
  const k = Math.max(2, Math.round(pts.length / 25)), out = [];
  let sum = 0;
  const prefix = [0];
  pts.forEach(p => { sum += p[1]; prefix.push(sum); });
  for (let i = 0; i < pts.length; i++) {
    const a = Math.max(0, i - k), b = Math.min(pts.length, i + k + 1);
    out.push([pts[i][0], (prefix[b] - prefix[a]) / (b - a)]);
  }
  return out;
}
// Draws chart `c` on its canvas: three rules, its highest and lowest numbers at the left, the line in its
// folder's colour (the raw line faint under a smoothed one), a square on the newest point, and the point
// `mark` (an index) pointed at or chosen with the keys.
function drawSeries(cv, c, mark = null) {
  const dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight;
  if (!w || !h) return;
  cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
  const g = cv.getContext("2d"); g.scale(dpr, dpr);
  const css = getComputedStyle(cv.closest(".chart") || document.documentElement);
  const ink = name => css.getPropertyValue(name).trim();
  const asked = chPoints(c.id), raw = asked || [];
  g.strokeStyle = ink("--rule"); g.lineWidth = 1;
  for (let k = 0; k < 3; k++) { const y = Math.round(6 + k * (h - 12) / 2) + 0.5; g.beginPath(); g.moveTo(0, y); g.lineTo(w, y); g.stroke(); }
  if (!asked) return;  // on their way
  if (!raw.length) {
    g.fillStyle = ink("--dim"); g.font = `12px ${ink("--sans") || "sans-serif"}`; g.textAlign = "center";
    g.fillText("No numbers yet", w / 2, h / 2 + 4);
    return;
  }
  const line = chLine(raw), smooth = line !== raw;
  let lo = Infinity, hi = -Infinity;
  for (const [, y] of raw) { lo = Math.min(lo, y); hi = Math.max(hi, y); }
  const top = chValue(hi, c.unit), bottom = chValue(lo, c.unit), flat = hi - lo < 1e-12 * Math.max(1, Math.abs(hi));
  if (flat) { const room = Math.max(1, Math.abs(hi) * 0.1); hi += room; lo -= room; }
  const pad = (hi - lo) * 0.08; lo -= pad; hi += pad;
  g.font = `11px ${ink("--sans") || "sans-serif"}`;
  const gutter = Math.ceil(Math.max(g.measureText(top).width, g.measureText(bottom).width)) + 8;
  const x0 = raw[0][0], x1 = raw[raw.length - 1][0];
  cv.__gutter = gutter;
  const X = x => gutter + (x1 > x0 ? (x - x0) / (x1 - x0) : 0.5) * (w - gutter - 6);
  const Y = y => h - 6 - (y - lo) / (hi - lo) * (h - 12);
  // The highest and lowest numbers at their heights (one, for a flat line).
  g.fillStyle = ink("--dim"); g.textAlign = "left";
  const label = (text, y) => g.fillText(text, 0, Math.max(10, Math.min(h - 2, y + 4)));
  if (flat) label(top, Y((hi + lo) / 2)); else { label(top, Y(hi - pad)); label(bottom, Y(lo + pad)); }
  const colour = ink("--line") || ink("--text");
  const stroke = (pts, width, alpha) => {
    g.beginPath();
    pts.forEach(([x, y], i) => { if (i === 0) g.moveTo(X(x), Y(y)); else g.lineTo(X(x), Y(y)); });
    g.globalAlpha = alpha; g.strokeStyle = colour; g.lineWidth = width; g.lineJoin = "round"; g.lineCap = "round"; g.stroke(); g.globalAlpha = 1;
  };
  const marked = mark != null && line[mark] ? line[mark] : null;
  if (marked) {
    const x = Math.round(X(marked[0])) + 0.5;
    g.strokeStyle = ink("--dim"); g.lineWidth = 1;
    g.beginPath(); g.moveTo(x, 0); g.lineTo(x, h); g.stroke();
  }
  if (smooth) stroke(raw, 1, 0.3);
  if (line.length === 1) { g.fillStyle = colour; g.fillRect(X(line[0][0]) - 3.5, Y(line[0][1]) - 3.5, 7, 7); }
  else stroke(line, 2, 1);
  const [lx, ly] = line[line.length - 1];
  g.fillStyle = colour; g.fillRect(X(lx) - 3.5, Y(ly) - 3.5, 7, 7);
  if (marked) {
    g.fillStyle = ink("--panel"); g.fillRect(X(marked[0]) - 6, Y(marked[1]) - 6, 12, 12);
    g.fillStyle = colour; g.fillRect(X(marked[0]) - 4, Y(marked[1]) - 4, 8, 8);
  }
}

// ---------- reading a chart: the pointer or the arrow keys ----------
function chMark(cv, index) {
  const c = chById(cv.dataset.chart), raw = chPoints(cv.dataset.chart) || [];
  if (!c || !raw.length) return;
  index = Math.max(0, Math.min(raw.length - 1, index));
  const line = chLine(raw), read = cv.parentElement.querySelector("[data-read]");
  if (S.chHover && S.chHover.cv !== cv) chUnmark();
  S.chHover = { cv, index, step: raw[index][0] };
  const smoothed = line !== raw ? ` (smoothed ${chValue(line[index][1], c.unit)})` : "";
  if (read) { read.textContent = `${chValue(raw[index][1], c.unit)}${smoothed} at ${steps(raw[index][0])} steps`; read.classList.add("on"); }
  drawSeries(cv, c, index);
}
function chUnmark() {
  const hover = S.chHover;
  S.chHover = null;
  if (!hover || !hover.cv.isConnected) return;
  const read = hover.cv.parentElement.querySelector("[data-read]"), c = chById(hover.cv.dataset.chart);
  if (read) { read.textContent = "agent steps"; read.classList.remove("on"); }
  if (c) drawSeries(hover.cv, c);
}
function chNearest(raw, step) {
  let best = 0;
  for (let i = 1; i < raw.length; i++) if (Math.abs(raw[i][0] - step) < Math.abs(raw[best][0] - step)) best = i;
  return best;
}
function chIndexAt(cv, clientX) {
  const raw = chPoints(cv.dataset.chart) || [];
  if (!raw.length) return null;
  const box = cv.getBoundingClientRect(), x0 = raw[0][0], x1 = raw[raw.length - 1][0];
  return chNearest(raw, x0 + Math.min(1, Math.max(0, (clientX - box.left - (cv.__gutter || 0)) / Math.max(1, box.width - (cv.__gutter || 0) - 6))) * (x1 - x0));
}
document.addEventListener("mousemove", e => {
  const cv = e.target.closest ? e.target.closest("canvas[data-chart]") : null;
  if (cv) { const i = chIndexAt(cv, e.clientX); if (i != null && !(S.chHover && S.chHover.cv === cv && S.chHover.index === i)) chMark(cv, i); }
  else if (S.chHover && document.activeElement !== S.chHover.cv) chUnmark();
});
document.addEventListener("keydown", e => {
  const cv = e.target.closest && e.target.closest("canvas[data-chart]");
  if (cv) {
    const n = (chPoints(cv.dataset.chart) || []).length, at = S.chHover && S.chHover.cv === cv ? S.chHover.index : n;
    const to = { ArrowLeft: at - 1, ArrowRight: at + 1, Home: 0, End: n - 1 }[e.key];
    if (to != null && n) { e.preventDefault(); chMark(cv, to); }
    else if (e.key === "Escape") chUnmark();
    return;
  }
  if (e.key === "Escape" && chState().menu && $("#modal").hidden) { const key = chState().menu; chState().menu = null; fillTelemetry(); const b = document.getElementById("ch-menu-" + key); if (b) b.focus(); }
});
document.addEventListener("focusout", e => { if (S.chHover && e.target === S.chHover.cv) chUnmark(); });
document.addEventListener("click", e => {  // a click anywhere else closes a chart's menu
  if (S.ch && S.ch.menu && !e.target.closest(".ch-menu, [data-chmenu]")) { S.ch.menu = null; fillTelemetry(); }
});
window.addEventListener("resize", () => {
  if (S.view !== "bot" || S.tab !== "overview" || !S.data || S.resizing) return;
  S.resizing = requestAnimationFrame(() => {
    S.resizing = 0; chUnmark();
    document.querySelectorAll("#telemetry canvas[data-chart]").forEach(cv => { const c = chById(cv.dataset.chart); if (c) drawSeries(cv, c); });
    drawCopyCharts();
  });
});

// ---------- choosing charts ----------
on("click", t => {
  if (t.dataset.chfold) {
    const open = chOpen(), id = t.dataset.chfold;
    if (open.has(id)) open.delete(id); else open.add(id);
    chSave({ open: [...open] }); fillTelemetry(); loadCharts(); return;
  }
  if (t.dataset.chpin) {
    chTogglePin(t.dataset.chpin); chState().menu = null;
    fillTelemetry(); loadCharts(); if ($("#ch-picker")) fillPicker(); return;
  }
  if (t.dataset.chmenu) { const st = chState(); st.menu = st.menu === t.dataset.chmenu ? null : t.dataset.chmenu; fillTelemetry(); const first = $(".ch-menu [role=menuitem]:not([disabled])"); if (first) first.focus(); return; }
  if (t.dataset.chmove) {
    // Moved among this bot's charts; the ones pinned on another bot's keep their places.
    const shown = chPinned(), at = shown.indexOf(t.dataset.chmove), by = Number(t.dataset.by);
    if (at < 0) return;
    const chosen = chPrefs().pinned, all = Array.isArray(chosen) ? chosen.filter((id, i) => chosen.indexOf(id) === i) : shown.slice();
    const moved = shown.slice();
    moved.splice(at, 1); moved.splice(Math.max(0, Math.min(moved.length, at + by)), 0, t.dataset.chmove);
    let k = 0;
    chSetPinned(all.map(id => shown.includes(id) ? moved[k++] : id)); fillTelemetry();
    const b = document.getElementById("ch-menu-" + chState().menu); if (b) b.focus();
    return;
  }
  if (t.dataset.chabout) { const id = t.dataset.chabout; chState().menu = null; fillTelemetry(); return chDetails(id); }
  if (t.dataset.chdefault != null) { chSetPinned(null); fillPicker(); fillTelemetry(); loadCharts(); return; }  // each bot's own
  return SKIP;
});
SEGS.chsmooth = value => { chSave({ smooth: value === "1" }); S.chHover = null; fillTelemetry(); };
SEGS.span = value => { S.span = value; S.chHover = null; loadCharts(); };

// The Add charts list: every chart by folder, searchable; a click adds it to Your charts or takes it out.
ACTIONS["ch-add"] = () => {
  chState().query = "";
  modal(`<h2 class="display">Add charts</h2>
    <div class="muted" style="font-size:13px;">Pick the charts to keep in <b>Your charts</b> at the top of the Overview. Click one again to take it out. Your choice is kept for every bot.</div>
    <label class="search"><span class="sr">Search charts</span>${ICON.search}<input id="ch-search" class="field" type="search" placeholder="Search charts" autocomplete="off"></label>
    <div id="ch-picker" class="ch-picker"></div>
    <div class="foot"><button type="button" class="btn ghost" data-chdefault="1">Back to the starting set</button><button type="button" class="btn primary" data-close="1">Done</button></div>`);
  $(".modal").classList.add("wide");
  fillPicker();
  $("#ch-search").focus();
};
function fillPicker() {
  const box = $("#ch-picker"), d = chData();
  if (!box) return;
  if (!d) return put(box, `<div class="empty">Loading the charts…</div>`);
  if (!d.charts.length) return put(box, `<div class="empty">The charts can be chosen once the bot has trained: they are made from what its training records.</div>`);
  const q = chState().query.trim().toLowerCase();
  const hit = (c, f) => !q || [c.title, c.about, c.id, f.title].some(s => String(s).toLowerCase().includes(q));
  const groups = d.folders.map(f => [f, d.charts.filter(c => c.folder === f.id && hit(c, f))]).filter(([, cs]) => cs.length);
  keepFocus(() => put(box, groups.length ? groups.map(([f, cs]) => `<section class="ch-pick-group" style="--fc:var(--c${f.colour})" aria-label="${esc(f.title)}">
      <div class="ch-pick-head"><span class="swatch" aria-hidden="true"></span>${esc(f.title)}<span class="muted">${cs.length}</span></div>
      ${cs.map(c => { const on = chIsPinned(c.id); return `<button type="button" class="ch-pick${on ? " on" : ""}" id="ch-pick-${esc(c.id)}" data-chpin="${esc(c.id)}" aria-pressed="${on}"><span class="box" aria-hidden="true"></span><span class="t"><b>${esc(c.title)}</b><span class="muted">${esc(c.about)}</span></span></button>`; }).join("")}
    </section>`).join("") : `<div class="empty">No chart matches “${esc(q)}”.</div>`));
}
on("input", el => { if (el.id === "ch-search") { chState().query = el.value; fillPicker(); } });

// A chart's numbers and details: what it shows, the nerd explanation, and its points as a table.
function chDetails(id) {
  const c = chById(id);
  if (!c) return;
  const pts = (chPoints(id) || []).slice().reverse(), f = chFolder(c.folder) || { title: "" };
  const rows = pts.slice(0, 200).map(([x, y]) => `<tr><td class="mono">${num(x)}</td><td class="mono">${esc(chValue(y, c.unit))}</td></tr>`).join("");
  modal(`<h2 class="display">${esc(c.title)}</h2>
    <div style="font-size:13.5px;">${esc(c.about)} <span class="muted">Folder: ${esc(f.title)}.</span></div>
    <details class="nerd"><summary class="btn link">Nerd explanation</summary><div class="guide" style="max-width:none;"><p>${codeMarks(esc(c.nerd || c.id))}</p><p>Column <code class="mono">${esc(c.id.startsWith("share_") ? "paid_abs_" + c.id.slice(6) : c.id)}</code> of the bot's metrics.csv. Each point is the average of the rounds in its stretch of agent steps (${pts.length} points${spanLoaded() ? ", " + (S.span === "hour" ? "last hour" : "last 10 minutes") : ""}).</p></div></details>
    ${rows ? `<div class="ch-table"><table><thead><tr><th>Agent steps</th><th>${esc(c.title)}</th></tr></thead><tbody>${rows}</tbody></table></div>` : `<div class="muted">No numbers yet.</div>`}
    <div class="foot"><button type="button" class="btn" data-close="1">Close</button></div>`);
}
