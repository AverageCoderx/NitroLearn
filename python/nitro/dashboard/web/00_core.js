// The page's script is these parts joined in name order into one script: they share their names. This part
// holds what every other one uses: the bridge to Python, small helpers, icons, the page's state, and the
// registries each part adds its pages, tabs, actions and event handlers to.
"use strict";
const BOOT = /*BOOT*/null;

// ---------- the bridge to Python (pywebview; tests put their own in window.__testApi) ----------
function bridge() { return (window.pywebview && window.pywebview.api) || window.__testApi || null; }
function ready() {
  return new Promise(resolve => {
    if (bridge()) return resolve();
    window.addEventListener("pywebviewready", () => resolve(), { once: true });
    const timer = setInterval(() => { if (bridge()) { clearInterval(timer); resolve(); } }, 50);
  });
}
async function call(method, ...args) {
  await ready();
  try {
    const out = await bridge()[method](...args);
    return out || {};
  } catch (err) {
    return { error: String(err && err.message || err) };
  }
}

// ---------- small helpers ----------
const $ = (s, root = document) => root.querySelector(s);
const esc = v => String(v == null ? "" : v).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
// Writes `html` into `el` only when it differs from what was written there last: a redraw that changes
// nothing never takes a click, the focus or a selection.
function put(el, html) {
  if (!el || el.__html === html) return false;
  el.__html = html; el.innerHTML = html; return true;
}
// Runs `draw` and gives the focused field back its focus, its cursor and any text typed but not yet sent.
function keepFocus(draw) {
  const a = document.activeElement, id = a && a.id, typing = a && (a.tagName === "INPUT" || a.tagName === "TEXTAREA");
  const value = typing ? a.value : null, start = typing ? a.selectionStart : null, end = typing ? a.selectionEnd : null;
  draw();
  const el = id && document.getElementById(id);
  if (!el || el === a) return;
  if (typing && el.value !== value) el.value = value;
  el.focus();
  try { if (start != null) el.setSelectionRange(start, end); } catch (err) { /* not a text field */ }
}
function steps(n) {
  n = Number(n || 0);
  if (n >= 1e9) return (n / 1e9).toFixed(2) + "B";
  if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + "K";
  return String(Math.round(n));
}
const num = (v, d = 0) => v == null ? "–" : Number(v).toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: d });
function ago(t) {
  if (!t) return "never";
  const s = Math.max(0, Date.now() / 1000 - t);
  if (s < 90) return "just now";
  if (s < 3600) return Math.round(s / 60) + " min ago";
  if (s < 86400 * 2) return Math.round(s / 3600) + " h ago";
  return Math.round(s / 86400) + " days ago";
}
function duration(s) {
  s = Math.max(0, Math.round(s || 0));
  const h = Math.floor(s / 3600), m = Math.floor(s % 3600 / 60);
  return h ? h + " h " + m + " min" : m + " min";
}
const STATUS = { training: "Training", stopped: "Stopped", starting: "Starting", copying: "Learning from replays", problem: "Settings problem" };
function statusWord(b) {
  if (b.stopping) return "Stopping";
  if (b.checking) return "Trying rewards";
  if (b.status === "stopped" && copiedOnly(b)) return "Replays copied";
  if (b.status === "stopped" && b.copy_so_far) return "Copied part-way";
  return b.status === "stopped" && !b.saves ? "Not started" : (STATUS[b.status] || b.status);
}
// Whether training on this PC runs on the processor alone (no graphics card it can use); false when unknown.
function noCard() { const m = S.home && S.home.machine; return !!m && m.graphics_card === false; }
// A bot training whose first round of training since its start has not ended: its numbers are not in yet.
function firstRoundWord() { return "Waiting for its first round of training" + (noCard() ? " (it can take several minutes on the processor)" : ""); }
// What a bot that was asked to stop does until its trainer has ended.
const STOPPING = "Stopping – saving after this step";
function copiedOnly(b) { return b.copied_replays && b.saves && !b.agent_steps && !b.copy_so_far; }
// What the start button does, in its words (bots.start_state): "Start from my-bot (save at 250.0M steps)", "Start
// training from scratch", "Carry on training", "Copy replays".
function startWord(b) { return b.start_word || "Start training"; }
function statusLine(b) {
  if (b.stopping) return STOPPING;
  if (b.applying) return b.applying;
  if (b.status === "training") return "Training · " + b.team_size + "v" + b.team_size;
  if (b.status === "copying") return "Learning from replays";
  if (b.status === "starting") return "Starting";
  if (b.status === "problem") return "Settings problem";
  if (b.checking) return "Trying rewards";
  return (copiedOnly(b) ? "Replays copied" : b.copy_so_far ? "Copied part-way" : b.saves ? "Stopped" : "Not started") + (b.team_size ? " · " + b.team_size + "v" + b.team_size : "");
}

const ICON = {
  note: '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 20h4L19 9l-4-4L4 16z"/><path d="m13 7 4 4"/></svg>',
  reset: '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 12a9 9 0 1 0 3-6.7"/><path d="M3 4v5h5"/></svg>',
  lock: '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><rect x="5" y="11" width="14" height="10" rx="2"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/></svg>',
  chev: '<svg class="chev" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m9 6 6 6-6 6"/></svg>',
  search: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>',
  eyeOff: '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M2 12s4-7 10-7c2 0 3.8.8 5.3 1.8M22 12s-4 7-10 7c-2 0-3.8-.8-5.3-1.8"/><path d="M3 3l18 18"/></svg>',
  eye: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/></svg>',
  play: '<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M7 4v16l13-8z"/></svg>',
  stop: '<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><rect x="5" y="5" width="14" height="14" rx="2"/></svg>',
  trash: '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 7h16"/><path d="M10 11v6M14 11v6"/><path d="M6 7l1 13h10l1-13"/><path d="M9 7V4h6v3"/></svg>',
  sun: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>',
  moon: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z"/></svg>',
};

function fmt(v) {
  const a = Math.abs(v);
  if (a >= 1000) return num(v, 0);
  if (a >= 10) return num(v, 1);
  if (a >= 0.1) return num(v, 3);
  return Number(v).toPrecision(3);
}

function allSections(list) { return list.flatMap(s => [s, ...allSections(s.sections || [])]); }

function show(v) {
  if (v == null) return "";
  if (Array.isArray(v)) return v.join(", ");
  return String(v);
}

// Escaped text with its `names` (a setting's name in a text) shown as code, never as raw backticks.
function codeMarks(html) { return String(html).replace(/`([^`\n]+)`/g, '<code class="mono">$1</code>'); }
// A "See more" text: paragraphs, each opening with its label ("What it does:"), and "- " lists.
function guideHTML(text) {
  return String(text).trim().split(/\n\s*\n/).map(block => {
    const lines = block.split("\n").map(l => l.trim()).filter(Boolean);
    const list = lines.filter(l => l.startsWith("- "));
    const prose = lines.filter(l => !l.startsWith("- ")).join(" ");
    const m = /^([A-Z][A-Za-z ,'-]{2,40}):\s*(.*)$/.exec(prose);
    const head = m ? `<b>${esc(m[1])}:</b> ${codeMarks(esc(m[2]))}` : codeMarks(esc(prose));
    return (prose ? `<p>${head}</p>` : "") + (list.length ? `<ul>${list.map(l => `<li>${codeMarks(esc(l.slice(2)))}</li>`).join("")}</ul>` : "");
  }).join("");
}

function canon(v) {
  if (Array.isArray(v)) return v.map(canon);
  if (v && typeof v === "object") { const o = {}; Object.keys(v).sort().forEach(k => { if (k !== "known") o[k] = canon(v[k]); }); return o; }
  if (typeof v === "string") { const n = Number(v.replace(/[_,\s]/g, "")); return v.trim() !== "" && !isNaN(n) ? n : v; }
  return v;
}

function pct(x) { return x == null ? "–" : (x * 100 >= 10 && x * 100 < 99.5 ? Math.round(x * 100) : x >= 1 ? 100 : (x * 100).toFixed(1)) + "%"; }

// A note being written: kept as typed (S.drafts) until saved or closed, so no redraw loses it.
function draftOr(key, saved) { return Object.prototype.hasOwnProperty.call(S.drafts, key) ? S.drafts[key] : saved; }

// ---------- state ----------
const S = {
  home: BOOT, view: "home", bot: null, data: null, tab: "overview", query: "",
  open: { run: true, ppo: true, env: true }, errors: {}, notesOpen: {}, helpOpen: {},
  consoleNext: 0, consoleText: "", showChanges: false, theme: (BOOT && BOOT.prefs && BOOT.prefs.theme) || "dark",
  rw: null, rwQuery: "", rwWindow: 60, rwNotesOpen: {}, rwAbout: {}, rwCatMore: {}, ls: null, lsMore: {}, wiz: null,
  typed: {}, drafts: {}, discards: 0, failed: null,
  // How long Try these rewards plays: shorter where training runs on the processor alone.
  rwTrySteps: BOOT && BOOT.machine && BOOT.machine.graphics_card === false ? 250000 : 1000000,
  rwView: null, rwShowHidden: false,
};
document.documentElement.dataset.theme = S.theme;

// ---------- what each page's file adds ----------
// VIEWS: a page's name (a button's data-go) to the function that draws it. ENTER: run when that page is opened.
// ACTIONS: a button's data-act to what it does (given the button). SEGS: a choice row's data-seg to what a pick
// does (given the value). TAB_VIEWS and TAB_REFRESH: a bot tab's name to the function that draws it (given its
// box), and to the one run on each refresh while it shows.
const VIEWS = {}, ENTER = {}, ACTIONS = {}, SEGS = {}, TAB_VIEWS = {}, TAB_REFRESH = {};
// Handlers of clicks (given the button), changes and typing (given the field), asked in file order. A click
// or change handler returns SKIP when the element is not its own; the first that does not ends the asking.
const SKIP = Symbol("skip");
const HANDLERS = { click: [], change: [], input: [] };
function on(kind, handler) { HANDLERS[kind].push(handler); }
