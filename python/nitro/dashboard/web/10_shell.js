// The shell: the top band (navigation, theme switch), toasts, dialogs, a bot's number plate, which page shows,
// and the one place the page's clicks, changes and typing are handed to the parts that own them.

// ---------- the top band ----------
function renderBand() {
  const current = S.view === "bot" ? "home" : S.view;
  document.querySelectorAll(".nav button").forEach(b => {
    if (b.dataset.go === current) b.setAttribute("aria-current", "page"); else b.removeAttribute("aria-current");
  });
  put($("#theme-switch"), S.theme === "dark" ? `${ICON.sun}Light` : `${ICON.moon}Dark`);
}

// A bot's number plate: its race number on its main colour, one corner cut, its second colour along the
// foot. `size` is its side in pixels.
function plate(bot, size = 54) {
  const l = (bot && bot.livery) || {};
  const paint = [["--plate", l.main], ["--plate-ink", l.ink], ["--plate-2", l.contrast]].filter(([, c]) => /^#[0-9a-f]{6}$/i.test(c || "")).map(([k, c]) => `${k}:${c};`).join("");
  return `<span class="plate display" style="--size:${Number(size) || 54}px;${paint}" title="Race number ${esc(l.number || "")}">${esc(l.number || "")}</span>`;
}

function toast(text, bad) {
  const el = document.createElement("div");
  el.className = "toast" + (bad ? " bad" : "");
  el.textContent = text;
  $("#toasts").appendChild(el);
  setTimeout(() => el.remove(), bad ? 9000 : Math.max(4500, String(text).length * 45));
}

// ---------- dialogs ----------
// A field being typed in sends its change when it loses focus: let it, before it is drawn anew.
function settle(root) { const a = document.activeElement; if (a && root && root.contains(a) && a.blur) a.blur(); }
// Opens a dialog holding `html`: an h2 first (its title band), the content, then a .foot with its buttons.
function modal(html) {
  const m = $("#modal");
  settle(m);
  m.hidden = false;
  m.innerHTML = `<div class="modal-back"><div class="modal" role="dialog" aria-modal="true">${html}</div></div>`;
  const first = m.querySelector("input, select, button.primary");
  if (first) first.focus();
}
function closeModal() { $("#modal").hidden = true; $("#modal").innerHTML = ""; }

function guideModal(title, entries) {
  modal(`<h2 class="display">${esc(title)}</h2>
    <div style="display:flex;flex-direction:column;gap:14px;max-height:65vh;overflow:auto;">${entries.map(([name, text]) => `<section><h3 style="margin:0 0 6px;">${esc(name)}</h3><div class="guide" style="max-width:none;">${guideHTML(text)}</div></section>`).join("") || '<div class="muted">No explanations yet.</div>'}</div>
    <div class="foot"><button type="button" class="btn" data-close="1">Close</button></div>`);
  $(".modal").classList.add("wide");
}

// ---------- rendering and loading ----------
function render() {
  renderBand();
  if (VIEWS[S.view]) VIEWS[S.view]();
  renderPending();
}
async function loadHome() {
  const asked = performance.now(), home = await call("home");
  if (!home.error) { S.home = home; S.homeAsked = asked; }  // when the shown answer was asked for
}
function setTheme(theme) {
  S.theme = theme; document.documentElement.dataset.theme = S.theme; call("set_pref", "theme", S.theme); render();
}
ACTIONS.theme = () => setTheme(S.theme === "dark" ? "light" : "dark");
SEGS.theme = value => setTheme(value);

// ---------- events ----------
// A dialog closes on a click on the dimmed backdrop only: one that both began and ended there (a drag that
// starts in a field and ends outside it is not a click on the backdrop), never on a click inside it.
let pressedOnBackdrop = false, releasedOnBackdrop = false;
document.addEventListener("mousedown", e => { pressedOnBackdrop = e.target.classList.contains("modal-back"); });
document.addEventListener("mouseup", e => { releasedOnBackdrop = e.target.classList.contains("modal-back"); });
document.addEventListener("click", e => {
  const t = e.target.closest("button, [data-sec], .modal-back");
  if (!t) return;
  if (t.classList.contains("modal-back")) { if (e.target === t && pressedOnBackdrop && releasedOnBackdrop) closeModal(); return; }
  if (t.dataset.close) return closeModal();
  if (t.dataset.go) {
    if (t.dataset.go === "rlbot") return openRLBot();
    S.view = t.dataset.go; S.bot = null; S.data = null;
    if (ENTER[S.view]) ENTER[S.view]();
    return loadHome().then(render);
  }
  if (t.dataset.bot) return openBot(t.dataset.bot);
  for (const handler of HANDLERS.click) if (handler(t) !== SKIP) return;
  if (t.parentElement && t.parentElement.dataset.seg) {
    const seg = t.parentElement.dataset.seg;
    t.parentElement.querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", String(b === t)));
    if (SEGS[seg]) SEGS[seg](t.dataset.v);
    return;
  }
  if (t.dataset.act && ACTIONS[t.dataset.act]) {
    if (t.disabled || t.dataset.busy) return;
    t.dataset.busy = "1"; t.disabled = true;  // a double click does it once (never two trainers of one bot)
    return Promise.resolve(ACTIONS[t.dataset.act](t)).finally(() => { if (t.isConnected) { t.disabled = false; delete t.dataset.busy; } });
  }
});
document.addEventListener("keydown", e => {
  if (e.key === "Escape" && !$("#modal").hidden) return closeModal();
  if (!$("#modal").hidden) {
    const inside = [...$("#modal").querySelectorAll("button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea")];
    if (e.key === "Tab" && inside.length) {  // the keyboard stays in the dialog, not the page behind it
      const first = inside[0], last = inside[inside.length - 1];
      if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
      else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
      else if (!$("#modal").contains(document.activeElement)) { e.preventDefault(); first.focus(); }
    }
    if (e.key === "Enter" && e.target.tagName === "INPUT" && e.target.type !== "checkbox") {
      const main = $("#modal .foot .btn.primary:not([disabled])");
      if (main) { e.preventDefault(); e.target.dispatchEvent(new Event("change", { bubbles: true })); main.click(); }
    }
  }
  const sec = e.target.closest && e.target.closest("[data-sec]");
  if (sec && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); sec.click(); }
  if (e.key === "Enter" && e.target.dataset && (e.target.dataset.set || e.target.dataset.rw || e.target.dataset.st) && e.target.tagName === "INPUT" && e.target.type === "text") e.target.blur();
});
document.addEventListener("change", e => {
  const el = e.target;
  if (!el.dataset) return;
  for (const handler of HANDLERS.change) if (handler(el) !== SKIP) return;
});
document.addEventListener("input", e => { HANDLERS.input.forEach(handler => handler(e.target)); });
