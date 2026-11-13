// Open RLBot: starts RLBot's program. Until NitroLearn knows where it is, a dialog says what RLBot is for and
// asks for it, or installs RLBot v5 with RLBot's own installer.
async function openRLBot() {
  const out = await call("open_rlbot");
  if (out.need_program) return rlbotModal();
  if (out.error) toast(out.error, true);
  else toast("Opening RLBot…");
}
function rlbotModal() {
  modal(`<h2 class="display">Open RLBot</h2>
    <div><b>RLBot plays your bot in a real Rocket League match on this PC, offline.</b> Make the bot's RLBot bot first (Make RLBot bot, on the bot's page), then start a match with it in RLBot.</div>
    <div>NitroLearn needs to know where RLBot's program is. Choose it once: from then on Open RLBot starts it.</div>
    <div class="banner"><div><b>No RLBot yet?</b> ${onWindows() ? "Install RLBot v5 from here, or get it" : "Get RLBot v5"} from its official site, <span class="mono">rlbot.org</span>, and then choose its program.</div></div>
    <div class="rli" data-where="dialog">${rlbotInstallHTML("dialog")}</div>`);
}
ACTIONS["rlbot-choose"] = async () => {
  const out = await call("choose_rlbot");
  if (out.error) toast(out.error, true);
  else if (out.opened) toast("RLBot is remembered. Opening it…");
  else if (out.need_program) toast("That file cannot be opened. Choose RLBot's program.", true);
  await loadHome();
  return render();
};
ACTIONS["rlbot-choose-close"] = () => { closeModal(); return ACTIONS["rlbot-choose"](); };
ACTIONS["rlbot-open"] = () => openRLBot();
ACTIONS["rlbot-open-close"] = () => { closeModal(); return openRLBot(); };

// ---------- installing RLBot v5 (Windows) ----------
// RLBot's own installer, downloaded from RLBot's official releases and opened; the person clicks through its
// windows. Shown wherever RLBot is not found yet: the welcome's RLBot step and the Open RLBot dialog (each a
// .rli box, redrawn in place while it runs). S.rli: { state: starting, running, done, failed or error, text
// (what it printed), next (where to read on from), error, details (the text shown) }.
const RLI_NOTE = "RLBot is a separate program with its own licence. It downloads from RLBot's official releases. Remove it later from Windows' Apps list.";
// RLBot v5's program is remembered and still there (the dashboard's own check: an older RLBot does not count).
function rlbotFound() { return !!(S.home && S.home.rlbot_ready); }
function rliBusy(r) { return !!r && (r.state === "starting" || r.state === "running"); }
// Why the install did not end with RLBot found: the last line it printed says it in plain words.
function rliProblem(r) {
  const lines = String(r.text || "").split("\n").map(l => l.trim()).filter(Boolean);
  const last = lines.length ? lines[lines.length - 1] : "";
  return /^RLBot/.test(last) ? last : "RLBot was not installed." + (last ? " " + last : "");
}
// Whether this PC runs Windows: RLBot's installer is a Windows program, so elsewhere only Choose RLBot is
// offered (true when the page does not know).
function onWindows() { const m = S.home && S.home.machine; return !m || m.windows !== false; }
const RLI_ELSEWHERE = "RLBot's installer is for Windows. On this system, get RLBot v5 from its official site, rlbot.org, then choose its program.";
function rlbotInstallHTML(where) {
  const r = S.rli || { state: "none" }, busy = rliBusy(r);
  if (!onWindows()) {
    if (where === "dialog") return `<div class="rli-note">${esc(RLI_ELSEWHERE)}</div><div class="foot"><button type="button" class="btn" data-close="1">Not now</button><button type="button" class="btn primary" data-act="rlbot-choose-close">Choose RLBot…</button></div>`;
    return r.state === "done" || rlbotFound() ? "" : `<div class="rli-note">${esc(RLI_ELSEWHERE)}</div>`;
  }
  const toggle = r.text ? `<button type="button" class="btn link rli-toggle" data-act="rlbot-install-details" aria-expanded="${r.details ? "true" : "false"}">Details</button>` : "";
  const text = r.text && r.details ? `<pre class="console mono rli-text">${esc(r.text)}</pre>` : "";
  const said = r.state === "error" ? `<div class="banner warn rli-said" role="alert"><b>${esc(r.error)}</b></div>`
    : r.state === "failed" ? `<div class="banner bad rli-said" role="alert"><b>${esc(rliProblem(r))}</b></div>`
    : busy ? `<div class="rli-said" role="status"><b>${r.state === "starting" ? "Starting RLBot's installer…" : "Installing RLBot…"}</b> Click through RLBot's installer window.</div>` : "";
  const button = `<button type="button" class="btn${where === "dialog" ? "" : " small"} rli-go" data-act="rlbot-install" ${busy ? "disabled" : ""}>${busy ? "Installing…" : "Install RLBot v5"}</button>`;
  if (where === "dialog") {
    if (r.state === "done") {
      return `<div class="banner rli-said" role="status"><span class="tag good">Installed</span><span>RLBot is installed. Open RLBot now starts it.</span></div>${toggle}${text}
        <div class="foot"><button type="button" class="btn" data-close="1">Close</button><button type="button" class="btn primary" data-act="rlbot-open-close">Open RLBot</button></div>`;
    }
    return `${said}${toggle}${text}<div class="rli-note">${esc(RLI_NOTE)}</div>
      <div class="foot"><button type="button" class="btn" data-close="1">Not now</button>${button}<button type="button" class="btn primary" data-act="rlbot-choose-close">Choose RLBot…</button></div>`;
  }
  if (r.state === "done" || rlbotFound()) return "";
  return `<div class="row">${button}${toggle}</div>${said}${text}<div class="rli-note">${esc(RLI_NOTE)}</div>`;
}
function drawRLBotInstall() {
  document.querySelectorAll(".rli").forEach(el => {
    if (put(el, rlbotInstallHTML(el.dataset.where))) { const pre = el.querySelector(".rli-text"); if (pre) pre.scrollTop = pre.scrollHeight; }
  });
}
ACTIONS["rlbot-install"] = async () => {
  if (rliBusy(S.rli)) return;
  S.rli = { state: "starting", text: "", next: 0, details: !!(S.rli && S.rli.details) };
  drawRLBotInstall();
  const out = await call("install_rlbot");
  if (out.error) { S.rli.state = "error"; S.rli.error = out.error; return drawRLBotInstall(); }
  S.rli.state = "running";
  drawRLBotInstall();
  return pollRLBotInstall();
};
ACTIONS["rlbot-install-details"] = () => { if (S.rli) S.rli.details = !S.rli.details; drawRLBotInstall(); };
// Every 1.5 s while it runs: what it printed since, and once it ends whether RLBot was found. Found, RLBot's
// program is remembered (Open RLBot starts it) and the page shows RLBot as set up.
async function pollRLBotInstall() {
  const r = S.rli;
  if (!r || r.state !== "running" || r.asking) return;
  r.asking = true;
  const out = await call("install_rlbot_state", r.next || 0);
  r.asking = false;
  if (out.error || S.rli !== r) return;
  r.text = ((r.text || "") + (out.text || "")).slice(-200000);
  r.next = out.next;
  if (out.state === "done" && out.program) {
    r.state = "done";
    if (((S.home && S.home.prefs) || {}).rlbot_program !== out.program) await call("set_pref", "rlbot_program", out.program);
    await loadHome();
    toast("RLBot is installed. Open RLBot now starts it.");
    if ($("#modal").hidden) render();
  } else if (out.state === "done" || out.state === "failed") {
    r.state = "failed";
    toast(rliProblem(r), true);
  }
  drawRLBotInstall();
}
