// The Check this PC page: the checks of `nitro doctor`, each with its state, its message and how to fix it.
const CHECK_STATES = { ok: ["good", "OK"], warning: ["hot", "Warning"], problem: ["bad", "Problem"] };
// The checks' names as this page says them (the command's own names stay in its report).
const CHECK_NAMES = { GPU: "Graphics card", "Graphics card": "Graphics card", "Rust core": "NitroLearn's core", "Collision meshes": "Arena files (collision meshes)", "Rust toolchain": "Rust (only to build rewards of your own)" };
// A check's words for this page: "graphics card" and "processor" for GPU and CPU, the page's own button for
// running the checks again, `code` as code and web addresses as links.
function checkWords(text) {
  const plain = String(text).replace(/run `nitro doctor` again/g, "press Check again").replace(/\bGPUs?\b/g, m => m.length > 3 ? "graphics cards" : "graphics card").replace(/\bCPU\b/g, "processor");
  return esc(plain).replace(/`([^`]+)`/g, '<code class="mono">$1</code>').replace(/https:\/\/[^\s<>"']+[^\s<>"'.,;:)]/g, url => `<button type="button" class="weblink" data-url="${url}" title="Opens in your browser">${url}</button>`);
}
function checkRow(c) {
  const [look, word] = CHECK_STATES[c.state] || ["", c.state];
  return `<div class="check-row" data-state="${esc(c.state)}" data-check="${esc(c.name)}">
    <span><span class="tag ${look}">${esc(word)}</span></span>
    <div><div class="name">${esc(CHECK_NAMES[c.name] || c.name)}</div><div class="msg">${checkWords(c.message)}</div>${c.fix ? `<div class="fix"><span class="cap">How to fix it</span>${checkWords(c.fix)}</div>` : ""}</div>
  </div>`;
}
on("click", t => {
  if (!t.dataset.url) return SKIP;
  call("open_url", t.dataset.url).then(out => { if (out.error) toast(out.error, true); });
});
function checkVerdict(d) {
  const n = d.warnings, warned = n ? ` ${n} ${n === 1 ? "warning" : "warnings"}: training works, but read ${n === 1 ? "it" : "them"}.` : "";
  if (d.problems) return `<div class="banner bad" role="alert"><b>${esc(d.verdict)}</b></div>`;
  return `<div class="banner${n ? " warn" : ""}" role="status"><span class="tag good">Ready</span><span><b>${esc(d.verdict)}</b>${warned}</span></div>`;
}
function renderDoctor() {
  const d = S.doctor || {};
  const button = `<button type="button" class="btn small primary" data-act="doctor-run" ${d.running ? "disabled" : ""}>${d.running ? "Checking…" : d.checks ? "Check again" : "Run the checks"}</button>`;
  $("#main").innerHTML = `<div class="heading"><h1 class="display">Check this PC</h1><div class="row">${button}</div></div>
    <div class="muted" style="font-size:13px;max-width:90ch;">Looks at what training needs on this PC: Python, the graphics card and its driver, NitroLearn's own core, the arena files, room on the disk, and on Windows the parts PyTorch and this window need and RLBot. It only looks: nothing is changed.</div>
    ${d.error ? `<div class="banner bad" role="alert"><pre>${esc(d.error)}</pre></div>` : ""}
    ${d.checks ? checkVerdict(d) + `<div class="checks">${d.checks.map(checkRow).join("")}</div>` : ""}
    ${!d.checks && !d.error ? `<div class="empty">${d.running ? "Checking this PC… it takes a few seconds, sometimes a minute the first time." : "Press <b>Run the checks</b> to see if this PC is ready to train."}</div>` : ""}
    <div class="muted" style="font-size:13px;">The same checks from a terminal: <code class="command">${esc(d.command || "nitro doctor")}</code></div>`;
}
VIEWS.doctor = () => renderDoctor();
// The first visit runs the checks; after that the button does.
ENTER.doctor = () => { if (!S.doctor) ACTIONS["doctor-run"](); };
ACTIONS["doctor-run"] = async () => {
  S.doctor = { running: true };
  if (S.view === "doctor") renderDoctor();
  const out = await call("check_pc");
  S.doctor = out.error ? { error: out.error } : out;
  if (S.view === "doctor") renderDoctor();
};
