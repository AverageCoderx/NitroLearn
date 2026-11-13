// The App settings page: the look, where RLBot is, where the bots are kept, updating NitroLearn (only when
// asked), what people who change NitroLearn's code need, and the About block (the logo, the version, how
// NitroLearn may be used and the licences of the pieces that come inside it).
async function loadAbout() {
  const out = await call("about");
  S.about = out.error ? { error: out.error } : out;
  if (S.view === "prefs") prefsView();
}
function aboutHTML() {
  const a = S.about;
  if (!a) return '<div class="set-hint">Loading…</div>';
  if (a.error) return `<div class="banner bad"><pre>${esc(a.error)}</pre></div>`;
  const logo = $("#band .lockup svg");
  return `<div class="about-block">
      <div class="about-lockup">${logo ? logo.outerHTML : ""}<span class="wordmark"><b>Nitro</b>Learn</span><span class="by">${esc(a.credit)}</span></div>
      <div class="about-facts">
        <div class="stat"><div class="k">Version</div><div class="v display" data-about-version>${esc(a.version)}</div></div>
        <div class="grow"><div><b>${esc(a.use)}</b> <span class="muted" data-about-licence>${esc(a.licence)}</span></div><div class="muted">${esc(a.fair_play)}</div></div>
      </div>
    </div>
    <div class="heading sub"><h3 class="cap">Bundled pieces and their licences</h3></div>
    <div class="set-rows about-licences">${(a.bundled || []).map(p => `<div class="setting res-row"><div class="what"><div class="name">${esc(p.name)}</div></div><div class="control"><span class="muted">${esc(p.licence)}</span></div></div>`).join("")}</div>`;
}

// ---------- updating NitroLearn ----------
// S.upd.phase: idle, checking, checked (S.upd.check), error, running, done or failed (S.upd.result).
function updateHTML() {
  const u = S.upd || { phase: "idle" }, c = u.check, a = S.about || {};
  const version = (c && c.installed) || a.version || "";
  const checkButton = label => `<button type="button" class="btn small" id="upd-check" data-act="upd-check" ${u.phase === "checking" ? "disabled" : ""}>${esc(label)}</button>`;
  const consoleBox = u.text ? `<pre class="console mono upd-console" id="updText">${esc(u.text)}</pre>` : "";
  if (u.phase === "idle" || u.phase === "checking") {
    return `<div class="row upd-line"><span class="grow">You have NitroLearn <b>${esc(version)}</b>.${u.phase === "checking" ? " Asking GitHub for the newest version…" : ""}</span>${checkButton(u.phase === "checking" ? "Checking…" : "Check for updates")}</div>`;
  }
  if (u.phase === "error") return `<div class="banner bad" role="alert" id="upd-said"><b>${esc(u.error)}</b></div><div class="row">${checkButton("Check again")}</div>`;
  if (u.phase === "checked" && c.state !== "newer") return `<div class="banner" role="status" id="upd-said"><span class="tag good">Up to date</span><span>${esc(c.line)}</span></div><div class="row">${checkButton("Check again")}</div>`;
  if (u.phase === "checked") {
    const why = c.blocked || c.busy;
    return `<div class="upd-news" id="upd-said"><b>${esc(c.line)}</b></div>
      ${c.release.notes ? `<pre class="upd-notes">${esc(c.release.notes)}</pre>` : ""}
      ${why ? `<div class="banner warn" role="status" id="upd-blocked">${esc(why)}</div>` : ""}
      <div class="row">${c.release.page ? '<button type="button" class="btn link" id="upd-page" data-act="upd-page">What is new, on GitHub</button>' : ""}<span class="grow"></span>${checkButton("Check again")}<button type="button" class="btn primary" id="upd-now" data-act="upd-confirm" ${why ? "disabled" : ""}>Update now</button></div>`;
  }
  if (u.phase === "running") return `<div class="banner" role="status" id="upd-said"><span class="tag strong">Updating</span><span>Updating to ${esc(c.release.version)}… keep this window open. It takes a few minutes, longer when PyTorch changes.</span></div>${consoleBox}`;
  const r = u.result || {};
  if (u.phase === "failed") {
    const words = r.error ? "The update stopped: " + r.error : `Updated from ${r.old} to ${r.new}, but the installer did not finish cleanly: the lines below say why. Run Install NitroLearn again to repair it.`;
    return `<div class="banner bad" role="alert" id="upd-said"><b>${esc(words)}</b></div>${consoleBox}<div class="row">${checkButton("Check again")}</div>`;
  }
  const later = r.installer === "later";
  const said = later ? `Updated the code from ${r.old} to ${r.new}. NitroLearn closes to finish the update: the installer runs in its own window, then the dashboard opens again.`
    : `Updated from ${r.old} to ${r.new}. Restart the dashboard to use it.`;
  const replaced = (r.replaced || []).length ? `<div class="upd-replaced"><b>Files of NitroLearn's that you had changed were replaced.</b> Your copies are kept in <span class="mono">${esc(r.backup)}</span>:<ul>${r.replaced.map(f => `<li class="mono">${esc(f)}</li>`).join("")}</ul></div>` : "";
  const removed = (r.removed || []).length ? `<details class="upd-removed" id="upd-removed"><summary><b>${r.removed.length} ${r.removed.length === 1 ? "file" : "files"} the new version no longer has ${r.removed.length === 1 ? "was" : "were"} removed</b> (NitroLearn's own, none of yours)</summary><ul>${r.removed.map(f => `<li class="mono">${esc(f)}</li>`).join("")}</ul></details>` : "";
  const button = u.manual ? `<span class="muted" id="upd-manual">${esc(u.manual)}</span>`
    : `<button type="button" class="btn primary" id="upd-restart" data-act="upd-restart" ${u.closing ? "disabled" : ""}>${u.closing ? "Closing…" : later ? "Close and finish" : "Restart"}</button>`;
  return `<div class="banner" role="status" id="upd-said"><span class="tag good">Updated</span><span>${esc(said)}</span></div>${replaced}${removed}<div class="row"><span class="grow"></span>${button}</div>${consoleBox}`;
}
function drawUpdate() {
  const box = $("#upd-box");
  if (!box) return;
  put(box, updateHTML());
  const pre = $("#updText");
  if (pre) pre.scrollTop = pre.scrollHeight;
}
ACTIONS["upd-check"] = async () => {
  S.upd = { phase: "checking" }; drawUpdate();
  const out = await call("update_check");
  S.upd = out.error ? { phase: "error", error: out.error } : { phase: "checked", check: out };
  drawUpdate();
};
ACTIONS["upd-page"] = () => call("open_url", S.upd.check.release.page);
ACTIONS["upd-confirm"] = () => {
  const c = S.upd.check;
  const how = c.git ? `This folder is a git checkout: git moves it to the release (${esc(c.release.tag)}). Changes of your own to the code stop the update; nothing of yours is overwritten.`
    : "NitroLearn's own files are replaced by the new version's, and files the new version no longer has are removed; nothing of yours is. A file of NitroLearn's that you changed is copied to .tmp/update/replaced first.";
  modal(`<h2 class="display">Update to NitroLearn ${esc(c.release.version)}?</h2>
    <div><b>Kept:</b> your bots, their settings, replays and saves, and every other file of yours in this folder.</div>
    <div>${how} Then the installer runs again for the parts the new version needs (a few minutes, longer when PyTorch changes).${c.finish_on_close ? " To finish, NitroLearn closes and opens again." : ""}</div>
    <div class="muted" style="font-size:13px;">Training must be stopped first. Bots keep working: an update never breaks a bot.</div>
    <div class="foot"><button type="button" class="btn" data-close="1">Cancel</button><button type="button" class="btn primary" id="upd-go" data-act="upd-go">Update now</button></div>`);
};
ACTIONS["upd-go"] = async () => {
  closeModal();
  const out = await call("update_start");
  if (out.error) { S.upd.check.busy = out.error; drawUpdate(); return toast(out.error, true); }
  S.upd = { phase: "running", check: S.upd.check, text: "", next: 0 };
  drawUpdate();
};
async function pollUpdate() {
  const u = S.upd;
  if (!u || u.phase !== "running") return;
  const out = await call("update_state", u.next || 0);
  if (out.error || S.upd !== u) return;
  u.text = ((u.text || "") + (out.text || "")).slice(-200000); u.next = out.next;
  if (out.state === "done" || out.state === "failed") {
    u.phase = out.state; u.result = out.result || {};
    toast(out.state === "done" ? "NitroLearn is updated." : "The update stopped; App settings says why.", out.state !== "done");
  }
  drawUpdate();
}
// An update whose installer was still to run when the window closed (Windows) is offered again at start.
async function askUnfinishedUpdate() {
  const out = await call("update_state", 0);
  if (!out.unfinished || (S.upd && S.upd.phase !== "idle")) return;
  S.upd = { phase: "done", result: out.result || {}, text: "", next: 0 };
  toast(`The update to ${out.result.new} is not finished: on App settings, press Close and finish.`);
  drawUpdate();
}
ACTIONS["upd-restart"] = async () => {
  const out = await call("update_restart");
  if (out.error) return toast(out.error, true);
  if (out.manual) S.upd.manual = out.manual; else S.upd.closing = true;
  drawUpdate();
};

// ---------- editing NitroLearn ----------
function editBuildHTML() {
  const b = (S.res && S.res.buildLive) || { state: "none", text: "" };
  const words = { none: "", running: "Building… (a few minutes)", done: "Built. Close the dashboard and open it again to use the new build.", failed: "The build failed: the lines below say why." }[b.state] || "";
  return (words ? `<div class="${b.state === "failed" ? "banner bad" : b.state === "done" ? "banner" : "muted"}" style="font-size:13px;">${esc(words)}</div>` : "")
    + (b.text ? `<pre class="console mono" id="editBuildText" style="height:24vh;">${esc(b.text)}</pre>` : "");
}
function drawEditBuild() {
  const running = !!(S.res && S.res.buildLive && S.res.buildLive.state === "running");
  const button = $("#edit-rebuild");
  if (button) button.disabled = running;
  put($("#edit-build"), editBuildHTML());
  const pre = $("#editBuildText");
  if (pre) pre.scrollTop = pre.scrollHeight;
}
ACTIONS["open-code"] = async () => { const out = await call("open_folder", "", "code"); if (out.error) toast(out.error, true); };
ACTIONS["open-rewards-guide"] = async () => { const out = await call("open_folder", "", "rewards-guide"); if (out.error) toast(out.error, true); };

function prefsView() {
  const prefs = (S.home && S.home.prefs) || {}, a = S.about || {};
  const row = (name, about, control) => `<div class="setting wide"><div class="what"><div class="name">${name}</div></div><div class="control">${control}</div><div class="help full">${about}</div></div>`;
  if (!S.about && !S.aboutAsked) { S.aboutAsked = true; loadAbout(); }
  const building = !!(S.res && S.res.buildLive && S.res.buildLive.state === "running");
  $("#main").innerHTML = `<div class="heading"><h1 class="display">App settings</h1></div>
    <div class="prefs">
      <section><div class="heading"><h2 class="display">This app</h2><span class="muted grow">How the dashboard looks and what it opens.</span></div>
        <div class="set-rows">
          ${row("Look", "Dark or light. The top band stays black in both.", `<div class="seg" data-seg="theme"><button type="button" data-v="dark" aria-pressed="${S.theme === "dark"}">Dark</button><button type="button" data-v="light" aria-pressed="${S.theme === "light"}">Light</button></div>`)}
          ${row("RLBot", "The program that plays your bot in a real Rocket League match, offline. Open RLBot (top right) starts it.", `<span class="mono" style="font-size:13px;overflow-wrap:anywhere;">${prefs.rlbot_program ? esc(prefs.rlbot_program) : "Not chosen yet"}</span><button type="button" class="btn small" data-act="rlbot-choose">Choose RLBot…</button>`)}
          ${row("Bots folder", "Where your bots, their saves and their settings are kept.", `<span class="mono" style="font-size:13px;overflow-wrap:anywhere;">${esc(S.home && S.home.runs)}</span>`)}
        </div>
      </section>
      <section id="update"><div class="heading"><h2 class="display">Update NitroLearn</h2><span class="muted grow">Checked only when you press the button, never in the background.</span></div>
        <div class="card pad upd-card" id="upd-box"></div>
      </section>
      <section id="editing"><div class="heading"><h2 class="display">Editing NitroLearn</h2><span class="muted grow">For people who change NitroLearn's own code.</span></div>
        <div class="set-rows">
          ${row("Code folder", "NitroLearn's folder: its code, settings files and docs.", `<span class="mono" style="font-size:13px;overflow-wrap:anywhere;">${esc(a.folder || "")}</span><button type="button" class="btn small" id="edit-open" data-act="open-code">Open the code folder</button>`)}
          ${row("Rebuild NitroLearn", "Builds NitroLearn's core again from the code in this folder, with your own Rust rewards from the Resources page. It needs Rust (the installer puts it in the folder when run with <span class='mono nowrap'>-Rust</span>) and, on Windows, Microsoft's C++ build tools: if one is missing, the build says what to install. A few minutes.", `<button type="button" class="btn small" id="edit-rebuild" data-act="res-build" ${building ? "disabled" : ""}>Rebuild NitroLearn</button>`)}
          <div id="edit-build" class="edit-build"></div>
          ${row("Your own rewards", "How to write a reward of your own in Rust, step by step: docs/your_own_rewards.md.", `<button type="button" class="btn small" id="edit-guide" data-act="open-rewards-guide">Open the guide</button>`)}
        </div>
      </section>
      <section><div class="heading"><h2 class="display">About</h2></div>${aboutHTML()}</section>
    </div>`;
  drawUpdate();
  drawEditBuild();
}
VIEWS.prefs = () => prefsView();
