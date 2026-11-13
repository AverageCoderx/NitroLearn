// The Replays in training tab ([replay_mix]): a share of the learning steps copy converted replays.
// The Replays in training tab. The settings are the bot's (staged and applied as any other);
// the tab adds how many converted replays there are to copy and the newest copy steps from the metrics.
function rmItem(name) { return allSections((S.data && S.data.settings) || []).flatMap(s => s.items).find(i => i.path === "replay_mix." + name); }
function rmVal(name) { const i = rmItem(name); return i ? valueOf(i) : undefined; }
function rmOn() { return !!rmVal("enabled") && Number(rmVal("share")) > 0; }
function rmFolders() { const v = rmVal("replays"); return Array.isArray(v) ? v.map(String) : []; }
async function loadReplayMix() {
  const bot = S.bot, out = await call("replay_mix", bot);
  if (S.bot !== bot) return;
  const first = !S.rm || S.rm.loading;
  S.rm = out.error ? { bot, error: out.error } : Object.assign(out, { bot });
  if (S.view !== "bot" || S.tab !== "replays") return;
  if (first || out.error) return renderReplayMix($("#tab"));
  renderReplayParts(); rerenderSetting("replay_mix.replays");  // its folders' counts
}
// After a change to one of its settings: the head at once, then the replays counted again with the change in.
function replayStaged() {
  if (!S.rm || S.rm.bot !== S.bot || !S.rm.sources) return;
  put($("#rm-head"), replayHeadHTML());
  loadReplayMix();
}
async function refreshReplayProgress() {
  if (!S.rm || S.rm.bot !== S.bot || S.rm.loading || S.rm.error) return;
  const bot = S.bot, out = await call("replay_mix_progress", bot);
  if (out.error || !S.rm || S.rm.bot !== bot) return;
  S.rm.progress = out.progress;
  renderReplayParts();
}
function renderReplayMix(box) {
  if (!S.rm || S.rm.bot !== S.bot) { S.rm = { bot: S.bot, loading: true }; loadReplayMix(); }
  if (S.rm.loading) { box.innerHTML = '<div class="card empty">Loading…</div>'; return; }
  if (S.rm.error) { box.innerHTML = `<div class="banner bad"><pre>${esc(S.rm.error)}</pre></div>`; return; }
  if (!rmItem("enabled")) { box.innerHTML = `<div class="banner bad"><div class="grow">The settings file has a mistake, so these settings cannot be shown here until it is fixed:<pre>${esc(S.data.problem || "")}</pre></div><button type="button" class="btn small" data-act="open-settings">Open the settings file</button></div>`; return; }
  box.innerHTML = `<div class="rw-layout">
    <div class="rw-main">
      <div class="settings-tools">
        <button type="button" class="btn small" data-act="rm-guide">How copying replays while training works</button>
        <button type="button" class="btn small ghost" data-go="resources">Converted replays (Resources page)</button>
      </div>
      <div class="muted" style="font-size:13px;max-width:104ch;">While the bot trains against itself, a share of its learning steps can copy how the players in your converted replays play, with a learning rate of their own, so a bot that is already good keeps picking up real play. Only the policy learns from them, never the value network. None of these settings is read only at the start: switch it on or off, or change it, at any time, also late in a bot's training. Changes wait until you press Apply; a training bot then saves and carries on training with them, its steps and everything it learned kept.</div>
      <div id="rm-head"></div>
      <div id="rm-store"></div>
      <section aria-label="Its settings"><div class="heading"><h3>Its settings</h3></div><div class="set-rows" id="rm-settings">${["enabled", "share", "lr", "minibatch", "kickoff_only_seconds"].map(k => rmItem(k)).filter(Boolean).map(i => settingHTML(i)).join("")}${rmFoldersHTML()}</div></section>
    </div>
    <aside class="rw-aside" aria-label="How the copying goes"><div id="rm-aside"></div></aside>
  </div>`;
  renderReplayParts();
}
function renderReplayParts() {
  if (!S.rm || !S.rm.sources || !$("#rm-head")) return;
  put($("#rm-head"), replayHeadHTML());
  put($("#rm-store"), replayStoreHTML());
  put($("#rm-aside"), replayAsideHTML());
}
function replayHeadHTML() {
  const on = rmOn(), share = Number(rmVal("share")) || 0;
  const per = share > 0 && share < 1 ? share / (1 - share) : 0;
  const ppo = name => { const i = allSections(S.data.settings || []).flatMap(s => s.items).find(x => x.path === "ppo." + name); return Number(i ? valueOf(i) : 0) || 0; };
  const policy = ppo("epochs") * ppo("minibatches");
  const what = on
    ? `${pct(share)} of the learning steps that move the policy copy replays: after each update's ${policy} policy steps (its passes times minibatches), about ${num(policy * per, 1)} copy steps of ${num(Number(rmVal("minibatch")) || 0)} decisions each, at a learning rate of ${esc(String(rmVal("lr")))}.`
    : rmVal("enabled") ? "Switched on, but with a share of 0 it takes no copy steps." : "Off: the bot learns from its own games only.";
  const advice = (S.rm.advice || []).map(a => `<div class="banner warn" style="font-size:13px;">${esc(a)}</div>`).join("");
  return `<section class="card pad rw-panel">
    <div class="row"><h3 class="grow">Copying replays while training</h3><span class="tag${on ? " good" : ""}">${on ? "On, " + pct(share) : "Off"}</span></div>
    <div style="font-size:13.5px;max-width:100ch;">${what}</div>${advice}
  </section>`;
}
function replayStoreHTML() {
  const src = S.rm.sources, n = x => Number(x || 0).toLocaleString("en-US");
  const sizes = (src.sizes || []).join(" and ");
  const others = Object.entries(src.store_counts || {}).filter(([k, c]) => c && !(src.sizes || []).includes(k)).map(([k, c]) => `${n(c)} ${k}`);
  const elsewhere = others.length ? ` The store holds ${others.join(", ")} replays, of other game sizes than this bot's.` : "";
  // The user's own folders by their path; the store's (its replays of one game size) in words.
  const where = (src.folders || []).map(f => src.named
    ? `<li><span class="mono" title="${esc(f.folder)}">${esc(f.short || f.folder)}</span>: ${f.missing ? "folder not found" : n(f.replays) + " converted replays"}</li>`
    : `<li title="${esc(f.folder)}">${esc(f.folder.split(/[\\/]/).pop())} replays in the store: ${f.missing || !f.replays ? "none yet" : n(f.replays)}</li>`).join("");
  if (!src.total) {
    const from = src.named ? "in its folders" : `of its size (${esc(sizes)}) in the store`;
    return `<div class="banner warn" role="alert"><div class="grow" style="font-size:13px;"><b>There are no converted replays ${from} yet.</b> ${rmOn() ? "Switched on like this, the bot has nothing to copy: it says so in its console at the start and at every save, and in its report, and trains as without it." : "Switched on, the bot would have nothing to copy."} Convert replays first on the Resources page (Converted replays), or with nitro convert-replays.${esc(elsewhere)}<ul style="margin:6px 0 0;padding-left:18px;">${where}</ul></div><button type="button" class="btn small" data-go="resources">Resources page</button></div>`;
  }
  const head = src.named
    ? `Its folders hold <b>${n(src.total)}</b> converted replays.`
    : `The store holds <b>${n(src.total)}</b> converted ${esc(sizes)} replays, this bot's game size${(src.sizes || []).length > 1 ? "s" : ""}.`;
  return `<section class="card pad rw-panel" id="rm-count"><div class="row" style="align-items:flex-end;gap:14px;"><div class="stat"><div class="k">Replays to copy</div><div class="v display">${n(src.total)}</div></div></div>
    <div style="font-size:13.5px;max-width:100ch;">${head} Each is copied once a cycle, in a random order, and the cycle starts again when all are used. Replays converted later join at the bot's next start (or Apply).${esc(elsewhere)}</div>
    <ul class="muted" style="font-size:12.5px;margin:0;padding-left:18px;">${where}</ul></section>`;
}
function rmFoldersHTML() {
  const item = rmItem("replays");
  if (!item) return "";
  const list = rmFolders(), src = (S.rm && S.rm.sources) || {}, n = x => Number(x || 0).toLocaleString("en-US");
  const counted = src.named ? src.folders || [] : [];
  const rows = list.map((f, i) => {
    const one = counted.find(c => c.folder === f) || counted[i];
    return `<div class="cp-folder"><span class="mono grow" title="${esc(f)}">${esc(f)}</span><span class="muted">${one ? (one.missing ? "not found" : n(one.replays) + " replays") : ""}</span><button type="button" class="btn small" data-act="rm-remove" data-i="${i}">Remove</button></div>`;
  }).join("");
  const none = list.length ? "" : `<div class="muted small">None chosen: the store's folder of this bot's game size (${esc((src.sizes || []).join(", "))}).</div>`;
  return settingHTML(item, false, {
    help: "Folders of converted replays (.nrpl.zst files, subfolders included). None chosen = the store's folder of this bot's game size; choose folders to copy only some replays, such as high-ranked games converted apart.",
    control: `<div class="cp-list">${rows}${none}<div class="row"><button type="button" class="btn small" data-act="rm-add">Add a folder…</button></div></div>`,
  });
}
async function rmEditFolders(list) {
  const item = rmItem("replays");
  if (item && JSON.stringify(item.value) === JSON.stringify(list)) return unstage(item.path);
  return stage("replay_mix.replays", list);
}
function replayAsideHTML() {
  const p = S.rm.progress, n = x => x == null ? "–" : x;
  const stat = (k, v, title) => `<div class="stat" title="${esc(title)}"><div class="k">${k}</div><div class="v display">${v}</div></div>`;
  if (!p) return `<section class="card pad rw-panel"><div class="heading"><h3>How the copying goes</h3></div><div class="muted" style="font-size:13px;">Once the bot trains with this on, this shows its newest update's copy steps: how many, their loss, how often its first choice is the player's, and any wait for replays to be made ready.</div></section>`;
  const last = p.last, newest = p.newest || {};
  const now = newest.steps == null ? "The newest update took no copy steps: copying replays was off, or there was nothing to copy (its console says why)."
    : newest.steps === 0 ? "The newest update took no copy steps: a value warm-up holds the policy still, or its share is carried to the next update." : "";
  const stats = last ? `<div class="cp-stats">
      ${stat("Copy steps per update", n(last.steps), "Copy steps the update took (share / (1 - share) per policy step, a fraction carried to the next)")}
      ${stat("Copy loss", last.loss == null ? "–" : Number(last.loss).toFixed(3), "The copy loss on the replays' decisions: lower is closer to the players")}
      ${stat("Chooses as the players did", last.agreement == null ? "–" : (100 * last.agreement).toFixed(1) + "%", "How often its first choice is the player's, on decisions it mostly has not learned from yet")}
      ${stat("Waited for replays", last.wait_seconds == null ? "–" : Number(last.wait_seconds).toFixed(1) + " s", "Seconds the update waited for replays to be made ready")}
    </div><div class="muted" style="font-size:12px;">From the update at ${steps(last.agent_steps)} agent steps${newest.agent_steps !== last.agent_steps ? ` (the newest is at ${steps(newest.agent_steps)})` : ""}.</div>` : "";
  return `<section class="card pad rw-panel" id="rm-progress"><div class="heading"><h3>How the copying goes</h3></div>${now ? `<div class="muted" style="font-size:13px;">${now}</div>` : ""}${stats}</section>`;
}

ACTIONS["rm-add"] = async () => {
  const out = await call("choose_replay_folder");
  if (out.error) return toast(out.error, true);
  if (out.folder && !rmFolders().includes(out.folder)) return rmEditFolders(rmFolders().concat([out.folder]));
  return;
};
ACTIONS["rm-remove"] = async el => { const list = rmFolders(); list.splice(Number(el.dataset.i), 1); return rmEditFolders(list); };
ACTIONS["rm-guide"] = () => guideModal("How copying replays while training works", ["enabled", "share", "lr", "minibatch", "kickoff_only_seconds", "replays"].map(k => [(rmItem(k) || { label: k }).label, ((S.rm && S.rm.guide) || {})["replay_mix." + k] || ""]).filter(([, v]) => v));
TAB_VIEWS.replays = renderReplayMix;
TAB_REFRESH.replays = () => refreshReplayProgress();
