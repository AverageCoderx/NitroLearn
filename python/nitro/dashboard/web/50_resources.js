// The Resources page: the files added to the library (settings files, reward sets, your own Rust rewards),
// notes on them, converting replays, and building NitroLearn with your own rewards. Its sections are listed
// on the left; a search narrows them all.
async function loadResources() {
  loadConverting();
  const out = await call("resources");
  S.res = Object.assign(S.res || { query: "", notesOpen: {} }, out.error ? { error: out.error } : out, { loaded: true });
  if (S.res.build === "running" && !S.res.buildLive) S.res.buildLive = { state: "running", text: "", next: 0 };
  if (S.view === "resources") renderResources();
}
function resShown(...texts) {
  const q = ((S.res && S.res.query) || "").trim().toLowerCase();
  return !q || texts.some(t => String(t || "").toLowerCase().includes(q));
}
function resRow(r) {
  const notes = S.res.notes || {}, note = notes[r.noteKey] || "", open = S.res.notesOpen[r.noteKey];
  const id = "res-note-" + r.noteKey.replace(/\W/g, "-");
  return `<div class="setting res-row">
    <div class="what"><div class="name">${esc(r.title)}${(r.chips || []).map(c => `<span class="badge">${esc(c)}</span>`).join("")}</div>${r.path ? `<div class="path mono">${esc(r.path)}</div>` : ""}</div>
    <div class="control">${r.removeId ? `<button type="button" class="iconbtn" data-resremove="${esc(r.removeId)}" title="Take off the list (the file stays)" aria-label="Take ${esc(r.title)} off the list">${ICON.trash}</button>` : ""}</div>
    ${r.about ? `<div class="help full">${esc(r.about)}</div>` : ""}
    ${r.problem ? `<div class="err help full">${esc(r.problem)}</div>` : ""}
    ${open ? "" : `<div class="links"><button type="button" class="more" data-resnote="${esc(r.noteKey)}" aria-label="${note ? "Change the note" : "Add a note"} for ${esc(r.title)}">${note ? "Change the note" : "Add a note"}</button></div>`}
    ${note && !open ? `<div class="notetext">${esc(note)}</div>` : ""}
    ${open ? `<div class="notebox"><textarea class="field" id="${id}" aria-label="Note for ${esc(r.title)}">${esc(note)}</textarea><div class="row"><button type="button" class="btn small primary" data-resnotesave="${esc(r.noteKey)}">Save note</button><button type="button" class="btn small" data-resnoteclose="${esc(r.noteKey)}">Close</button></div></div>` : ""}
  </div>`;
}
// One section of the page: { id, title, about, rows } listed as ruled rows, or { id, title, html } drawn by
// its own function (`count` is then what its line in the list on the left says). `lead` goes above the rows.
function resSectionHTML(sec) {
  if (sec.html != null) return sec.html ? `<div class="res-section" id="res-${sec.id}">${sec.html}</div>` : "";
  const shown = resRowsShown(sec.rows);
  if (!shown.length && S.res.query) return "";
  return `<section class="res-section" id="res-${sec.id}"><div class="heading"><h2 class="display">${esc(sec.title)}</h2><span class="muted grow">${esc(sec.about)}</span></div>
    ${sec.lead || ""}<div class="set-rows">${shown.map(resRow).join("") || '<div class="set-hint">None yet.</div>'}</div></section>`;
}
function resRowsShown(rows) { return rows.filter(r => resShown(r.title, r.about, r.path, ((S.res.notes || {})[r.noteKey]) || "")); }
function renderResources() {
  const top = `<div class="heading"><h1 class="display">Resources</h1><div class="row"><button type="button" class="btn small primary" data-act="res-add">Add a file…</button></div></div>`;
  if (!S.res || !S.res.loaded) { $("#main").innerHTML = top + '<div class="empty">Loading…</div>'; if (!S.res || !S.res.loading) { S.res = Object.assign(S.res || { query: "", notesOpen: {} }, { loading: true }); loadResources(); } return; }
  if (S.res.error) { $("#main").innerHTML = top + `<div class="banner bad"><pre>${esc(S.res.error)}</pre></div>`; return; }
  const files = S.res.files || [];
  const added = kind => files.filter(f => f.kind === kind || (kind === "rust" && f.kind.startsWith("rust_")));
  const built = f => f.kind === "rust_reward" ? (f.built ? ["in this build"] : ["not built yet"]) : f.kind.startsWith("rust_") ? ["building these comes later"] : [];
  const fileRow = f => ({ title: f.title, chips: [KIND_TITLES[f.kind] || f.kind].concat(built(f), f.built_in ? ["comes with NitroLearn"] : [], f.trick_move ? ["trick move"] : []), about: f.kind === "rust_reward" ? (f.built ? "Built into this NitroLearn: it is in every bot's Add reward list." : "Press Build to compile it into NitroLearn.") : f.about, problem: f.problem, path: f.path, noteKey: "file:" + f.id, removeId: f.built_in ? "" : f.id });
  const sources = (S.res.sources || []).map(s => ({ title: s.title, chips: [s.group === "preset" ? "preset" : s.group === "my-settings" ? "my-settings" : "added"], about: s.about, noteKey: "file:" + (s.group === "preset" ? s.id : (files.find(f => "file:" + f.path === s.id) || { id: s.id.replace(/\W/g, "").slice(-40) }).id), removeId: s.group === "library" ? (files.find(f => "file:" + f.path === s.id) || {}).id : "" }));
  const brokenSettings = added("settings").filter(f => f.problem).map(fileRow);
  const obs = S.res.observation || { options: [], help: "" };
  const kept = Object.values(((S.conv || {}).store || {}).counts || {}).reduce((a, v) => a + v, 0);
  const sections = [
    { id: "replays", title: "Converted replays", html: resShown("converted replays convert") ? convertHTML() : "", count: kept },
    { id: "settings", title: "Settings files", about: "Starting points for new bots: the presets, your files in my-settings, and settings files you added.", rows: sources.concat(brokenSettings) },
    { id: "sets", title: "Reward sets", about: "Rewards kept apart from a bot, to load into one from its Rewards tab (which can also save one).", rows: added("rewards").map(fileRow) },
    { id: "rust", title: "Your Rust files", about: "Your own rewards, compiled into NitroLearn by Build (observation builders and action parsers come later). Start from the example, docs/your_own_rewards.md.", rows: added("rust").map(fileRow), lead: buildHTML(added("rust")) },
    { id: "rewards", title: "Built in: rewards", about: "Every reward a bot can use, with what it pays for.", rows: (S.res.rewards || []).map(k => ({ title: k.title, chips: [k.name].concat(k.every_step ? ["every step"] : [], k.trick_move ? ["trick move"] : []), about: k.about, noteKey: "reward:" + k.name })) },
    { id: "observations", title: "Built in: observations", about: obs.help, rows: obs.options.map(o => ({ title: o, chips: ["env.observation"], noteKey: "observation:" + o })) },
  ];
  const nav = sections.map(sec => {
    const n = sec.rows ? resRowsShown(sec.rows).length : sec.count;
    const gone = sec.rows ? !n && !!S.res.query : !sec.html;
    return `<button type="button" class="set-group" data-resjump="${sec.id}" ${gone ? "disabled" : ""}><span class="t">${esc(sec.title)}</span><span class="n">${Number(n || 0).toLocaleString("en-US")}</span></button>`;
  }).join("");
  const body = sections.map(resSectionHTML).join("");
  $("#main").innerHTML = top + `
    <div class="muted" style="font-size:13px;max-width:104ch;">Files you add are recognised by what they hold and offered where they fit: settings files as starting points in the new-bot form, reward sets in a bot's Rewards tab. The files stay where they are. Notes can go on anything here, the built-in rewards and observations too.</div>
    <div class="set-layout">
      <aside class="set-nav" aria-label="Sections of this page">
        <label class="search"><span class="sr">Search resources</span>${ICON.search}<input id="res-search" class="field" type="search" placeholder="Search resources and notes" value="${esc(S.res.query)}"></label>
        <nav class="set-groups" aria-label="Sections">${nav}</nav>
      </aside>
      <div class="set-main res-main">${body || '<div class="card empty">Nothing here matches that search.</div>'}</div>
    </div>`;
  placeSettingsNav();
  const box = $("#res-search");
  if (S.res.focusSearch && box) { box.focus(); box.setSelectionRange(box.value.length, box.value.length); }
}
// ---------- converted replays: replays converted once, for every bot's replay copy ----------
async function loadConverting() {
  const after = S.conv && S.conv.state === "running" ? S.conv.next || 0 : 0;
  const out = await call("converting", after);
  if (out.error) return;
  const text = after && S.conv ? (S.conv.text || "") + (out.text || "") : out.text || "";
  S.conv = Object.assign(S.conv || { limit: "" }, out, { text: text.slice(-60000), loaded: true });
  if (S.view === "resources") renderConvert();
}
function renderConvert() { const box = $("#convert"); if (box) { box.outerHTML = convertHTML(); const pre = $("#convText"); if (pre) pre.scrollTop = pre.scrollHeight; } }
async function countConvert(list) {
  const key = JSON.stringify(list);
  S.cvCounting = key;
  const out = await call("count_replays", list);
  if (S.cvCounting !== key) return;
  S.cvCounting = null;
  if (!out.error) { S.cvCount = { key, out }; renderConvert(); }
}
function convertHTML() {
  if (!S.conv || !S.conv.loaded) { if (!S.convLoading) { S.convLoading = true; loadConverting().finally(() => { S.convLoading = false; }); } return '<section id="convert"><div class="heading"><h2 class="display">Converted replays</h2></div><div class="set-hint">Loading…</div></section>'; }
  const c = S.conv, n = x => Number(x || 0).toLocaleString("en-US"), gb = b => (Number(b || 0) / 2 ** 30).toFixed(1);
  const counts = Object.entries((c.store || {}).counts || {}).filter(([, v]) => v > 0);
  const kept = counts.reduce((a, [, v]) => a + v, 0);
  const list = c.folders || [], key = JSON.stringify(list), counted = S.cvCount && S.cvCount.key === key ? S.cvCount.out : null;
  if (!counted && S.cvCounting !== key) countConvert(list);
  const rows = list.map((f, i) => { const one = counted && counted.folders[i]; return `<div class="cp-folder"><span class="mono grow" title="${esc(f)}">${esc(f)}</span><span class="muted">${one ? (one.missing ? "not found" : n(one.replays) + " replays") : "counting…"}</span><button type="button" class="btn small" data-act="cv-remove" data-i="${i}" ${c.state === "running" ? "disabled" : ""}>Remove</button></div>`; }).join("");
  const own = list.length ? "" : `<div class="muted" style="font-size:13px;">None chosen: the game's own replay folders${counted ? ", " + n(counted.total) + " replays there" : ""}.</div>`;
  const p = c.progress, running = c.state === "running";
  const share = p && p.todo ? Math.min(1, p.done / p.todo) : 0;
  const left = p && p.seconds_left != null ? (p.seconds_left >= 3600 ? Math.floor(p.seconds_left / 3600) + " h " + Math.round(p.seconds_left % 3600 / 60) + " min" : Math.max(1, Math.round(p.seconds_left / 60)) + " min") : "";
  const words = { none: "", running: "Converting…", done: "Done.", stopped: "Stopped. What was converted is kept; Convert carries on with the rest.", failed: "It stopped with an error: the lines below say why." }[c.state] || "";
  const progress = p && (running || c.state !== "none") ? `<div style="display:flex;flex-direction:column;gap:6px;">
      <div class="row"><b>${n(p.done)} of ${n(p.todo)} converted</b><span class="muted">${p.per_minute} a minute${running && left ? " · about " + left + " left" : ""}</span></div>
      <div class="bar" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${Math.round(100 * share)}"><span style="width:${(100 * share).toFixed(1)}%"></span></div>
      <div class="muted" style="font-size:13px;">${n(p.found)} replays in the folders, ${n(p.found - p.todo)} converted before or not usable. Kept now: ${n(p.kept)}${Object.keys(p.by_size || {}).length ? " (" + Object.entries(p.by_size).sort().map(([k, v]) => k + " " + n(v)).join(", ") + ")" : ""}${p.failed ? "; " + n(p.failed) + " could not be converted" : ""}.</div>
      ${p.disk_full ? '<div class="banner bad" style="font-size:13px;">The disk has under 20 GB free, so converted replays are no longer kept. Free some space and press Convert again.</div>' : ""}
    </div>` : "";
  const stat = (k, v) => `<div class="stat"><div class="k">${k}</div><div class="v display">${v}</div></div>`;
  return `<section id="convert"><div class="heading"><h2 class="display">Converted replays</h2><span class="muted grow">Replays are converted once and kept, for every bot that learns from them.</span></div>
   <div class="card pad rw-panel res-card">
    <div class="row" style="gap:12px 34px;align-items:flex-end;">${stat("Kept", n(kept))}${stat("Their size", gb((c.store || {}).bytes) + " GB")}${stat("Free on the disk", gb((c.store || {}).free) + " GB")}</div>
    <div class="ink2" style="font-size:13px;max-width:110ch;">Converting a replay (re-simulating it to recover every press) is the slow part of learning from replays. Every replay converted, here or by any bot's replay copy, is kept by game size, and every copy uses the kept ones first, so no replay is ever converted twice. Converting here learns nothing: it uses all ${n(c.threads)} threads of this PC at low priority, so the PC stays usable. About 8-10 MB a 5-minute replay.</div>
    ${counts.length ? `<div class="chips">${counts.map(([k, v]) => `<span class="chip">${esc(k)}: ${n(v)}</span>`).join("")}</div>` : ""}
    <div class="cp-list" style="max-width:860px;"><span class="cap muted">Folders to convert</span>${rows}${own}<div class="row"><button type="button" class="btn small" data-act="cv-add" ${running ? "disabled" : ""}>Add a folder…</button>${counted && list.length > 1 ? `<span class="muted">${n(counted.total)} replays in all</span>` : ""}</div></div>
    <div class="row"><label class="muted" for="cv-limit" style="font-size:13px;">How many (newest first; empty for all)</label><input id="cv-limit" class="field" style="width:110px;" inputmode="numeric" value="${esc(c.limit || "")}" ${running ? "disabled" : ""}>
      <span class="grow"></span>${running ? '<button type="button" class="btn" data-act="cv-stop">Stop</button>' : '<button type="button" class="btn primary" data-act="cv-start">Convert</button>'}</div>
    ${words ? `<div class="${c.state === "failed" ? "banner bad" : "muted"}" style="font-size:13px;">${esc(words)}</div>` : ""}
    ${progress}
    ${c.text ? `<pre class="console mono" id="convText" style="height:18vh;">${esc(c.text)}</pre>` : ""}
   </div></section>`;
}
async function pollConverting() { if (S.view === "resources" && S.conv && S.conv.state === "running") await loadConverting(); }
function buildHTML(rust) {
  const b = S.res.buildLive || { state: S.res.build || "none", text: "" };
  const rewards = rust.filter(f => f.kind === "rust_reward" && !f.problem);
  const stale = rewards.some(f => !f.built);
  if (!rewards.length && b.state === "none") return "";
  const words = { none: stale ? "Some of your rewards are not built yet." : "Your rewards are built in.", running: "Building… (a few minutes)", done: "Built. Close the dashboard and open it again to use the new rewards.", failed: "The build failed: the lines below say why (for your own file, its name and line)." }[b.state];
  return `<section class="card pad rw-panel res-card">
    <div class="row"><h3 class="grow">Build NitroLearn with your rewards</h3><button type="button" class="btn ${stale ? "primary" : ""}" data-act="res-build" ${b.state === "running" ? "disabled" : ""}>Build</button></div>
    <div class="ink2" style="font-size:13px;">Compiles ${rewards.length} reward file${rewards.length === 1 ? "" : "s"} of yours into NitroLearn. It needs Rust and, on Windows, Microsoft's C++ build tools: if one is missing, the build says what to install. In this build: ${esc((S.res.own_rewards || []).join(", ") || "none")}. Bots training meanwhile carry on with the old build until they start again.</div>
    <div class="${b.state === "failed" ? "banner bad" : b.state === "done" ? "banner" : "muted"}" style="font-size:13px;">${esc(words)}</div>
    ${b.text ? `<pre class="console mono" id="buildText" style="height:28vh;">${esc(b.text)}</pre>` : ""}
  </section>`;
}
async function pollBuild() {
  if (!S.res || !S.res.buildLive || S.res.buildLive.state !== "running") return;
  const out = await call("build_state", S.res.buildLive.next || 0);
  if (out.error || !S.res || !S.res.buildLive) return;
  S.res.buildLive.text = ((S.res.buildLive.text || "") + (out.text || "")).slice(-200000);
  S.res.buildLive.next = out.next; S.res.buildLive.state = out.state;
  if (S.view === "resources") { renderResources(); const pre = $("#buildText"); if (pre) pre.scrollTop = pre.scrollHeight; }
  if (S.view === "prefs") drawEditBuild();
  if (out.state === "done") toast("Built. Close the dashboard and open it again to use the new rewards.");
  if (out.state === "failed") toast("The build failed; the Resources page says why.", true);
}
const KIND_TITLES = { settings: "settings file", rewards: "reward set", rust_reward: "Rust reward", rust_observation: "Rust observation", rust_actions: "Rust action parser" };

ACTIONS["cv-add"] = async () => {
  const out = await call("choose_replay_folder");
  if (out.error) return toast(out.error, true);
  if (out.folder && !(S.conv.folders || []).includes(out.folder)) { const f = await call("convert_folders", (S.conv.folders || []).concat([out.folder])); if (!f.error) S.conv.folders = f.folders; }
  return renderConvert();
};
ACTIONS["cv-remove"] = async el => { const list = (S.conv.folders || []).slice(); list.splice(Number(el.dataset.i), 1); const f = await call("convert_folders", list); if (!f.error) S.conv.folders = f.folders; return renderConvert(); };
ACTIONS["cv-start"] = async () => {
  const raw = ($("#cv-limit").value || "").trim().replace(/[,_ ]/g, "");
  if (raw && !/^\d+$/.test(raw)) return toast("How many: a whole number, or empty for all.", true);
  S.conv.limit = raw;
  const out = await call("convert_start", S.conv.folders || [], Number(raw || 0));
  if (out.error) return toast(out.error, true);
  toast("Converting replays (learning nothing)…");
  S.conv.state = "running"; S.conv.text = ""; S.conv.next = 0; S.conv.progress = null;
  return renderConvert();
};
ACTIONS["cv-stop"] = async () => { const out = await call("stop_job", ".convert", "convert"); if (out.error) return toast(out.error, true); toast("Stopped. What was converted is kept."); return loadConverting(); };
ACTIONS["res-build"] = async () => {
  const out = await call("build_rust");
  if (out.error) return toast(out.error, true);
  S.res = S.res || { query: "", notesOpen: {} };  // also started from App settings (Rebuild NitroLearn)
  S.res.buildLive = { state: "running", text: "", next: 0 };
  toast("Building with " + (out.files.join(", ") || "no files of yours") + "…");
  return S.view === "prefs" ? drawEditBuild() : renderResources();
};
ACTIONS["res-add"] = async () => {
  const out = await call("add_resources");
  if (out.error) return toast(out.error, true);
  (out.refused || []).forEach(r => toast(r, true));
  if ((out.added || []).length) toast("Added " + out.added.map(a => a.title + " (" + (KIND_TITLES[a.kind] || a.kind) + ")").join(", ") + ".");
  S.res = Object.assign(S.res || {}, { loaded: false, loading: false }); return renderResources();
};
on("click", t => {
  if (t.dataset.resjump) {
    const el = document.getElementById("res-" + t.dataset.resjump), band = $("#band");
    if (el) window.scrollTo(0, el.getBoundingClientRect().top + window.scrollY - (band ? band.offsetHeight : 0) - 14);
    return;
  }
  if (t.dataset.resnote) { S.res.notesOpen[t.dataset.resnote] = true; renderResources(); const ta = $("#res-note-" + t.dataset.resnote.replace(/\W/g, "-")); if (ta) ta.focus(); return; }
  if (t.dataset.resnoteclose) { delete S.res.notesOpen[t.dataset.resnoteclose]; return renderResources(); }
  if (t.dataset.resnotesave) {
    const key = t.dataset.resnotesave, ta = $("#res-note-" + key.replace(/\W/g, "-"));
    return call("set_library_note", key, ta ? ta.value : "").then(out => { if (out.error) return toast(out.error, true); S.res.notes = out.notes; delete S.res.notesOpen[key]; renderResources(); });
  }
  if (t.dataset.resremove) return call("remove_resource", t.dataset.resremove).then(out => { if (out.error) return toast(out.error, true); S.res.loaded = false; S.res.loading = false; renderResources(); });
  return SKIP;
});
on("input", el => {
  if (el.id === "res-search") { S.res.query = el.value; S.res.focusSearch = true; renderResources(); S.res.focusSearch = false; }
});
VIEWS.resources = () => renderResources();
ENTER.resources = () => { S.res = { query: (S.res && S.res.query) || "", notesOpen: {} }; };
