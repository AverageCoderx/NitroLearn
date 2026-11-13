// The Rewards tab: the bot's rewards and their options, what each reward pays, trying rewards on the saved bot, the
// Add reward list and reward sets.
const WINDOWS = [[60, "Last minute"], [600, "10 minutes"], [3600, "Hour"]];
function kindOf(name) {
  return ((S.rw && S.rw.catalog) || []).find(k => k.name === name) || { name, title: name, about: "", params: [], every_step: false, group: false };
}

S.rwNerd = {}; S.rwCatNerd = {}; S.rwCatGroup = ""; S.rwMoveOpen = null;
ICON.grip = '<svg width="14" height="16" viewBox="0 0 14 16" fill="currentColor" aria-hidden="true"><circle cx="4" cy="3" r="1.6"/><circle cx="10" cy="3" r="1.6"/><circle cx="4" cy="8" r="1.6"/><circle cx="10" cy="8" r="1.6"/><circle cx="4" cy="13" r="1.6"/><circle cx="10" cy="13" r="1.6"/></svg>';
const GUIDE_NAMES = { "reward.weight": "Weight", "reward.zero_sum": "Zero-sum", "reward.zero_sum.team_spirit": "Team spirit", "reward.zero_sum.opponent_scale": "Opponent share", "reward.shares": "Sorting and hiding in what each reward pays", "start.weight": "Weight (share of games)" };

const originalOf = e => (e.from != null && S.rw.original) ? S.rw.original[e.from] : null;
const isChanged = e => { const o = originalOf(e); return !o || JSON.stringify(canon(e.view)) !== JSON.stringify(canon(o)); };
// A reward's key: its name, with #2, #3, ... for a second or third of that name (its note and its place in
// the folders are kept under it; reward_layout.py).
function rewardKey(i) {
  const list = S.rw.rewards, name = list[i].view.name;
  const n = list.slice(0, i + 1).filter(e => e.view.name === name).length;
  return name + (n > 1 ? "#" + n : "");
}
function noteKey(i) { return "reward:" + rewardKey(i); }
function shareOf(label) { return label ? (((S.rw.shares || {}).rewards) || []).find(r => r.label === label) : null; }
function alerted(label) { return (((S.rw.shares || {}).alerts) || []).some(a => a.label === label && a.level === "warning"); }

// Whether a reward matches a search, the same in the Add reward list and in the bot's rewards: its name in
// the file, its plain title, what it pays for or the file it comes from (`more`: other words of its own).
function rewardMatches(k, query, more) {
  const q = String(query || "").trim().toLowerCase();
  return !q || [k.name, k.title, k.about, k.family || ""].concat(more || []).some(s => String(s || "").toLowerCase().includes(q));
}
// `text` escaped, with each place the search found marked.
function marked(text, query) {
  const s = String(text == null ? "" : text), q = String(query || "").trim().toLowerCase();
  if (!q) return codeMarks(esc(s));
  const low = s.toLowerCase();
  let out = "", at = 0, i;
  while ((i = low.indexOf(q, at)) !== -1) { out += esc(s.slice(at, i)) + `<mark class="hit">${esc(s.slice(i, i + q.length))}</mark>`; at = i + q.length; }
  return codeMarks(out + esc(s.slice(at)));
}
function rewardShown(e) { return rewardMatches(kindOf(e.view.name), S.rwQuery, [e.label || ""]); }

// What makes a reward the same as another (rewards.identity): its name; an "all of" group's parts too.
function rewardIdentity(view) {
  if (view.name !== "all_of") return view.name || null;
  const parts = (view.parts || []).map(p => JSON.stringify(canon(p))).sort();
  return parts.length ? "all_of:" + parts.join(",") : null;
}
// The rewards `list` holds more often than once and more often than the settings file does (rewards.added_copies).
function addedCopies(list) {
  const count = views => views.reduce((m, v) => { const id = rewardIdentity(v); if (id) m.set(id, (m.get(id) || 0) + 1); return m; }, new Map());
  const had = count(S.rw.original || []), now = count(list.map(e => e.view));
  return [...now].filter(([id, n]) => n > Math.max(1, had.get(id) || 0)).map(([id]) => id);
}
function rewardIndex(name) { return ((S.rw && S.rw.rewards) || []).findIndex(e => e.view.name === name); }
async function loadRewards() {
  const bot = S.bot;
  const out = await call("rewards", bot, S.rwWindow);
  if (S.bot !== bot) return;
  S.rw = out.error ? { bot, error: out.error } : Object.assign(out, { bot });
  if (S.view === "bot" && S.tab === "rewards") renderRewards($("#tab"));
}
function rewardsSwitchTransfer() {
  const st = (S.data && S.data.start) || {}, parts = S.data && tfItem("parts");
  return st.kind === "from" && !st.done && parts && parts.value === "all";
}
function renderRewards(box) {
  if (!S.rw || S.rw.bot !== S.bot) { S.rw = { bot: S.bot, loading: true }; S.rwNerd = {}; loadRewards(); }
  if (S.rw.loading) { box.innerHTML = '<div class="card empty">Loading the rewards…</div>'; return; }
  if (S.rw.error) { box.innerHTML = `<div class="banner bad"><pre>${esc(S.rw.error)}</pre></div>`; return; }
  box.innerHTML = `<div class="rw-layout">
    <div class="rw-main">
      <div class="settings-tools">
        <label class="search"><span class="sr">Search rewards</span>${ICON.search}<input id="rw-search" class="field" type="search" placeholder="Search rewards" title="Searches the names, what each pays for, and the folder names. Escape clears it." value="${esc(S.rwQuery)}"></label>
        <button type="button" class="btn primary small" data-act="rw-add">+ Add reward</button>
        <button type="button" class="btn small" data-act="rw-folder-new" title="Group rewards on this page. Only changes how this list looks: no Apply needed, training stays the same.">+ New folder</button>
        <button type="button" class="btn small" data-act="rw-load">Load a reward set</button>
        <button type="button" class="btn small" data-act="rw-save">Save as a reward set</button>
        <button type="button" class="btn small ghost" data-act="rw-guide">How the reward options work</button>
      </div>
      <div class="muted" style="font-size:13px;">${S.rw.in_file ? "" : "This bot uses the default rewards (its settings list none); a change writes them all into its settings. "}${rewardsSwitchTransfer() ? `It starts from ${esc(S.data.start.source)} taking everything: applying new rewards switches it to ${esc(S.data.start.source)}'s policy only (Settings tab, Transfer learning). ` : ""}Changes wait until you press Apply. Folders and order are for this page only and save at once.</div>
      ${(S.rw.general || []).length ? `<div class="banner bad"><pre>${esc(S.rw.general.join("\n"))}</pre></div>` : ""}
      <div id="rw-dups"></div>
      <div id="rw-list" class="rw-list"></div>
    </div>
    <aside class="rw-aside" aria-label="What each reward pays"><div id="rw-aside"></div><div id="rw-try"></div></aside>
  </div>`;
  renderRewardList(); renderShares(); loadTry();
}
function renderRewardList() {
  const list = $("#rw-list");
  if (!list || !S.rw || !S.rw.rewards) return;
  const p = placed(), q = S.rwQuery.trim(), low = q.toLowerCase();
  const card = (i, where, at, of) => rewardHTML(S.rw.rewards[i], i, { key: p.keys[i], where, first: at === 0, last: at === of - 1, folders: p.folders });
  const cards = (items, where) => items.filter(i => where.hit || rewardShown(S.rw.rewards[i])).map(i => card(i, where.id, items.indexOf(i), items.length)).join("");
  const folders = p.folders.map(({ f, items }) => {
    const hit = !!q && f.name.toLowerCase().includes(low), inner = cards(items, { id: f.id, hit });
    if (q && !inner) return "";
    const open = !!q || !f.collapsed;
    return `<section class="rw-folder${open ? "" : " closed"}" data-rwfolder="${esc(f.id)}">
      <div class="rw-folder-head" data-rwdrop="${esc(f.id)}">
        <button type="button" class="rw-fold" data-rwfold="${esc(f.id)}" aria-expanded="${open}" ${q ? 'title="Open while searching"' : ""}>${ICON.chev}<span class="name">${marked(f.name, q)}</span><span class="tag">${items.length} reward${items.length === 1 ? "" : "s"}</span></button>
        <button type="button" class="btn small ghost" data-rwfolderrename="${esc(f.id)}" aria-label="Rename the folder ${esc(f.name)}">Rename</button>
        <button type="button" class="iconbtn" data-rwfolderdel="${esc(f.id)}" title="Delete the folder (its rewards stay, out of any folder)" aria-label="Delete the folder ${esc(f.name)}">${ICON.trash}</button>
      </div>
      ${open ? `<div class="rw-folder-body" data-rwdrop="${esc(f.id)}">${inner || '<div class="rw-drop-hint">Empty. Drag rewards here, or use a reward\'s "Move to…".</div>'}</div>` : ""}
    </section>`;
  }).join("");
  const loose = cards(p.loose, { id: "", hit: false });
  const looseHead = p.folders.length && (loose || !q) ? `<div class="rw-loose-head" data-rwdrop=""><span class="name">Not in a folder</span><span class="tag">${p.loose.length}</span></div>` : "";
  const looseHint = p.folders.length && !p.loose.length && !q ?'<div class="rw-drop-hint">Drag a reward here to take it out of its folder.</div>' : "";
  const html = folders || loose || looseHint ? folders + looseHead + `<div class="rw-loose" data-rwdrop="">${loose || looseHint}</div>` : '<div class="empty">No reward matches that search.</div>';
  keepFocus(() => put(list, html));
  renderDuplicates();
}

// ---------- folders and order: a view only, kept in the bot's folder (reward_layout.py), no Apply ----------
function rwLayout() { if (!S.rw.layout) S.rw.layout = { folders: [], loose: [] }; return S.rw.layout; }
// The rewards as the list shows them: each folder with its rewards (their places in S.rw.rewards), then the
// rewards in no folder, those the layout does not name last, in the settings' order.
function placed() {
  const keys = S.rw.rewards.map((e, i) => rewardKey(i)), at = new Map(keys.map((k, i) => [k, i])), used = new Set();
  const take = list => (list || []).filter(k => at.has(k) && !used.has(k) && used.add(k)).map(k => at.get(k));
  const folders = rwLayout().folders.map(f => ({ f, items: take(f.rewards) }));
  const loose = take(rwLayout().loose);
  keys.forEach((k, i) => { if (!used.has(k)) loose.push(i); });
  return { folders, loose, keys };
}
// The layout with every reward shown in its place; keys of rewards not shown now (one removed, waiting for
// Apply) stay where they were, so a Discard brings them back to their folder.
function layoutNow() {
  const p = placed(), shown = new Set(p.keys), L = rwLayout();
  const rest = list => (list || []).filter(k => !shown.has(k));
  return {
    folders: p.folders.map(({ f, items }) => Object.assign({}, f, { rewards: items.map(i => p.keys[i]).concat(rest(f.rewards)) })),
    loose: p.loose.map(i => p.keys[i]).concat(rest(L.loose)),
  };
}
function listOf(L, where) { if (!where) return L.loose; const f = L.folders.find(x => x.id === where); return f ? f.rewards : null; }
function folderOf(key) { const f = rwLayout().folders.find(x => (x.rewards || []).includes(key)); return f || null; }
let rwLayoutSeq = 0;
async function saveLayout(L) {
  const bot = S.bot, seq = ++rwLayoutSeq, was = S.rw.layout;
  S.rw.layout = L; renderRewardList(); renderShares(); renderTry();
  const out = await call("set_reward_layout", bot, L);
  if (seq !== rwLayoutSeq || S.bot !== bot || !S.rw) return out;
  if (out.error) { S.rw.layout = was; toast(out.error, true); }
  else S.rw.layout = out.layout;
  renderRewardList(); renderShares(); renderTry();
  return out;
}
// Moves the reward `key` into the folder `to` ("" for no folder), before the reward `before` (null: at the end).
function moveReward(key, to, before) {
  if (before === key) return;
  const L = layoutNow();
  L.folders.forEach(f => { f.rewards = f.rewards.filter(k => k !== key); });
  L.loose = L.loose.filter(k => k !== key);
  const list = listOf(L, to);
  if (!list) return;
  const at = before ? list.indexOf(before) : -1;
  list.splice(at < 0 ? placedCount(to) : at, 0, key);
  return saveLayout(L);
}
// How many rewards of the list `where` show now (its keys of rewards not shown come after them).
function placedCount(where) {
  const p = placed();
  if (!where) return p.loose.length;
  const f = p.folders.find(x => x.f.id === where);
  return f ? f.items.length : 0;
}
// Moves the reward `key` one place up (-1) or down (1) in its list.
function stepReward(key, by) {
  const L = layoutNow(), f = L.folders.find(x => x.rewards.includes(key)), list = f ? f.rewards : L.loose;
  const at = list.indexOf(key), to = at + by;
  if (at < 0 || to < 0 || to >= placedCount(f ? f.id : "")) return;
  list.splice(at, 1); list.splice(to, 0, key);
  return saveLayout(L);
}
function folderModal(id, thenMove) {
  const f = id ? rwLayout().folders.find(x => x.id === id) : null;
  modal(`<h2 class="display">${f ? "Rename the folder" : "New folder"}</h2>
    <label class="f">Name<span class="h">Folders group the rewards on this page only: nothing changes in training, and no Apply is needed.</span><input id="rw-folder-name" class="field" type="text" autocomplete="off" maxlength="60" value="${esc(f ? f.name : "")}"></label>
    <div class="foot"><button type="button" class="btn" data-close="1">Cancel</button><button type="button" class="btn primary" data-act="rw-folder-go" data-id="${esc(id || "")}" data-move="${esc(thenMove || "")}">${f ? "Rename" : "Make the folder"}</button></div>`);
  const input = $("#rw-folder-name"); input.focus(); input.select();
}
function folderGo(t) {
  const name = $("#rw-folder-name").value.replace(/\s+/g, " ").trim(), id = t.dataset.id, L = layoutNow();
  if (!name) return toast("Give the folder a name.", true);
  if (L.folders.some(f => f.id !== id && f.name.toLowerCase() === name.toLowerCase())) return toast(`There is already a folder called '${name}'.`, true);
  if (id) { const f = L.folders.find(x => x.id === id); if (f) f.name = name; }
  else {
    let n = 1; while (L.folders.some(f => f.id === "f" + n)) n++;
    const f = { id: "f" + n, name, collapsed: false, rewards: [] };
    const move = t.dataset.move;
    if (move) { L.folders.forEach(x => { x.rewards = x.rewards.filter(k => k !== move); }); L.loose = L.loose.filter(k => k !== move); f.rewards.push(move); }
    L.folders.push(f);
  }
  closeModal();
  return saveLayout(L);
}
function deleteFolder(id) {
  const L = layoutNow(), f = L.folders.find(x => x.id === id);
  if (!f) return;
  L.folders = L.folders.filter(x => x !== f);
  L.loose = L.loose.concat(f.rewards);
  toast(`Folder '${f.name}' deleted. Its rewards are back in the list, out of any folder.`);
  return saveLayout(L);
}
function foldFolder(id, open) {
  const L = layoutNow(), f = L.folders.find(x => x.id === id);
  if (!f) return;
  f.collapsed = open == null ? !f.collapsed : !open;
  return saveLayout(L);
}
// Shows the reward at place `i`: out of a closed folder and past the search, scrolled to and lit up.
async function jumpToReward(i) {
  const key = rewardKey(i), f = folderOf(key);
  if (S.rwQuery && !rewardShown(S.rw.rewards[i])) { S.rwQuery = ""; const box = $("#rw-search"); if (box) box.value = ""; }
  if (f && f.collapsed) foldFolder(f.id, true);
  renderRewardList();
  const card = document.querySelector(`#rw-list [data-reward="${i}"]`);
  if (!card) return;
  card.scrollIntoView({ block: "center" });
  card.classList.remove("flash"); void card.offsetWidth; card.classList.add("flash");
  card.focus({ preventScroll: true });
  setTimeout(() => card.classList.remove("flash"), 1800);
}
// The warning for a reward listed more than once (each copy pays on its own), with the one click that removes
// the extra copies, keeping the first: a change like any other, waiting for Apply.
function renderDuplicates() {
  const box = $("#rw-dups");
  if (!box) return;
  const dups = S.rw.duplicates || [];
  put(box, dups.length ? `<div class="banner warn rw-dups" role="alert"><div class="rw-dups-in">
      <b>The same reward is in this bot's settings more than once.</b>
      <span class="muted">Each copy pays on its own, so it counts ${dups.some(d => d.indexes.length > 2) ? "several times" : "twice"}. Removing the extra copy keeps the first one; it waits for Apply like any change.</span>
      ${dups.map(d => `<div class="row"><span><b>${esc(kindOf(d.name).title)}</b> · ${d.indexes.length} copies</span><button type="button" class="btn small strong" data-rwdedupe="${d.indexes.join(",")}">Remove the extra cop${d.indexes.length > 2 ? "ies" : "y"}</button></div>`).join("")}
    </div></div>` : "");
}
function extraCopies() { return new Set((S.rw.duplicates || []).flatMap(d => d.indexes.slice(1))); }
function moveMenuHTML(i, place) {
  const k = kindOf(S.rw.rewards[i].view.name), key = place.key;
  const item = (attrs, label, off) => `<button type="button" role="menuitem" data-rwkey="${esc(key)}" ${attrs} ${off ? "disabled" : ""}>${label}</button>`;
  return `<div class="rw-menu" role="menu" aria-label="Move ${esc(k.title)}">
    ${item('data-rwmv="up"', "↑ Up", place.first)}${item('data-rwmv="down"', "↓ Down", place.last)}
    <span class="cap muted">To a folder</span>
    ${place.folders.map(({ f }) => item(`data-rwmv="to" data-to="${esc(f.id)}"`, esc(f.name) + (place.where === f.id ? " (here)" : ""), place.where === f.id)).join("")}
    ${place.folders.length ? item('data-rwmv="to" data-to=""', "No folder" + (place.where ? "" : " (here)"), !place.where) : ""}
    ${item('data-rwmv="new"', "New folder…")}
  </div>`;
}
function rwInput(i, key, value, label, extra) {
  const id = `rw-${i}-${key.replace(/\W/g, "-")}`;
  return `<label class="rw-field" for="${id}">${esc(label)}<input id="${id}" class="field mono" type="text" inputmode="decimal" data-rw="${i}" data-k="${esc(key)}" value="${esc(value == null ? "" : value)}" ${extra || ""}></label>`;
}
function rwCheck(i, key, on, label, title) {
  const id = `rw-${i}-${key.replace(/\W/g, "-")}`;
  return `<label class="rw-opt" for="${id}" ${title ? `title="${esc(title)}"` : ""}><input id="${id}" type="checkbox" data-rw="${i}" data-k="${esc(key)}" ${on ? "checked" : ""}>${esc(label)}</label>`;
}
// A reward's "Nerd explanation": a written text when there is one, then its name and settings as the file
// holds them and its block to copy (`view`: the reward as the card has it now).
function rewardBlock(k, view) {
  const lines = ["[[rewards]]", `name = ${JSON.stringify(k.name)}`, `weight = ${tomlValue("float", view.weight) || "1.0"}`];
  (k.params || []).forEach(p => {
    const v = (view.params || {})[p.path];
    if (v != null && v !== "" && !sameValue(v, p.default)) lines.push(`${p.path} = ${tomlValue(p.kind, v)}`);
  });
  if (view.zero_sum) lines.push(`zero_sum = { team_spirit = ${tomlValue("number", view.zero_sum.team_spirit) || "0"}, opponent_scale = ${tomlValue("number", view.zero_sum.opponent_scale) || "1"} }`);
  if (k.group) lines.push(`parts = [${(view.parts || []).map(p => `{ name = ${JSON.stringify(p.name)} }`).join(", ")}]`);
  if (view.enabled === false) lines.push("enabled = false");
  return lines.join("\n");
}
function rewardNerdHTML(k, view) {
  const about = p => [SETTING_KINDS[p.kind] || p.kind, p.default != null ? "default " + tomlValue(p.kind, p.default) : p.nullable ? "may be left out" : "", limitWords(p).toLowerCase(), (p.options || []).length ? "one of " + p.options.join(", ") : ""].filter(Boolean).join(" · ");
  const settings = [["weight", "A number · default 1.0 · what the reward's pay is multiplied by"]]
    .concat((k.params || []).map(p => [p.path, about(p)]))
    .concat([["zero_sum", "Left out, or { team_spirit = 0 to 1, opponent_scale = 0 or more }"], ["enabled", "true or false · default true"]]);
  return `<div class="nerd"><span class="cap">Nerd explanation</span>
    ${k.nerd ? `<div class="nerd-text">${nerdTextHTML(k.nerd)}</div>` : ""}
    <dl class="facts">
      <div><dt class="cap">Name in the file</dt><dd><span class="mono">${esc(k.name)}</span>, in a <span class="mono">[[rewards]]</span> block of the bot's settings file</dd></div>
      <div><dt class="cap">Paid</dt><dd>${k.every_step ? "On every step" : k.group ? "When every one of its parts pays" : "On the steps where it happens"}</dd></div>
      ${settings.map(([key, words]) => `<div><dt class="mono">${esc(key)}</dt><dd>${esc(words)}</dd></div>`).join("")}
    </dl>
    <span class="cap muted">In the bot's settings file</span>${codeBlockHTML(rewardBlock(k, view))}
    <span class="cap muted">Check the file, then carry on training</span>${codeBlockHTML(botCommands())}
  </div>`;
}
function rewardHTML(e, i, place) {
  const v = e.view, k = kindOf(v.name), on = v.enabled !== false;
  const changed = isChanged(e), fresh = !originalOf(e);
  const share = shareOf(e.label);
  const note = (S.data.notes || {})[noteKey(i)] || "";
  const zs = v.zero_sum, params = k.params || [];
  const partNames = (v.parts || []).map(p => p.name);
  const partChoices = ((S.rw.catalog) || []).filter(c => !c.group && !partNames.includes(c.name));
  const q = S.rwQuery, menu = S.rwMoveOpen === place.key;
  return `<section class="reward${on ? "" : " off"}${changed ? " changed" : ""}" data-reward="${i}" data-rwkey="${esc(place.key)}" data-rwin="${esc(place.where)}" tabindex="-1">
    <div class="head">
      <span class="iconbtn grip" role="button" tabindex="0" id="rw-${i}-grip" draggable="true" data-rwgrip="${esc(place.key)}" title="Drag to move it, or press the up and down arrow keys" aria-label="Move ${esc(k.title)}: drag it, or press the up and down arrow keys">${ICON.grip}</span>
      <input class="sw" type="checkbox" id="rw-${i}-enabled" data-rw="${i}" data-k="enabled" ${on ? "checked" : ""} aria-label="${esc(k.title)} on or off">
      <div class="who"><span class="title">${marked(k.title, q)}</span><span class="kind mono">${marked(v.name, q)}</span>${fresh ? '<span class="badge new">new</span>' : changed ? '<span class="badge hot">changed</span>' : ""}${on ? "" : '<span class="badge">off</span>'}${k.every_step ? '<span class="badge" title="Paid on every step">every step</span>' : ""}${ownBadge(k)}${extraCopies().has(i) ? '<span class="tag bad" title="The same reward is listed above too">extra copy</span>' : ""}</div>
      ${share ? `<span class="sharechip${alerted(e.label) ? " warn" : ""}" title="Share of all reward paid, ${esc(windowWords())}">${pct(share.share)}</span>` : ""}
      ${changed && !fresh ? `<button type="button" class="iconbtn" data-rwundo="${i}" title="Undo the changes to this reward" aria-label="Undo the changes to ${esc(k.title)}">${ICON.reset}</button>` : ""}
      <button type="button" class="iconbtn" data-rwdel="${i}" title="Remove this reward" aria-label="Remove ${esc(k.title)}">${ICON.trash}</button>
    </div>
    ${k.about ? `<div class="about${S.rwAbout[i] ? " full" : ""}">${marked(k.about, q)}</div>` : ""}
    <div class="links">
      ${k.more || (k.about || "").length > 150 ? `<button type="button" class="more" data-rwabout="${i}" aria-expanded="${!!S.rwAbout[i]}">${S.rwAbout[i] ? "Show less" : "See more"}</button>` : ""}
      <button type="button" class="more" data-rwnerd="${i}" aria-expanded="${!!S.rwNerd[i]}">${S.rwNerd[i] ? "Close the nerd explanation" : "Nerd explanation"}</button>
      ${S.rwNotesOpen[i] ? "" : `<button type="button" class="more" data-rwnote="${i}">${note ? "Change the note" : "Add a note"}</button>`}
      <span class="rw-move"><button type="button" class="more" id="rw-${i}-move" data-rwmove="${esc(place.key)}" aria-haspopup="menu" aria-expanded="${menu}">Move to…</button>${menu ? moveMenuHTML(i, place) : ""}</span>
    </div>
    ${S.rwAbout[i] && k.more ? `<div class="guide">${guideHTML(k.more)}</div>` : ""}
    ${S.rwNerd[i] ? rewardNerdHTML(k, v) : ""}
    ${!v.known ? `<div class="rw-problems">'${esc(v.name)}' is not a reward NitroLearn knows.</div>` : ""}
    <div class="rw-grid">
      ${rwInput(i, "weight", v.weight, "Weight", `title="${esc((S.rw.help || {}).weight || "")}"`)}
      ${params.map(p => { const set = (v.params || {})[p.path], shown = set == null ? p.default : set;  // a setting left out of the file takes its default
        return p.kind === "bool" ? rwCheck(i, "params." + p.path, !!shown, p.label, p.help) : rwInput(i, "params." + p.path, set, p.label + (p.nullable && p.default == null ? " (optional)" : ""), `title="${esc(p.help)}"` + (set == null && p.default != null ? ` placeholder="${esc(p.default)}"` : "")); }).join("")}
      ${rwCheck(i, "zero_sum.on", !!zs, "Zero-sum", (S.rw.help || {}).zero_sum || "Shares the reward with teammates and subtracts what the opponents get, so it cannot be farmed when it helps nobody.")}
    </div>
    ${zs ? `<div class="rw-sub"><span class="lead">Zero-sum: team spirit 0 = each player keeps their own reward, 1 = the team shares its average. The opponents' average is subtracted, times the opponent share.</span>
      ${rwInput(i, "zero_sum.team_spirit", zs.team_spirit, "Team spirit (0 to 1)", `title="${esc((S.rw.help || {})["zero_sum.team_spirit"] || "")}"`)}
      ${rwInput(i, "zero_sum.opponent_scale", zs.opponent_scale, "Opponent share", `title="${esc((S.rw.help || {})["zero_sum.opponent_scale"] || "")}"`)}
    </div>` : ""}
    ${k.group ? `<div class="rw-sub"><span class="lead">Parts: pays only when every part is positive (an "and"). At least two.</span>
      ${(v.parts || []).map((p, j) => `<span class="chip">${esc(kindOf(p.name).title)}<button type="button" class="more" data-rwpartdel="${i}:${j}" aria-label="Remove part ${esc(p.name)}">×</button></span>`).join("")}
      <select class="field" style="width:auto;" id="rw-${i}-part-add" data-rw="${i}" data-k="parts.add" aria-label="Add a part"><option value="">+ add a part…</option>${partChoices.map(c => `<option value="${esc(c.name)}">${esc(c.title)}</option>`).join("")}</select>
    </div>` : ""}
    ${(e.problems || []).length ? `<div class="rw-problems" role="alert">${e.problems.map(p => `<span>${esc(p)}</span>`).join("")}</div>` : ""}
    ${note && !S.rwNotesOpen[i] ? `<div class="notetext">${esc(note)}</div>` : ""}
    ${S.rwNotesOpen[i] ? `<div class="notebox"><textarea class="field" id="rw-note-${i}" aria-label="Note for ${esc(k.title)}">${esc(draftOr(noteKey(i), note))}</textarea><div class="row"><button type="button" class="btn small primary" data-rwnotesave="${i}">Save note</button><button type="button" class="btn small" data-rwnoteclose="${i}">Close</button>${noteStateHTML(noteKey(i))}</div></div>` : ""}
  </section>`;
}

// The folder a ranked reward is in, as a small tag.
function folderTag(label) {
  const i = ((S.rw && S.rw.rewards) || []).findIndex(x => x.label === label), f = i >= 0 ? folderOf(rewardKey(i)) : null;
  return f ? `<span class="tag rw-in" title="In the folder ${esc(f.name)}">${esc(f.name)}</span> ` : "";
}
// A zero-sum reward's size before zero-sum (what the bots earn themselves) beside its share after it.
function zeroSumLine(r) {
  const e = ((S.rw && S.rw.rewards) || []).find(x => x.label === r.label);
  if (!e || !e.view.zero_sum || r.own_share == null) return "";
  return `<span class="line" title="Zero-sum mixes in the teammates' and opponents' pay, so what the bots earn from it themselves can be larger or smaller than what it pays in the end.">Before zero-sum: ${pct(r.own_share)} (earns ${fmt(r.own_per_step)} a step)</span>`;
}
function windowWords() { return "the last " + ((WINDOWS.find(w => w[0] === S.rwWindow) || [0, ""])[1] || "").toLowerCase().replace("last ", ""); }
// How the two reward lists are sorted and filtered: view only. Kept in the browser's storage and in the
// dashboard's preferences, as pywebview's private window forgets its storage when it closes.
const RW_SORTS = [["share", "Share of reward"], ["fires", "How often it triggers"], ["raw", "Size before its weight"], ["az", "A–Z"]];
const CONTINUOUS_RATE = 0.5;  // a reward the catalog does not mark "every step" counts as continuous above this
const VIEW_KEY = "nitro.rewardView";
function rwView() {
  if (S.rwView) return S.rwView;
  let v = null;
  try { v = JSON.parse(localStorage.getItem(VIEW_KEY) || "null"); } catch (err) { v = null; }
  v = v || (S.home && S.home.prefs && S.home.prefs.reward_view) || {};
  S.rwView = { sort: RW_SORTS.some(o => o[0] === v.sort) ? v.sort : "share", bots: v.bots && typeof v.bots === "object" ? v.bots : {} };
  return S.rwView;
}
function saveView() {
  try { localStorage.setItem(VIEW_KEY, JSON.stringify(rwView())); } catch (err) { /* no storage here */ }
  call("set_pref", "reward_view", rwView());
  renderShares(); renderTry();
}
function botView() { const b = rwView().bots[S.bot] || {}; return { hideContinuous: !!b.hide_continuous, hidden: Array.isArray(b.hidden) ? b.hidden : [] }; }
function setBotView(change) {
  const v = botView();
  rwView().bots[S.bot] = Object.assign({ hide_continuous: v.hideContinuous, hidden: v.hidden }, change);
  saveView();
}
// A metrics label's reward: `speed_2` is the second speed reward unless a reward is called speed_2.
function labelKind(label) {
  const m = /^(.*?)(?:_(\d+))?$/.exec(label), base = kindOf(label).about ? label : m[1];
  return { k: kindOf(base), n: base !== label ? m[2] : "" };
}
function rewardTitle(label) { const { k, n } = labelKind(label); return k.about ? k.title + (n ? " " + n : "") : label; }
function isContinuous(r) { return !!labelKind(r.label).k.every_step || (r.fire_rate != null && r.fire_rate > CONTINUOUS_RATE); }
// The rows to show, sorted; with some hidden, the shares are over the shown rows (adding up to 100%).
function arrange(list) {
  const v = botView(), hidden = new Set(v.hidden);
  const off = r => hidden.has(r.label) || (v.hideContinuous && isContinuous(r));
  const kept = list.filter(r => !off(r)), filtered = kept.length < list.length;
  const total = key => kept.reduce((a, r) => a + (r[key] || 0), 0);
  const sums = { share: total("share"), share_before: total("share_before"), own_share: total("own_share") };
  const rows = kept.map(r => {
    if (!filtered) return r;
    const o = Object.assign({}, r);
    Object.keys(sums).forEach(key => { if (o[key] != null) o[key] = sums[key] ? o[key] / sums[key] : 0; });
    return o;
  });
  const size = x => x == null ? -Infinity : Math.abs(x);  // rows without the number go last
  const by = {
    share: (a, b) => b.share - a.share,
    fires: (a, b) => size(b.fires_per_minute) - size(a.fires_per_minute) || b.share - a.share,
    raw: (a, b) => size(b.raw_per_step) - size(a.raw_per_step) || b.share - a.share,
    az: (a, b) => rewardTitle(a.label).localeCompare(rewardTitle(b.label)),
  }[rwView().sort];
  rows.sort(by);
  return { rows, filtered, hidden: list.filter(r => hidden.has(r.label)), continuous: v.hideContinuous ? list.filter(r => !hidden.has(r.label) && isContinuous(r)).length : 0 };
}
function viewControls(id) {
  const v = botView();
  return `<div class="rw-view"><label class="row" style="gap:6px;" for="${id}-sort">Sort by <select id="${id}-sort" class="field" data-rwsort="1">${RW_SORTS.map(([k, l]) => `<option value="${k}" ${rwView().sort === k ? "selected" : ""}>${l}</option>`).join("")}</select></label>
    <label class="rw-opt" for="${id}-cont" title="Hides rewards marked &quot;every step&quot; (paid on every step, like speed toward the ball) and any other reward that fires on more than ${Math.round(CONTINUOUS_RATE * 100)}% of the steps. View only: training still pays them."><input id="${id}-cont" type="checkbox" data-rwcont="1" ${v.hideContinuous ? "checked" : ""}>Hide continuous rewards</label></div>`;
}
function hiddenLine(view) {
  const n = view.hidden.length;
  const said = view.filtered ? `<span>Percentages are the share of shown rewards, so they add up to 100%${view.continuous ? ` (${view.continuous} continuous hidden)` : ""}. View only: training still pays them all.</span>` : "";
  const button = n ? `<div><button type="button" class="btn small ghost" data-rwshowhidden="1" aria-expanded="${S.rwShowHidden}">${S.rwShowHidden ? "Close the hidden list" : `Show ${n} hidden`}</button></div>` : "";
  const list = n && S.rwShowHidden ? `<div class="chips">${view.hidden.map(r => `<button type="button" class="chip" data-rwunhide="${esc(r.label)}" title="Show it again">${esc(rewardTitle(r.label))} ↺</button>`).join("")}<button type="button" class="chip" data-rwunhide="">Show all</button></div>` : "";
  return said || n ? `<div class="muted rw-hidden">${said}${button}${list}</div>` : "";
}
// How often a reward fires before its weight: in a minute of one car's play, with the share of steps when high.
function firesLine(r) {
  if (r.fires_per_minute == null) return "";
  const x = r.fires_per_minute, share = r.fire_rate >= 0.1 ? ` (${pct(r.fire_rate)} of steps)` : "";
  return `fires ${num(x, x >= 10 ? 0 : x >= 1 ? 1 : 2)} a minute${share}`;
}
function hideButton(r) { return `<button type="button" class="iconbtn hide" data-rwhide="${esc(r.label)}" title="Hide from these lists (view only)" aria-label="Hide ${esc(rewardTitle(r.label))}">${ICON.eyeOff}</button>`; }
function renderShares() {
  const box = $("#rw-aside");
  if (!box || !S.rw) return;
  const sh = S.rw.shares || {}, all = sh.rewards || [], alerts = sh.alerts || [];
  const view = arrange(all), ranked = view.rows, of = view.filtered ? "share of shown rewards" : "share of all reward paid";
  box.innerHTML = `<section class="card pad rw-panel">
    <div class="heading"><h3>What each reward pays</h3></div>
    <div class="seg" data-seg="rwwin">${WINDOWS.map(([s, l]) => `<button type="button" data-v="${s}" aria-pressed="${S.rwWindow === s}">${l}</button>`).join("")}</div>
    <div class="muted" style="font-size:12.5px;">Each reward's share of all reward paid over ${esc(windowWords())} of training, by size (a penalty counts too), after zero-sum; a zero-sum reward also shows its share before it. One share that dominates, or shoots up, can mean the bot farms it instead of playing. "Fires" counts how often it pays before its weight, so a small weight does not hide how often it happens.</div>
    ${alerts.map(a => `<div class="banner warn" style="font-size:13px;">${esc(a.text)}</div>`).join("")}
    ${all.length ? viewControls("rw-sh") : ""}
    ${all.length && sh.fires_note ? `<div class="muted" style="font-size:12px;">${esc(sh.fires_note)}</div>` : ""}
    ${all.length ? hiddenLine(view) : ""}
    ${ranked.length ? `<div class="shares">${ranked.map(r => {
        const trend = r.share_before == null ? "" : r.share > r.share_before * 1.25 + 0.01 ? '<span class="up" title="Up since the window before">▲</span>' : r.share < r.share_before * 0.8 - 0.01 ? '<span class="down" title="Down since the window before">▼</span>' : "";
        const pays = r.paid_per_step, fires = firesLine(r);
        return `<div class="share hideable"><span class="n">${esc(rewardTitle(r.label))}</span><span class="pct num" title="${of}">${trend} ${pct(r.share)}</span>${hideButton(r)}
          <div class="bar${alerted(r.label) ? " warn" : ""}"><span style="width:${Math.max(1, Math.min(100, r.share * 100)).toFixed(1)}%"></span></div>
          <span class="line">${folderTag(r.label)}${Math.abs(pays) < 1e-9 && r.size_per_step > 0 ? "evens out between the teams" : (pays >= 0 ? "pays " : "costs ") + fmt(Math.abs(pays)) + " a step"}${r.share_before != null ? " · was " + pct(r.share_before) : ""}${fires ? " · " + fires : ""}</span>${zeroSumLine(r)}</div>`;
      }).join("")}</div>
      <div class="muted" style="font-size:12px;">Over ${num(sh.seconds, 0)} s of training (${steps(sh.agent_steps)} steps)${sh.updated ? ", updated " + esc(ago(sh.updated)) : ""}.</div>`
      : all.length ? `<div class="muted">Every reward is hidden.</div>` : `<div class="muted">${esc(sh.note || "No training yet.")}</div>`}
  </section>`;
}
// The rewards tried on the frozen bot (nitro reward-check).
async function loadTry() {
  const bot = S.bot;
  const out = await call("reward_check", bot);
  if (S.bot !== bot || out.error && !("playing" in out)) return;
  const was = S.rwTry && S.rwTry.bot === bot && S.rwTry.playing;
  S.rwTry = Object.assign(out, { bot });
  renderTry();
  clearTimeout(S.rwTryTimer);  // one poll at a time, however often the tab is drawn
  if (out.playing) S.rwTryTimer = setTimeout(() => { if (S.view === "bot" && S.tab === "rewards" && S.bot === bot) loadTry(); }, 2000);
  else if (was && out.stopped) toast("Stopped trying the rewards on " + bot + ". Nothing of it changed.");
  else if (was && !out.error) toast("The rewards were tried on " + bot + ".");
}
// How long the check takes: while it plays, the time left once a round is done; before, a rough guess from
// the bot's training speed (playing alone takes about 60% of a round of training).
const TRY_STEPS = [[250000, "250,000 steps"], [1000000, "1 million steps"], [5000000, "5 million steps"]];
function tryTimeWord(t, b) {
  if (t.playing) return t.seconds_left != null ? "About " + minutesWord(t.seconds_left) + " left" : "Setting up the games…";
  const speed = b.steps_per_second;
  return speed ? "Roughly " + minutesWord(0.6 * S.rwTrySteps / speed) + " at this bot's speed" : "";
}
function minutesWord(seconds) { const m = Math.round(seconds / 60); return m < 1 ? "under a minute" : m === 1 ? "1 minute" : m + " minutes"; }
function renderTry() {
  const box = $("#rw-try");
  if (!box || !S.rwTry || S.rwTry.bot !== S.bot) return;
  const t = S.rwTry, r = t.result, b = S.data || {};
  const busy = b.status === "training" || b.status === "copying" || b.status === "starting";
  const why = busy ? "Stop the bot first: it is running." : !b.saves ? (b.copy_first ? "Press Copy replays first: the check plays the copied bot." : "It has no save yet.") : "";
  const all = r && r.ranked ? r.ranked : [], view = arrange(all), tried = view.rows;
  const counted = all.some(x => x.fires_per_minute != null);
  put(box, `<section class="card pad rw-panel">
    <div class="heading"><h3>Try these rewards</h3></div>
    <div class="muted" style="font-size:12.5px;">The bot as last saved plays itself, learning nothing (both networks frozen), and this says what each reward above would pay it, waiting changes included. Balance the weights here before training. Nothing of the bot changes.</div>
    <div class="row">
      <select id="rw-try-steps" class="field" style="width:auto;" aria-label="How long to play" ${t.playing ? "disabled" : ""}>${TRY_STEPS.map(([v, l]) => `<option value="${v}" ${S.rwTrySteps === v ? "selected" : ""}>${l}</option>`).join("")}</select>
      <button type="button" class="btn primary small" data-act="rw-try" ${t.playing || why ? "disabled" : ""}>${t.playing ? "Playing…" + (t.progress != null ? " " + Math.round(t.progress * 100) + "%" : "") : "Try these rewards"}</button>
      ${t.playing ? '<button type="button" class="btn small" data-act="rw-try-stop">Stop</button>' : ""}
      <span class="muted" id="rw-try-time" style="font-size:12.5px;">${esc(tryTimeWord(t, b))}</span>
    </div>
    ${why && !t.playing ? `<div class="muted" style="font-size:12.5px;">${esc(why)}</div>` : ""}
    ${t.error && !t.playing ? `<div class="banner bad" style="font-size:13px;">${esc(t.error)}</div>` : ""}
    ${(r && r.alerts || []).map(a => `<div class="banner${a.level === "warning" ? " warn" : ""}" style="font-size:13px;"><span>${esc(a.text)}</span></div>`).join("")}
    ${all.length ? viewControls("rw-try") : ""}
    ${all.length && !counted ? `<div class="muted" style="font-size:12px;">Trigger counts appear once the check is tried with this version of NitroLearn.</div>` : ""}
    ${all.length ? hiddenLine(view) : ""}
    ${tried.length ? `<div class="shares">${tried.map(x => `<div class="share hideable"><span class="n">${esc(rewardTitle(x.label))}</span><span class="pct num" title="${view.filtered ? "share of shown rewards" : "share of all reward paid"}">${pct(x.share)}</span>${hideButton(x)}
        <div class="bar"><span style="width:${Math.max(1, Math.min(100, x.share * 100)).toFixed(1)}%"></span></div>
        <span class="line">${folderTag(x.label)}${Math.abs(x.paid_per_step) < 1e-9 && x.size_per_step > 0 ? "evens out between the teams" : (x.paid_per_step >= 0 ? "pays " : "costs ") + fmt(Math.abs(x.paid_per_step)) + " a step, " + fmt(Math.abs(x.paid_per_minute)) + " a minute"} · before its weight ${fmt(x.raw_per_step)} a step${firesLine(x) ? " · " + firesLine(x) : ""}</span></div>`).join("")}</div>` : all.length ? `<div class="muted">Every reward is hidden.</div>` : ""}
    ${all.length ? `<div class="muted" style="font-size:12px;">${esc(r.bot)} as saved at ${steps(r.saved_steps)} steps, over ${steps(r.agent_steps)} steps (${num(r.minutes_of_play, 0)} minutes of one car's play), ${esc(ago(r.when))}.</div>` : ""}
  </section>`);
}
async function refreshShares() {
  if (S.view !== "bot" || S.tab !== "rewards" || !S.rw || !S.rw.rewards) return;
  const out = await call("reward_shares", S.bot, S.rwWindow);
  if (!out.error && S.rw) { S.rw.shares = out; renderShares(); }
}
function newView(k) {
  const view = { name: k.name, known: true, enabled: true, weight: 1, zero_sum: null, params: {} };
  (k.params || []).forEach(p => { view.params[p.path] = k.own ? null : p.default; });  // a reward of one's own: its defaults, unwritten
  if (k.group) view.parts = [];
  return view;
}
function rewardEdit(i, key, raw) {
  const e = S.rw.rewards[i];
  if (!e) return;
  const v = e.view;
  if (key === "enabled") v.enabled = !!raw;
  else if (key === "weight") v.weight = raw;
  else if (key.startsWith("params.")) v.params[key.slice(7)] = raw;
  else if (key === "zero_sum.on") v.zero_sum = raw ? { team_spirit: 0, opponent_scale: 1 } : null;
  else if (key.startsWith("zero_sum.")) { if (v.zero_sum) v.zero_sum[key.slice(9)] = raw; }
  else if (key === "parts.add") { if (raw) { const was = v.parts; v.parts = [...(v.parts || []), { name: raw }]; if (!partsKept(v, was)) return renderRewardList(); } }
  return stageRewards();
}
// A group whose parts now match another group's is two of the same reward: its parts go back as they were.
function partsKept(view, was) {
  if (!addedCopies(S.rw.rewards).length) return true;
  view.parts = was;
  toast("Another 'All of' group has exactly these parts: a reward can be in a bot only once.", true);
  return false;
}
// Edits go to Python one at a time, each with every reward as it stands; an answer overtaken by a newer edit
// is dropped (the newer one carries everything).
let rwSeq = 0, rwChain = Promise.resolve();
function stageRewards() {
  const bot = S.bot, seq = ++rwSeq;
  rwChain = rwChain.then(async () => {
    if (seq !== rwSeq || S.bot !== bot || !S.rw || !S.rw.rewards) return;
    const out = await call("stage_rewards", bot, S.rw.rewards.map(e => ({ from: e.from, view: e.view })));
    if (seq !== rwSeq || S.bot !== bot) return;
    if (out.error) return toast(out.error, true);
    Object.assign(S.rw, out);
    S.data.reward_changes = out.changes || [];
    renderRewardList(); renderPending(); updateTabs();
    if ((out.general || []).length) renderRewards($("#tab"));
  });
  return rwChain;
}
function addRewardModal() {
  S.rwCatQuery = ""; S.rwCatGroup = "";
  modal(`<h2 class="display">Add a reward</h2>
    <label class="search" style="max-width:none;"><span class="sr">Search the rewards</span>${ICON.search}<input id="rw-cat-q" class="field" type="search" placeholder="Search (boost, touch, aerial, …)"></label>
    <div class="seg" id="rw-cat-groups" role="group" aria-label="Which rewards to list"></div>
    <div class="catalog" id="rw-catalog"></div>
    <div class="muted" style="font-size:12.5px;">"Trick moves" pay for one trick move each; "More rewards" are the others. Both come with NitroLearn (the list, with what each pays and who made it: docs/rewards.md). "Yours" are compiled in from your own Rust files (Resources page).</div>
    <div class="foot"><button type="button" class="btn" data-close="1">Close</button></div>`);
  $(".modal").classList.add("wide");
  renderCatalog();
  $("#rw-cat-q").focus();
}
// Built in, then the rewards that come with NitroLearn (each file a family) in their two groups, then the
// user's own files.
const CATALOG_GROUPS = [["builtin", "Built in"], ["moves", "Trick moves (they come with NitroLearn)"], ["library", "More rewards (they come with NitroLearn)"], ["own", "Your own Rust files"]];
function catalogGroup(k) { return !k.own || k.source === "example" ? "builtin" : k.source === "library" ? (k.trick_move ? "moves" : "library") : "own"; }
function ownBadge(k) { return !k.own || k.source === "example" ? "" : k.source === "library" ? `<span class="badge new" title="${esc(k.file || "")}">${k.trick_move ? "trick move" : "comes with NitroLearn"} · ${esc(k.family || "")}</span>` : `<span class="badge new" title="${esc(k.file || "")}">yours</span>`; }
// The rewards the search finds, in every group.
function catalogKinds() { return ((S.rw && S.rw.catalog) || []).filter(k => rewardMatches(k, S.rwCatQuery)); }
// The Add reward list and the row that picks one group of it, each with how many rewards the search finds.
function renderCatalog() {
  const box = $("#rw-catalog");
  if (!box) return;
  const kinds = catalogKinds(), chosen = S.rwCatGroup || "";
  put($("#rw-cat-groups"), [["", "All"]].concat(CATALOG_GROUPS).map(([g, title]) => {
    const n = g ? kinds.filter(k => catalogGroup(k) === g).length : kinds.length;
    return g && !n && chosen !== g ? "" : `<button type="button" data-rwcatgroup="${g}" aria-pressed="${chosen === g}">${esc(title.split(" (")[0])} <span class="n">${n}</span></button>`;
  }).join(""));
  box.innerHTML = catalogHTML(); box.scrollTop = 0;
}
function catalogHTML() {
  const kinds = catalogKinds(), chosen = S.rwCatGroup || "";
  const used = new Set(((S.rw && S.rw.rewards) || []).map(e => e.view.name)), q = S.rwCatQuery;
  // A reward the bot has already: shown as Added, and picking it shows the bot's own (an "all of" group can
  // be added again with other parts).
  const added = k => !k.group && used.has(k.name);
  const item = k => `<div class="catalog-item${added(k) ? " added" : ""}"><button type="button" class="catalog-pick" ${added(k) ? `data-rwjump="${esc(k.name)}" title="Already in this bot: show it"` : `data-rwadd="${esc(k.name)}" title="Add ${esc(k.title)}"`}>
      <span class="row" style="gap:8px;"><b>${marked(k.title, q)}</b><span class="kind mono muted" style="font-size:11.5px;">${marked(k.name, q)}</span>${k.every_step ? '<span class="badge">every step</span>' : ""}${k.group ? '<span class="badge">group</span>' : ""}${ownBadge(k)}${added(k) ? '<span class="tag strong">✓ Added</span>' : used.has(k.name) ? '<span class="badge">in use</span>' : ""}</span>
      <span class="about">${marked(k.about, q)}</span><span class="pick-word" aria-hidden="true">${added(k) ? "Show it in the list" : "+ Add to this bot"}</span></button>
      <span class="links">${k.more ? `<button type="button" class="more" data-rwcatmore="${esc(k.name)}" aria-expanded="${!!S.rwCatMore[k.name]}">${S.rwCatMore[k.name] ? "Show less" : "See more"}</button>` : ""}<button type="button" class="more" data-rwcatnerd="${esc(k.name)}" aria-expanded="${!!S.rwCatNerd[k.name]}">${S.rwCatNerd[k.name] ? "Close the nerd explanation" : "Nerd explanation"}</button></span>
      ${k.more && S.rwCatMore[k.name] ? `<div class="guide" style="max-width:none;">${guideHTML(k.more)}</div>` : ""}
      ${S.rwCatNerd[k.name] ? rewardNerdHTML(k, newView(k)) : ""}</div>`;
  const html = CATALOG_GROUPS.filter(([g]) => !chosen || g === chosen).map(([g, title]) => {
    const list = kinds.filter(k => catalogGroup(k) === g);
    if (g === "library" || g === "moves") list.sort((a, b) => (a.family || "").localeCompare(b.family || "") || a.name.localeCompare(b.name));
    return list.length ? `<div class="catalog-head">${esc(title)} <span class="muted">${list.length}</span></div>` + list.map(item).join("") : "";
  }).join("");
  return html || '<div class="muted" style="padding:14px 2px;">No reward matches that search.</div>';
}

async function rewardSetModal() {
  const out = await call("resources");
  if (out.error) return toast(out.error, true);
  const sets = (out.files || []).filter(f => f.kind === "rewards");
  // Grouped as the Add reward list: the two groups that come with NitroLearn, then added or saved.
  const setGroups = [["Trick moves", s => s.built_in && s.trick_move], ["More reward sets", s => s.built_in && !s.trick_move], ["Added or saved", s => !s.built_in]];
  const settingItem = s => `<div class="catalog-item" style="cursor:default;"><span class="row" style="gap:8px;"><b>${esc(s.title)}</b>${s.built_in ? `<span class="badge new">${s.trick_move ? "trick move · " : ""}comes with NitroLearn</span>` : ""}<span class="mono muted" style="font-size:11.5px;">${esc(s.path)}</span></span><span class="about">${esc(s.about)}</span>${(out.notes || {})["file:" + s.id] ? `<span class="notetext" style="margin-top:0;">${esc(out.notes["file:" + s.id])}</span>` : ""}${s.problem ? `<span class="about" style="color:var(--bad-text);">${esc(s.problem)}</span>` : `<span class="row"><button type="button" class="btn small strong" data-rwset="${esc(s.id)}" data-how="replace">Use these instead</button><button type="button" class="btn small" data-rwset="${esc(s.id)}" data-how="add">Add them to this bot's</button></span>`}</div>`;
  const setList = setGroups.map(([title, test]) => { const list = sets.filter(test); return list.length ? `<div class="catalog-head">${esc(title)} <span class="muted">${list.length}</span></div>` + list.map(settingItem).join("") : ""; }).join("");
  modal(`<h2 class="display">Load a reward set</h2>
    <div class="muted">Reward sets are added on the Resources page, or saved from a bot here. Loading one is a change like any other: it waits for Apply.</div>
    <div class="catalog">${setList || '<div class="muted">No reward sets yet.</div>'}</div>
    <div class="foot"><button type="button" class="btn" data-close="1">Close</button></div>`);
  $(".modal").classList.add("wide");
}
async function useRewardSet(ident, how) {
  const out = await call("reward_set", ident);
  if (out.error) return toast(out.error, true);
  // A reward the bot (or the set itself) has already is left out: a reward can be in a bot only once.
  const have = new Set(how === "replace" ? [] : S.rw.rewards.map(e => rewardIdentity(e.view)).filter(Boolean));
  const fresh = [], left = [];
  (out.rewards || []).forEach(view => {
    const id = rewardIdentity(view);
    if (id && have.has(id)) left.push(kindOf(view.name).title);
    else { fresh.push({ from: null, view }); if (id) have.add(id); }
  });
  closeModal();
  if (how !== "replace" && !fresh.length) return toast(`Nothing added: this bot has every reward of that set already (${left.join(", ")}).`);
  S.rw.rewards = how === "replace" ? fresh : S.rw.rewards.concat(fresh);
  S.rwNotesOpen = {}; S.rwAbout = {}; S.rwNerd = {};
  await stageRewards(); renderRewards($("#tab"));
  const skipped = left.length ? ` Left out, as ${how === "replace" ? "the set lists them twice" : "this bot has them already"}: ${left.join(", ")}.` : "";
  toast((how === "replace" ? "Reward set loaded in place of the bot's rewards; Apply to keep it." : `Reward set added (${fresh.length} reward${fresh.length === 1 ? "" : "s"}); Apply to keep it.`) + skipped);
}
function saveSetModal() {
  modal(`<h2 class="display">Save as a reward set</h2>
    <label class="f">Name<span class="h">Saved in my-settings\\reward-sets, and listed on the Resources page.</span><input id="rs-name" class="field" type="text" autocomplete="off" value="${esc(S.bot)} rewards"></label>
    <div class="muted" style="font-size:13px;">It saves the rewards as this tab shows them, waiting changes included.</div>
    <div class="foot"><button type="button" class="btn" data-close="1">Cancel</button><button type="button" class="btn primary" data-act="rw-save-go">Save</button></div>`);
}

ACTIONS["rw-try"] = async () => {
  const chosen = S.rwTrySteps;
  const out = await call("try_rewards", S.bot, chosen);
  if (out.error) return toast(out.error, true);
  S.rwTry = Object.assign(S.rwTry || {}, { playing: true, progress: 0, error: "", steps: chosen });
  renderTry(); toast("Playing " + S.bot + " with these rewards, learning nothing…");
  clearTimeout(S.rwTryTimer);
  S.rwTryTimer = setTimeout(loadTry, 1500);
  return;
};
ACTIONS["rw-try-stop"] = async () => {
  const out = await call("stop_job", S.bot, "check");
  if (out.error) return toast(out.error, true);
  clearTimeout(S.rwTryTimer);
  return loadTry();
};
ACTIONS["rw-add"] = () => addRewardModal();
ACTIONS["rw-guide"] = () => guideModal("How the reward options work", Object.entries((S.rw && S.rw.guide) || {}).map(([k, v]) => [GUIDE_NAMES[k] || k, v]));
ACTIONS["rw-load"] = () => rewardSetModal();
ACTIONS["rw-save"] = () => saveSetModal();
ACTIONS["rw-folder-new"] = () => folderModal("");
ACTIONS["rw-folder-go"] = t => folderGo(t);
ACTIONS["rw-save-go"] = async () => {
  const out = await call("save_reward_set", S.bot, $("#rs-name").value.trim(), S.rw.rewards.map(e => ({ from: e.from, view: e.view })));
  if (out.error) return toast(out.error, true);
  closeModal(); return toast("Saved " + out.saved + ".");
};
on("click", t => {
  if (t.dataset.rwset) return useRewardSet(t.dataset.rwset, t.dataset.how);
  if (t.dataset.rwjump != null || t.dataset.rwadd != null && !kindOf(t.dataset.rwadd).group && rewardIndex(t.dataset.rwadd) >= 0) {
    const i = rewardIndex(t.dataset.rwjump != null ? t.dataset.rwjump : t.dataset.rwadd);
    closeModal();
    return i >= 0 ? jumpToReward(i) : undefined;
  }
  if (t.dataset.rwadd != null) {
    const k = kindOf(t.dataset.rwadd);
    S.rw.rewards.push({ from: null, view: newView(k) });
    closeModal(); S.rwQuery = "";
    return stageRewards().then(() => { renderRewards($("#tab")); jumpToReward(S.rw.rewards.length - 1); });
  }
  if (t.dataset.rwdedupe != null) {
    const at = t.dataset.rwdedupe.split(",").map(Number), name = (S.rw.rewards[at[0]] || { view: {} }).view.name;
    at.slice(1).sort((a, b) => b - a).forEach(i => S.rw.rewards.splice(i, 1));
    S.rwNotesOpen = {}; S.rwAbout = {}; S.rwNerd = {};
    toast(`Removed the extra cop${at.length > 2 ? "ies" : "y"} of ${kindOf(name).title}; Apply to keep it.`);
    return stageRewards();
  }
  if (t.dataset.rwmove != null) {
    S.rwMoveOpen = S.rwMoveOpen === t.dataset.rwmove ? null : t.dataset.rwmove;
    renderRewardList();
    const first = $("#rw-list .rw-menu [role=menuitem]:not([disabled])");
    if (first) first.focus();
    return;
  }
  if (t.dataset.rwmv) {
    const key = t.dataset.rwkey, how = t.dataset.rwmv;
    S.rwMoveOpen = null;
    if (how === "new") { renderRewardList(); return folderModal("", key); }
    const done = how === "up" ? stepReward(key, -1) : how === "down" ? stepReward(key, 1) : moveReward(key, t.dataset.to, null);
    const grip = document.querySelector(`#rw-list [data-rwgrip="${CSS.escape(key)}"]`);
    if (grip) grip.focus();
    return done;
  }
  if (t.dataset.rwfold != null) return foldFolder(t.dataset.rwfold);
  if (t.dataset.rwfolderrename != null) return folderModal(t.dataset.rwfolderrename);
  if (t.dataset.rwfolderdel != null) return deleteFolder(t.dataset.rwfolderdel);
  if (t.dataset.rwhide != null) { const v = botView(); return setBotView({ hidden: v.hidden.filter(l => l !== t.dataset.rwhide).concat([t.dataset.rwhide]) }); }
  if (t.dataset.rwunhide != null) { const l = t.dataset.rwunhide; return setBotView({ hidden: l ? botView().hidden.filter(x => x !== l) : [] }); }
  if (t.dataset.rwshowhidden) { S.rwShowHidden = !S.rwShowHidden; renderShares(); return renderTry(); }
  if (t.dataset.rwdel != null) { S.rw.rewards.splice(+t.dataset.rwdel, 1); S.rwNotesOpen = {}; S.rwAbout = {}; S.rwNerd = {}; return stageRewards(); }
  if (t.dataset.rwundo != null) { const e = S.rw.rewards[+t.dataset.rwundo], o = originalOf(e); if (o) e.view = JSON.parse(JSON.stringify(o)); return stageRewards(); }
  if (t.dataset.rwabout != null) { S.rwAbout[t.dataset.rwabout] = !S.rwAbout[t.dataset.rwabout]; return renderRewardList(); }
  if (t.dataset.rwnerd != null) { S.rwNerd[t.dataset.rwnerd] = !S.rwNerd[t.dataset.rwnerd]; return renderRewardList(); }
  if (t.dataset.rwcatgroup != null) { S.rwCatGroup = t.dataset.rwcatgroup; return renderCatalog(); }
  if (t.dataset.rwcatmore != null) { const n = t.dataset.rwcatmore, box = $("#rw-catalog"), top = box.scrollTop; S.rwCatMore[n] = !S.rwCatMore[n]; box.innerHTML = catalogHTML(); box.scrollTop = top; return; }
  if (t.dataset.rwcatnerd != null) { const n = t.dataset.rwcatnerd, box = $("#rw-catalog"), top = box.scrollTop; S.rwCatNerd[n] = !S.rwCatNerd[n]; box.innerHTML = catalogHTML(); box.scrollTop = top; return; }
  if (t.dataset.rwnote != null) { S.rwNotesOpen[t.dataset.rwnote] = true; renderRewardList(); const ta = $("#rw-note-" + t.dataset.rwnote); if (ta) ta.focus(); return; }
  if (t.dataset.rwnoteclose != null) { delete S.rwNotesOpen[t.dataset.rwnoteclose]; delete S.drafts[noteKey(+t.dataset.rwnoteclose)]; return renderRewardList(); }
  if (t.dataset.rwnotesave != null) {
    const i = +t.dataset.rwnotesave, ta = $("#rw-note-" + i);
    return call("set_note", S.bot, noteKey(i), ta ? ta.value : "").then(out => { if (out.error) return toast(out.error, true); S.data.notes = out.notes; delete S.rwNotesOpen[i]; delete S.drafts[noteKey(i)]; renderRewardList(); });
  }
  if (t.dataset.rwpartdel) {
    const [i, j] = t.dataset.rwpartdel.split(":").map(Number), v = S.rw.rewards[i].view, was = v.parts;
    v.parts = was.filter((p, n) => n !== j);
    return partsKept(v, was) ? stageRewards() : renderRewardList();
  }
  return SKIP;
});
on("change", el => {
  if (el.dataset.rw != null) return rewardEdit(+el.dataset.rw, el.dataset.k, el.type === "checkbox" ? el.checked : el.value);
  if (el.id === "rw-try-steps") { S.rwTrySteps = Number(el.value); return renderTry(); }
  if (el.dataset.rwsort) { rwView().sort = el.value; return saveView(); }
  if (el.dataset.rwcont) return setBotView({ hide_continuous: el.checked });
  return SKIP;
});
on("input", el => {
  if (el.id && el.id.startsWith("rw-note-")) { const key = noteKey(+el.id.slice(8)); S.drafts[key] = el.value; autosaveNote(key, el.value); }
  if (el.id === "rw-search") { S.rwQuery = el.value; renderRewardList(); }
  if (el.id === "rw-cat-q") { S.rwCatQuery = el.value; renderCatalog(); }
});
SEGS.rwwin = value => { S.rwWindow = Number(value); return refreshShares(); };
TAB_VIEWS.rewards = renderRewards;
TAB_REFRESH.rewards = () => { refreshShares(); renderTry(); };

// The keyboard: Escape clears the search and closes a Move to… menu; the arrow keys move a reward whose
// handle has the focus.
document.addEventListener("keydown", e => {
  if (!$("#modal").hidden) return;
  if (e.key === "Escape" && e.target.id === "rw-search") { e.preventDefault(); e.target.value = ""; S.rwQuery = ""; return renderRewardList(); }
  if (e.key === "Escape" && S.rwMoveOpen && e.target.closest && e.target.closest("#rw-list")) {
    const key = S.rwMoveOpen;
    S.rwMoveOpen = null; renderRewardList();
    const button = document.querySelector(`#rw-list [data-rwmove="${CSS.escape(key)}"]`);
    if (button) button.focus();
    return;
  }
  const grip = e.target.dataset && e.target.dataset.rwgrip;
  if (grip && (e.key === "ArrowUp" || e.key === "ArrowDown")) { e.preventDefault(); stepReward(grip, e.key === "ArrowUp" ? -1 : 1); }
});
document.addEventListener("mousedown", e => {
  if (S.rwMoveOpen && !e.target.closest(".rw-menu, [data-rwmove]")) { S.rwMoveOpen = null; renderRewardList(); }
});

// Dragging a reward by its handle: dropped on a reward it goes before or after it (by the pointer's height),
// on a folder's head or the space below a list at the end of that folder or list.
let rwDrag = null;
function dropSpot(e) {
  const card = e.target.closest && e.target.closest("#rw-list .reward");
  if (card) { const r = card.getBoundingClientRect(); return { el: card, key: card.dataset.rwkey, to: card.dataset.rwin, after: e.clientY > r.top + r.height / 2 }; }
  const zone = e.target.closest && e.target.closest("#rw-list [data-rwdrop]");
  return zone ? { el: zone, key: null, to: zone.dataset.rwdrop } : null;
}
function clearDropMarks() { document.querySelectorAll("#rw-list .drop-before, #rw-list .drop-after, #rw-list .drop-into").forEach(el => el.classList.remove("drop-before", "drop-after", "drop-into")); }
document.addEventListener("dragstart", e => {
  const grip = e.target.closest && e.target.closest("[data-rwgrip]");
  if (!grip) return;
  rwDrag = grip.dataset.rwgrip; S.rwMoveOpen = null;
  e.dataTransfer.effectAllowed = "move";
  e.dataTransfer.setData("text/plain", rwDrag);
  const card = grip.closest(".reward");
  if (card) { e.dataTransfer.setDragImage(card, 24, 24); setTimeout(() => card.classList.add("dragging"), 0); }
  $("#rw-list").classList.add("dragging");
});
document.addEventListener("dragover", e => {
  if (!rwDrag) return;
  const spot = dropSpot(e);
  clearDropMarks();
  if (!spot) return;
  e.preventDefault(); e.dataTransfer.dropEffect = "move";
  spot.el.classList.add(spot.key ? (spot.after ? "drop-after" : "drop-before") : "drop-into");
});
document.addEventListener("drop", e => {
  if (!rwDrag) return;
  const spot = dropSpot(e), key = rwDrag;
  e.preventDefault(); clearDropMarks(); rwDrag = null;
  if (!spot || spot.key === key) return;
  let before = spot.key;
  if (spot.key && spot.after) {  // after it: before the next reward of its list, or at the end
    const p = placed(), list = spot.to ? (p.folders.find(x => x.f.id === spot.to) || { items: [] }).items : p.loose;
    const at = list.map(i => p.keys[i]).indexOf(spot.key);
    before = list[at + 1] != null ? p.keys[list[at + 1]] : null;
  }
  if (before === key) return;
  moveReward(key, spot.to, before);
});
document.addEventListener("dragend", () => {
  rwDrag = null; clearDropMarks();
  document.querySelectorAll("#rw-list.dragging, #rw-list .dragging").forEach(el => el.classList.remove("dragging"));
});
