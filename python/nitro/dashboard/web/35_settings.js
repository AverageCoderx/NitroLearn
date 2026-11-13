// The Settings tab: the settings in plain-named groups (nitro/dashboard/layout.py), Basic or with the advanced
// ones shown, a search over all of them, and one ruled row a setting: its plain title, its name in the file, its
// control, its help, "See more", "Nerd explanation", a note, and its change waiting for Apply. The Transfer
// learning, Learning from replays and Self-play cards lead their groups.

// The groups and the Basics list, asked once (settings_layout); `null` until the answer is in.
let SET_LAYOUT = null, SET_BOX = null;
call("settings_layout").then(out => {
  SET_LAYOUT = out.error || !out.groups ? { groups: [], basics: [] } : out;
  if (SET_BOX && SET_BOX.isConnected && S.view === "bot" && S.tab === "settings" && S.data) renderSettings(SET_BOX);
});
S.nerdOpen = {};

function settingItems() { return allSections((S.data && S.data.settings) || []).flatMap(s => s.items); }
function settingItem(path) { return settingItems().find(i => i.path === path); }
// A setting's plain title, or the name made from its key.
function settingName(item) { return item.title || item.label; }
// The search's matches (rankSettings), path -> score, worked out once for each query and bot data.
let SET_RANKS = { q: null, data: null, scores: new Map(), first: "" };
function searchRanks() {
  const q = S.query.trim();
  if (SET_RANKS.q !== q || SET_RANKS.data !== S.data) {
    const ranked = q ? rankSettings(settingItems(), q) : [];
    SET_RANKS = { q, data: S.data, scores: new Map(ranked.map(x => [x.item.path, x.score])), first: ranked.length ? ranked[0].item.path : "" };
  }
  return SET_RANKS;
}
function shown(item) { return !S.query.trim() || searchRanks().scores.has(item.path); }
function matchScore(item) { return searchRanks().scores.get(item.path) || 0; }
function valueOf(item) {
  const p = S.data.pending || {};
  return Object.prototype.hasOwnProperty.call(p, item.path) ? p[item.path] : item.value;
}
function settingPending(path) { return Object.prototype.hasOwnProperty.call((S.data && S.data.pending) || {}, path); }
// A value in words: a switch as on or off, a large whole number with its thousands marked.
function showValue(item, v) {
  if (item.kind === "bool") return v ? "on" : "off";
  if (item.kind === "int" && v != null && v !== "" && !isNaN(Number(v)) && Math.abs(Number(v)) >= 10000) return Number(v).toLocaleString("en-US");
  return show(v);
}
function sameValue(a, b) { return JSON.stringify(canon(a == null ? "" : a)) === JSON.stringify(canon(b == null ? "" : b)); }

// ---------- groups and levels ----------
function setGroups() {
  const known = (SET_LAYOUT && SET_LAYOUT.groups) || [], ids = new Set(known.map(g => g.id));
  const extra = [...new Set(settingItems().map(i => i.group || "other"))].filter(id => !ids.has(id));
  return known.concat(extra.map(id => ({ id, title: id.replace(/_/g, " "), about: "" })));
}
function groupItems(id) {
  const items = settingItems();
  if (id === "basics") return ((SET_LAYOUT && SET_LAYOUT.basics) || []).map(p => items.find(i => i.path === p)).filter(Boolean);
  return items.filter(i => (i.group || "other") === id).sort((a, b) => (a.order || 0) - (b.order || 0));
}
// With "Show advanced" off an advanced setting still shows while its change waits for Apply.
function levelShown(item) { return S.setAdvanced || item.level !== "advanced" || settingPending(item.path) || !!S.errors[item.path]; }
function advancedCount() { return settingItems().filter(i => i.level === "advanced").length; }
function currentSetGroup() {
  const groups = setGroups();
  return groups.find(g => g.id === S.setGroup) || groups.find(g => groupItems(g.id).length) || groups[0] || { id: "", title: "", about: "" };
}

// ---------- one setting's control ----------
function controlHTML(item) {
  const own = item.path.startsWith("start_from.") && S.data.transfer ? transferControl(item) : "";
  if (own) return own;
  const v = valueOf(item), dis = item.fixed ? "disabled" : "", id = "f-" + item.path.replace(/\W/g, "-");
  const typed = S.typed[item.path];  // a value refused: kept in its box, beside the reason
  const label = `aria-label="${esc(settingName(item))}"`;
  if (item.kind === "bool") return `<input id="${id}" class="sw" type="checkbox" data-set="${esc(item.path)}" ${v ? "checked" : ""} ${dis} ${label}>`;
  if (item.kind === "choice") {
    const opts = (item.nullable ? [""] : []).concat(item.options);
    return `<select id="${id}" class="field" data-set="${esc(item.path)}" ${dis} ${label}>${opts.map(o => `<option value="${esc(o)}" ${String(o) === show(v) ? "selected" : ""}>${o === "" ? "not set" : esc(o)}</option>`).join("")}</select>`;
  }
  if (item.kind === "other") return `<div class="muted" style="font-size:13px;">Changed in the settings file for now.</div><code class="mono" style="font-size:12px;">${esc(JSON.stringify(v))}</code>`;
  const numeric = item.kind === "int" || item.kind === "float";
  const hint = numeric && limitWords(item) ? `placeholder="${esc(limitWords(item).toLowerCase())}"` : item.nullable ? 'placeholder="not set"' : "";
  return `<input id="${id}" class="field${numeric ? " mono number" : ""}" type="text" ${numeric ? 'inputmode="decimal"' : ""} data-set="${esc(item.path)}" value="${esc(typed != null ? typed : showValue(item, v))}" ${hint} ${dis} ${label}>`;
}

// ---------- "Nerd explanation": a written text when there is one, then the facts and the lines to copy ----------
function codeBlockHTML(text) { return `<div class="codeblock"><pre class="mono">${esc(text)}</pre><button type="button" class="btn small ghost" data-copy="1" title="Copy these lines">Copy</button></div>`; }
// A written text: paragraphs as "See more" has them, with blocks of code between lines of three backticks.
function nerdTextHTML(text) {
  return String(text).split(/```[^\n]*\n?/).map((part, i) => i % 2 ? codeBlockHTML(part.replace(/\s+$/, "")) : part.trim() ? guideHTML(part) : "").join("");
}
function copyLines(text) {
  const said = ok => toast(ok ? "Copied." : "Could not copy it: select the lines and press Ctrl+C.", !ok);
  const area = document.createElement("textarea");
  area.value = text; area.style.position = "fixed"; area.style.opacity = "0";
  document.body.appendChild(area); area.select();
  let ok = false;
  try { ok = document.execCommand("copy"); } catch (err) { ok = false; }
  area.remove();
  if (ok) return said(true);
  if (navigator.clipboard && navigator.clipboard.writeText) return navigator.clipboard.writeText(text).then(() => said(true), () => said(false));
  return said(false);
}
// A value as a settings file writes it; "" for one that is left out.
function tomlValue(kind, v) {
  if (v == null) return "";
  if (Array.isArray(v)) return "[" + v.map(x => tomlValue(typeof x === "number" ? "number" : "text", x)).join(", ") + "]";
  if (kind === "bool" || typeof v === "boolean") return v ? "true" : "false";
  const n = Number(v), isNumber = String(v).trim() !== "" && !isNaN(n);
  if (kind === "int" && isNumber) return Math.abs(n) >= 100000 ? String(Math.round(n)).replace(/\B(?=(\d{3})+(?!\d))/g, "_") : String(Math.round(n));
  if (kind === "float" && isNumber) return Number.isInteger(n) ? n.toFixed(1) : String(n);
  if (kind === "number" && isNumber) return String(n);
  return JSON.stringify(String(v));
}
const SETTING_KINDS = { bool: "On or off (true or false)", int: "A whole number", float: "A number", choice: "One of its choices", numbers: "A list of numbers", words: "A list of words", path: "A path, in quotes", text: "Text, in quotes", other: "A table of its own" };
// A setting's limits in words; a largest value beyond what a number field holds exactly is no limit to name.
function limitWords(item) {
  const lo = item.minimum, hi = item.maximum != null && Math.abs(item.maximum) > Number.MAX_SAFE_INTEGER ? null : item.maximum;
  return lo != null && hi != null ? `From ${lo} to ${hi}` : lo != null ? `${lo} or more` : hi != null ? `${hi} or less` : "";
}
function settingFileBlock(item) {
  const at = item.path.lastIndexOf("."), section = item.path.slice(0, at), key = item.path.slice(at + 1);
  const now = valueOf(item), written = tomlValue(item.kind, now == null ? item.default : now);
  return `[${section}]\n` + (written === "" ? `# ${key} is left out: not set` : `${key} = ${written}`);
}
function botCommands() {
  const runs = String((S.home && S.home.runs) || "runs"), sep = runs.includes("\\") ? "\\" : "/";
  const file = runs + sep + S.bot + sep + "settings.toml";
  return `nitro check-settings ${/\s/.test(file) ? `"${file}"` : file}\nnitro resume ${S.bot}`;
}
function settingNerdHTML(item) {
  const at = item.path.lastIndexOf("."), section = item.path.slice(0, at), key = item.path.slice(at + 1);
  const when = item.applies === "saves" ? (item.fixed ? "Fixed: this bot has a save, and its saved networks were made with this value." : "Until the bot's first save. After that it is fixed: its saved networks are made with this value.")
    : item.applies === "first_start" ? "At the bot's first start only: it is read when the bot is made, before it has a save of its own."
    : "From the bot's next start. Apply writes it into the file; a bot that is training saves and starts again with it.";
  const facts = [
    ["Name in the file", `<span class="mono">${esc(key)}</span> in <span class="mono">[${esc(section)}]</span>`],
    ["Type", esc(SETTING_KINDS[item.kind] || item.kind) + (item.nullable ? ", or left out" : "")],
    ["Default", item.default == null ? "Not set" : `<span class="mono">${esc(tomlValue(item.kind, item.default))}</span>`],
    item.options && item.options.length ? ["Choices", item.options.map(o => `<span class="mono">${esc(o)}</span>`).join(", ")] : null,
    limitWords(item) ? ["Limits", esc(limitWords(item))] : null,
    ["Takes effect", esc(when)],
  ].filter(Boolean);
  return `<div class="nerd"><span class="cap">Nerd explanation</span>
    ${item.nerd ? `<div class="nerd-text">${nerdTextHTML(item.nerd)}</div>` : ""}
    ${item.plain && item.help ? `<div class="nerd-text" data-full-help><p>${esc(item.help)}</p></div>` : ""}
    <dl class="facts">${facts.map(([k, v]) => `<div><dt class="cap">${k}</dt><dd>${v}</dd></div>`).join("")}</dl>
    <span class="cap muted">In the bot's settings file</span>${codeBlockHTML(settingFileBlock(item))}
    <span class="cap muted">Check the file, then carry on training</span>${codeBlockHTML(botCommands())}
  </div>`;
}

// ---------- one setting's row ----------
// `quiet`: why a fixed setting is fixed is said once above it (the transfer card), not on every row.
// `own`: a card's own words or control for the row ({ name, help, control }), in place of the setting's.
function settingHTML(item, quiet, own) {
  const path = item.path, changed = settingPending(path);
  const note = (S.data.notes || {})["setting:" + path] || "", noteOpen = S.notesOpen[path];
  const name = (own && own.name) || settingName(item);
  const said = S.errors[path] || "", err = said.startsWith(item.label + ": ") ? said.slice(item.label.length + 2) : said;
  const helpOpen = S.helpOpen[path], nerdOpen = S.nerdOpen[path];
  const help = (own && own.help) || item.plain || item.help || "";
  // The default shows beside a value that differs from it, unless the row is a card's own or already says "Was".
  const offDefault = !item.fixed && item.kind !== "other" && !path.startsWith("start_from.") && !(own && own.control) && !sameValue(valueOf(item), item.default) && !(changed && sameValue(item.value, item.default));
  const wide = ["words", "numbers", "path", "text"].includes(item.kind) || (path.startsWith("start_from.") && item.kind !== "bool");
  const tags = (changed ? '<span class="tag hot">Changed</span>' : "")
    + (item.fixed ? `<span class="tag" title="${esc(item.why_fixed)}">${ICON.lock}Fixed</span>` : item.applies === "saves" ? `<span class="tag" title="The bot's saved networks are made with this, so it cannot change after the first save.">${ICON.lock}Fixed once it has a save</span>` : "")
    + (item.level === "advanced" ? '<span class="tag">Advanced</span>' : "");
  return `<div class="setting${changed ? " changed" : ""}${wide ? " wide" : ""}" data-item="${esc(path)}" ${quiet && item.fixed ? 'data-quiet="1"' : ""}>
    <div class="what"><div class="name">${esc(name)}${tags}</div><div class="path mono">${esc(path)}</div></div>
    <div class="control">${own && own.control ? own.control : controlHTML(item)}
      ${changed ? `<span class="was">Was ${esc(showValue(item, item.value) || "not set")} · <button type="button" class="more" data-undo="${esc(path)}" aria-label="Undo the change to ${esc(name)}">Undo</button></span>` : ""}
      ${offDefault ? `<span class="was">Default ${esc(showValue(item, item.default) || "not set")} · <button type="button" class="more" data-default="${esc(path)}" aria-label="Put ${esc(name)} back to its default">Back to default</button></span>` : ""}
      ${item.fixed && !quiet ? `<span class="was">${esc(item.why_fixed)}</span>` : ""}
      ${err ? `<span class="err" role="alert">${esc(err)}</span>` : ""}
    </div>
    ${help ? `<div class="help${helpOpen || (own && own.help) ? " full" : ""}">${esc(help)}</div>` : ""}
    <div class="links">
      ${item.more || (!own && help.length > 140) ? `<button type="button" class="more" data-help="${esc(path)}" aria-expanded="${!!helpOpen}">${helpOpen ? "Show less" : "See more"}</button>` : ""}
      <button type="button" class="more" data-nerd="${esc(path)}" aria-expanded="${!!nerdOpen}">${nerdOpen ? "Close the nerd explanation" : "Nerd explanation"}</button>
      ${noteOpen ? "" : `<button type="button" class="more" data-noteopen="${esc(path)}">${note ? "Change the note" : "Add a note"}</button>`}
    </div>
    ${helpOpen && item.more ? `<div class="guide">${guideHTML(item.more)}</div>` : ""}
    ${nerdOpen ? settingNerdHTML(item) : ""}
    ${note && !noteOpen ? `<div class="notetext">${esc(note)}</div>` : ""}
    ${noteOpen ? `<div class="notebox"><textarea class="field" data-note="${esc(path)}" aria-label="Note for ${esc(name)}">${esc(draftOr("setting:" + path, note))}</textarea><div class="row"><button type="button" class="btn small primary" data-savenote="${esc(path)}">Save note</button><button type="button" class="btn small" data-note-close="${esc(path)}">Close</button>${noteStateHTML("setting:" + path)}</div></div>` : ""}
  </div>`;
}
// `quiet`: a reason said once above these rows; the rows it fixes do not repeat it.
function settingRowsHTML(items, quiet) { return items.map(i => i.path === "replay_mix.replays" ? rmFoldersHTML() : settingHTML(i, tfQuiet(i) || (!!quiet && i.why_fixed === quiet))).join(""); }
// Why a group's fixed settings are fixed, said once above them when most of its rows share the reason.
function fixedReason(items) {
  const reasons = items.filter(i => i.fixed).map(i => i.why_fixed);
  const shared = reasons.find(r => reasons.filter(x => x === r).length > items.length / 2);
  return shared || "";
}
function fixedBanner(reason) { return reason ? `<div class="set-fixed">${ICON.lock}<span><b>The settings marked Fixed.</b> ${esc(reason)}</span></div>` : ""; }

function rerenderSetting(path) {
  const el = document.querySelector(`[data-item="${CSS.escape(path)}"]`);
  const item = settingItem(path);
  if (!el || !item) return;
  if (path === "selfplay.enabled" && el.closest("#selfplay-card")) return renderSelfplay();
  el.outerHTML = path === "replay_mix.replays" ? rmFoldersHTML() : settingHTML(item, tfQuiet(item) || !!el.dataset.quiet);
}

// ---------- the tab ----------
function renderSettings(box) {
  SET_BOX = box;
  if (S.setBot !== S.bot) { S.setBot = S.bot; S.setGroup = "basics"; S.nerdOpen = {}; }
  if (S.setAdvanced == null) S.setAdvanced = !!(((S.home && S.home.prefs) || {}).settings_advanced);
  const secs = S.data.settings || [];
  if (!secs.length && S.data.problem) {
    box.innerHTML = `<div class="banner bad"><div class="grow">The settings file has a mistake, so its settings cannot be shown here until it is fixed:<pre>${esc(S.data.problem)}</pre></div><button type="button" class="btn small" data-act="open-settings">Open the settings file</button></div>`;
    return;
  }
  if (!SET_LAYOUT) { box.innerHTML = '<div class="card empty">Loading the settings…</div>'; return; }
  const target = takeSearchTarget();
  const scroll = window.scrollY;
  keepFocus(() => {
    box.innerHTML = `<div class="set-layout">
      <aside class="set-nav" aria-label="Groups of settings">
        <label class="search"><span class="sr">Search settings</span>${ICON.search}<input id="settings-search" class="field" type="search" placeholder="Search settings" value="${esc(S.query)}"></label>
        <label class="set-adv" for="set-advanced"><input id="set-advanced" class="sw" type="checkbox" ${S.setAdvanced ? "checked" : ""}><span><b>Show advanced</b><span class="muted" id="set-adv-count"></span></span></label>
        <nav id="set-groups" class="set-groups" aria-label="Groups"></nav>
      </aside>
      <div class="set-main" id="set-main"></div>
    </div>`;
    renderSettingsParts();
  });
  placeSettingsNav();
  window.scrollTo(0, scroll);
  if (target) showSetting(target);
}
// The group list, the advanced count and the chosen group's (or the search's) rows.
function renderSettingsParts() {
  const main = $("#set-main");
  if (!main) return;
  updateSettingsNav();
  main.innerHTML = S.query.trim() ? settingsSearchHTML() : settingsGroupHTML(currentSetGroup());
}
function updateSettingsNav() {
  const nav = $("#set-groups");
  if (!nav) return;
  const q = S.query.trim(), current = currentSetGroup().id, pending = Object.keys((S.data && S.data.pending) || {});
  put(nav, setGroups().map(g => {
    const all = groupItems(g.id), count = q ? (g.id === "basics" ? "" : all.filter(shown).length) : all.filter(levelShown).length;
    const changed = all.some(i => pending.includes(i.path));
    return `<button type="button" class="set-group${q && count === 0 ? " none" : ""}" data-setgroup="${esc(g.id)}" ${!q && g.id === current ? 'aria-current="true"' : ""}><span class="t">${esc(g.title)}</span>${changed ? '<span class="mark" title="Holds changes waiting for Apply"><span class="sr">has changes waiting</span></span>' : ""}<span class="n">${count}</span></button>`;
  }).join(""));
  const n = advancedCount();
  put($("#set-adv-count"), S.setAdvanced ? `${n} advanced settings shown` : `${n} more, for fine-tuning`);
}
// The group list stays in view under the top band, however tall the band is.
function placeSettingsNav() {
  const band = $("#band");
  if (band) document.documentElement.style.setProperty("--set-top", band.offsetHeight + 14 + "px");
}
window.addEventListener("resize", placeSettingsNav);

function hiddenHint(all) {
  const hidden = all.length - all.filter(levelShown).length;
  return hidden ? `<div class="set-hint"><span>${hidden} advanced ${hidden === 1 ? "setting" : "settings"} in this group ${hidden === 1 ? "is" : "are"} hidden.</span><button type="button" class="more" data-act="set-show-advanced">Show advanced</button></div>` : "";
}
function settingsGroupHTML(g) {
  const all = groupItems(g.id), vis = all.filter(levelShown);
  const head = `<div class="heading set-head"><h2 class="display">${esc(g.title)}</h2><span class="muted grow">${esc(g.about)}${g.id === "basics" ? " Edits wait until you press Apply." : ""}</span></div>`;
  let body;
  if (g.id === "start_from") body = transferHTML();
  else if (g.id === "replays_first") body = copySetupHTML();
  else if (g.id === "selfplay") body = selfplayHTML() + `<div class="set-rows">${settingRowsHTML(vis.filter(i => i.level === "advanced"))}</div>` + hiddenHint(all);
  else if (g.id === "replay_mix") body = replayLeadHTML() + `<div class="set-rows">${settingRowsHTML(vis)}</div>` + hiddenHint(all);
  else {
    const reason = fixedReason(vis);
    body = fixedBanner(reason) + `<div class="set-rows">${settingRowsHTML(vis, reason) || '<div class="set-hint">Nothing to change here at the Basic level.</div>'}</div>` + hiddenHint(all);
  }
  return head + body;
}
// A search shows every match, advanced ones too, under its group's name: the group of the best match first,
// the best match first in each. Enter, or a click on a match's name, opens it in its group.
function settingsSearchHTML() {
  const best = list => Math.max(0, ...list.map(matchScore));
  const groups = setGroups().filter(g => g.id !== "basics").map(g => ({ g, found: groupItems(g.id).filter(shown).sort((a, b) => matchScore(b) - matchScore(a)) }));
  groups.sort((a, b) => best(b.found) - best(a.found));
  const parts = groups.map(({ g, found }) => {
    let body = "";
    if (g.id === "start_from") body = transferHTML();
    else if (g.id === "replays_first") body = copySetupHTML();
    else if (found.length) body = (g.id === "replay_mix" ? replayHintHTML() : "") + `<div class="set-rows">${settingRowsHTML(found)}</div>`;
    return body ? `<div class="set-found"><div class="heading"><h3 class="cap">${esc(g.title)}</h3>${found.length ? `<span class="muted">${found.length} ${found.length === 1 ? "match" : "matches"}</span>` : ""}</div>${body}</div>` : "";
  }).join("");
  return parts || '<div class="card empty">No setting matches that search.</div>';
}
// The search's best match, or "".
function firstMatch() { return S.query.trim() ? searchRanks().first : ""; }
// Copying replays while training has its own tab: a search that finds its settings points there.
function replayHintHTML() {
  return `<div class="banner" style="font-size:13px;"><div class="grow">Copying replays while training has its own tab.</div><button type="button" class="btn small" data-tab="replays">Replays in training</button></div>`;
}
function replayLeadHTML() {
  const on = rmOn(), share = Number(rmVal("share")) || 0;
  return `<section class="card set-card" aria-label="Replays in training">
    <div class="set-card-head"><div class="grow"><h3>Copying replays while training</h3><div class="muted">A share of the bot's learning steps copies how the players in your converted replays play. Its own tab shows how many replays there are to copy and how the copying goes.</div></div><span class="tag${on ? " good" : ""}">${on ? "On, " + pct(share) : "Off"}</span></div>
    <div class="set-card-line"><button type="button" class="btn small" data-tab="replays">Open the Replays in training tab</button></div>
  </section>`;
}

// Another part's search asks for one setting (S.searchTarget, its path): its group opens, with the advanced
// settings if it is one, and it is scrolled to and marked for a moment.
function takeSearchTarget() {
  const path = S.searchTarget, item = path && settingItem(path);
  delete S.searchTarget;
  if (!item) return "";
  S.query = "";
  S.setGroup = item.group || S.setGroup;
  if (item.level === "advanced") S.setAdvanced = true;
  if (item.group === "replays_first") S.cpMore = true;
  return path;
}
// For other parts: shows the setting at `path` in the Settings tab, now if the tab is open, else when it
// next opens (openBot(name, "settings") after this opens it on another bot).
function goToSetting(path) {
  S.searchTarget = path;
  if (SET_BOX && SET_BOX.isConnected && S.view === "bot" && S.tab === "settings" && S.data) renderSettings(SET_BOX);
}
function showSetting(path) {
  const group = { start_from: "#transfer", replays_first: "#copysetup", selfplay: "#selfplay-card" }[(settingItem(path) || {}).group];
  const el = document.querySelector(`[data-item="${CSS.escape(path)}"]`) || (group && $(group));
  if (!el) return;
  el.scrollIntoView({ block: "center" });
  el.classList.add("set-flash");
  setTimeout(() => el.classList.remove("set-flash"), 2600);
  focusSetting(path, el);
}
// Puts the keyboard in the setting's field: its own (f-<path>), or the first one in its row.
function focusSetting(path, row) {
  const own = document.getElementById("f-" + path.replace(/\W/g, "-"));
  const field = own && !own.disabled ? own : row && [...row.querySelectorAll("input, select, textarea, button")].find(f => !f.disabled);
  if (field) field.focus({ preventScroll: true });
}
// Opens the setting at `path` in its group, from this tab's own search.
function openFoundSetting(path) {
  const search = $("#settings-search");
  if (search) search.value = "";
  goToSetting(path);
}
function openSettingsGroup(id) {
  S.query = ""; S.setGroup = id;
  const search = $("#settings-search");
  if (search) search.value = "";
  renderSettingsParts();
  const top = SET_BOX && SET_BOX.isConnected ? SET_BOX.getBoundingClientRect().top : 0, band = $("#band");
  if (top < 0) window.scrollBy(0, top - (band ? band.offsetHeight : 0) - 14);
}
function setAdvanced(on) {
  S.setAdvanced = !!on;
  if (S.home && S.home.prefs) S.home.prefs.settings_advanced = S.setAdvanced;
  call("set_pref", "settings_advanced", S.setAdvanced);
  const box = $("#set-advanced");
  if (box) box.checked = S.setAdvanced;
  renderSettingsParts();
}
async function backToDefault(path) {
  const item = settingItem(path);
  if (!item) return;
  if (sameValue(item.value, item.default)) return unstage(path);
  return stage(path, item.default == null ? "" : item.default);
}

// ---------- self-play ----------
// The group's card: the switch with what it does in plain words; the settings that matter show once it is on.
function spOn() { const en = settingItem("selfplay.enabled"); return !!(en && valueOf(en)); }
function selfplayHTML() {
  const en = settingItem("selfplay.enabled");
  if (!en) return "";
  const on = spOn();
  const rest = groupItems("selfplay").filter(i => i.level !== "advanced" && i.path !== en.path);
  const what = "The bot always plays against itself as it is now. With this on, some of its games are against saved past versions of it instead, which helps it not to forget how to beat the ways it used to play. The past versions are saved, rated and picked by NitroLearn. It is off until you try it: only full games can show whether it helps your bot.";
  return `<section class="card set-card" id="selfplay-card" aria-label="Self-play">
    <div class="set-rows">${settingHTML(en, false, { help: what })}${on ? settingRowsHTML(rest) : '<div class="set-hint">Switch it on to choose how many games are against past versions and how often one is kept.</div>'}</div>
  </section>`;
}
function renderSelfplay() { const box = $("#selfplay-card"); if (box) keepFocus(() => { box.outerHTML = selfplayHTML(); }); }

// ---------- transfer learning (start_from) ----------
// Where the bot starts from, as the card of its group: the switch, the bot and save it starts from and what it
// takes; read-only after the first save.
function tfItem(name) { return settingItem("start_from." + name); }
function tfOn() {
  const en = tfItem("enabled"), run = tfItem("run");
  return !!(en && run && valueOf(en) && (valueOf(run) !== "" || S.tfPick === S.bot));
}
// After the first save the card's line says once why where it started from is fixed.
function tfQuiet(item) { return item.path.startsWith("start_from.") && !!(S.data.start || {}).done; }
function tfChoice() {
  const run = valueOf(tfItem("run"));
  return ((S.data.transfer || {}).choices || []).find(c => c.value === run);
}
function transferControl(item) {
  const v = valueOf(item), dis = item.fixed ? "disabled" : "", id = "f-" + item.path.replace(/\W/g, "-"), label = `aria-label="${esc(item.label)}"`;
  const opt = (value, text, sel) => `<option value="${esc(value)}" ${sel ? "selected" : ""}>${esc(text)}</option>`;
  const n = x => Number(x).toLocaleString("en-US"), choice = tfChoice(), st = S.data.start || {};
  if (item.path === "start_from.enabled") return `<input id="${id}" class="sw" type="checkbox" data-tf="enabled" ${tfOn() ? "checked" : ""} ${dis} ${label}>`;
  if (item.path === "start_from.run") {
    const choices = (S.data.transfer || {}).choices || [];
    const opts = v === "" ? [opt("", "Pick a bot…", true)] : choices.some(c => c.value === v) ? [] : [opt(v, (st.source || v) + (item.fixed ? "" : " (not found)"), true)];
    return `<select id="${id}" class="field" data-tf="run" ${dis} ${label}>${opts.concat(choices.map(c => opt(c.value, c.label, c.value === v))).join("")}</select>`;
  }
  if (item.path === "start_from.at") {
    const saves = choice ? choice.saves : [], at = Number(v);
    const opts = [opt(0, saves.length ? "Its newest save when this bot starts (now " + n(saves[0]) + " steps)" : "Its newest save", at === 0)];
    if (at && !saves.includes(at)) opts.push(opt(at, "Save at " + n(at) + " steps" + (item.fixed ? "" : " (not there any more)"), true));
    saves.forEach((s, i) => opts.push(opt(s, "Save at " + n(s) + " steps" + (i === 0 ? " (newest)" : ""), at === s)));
    return `<select id="${id}" class="field" data-set="start_from.at" ${dis} ${label}>${opts.join("")}</select>`;
  }
  if (item.path === "start_from.parts") {
    return `<select id="${id}" class="field" data-set="start_from.parts" ${dis} ${label}>${opt("all", "Everything it learned (same rewards)", v === "all")}${opt("policy", "The policy only, a fresh value network (new rewards)", v === "policy")}${opt("matching", "The policy and every part of its value network that fits (same rewards, another value network)", v === "matching")}</select>`;
  }
  return "";
}
function transferHTML() {
  const sec = (S.data.settings || []).find(s => s.path === "start_from");
  if (!sec) return "";
  const q = S.query.trim().toLowerCase();
  if (q && !sec.items.some(shown) && !"transfer learning".includes(q)) return "";
  const st = S.data.start || {}, tf = S.data.transfer || {}, on = tfOn();
  const waiting = Object.keys(S.data.pending || {}).some(k => k.startsWith("start_from."));
  let names = ["enabled"];
  if (st.done) names = st.kind === "from" ? ["enabled", "run", "at", "parts"] : [];
  else if (st.kind === "replays") names = [];
  else if (on) names = names.concat(["run", "at", "parts"]);
  const rows = names.map(tfItem).filter(Boolean).map(item => settingHTML(item, tfQuiet(item))).join("");
  const pill = st.done ? st.chip : on ? "On" : "Off";
  return `<section class="card set-card" id="transfer" aria-label="Transfer learning">
    <div class="set-card-head"><div class="grow"><h3>Transfer learning</h3><div class="muted">Start this bot from another bot's trained networks instead of from scratch. Done once, at its first start; after that it trains on its own networks.</div></div><span class="tag${!st.done && on ? " good" : ""}" data-tf-state>${esc(pill)}</span></div>
    <div class="set-card-line"><div data-tf-line>${esc(st.line || "")}</div>
      ${waiting ? '<div class="muted small">The changes below wait for Apply; the line above is what the bot does now.</div>' : ""}
      ${!st.done && st.problem ? `<div class="err" role="alert">${esc(st.problem)}.</div>` : ""}
      ${!st.done && on && tf.note ? `<div class="notetext" data-tf-note>${esc(tf.note)}</div>` : ""}
      ${!st.done && on && valueOf(tfItem("run")) === "" ? '<div class="muted small">Pick the bot to start from below.</div>' : ""}
    </div>
    ${rows ? `<div class="set-rows">${rows}</div>` : ""}
  </section>`;
}
function renderTransfer() { const box = $("#transfer"); if (box) keepFocus(() => { box.outerHTML = transferHTML(); }); }
async function transferEdit(what, value) {
  if (what === "enabled") {
    const item = tfItem("enabled");
    S.tfPick = value && valueOf(tfItem("run")) === "" ? S.bot : null;
    if (Boolean(value) === Boolean(item.value)) await unstage(item.path); else await stage(item.path, value);
    if (value && cpOn()) { await copyEdit("enabled", false); toast("Learning from replays switched off: a bot starts from replays or from another bot, not both."); }
  } else {
    await stage("start_from.run", value);
    const c = ((S.data.transfer || {}).choices || []).find(x => x.value === value);
    await stage("start_from.at", c && c.saves.length ? c.saves[0] : 0);  // its newest save, written out
  }
  renderTransfer();
}
// ---------- learning from replays ([pretrain]) ----------
// The card of its group: the switch, the replay folders, the game sizes, how many replays and the skill checks;
// the other copy settings fold away under them. Read-only once the bot has a save.
const CP_CARD = ["enabled", "replays", "team_sizes", "limit", "skill_every", "skill_points"];
function cpItem(name) { return settingItem("pretrain." + name); }
function cpVal(name) { const i = cpItem(name); return i ? valueOf(i) : undefined; }
function cpOn() { return !!cpVal("enabled"); }
function cpFolders() { const v = cpVal("replays"); return Array.isArray(v) ? v.map(String) : []; }
function cpRoom() { const i = settingItem("env.obs_team_size"); return Number(i ? valueOf(i) : S.data.obs_team_size || 0); }
function cpRow(path, name, help, control, wide) {
  const err = S.errors[path], changed = settingPending(path);
  return `<div class="setting${wide ? " wide" : ""}${changed ? " changed" : ""}" data-item="${esc(path)}"><div class="what"><div class="name">${esc(name)}${changed ? '<span class="tag hot">Changed</span>' : ""}</div><div class="path mono">${esc(path)}</div></div>
    <div class="control">${control}${changed ? `<span class="was">Waiting for Apply · <button type="button" class="more" data-undo="${esc(path)}" aria-label="Undo the change to ${esc(name)}">Undo</button></span>` : ""}${err ? `<span class="err" role="alert">${esc(err)}</span>` : ""}</div>
    <div class="help full">${help}</div></div>`;
}
function copyFoldersHTML() {
  const list = cpFolders(), key = JSON.stringify(list), c = S.cpCount && S.cpCount.key === key ? S.cpCount.out : null;
  if (!c && S.cpCounting !== key) countReplays(list);
  const n = x => Number(x).toLocaleString("en-US");
  const rows = list.map((f, i) => {
    const one = c && c.folders[i];
    return `<div class="cp-folder"><span class="mono grow" title="${esc(f)}">${esc(f)}</span><span class="muted">${one ? (one.missing ? "not found" : n(one.replays) + " replays") : "counting…"}</span><button type="button" class="btn small" data-act="cp-remove" data-i="${i}">Remove</button></div>`;
  }).join("");
  const own = c && c.default ? `<div class="muted small">None chosen: the game's own replay folders (Documents\\My Games\\Rocket League\\TAGame\\DemosEpic and Demos), ${n(c.total)} replays there.</div>` : list.length ? "" : '<div class="muted small">None chosen: the game\'s own replay folders.</div>';
  const total = c && list.length > 1 ? `<span class="muted">${n(c.total)} replays in all</span>` : "";
  return cpRow("pretrain.replays", "Replay folders", "The .replay files directly in these folders (not in folders inside them). Add the folder your replays are in; with several, they are taken in turn.",
    `<div class="cp-list">${rows}${own}<div class="row"><button type="button" class="btn small" data-act="cp-add">Add a folder…</button>${total}</div></div>`, true);
}
function copySizesHTML() {
  const own = Number(S.data.team_size || 1), room = cpRoom(), v = cpVal("team_sizes");
  const picked = Array.isArray(v) && v.length ? v.map(Number) : [own];
  const boxes = [1, 2, 3].map(k => {
    const ok = room ? k <= room : k === own;
    const why = ok ? "" : room ? `Its observation has room for ${room} a side` : `Only ${own}v${own}: other sizes need a padded observation (Observation room, in the Games group)`;
    return `<label class="check" ${why ? `title="${esc(why)}"` : ""}><input type="checkbox" data-cpsize="${k}" ${picked.includes(k) ? "checked" : ""} ${ok ? "" : "disabled"}><span>${k}v${k}</span></label>`;
  }).join("");
  return cpRow("pretrain.team_sizes", "Game sizes", `Only replays of these sizes are copied; the rest are skipped (each replay's size shows when it is converted). The bot plays ${own}v${own}.`, `<div class="cp-sizes">${boxes}</div>`);
}
function copySkillHTML() {
  const every = Number(cpVal("skill_every") || 0), on = every > 0;
  const row = cpRow("pretrain.skill_every", "Skill checks", "Every few rounds the copy plays earlier versions of itself (where it started and the last check's) from kickoffs, and gets a rating from the goals: whether it is getting better at the game, not only closer to the players. About a minute each on a PC.",
    `<input class="sw" type="checkbox" data-cp="skill" ${on ? "checked" : ""} aria-label="Skill checks">`);
  return row + (on ? [cpItem("skill_every"), cpItem("skill_points")].filter(Boolean).map(i => settingHTML(i)).join("") : "");
}
function copySetupHTML() {
  const sec = (S.data.settings || []).find(s => s.path === "pretrain");
  if (!sec) return "";
  const q = S.query.trim().toLowerCase();
  if (q && !sec.items.some(shown) && !"learning from replays".includes(q)) return "";
  const st = S.data.start || {}, on = cpOn(), done = !!st.done, copied = st.kind === "replays";
  const waiting = Object.keys(S.data.pending || {}).some(k => k.startsWith("pretrain."));
  const pill = done ? (copied ? "Copied" : "Not used") : on ? "On" : "Off";
  let line, body = "";
  if (done) {
    line = copied ? "This bot started from a copy of the players in its replays. The copy is made once, before training, so these settings no longer change anything, except the value warm-up below, which acts while it trains; the Overview tab shows how both went." : "This bot did not learn from replays, and it has its own trained networks now (its saves). New bot from this one makes a bot that can.";
    body = copied && cpItem("value_warmup_steps") ? settingHTML(cpItem("value_warmup_steps")) : "";
  } else {
    line = waiting ? '<span class="muted small">The changes below wait for Apply.</span>' : "";
    body = cpRow("pretrain.enabled", "Learn from replays first", `Its first start copies how the players in your replays play: it learns to choose what they chose. Then it saves and stops; Start training trains it from the copy. ${on ? "" : "Switching this on switches transfer learning off."}`,
      `<input class="sw" type="checkbox" data-cp="enabled" ${on ? "checked" : ""} aria-label="Learn from replays first">`);
    if (on) {
      const more = groupItems("replays_first").filter(i => !CP_CARD.includes(i.path.slice(9)));
      const open = S.cpMore || S.setAdvanced || q;
      body += copyFoldersHTML() + copySizesHTML() + (cpItem("limit") ? settingHTML(cpItem("limit")) : "") + copySkillHTML()
        + (S.setAdvanced || q ? "" : `<div class="set-hint"><span>${more.length} more copy settings, for fine-tuning.</span><button type="button" class="btn small" data-act="cp-more" aria-expanded="${!!S.cpMore}">${S.cpMore ? "Fewer copy settings" : "More copy settings"}</button></div>`)
        + (open ? more.map(i => settingHTML(i)).join("") : "");
    }
  }
  return `<section class="card set-card" id="copysetup" aria-label="Learning from replays">
    <div class="set-card-head"><div class="grow"><h3>Learning from replays</h3><div class="muted">Start this bot from a copy of how the players in your replays play, instead of from scratch. Done once, at its first start; the Overview tab shows its progress.</div></div><span class="tag${!done && on ? " good" : ""}">${esc(pill)}</span></div>
    ${line ? `<div class="set-card-line">${line}</div>` : ""}
    ${body ? `<div class="set-rows">${body}</div>` : ""}
  </section>`;
}
function renderCopySetup() { const box = $("#copysetup"); if (box) keepFocus(() => { box.outerHTML = copySetupHTML(); }); }
async function countReplays(list) {
  const key = JSON.stringify(list), bot = S.bot;
  S.cpCounting = key;
  const out = await call("count_replays", list);
  if (S.bot !== bot || S.cpCounting !== key) return;
  S.cpCounting = null;
  if (!out.error) { S.cpCount = { key, out }; renderCopySetup(); }
}
async function copyEdit(what, value) {
  const set = async (path, v) => { const item = cpItem(path.slice(9)); if (item && JSON.stringify(item.value) === JSON.stringify(v)) await unstage(path); else await stage(path, v); };
  if (what === "enabled") {
    await set("pretrain.enabled", !!value);
    if (value) {
      if (tfOn()) { await stage("start_from.enabled", false); toast("Transfer learning switched off: a bot starts from replays or from another bot, not both."); }
      if (!Number(cpVal("batch_replays"))) await set("pretrain.batch_replays", 100);  // the large mode: any number of replays
    }
  } else if (what === "skill") {
    await set("pretrain.skill_every", value ? 10 : 0);
  } else if (what === "size") {
    const own = Number(S.data.team_size || 1), v = cpVal("team_sizes");
    const picked = new Set(Array.isArray(v) && v.length ? v.map(Number) : [own]);
    if (value.on) picked.add(value.size); else picked.delete(value.size);
    if (!picked.size) { toast("At least one game size is copied.", true); return renderCopySetup(); }
    await set("pretrain.team_sizes", [...picked].sort());
  } else if (what === "folders") {
    await set("pretrain.replays", value);
  }
  renderCopySetup(); renderTransfer();
}

ACTIONS["cp-add"] = async () => {
  const out = await call("choose_replay_folder");
  if (out.error) return toast(out.error, true);
  if (out.folder && !cpFolders().includes(out.folder)) return copyEdit("folders", cpFolders().concat([out.folder]));
  return;
};
ACTIONS["cp-remove"] = async el => { const list = cpFolders(); list.splice(Number(el.dataset.i), 1); return copyEdit("folders", list); };
ACTIONS["cp-more"] = async () => { S.cpMore = !S.cpMore; return renderCopySetup(); };
ACTIONS["set-show-advanced"] = async () => setAdvanced(true);
on("click", t => {
  if (t.dataset.setgroup) return openSettingsGroup(t.dataset.setgroup);
  if (t.dataset.help) { S.helpOpen[t.dataset.help] = !S.helpOpen[t.dataset.help]; return rerenderSetting(t.dataset.help); }
  if (t.dataset.nerd) { S.nerdOpen[t.dataset.nerd] = !S.nerdOpen[t.dataset.nerd]; return rerenderSetting(t.dataset.nerd); }
  if (t.dataset.default) return backToDefault(t.dataset.default);
  if (t.dataset.copy) { const pre = t.parentElement.querySelector("pre"); return pre ? copyLines(pre.textContent) : undefined; }
  if (t.dataset.noteopen) { S.notesOpen[t.dataset.noteopen] = true; rerenderSetting(t.dataset.noteopen); const ta = document.querySelector(`[data-note="${CSS.escape(t.dataset.noteopen)}"]`); if (ta) ta.focus(); return; }
  if (t.dataset.noteClose) { delete S.notesOpen[t.dataset.noteClose]; delete S.drafts["setting:" + t.dataset.noteClose]; return rerenderSetting(t.dataset.noteClose); }
  if (t.dataset.savenote) {
    const path = t.dataset.savenote, ta = document.querySelector(`[data-note="${CSS.escape(path)}"]`);
    return call("set_note", S.bot, "setting:" + path, ta ? ta.value : "").then(out => { if (out.error) return toast(out.error, true); S.data.notes = out.notes; delete S.notesOpen[path]; delete S.drafts["setting:" + path]; rerenderSetting(path); });
  }
  return SKIP;
});
on("change", el => {
  if (el.id === "set-advanced") return setAdvanced(el.checked);
  if (el.dataset.tf) return transferEdit(el.dataset.tf, el.type === "checkbox" ? el.checked : el.value);
  if (el.dataset.cp) return copyEdit(el.dataset.cp, el.checked);
  if (el.dataset.cpsize) return copyEdit("size", { size: Number(el.dataset.cpsize), on: el.checked });
  return SKIP;
});
document.addEventListener("keydown", e => {
  if (e.target.id !== "settings-search" || e.key !== "Enter") return;
  e.preventDefault();
  const path = firstMatch();
  if (path) openFoundSetting(path);
});
// A click on a found setting's name (not on its control or links) opens it in its group.
document.addEventListener("click", e => {
  const what = e.target.closest && e.target.closest(".set-found .setting > .what");
  if (!what || !S.query.trim()) return;
  const row = what.closest("[data-item]");
  if (row) openFoundSetting(row.dataset.item);
});
on("input", el => {
  if (el.dataset && el.dataset.note) { S.drafts["setting:" + el.dataset.note] = el.value; autosaveNote("setting:" + el.dataset.note, el.value); }
  if (el.id === "settings-search") { S.query = el.value; renderSettingsParts(); }
});
TAB_VIEWS.settings = renderSettings;
