// The search box in the top band ("Search or run anything"): bots, pages, the open bot's settings and every
// action that makes sense now. Ctrl+K or / puts the keyboard in it; arrows and Enter choose, Escape closes.

// ---------- what can be run ----------
// COMMANDS is the list of actions the search offers; each part of the page adds its own:
//   { id, label, keywords, command, about, when, run }
// `label` may be a function (the Start button's words follow the bot); `keywords` are other words it is found
// by; `command` is the simple terminal command for the same thing, when there is one ("{bot}" becomes the open
// bot's name), `about` a few words shown when there is none; `when()` says whether it applies now (always,
// when left out); `run()` does it.
const COMMANDS = [];
// The bot whose page is open, or null.
function shownBot() { return S.view === "bot" && S.bot && S.data && S.data.name === S.bot ? S.data : null; }
function goTo(view) { const b = document.querySelector(`.nav [data-go="${view}"]`); if (b) b.click(); }
function showTab(tab) { const b = document.querySelector(`[role=tab][data-tab="${tab}"]`); if (b) b.click(); }
function botIsBusy() { const b = shownBot(); return !!b && (b.status === "training" || b.status === "copying" || b.status === "starting"); }
COMMANDS.push(
  { id: "new-bot", label: "New bot", keywords: "make create add wizard starter", command: "nitro new <settings file> --name <name>", run: () => ACTIONS.newbot() },
  { id: "unpack", label: "Add a packed bot", keywords: "unpack import cloud another computer", command: "nitro unpack <file>", run: () => ACTIONS.unpack() },
  { id: "open-rlbot", label: "Open RLBot", keywords: "match play real game rocket league", about: "plays your bot in a real match, offline", run: () => openRLBot() },
  { id: "install-rlbot", label: "Install RLBot v5", keywords: "rlbot download installer setup get", command: "nitro install-rlbot", when: () => !rlbotFound() && onWindows(), run: () => { rlbotModal(); return ACTIONS["rlbot-install"](); } },
  { id: "choose-rlbot", label: "Choose where RLBot is", keywords: "rlbot program path", about: "asked once, then Open RLBot starts it", run: () => ACTIONS["rlbot-choose"]() },
  { id: "check-pc", label: "Check this PC", keywords: "doctor ready graphics card gpu arena files meshes problems", command: "nitro doctor", run: () => { goTo("doctor"); if (S.doctor && !S.doctor.running) ACTIONS["doctor-run"](); } },
  { id: "update", label: "Update NitroLearn", keywords: "update upgrade new version release download check for updates", command: "nitro update", run: () => { goTo("prefs"); const u = S.upd; if (!u || !["checking", "running"].includes(u.phase)) ACTIONS["upd-check"](); setTimeout(() => { const box = $("#update"); if (box) box.scrollIntoView({ block: "start" }); }, 0); } },
  { id: "code-folder", label: "Open the code folder", keywords: "code source folder edit change nitrolearn developer", about: "NitroLearn's own folder", run: () => ACTIONS["open-code"]() },
  { id: "theme", label: () => S.theme === "dark" ? "Switch to the light look" : "Switch to the dark look", keywords: "theme dark light colours mode", about: "the top band stays black", run: () => ACTIONS.theme() },
  { id: "start", label: () => startWord(shownBot() || {}), keywords: "start train training resume carry on run", command: "nitro resume {bot}", when: () => !!shownBot() && !botIsBusy() && shownBot().status !== "problem", run: () => ACTIONS.start() },
  { id: "stop", label: "Stop and save", keywords: "stop pause end training", about: "saves the bot, then stops training", when: () => { const b = shownBot(); return !!b && (b.status === "training" || b.status === "copying"); }, run: () => ACTIONS.stop() },
  { id: "watch", label: "Watch it play", keywords: "watch game match viewer see", command: "nitro watch {bot}", when: () => !!shownBot() && shownBot().saves > 0, run: () => ACTIONS.watch() },
  { id: "export", label: "Make RLBot bot", keywords: "export rlbot bot folder", command: "nitro export {bot}", when: () => !!shownBot() && shownBot().saves > 0 && !shownBot().exporting, run: () => ACTIONS.export() },
  { id: "pack", label: "Pack for another computer", keywords: "pack cloud move file zip", command: "nitro pack {bot}", when: () => !!shownBot(), run: () => ACTIONS.pack() },
  { id: "copy", label: "New bot from this one", keywords: "copy clone duplicate transfer learning", about: "its settings, with or without what it learned", when: () => !!shownBot(), run: () => ACTIONS.copy() },
  { id: "charts", label: "Add charts", keywords: "charts graphs telemetry plots add pin overview", about: "choose the charts on its Overview", when: () => !!shownBot(), run: () => { if (S.tab !== "overview") { S.tab = "overview"; renderBot(); } ACTIONS["ch-add"](); } },
  { id: "report", label: "Progress report", keywords: "report charts page html", about: "a page with every chart", when: () => !!shownBot() && !!shownBot().report, run: () => ACTIONS.report() },
  { id: "wandb", label: "W&B charts", keywords: "wandb weights biases charts", about: "its page on Weights & Biases", when: () => !!shownBot() && !!shownBot().wandb, run: () => ACTIONS.wandb() },
  { id: "folder", label: "Open its folder", keywords: "folder files saves explorer", about: "saves, settings and logs", when: () => !!shownBot(), run: () => ACTIONS.folder() },
  { id: "settings-file", label: "Open its settings file", keywords: "settings toml file editor", about: "in this PC's editor", when: () => !!shownBot(), run: () => ACTIONS["open-settings"]() },
  { id: "apply", label: "Apply the waiting changes", keywords: "apply pending changes save", about: "asks first", when: () => !!shownBot() && changeLines().length > 0, run: () => ACTIONS.apply() },
);

// ---------- what is searched ----------
const FINDER_PAGES = [["home", "Garage", "your bots", "home start"], ["resources", "Resources", "your files, rewards and replays", "library settings"], ["doctor", "Check this PC", "is it ready to train", "doctor"], ["prefs", "App settings", "the look, RLBot, updates, the code", "theme preferences update editing"], ["help", "Help", "support, bug reports and ideas on the Vanta Discord server", "help support bug report idea suggestion discord vanta"]];
// One entry a thing that can be found: its group, its label, a one-line hint, the words it is found by (lower
// case: the label first, then names, then the rest), and what choosing it does.
function finderEntry(group, label, hint, words, run, mono, sub) {
  return { group, label, sub: sub || "", hint, mono: !!mono, text: words.map(w => String(w || "").toLowerCase()), run };
}
// The open bot's settings, read once for a bot: each by its plain title, found as the Settings tab's own
// search finds it (settingMatch).
function settingsIndex() {
  const b = shownBot();
  if (!b) return [];
  const sections = allSections(b.settings || []), count = sections.reduce((n, s) => n + s.items.length, 0);
  const key = b.name + ":" + count;
  if (!S.finderIndex || S.finderIndex.key !== key) {
    const items = sections.flatMap(s => s.items);
    const entries = sections.flatMap(s => s.items.map(i => Object.assign(finderEntry("Settings", i.title || i.label, i.path, [i.title || i.label, i.path, i.help], () => openSetting(i.path), true, s.label), { item: i })));
    S.finderIndex = { key, items, entries };
  }
  return S.finderIndex.entries;
}
function finderEntries() {
  const b = shownBot();
  const bots = ((S.home && S.home.bots) || []).map(x => finderEntry("Bots", x.name, statusLine(x), [x.name, statusLine(x)], () => openBot(x.name)));
  const pages = FINDER_PAGES.map(([view, label, about, words]) => finderEntry("Pages", label, about, [label, about + " " + words], () => goTo(view)));
  if (b && typeof TABS === "object") Object.entries(TABS).forEach(([tab, label]) => pages.push(finderEntry("Pages", `${b.name} · ${label}`, "", [label, b.name + " " + tab], () => showTab(tab))));
  const actions = COMMANDS.filter(c => !c.when || c.when()).map(c => {
    const label = typeof c.label === "function" ? c.label() : c.label;
    const command = (c.command || "").replace("{bot}", b ? b.name : "<bot>");
    return finderEntry("Actions", label, command || c.about || "", [label, c.keywords, command], () => c.run(), !!command);
  });
  return { Bots: bots, Pages: pages, Settings: settingsIndex(), Actions: actions };
}
// How well an entry fits the words typed: 0 when one of them is nowhere in it; the label counts most.
function finderScore(item, words) {
  let score = 0;
  for (const word of words) {
    const [label, ...rest] = item.text;
    const at = label.indexOf(word);
    let best = at === 0 ? 100 : at > 0 && /[\s·._-]/.test(label[at - 1]) ? 80 : at > 0 ? 60 : 0;
    rest.forEach((text, i) => { if (!best && text.includes(word)) best = 40 - 10 * Math.min(i, 2); });
    if (!best) return 0;
    score += best;
  }
  return score;
}
const FINDER_MOST = { Bots: 6, Pages: 8, Settings: 9, Actions: 9 };
function finderResults(query) {
  const words = query.toLowerCase().split(/\s+/).filter(Boolean), all = finderEntries();
  const out = [];
  for (const group of ["Bots", "Pages", "Settings", "Actions"]) {
    let list = all[group];
    if (words.length && group === "Settings") {
      const byPath = new Map(list.map(e => [e.item.path, e]));
      list = list.length ? rankSettings(S.finderIndex.items, query).map(x => byPath.get(x.item.path)) : [];
    } else if (words.length) list = list.map(item => [finderScore(item, words), item]).filter(([score]) => score > 0).sort((a, b) => b[0] - a[0]).map(([, item]) => item);
    else if (group === "Settings") list = [];  // typed for: there are too many to list
    out.push(...list.slice(0, FINDER_MOST[group]));
  }
  return out;
}

// ---------- opening a setting ----------
// Opens the open bot's Settings tab at the setting `path`: the tab opens its group, scrolls to it and marks
// it (goToSetting); its field then gets the keyboard.
function openSetting(path) {
  if (path.startsWith("replay_mix.")) showTab("replays");
  else if (S.view === "bot" && S.tab === "settings") goToSetting(path);
  else { S.searchTarget = path; showTab("settings"); }
  let tries = 0;
  const look = () => {
    const el = document.getElementById("f-" + path.replace(/\W/g, "-")) || document.getElementById("f-" + path.replace(".", "-"));
    if (!el) {
      if (++tries < 25) setTimeout(look, 60);
      return;
    }
    if (path.startsWith("replay_mix.")) el.scrollIntoView({ block: "center" });
    if (el.focus && !el.disabled) el.focus({ preventScroll: true });
  };
  look();
}

// ---------- the box ----------
const FINDER = { list: [], at: 0, open: false };
function finderBox() { return $("#finder-q"); }
function drawFinder() {
  const box = $("#finder-list"), input = finderBox();
  if (!box || !input) return;
  box.hidden = !FINDER.open;
  input.setAttribute("aria-expanded", String(FINDER.open));
  if (!FINDER.open) { input.removeAttribute("aria-activedescendant"); return; }
  let group = "";
  const rows = FINDER.list.map((item, i) => {
    const head = item.group !== group ? `<div class="finder-group cap" role="presentation">${esc(item.group)}</div>` : "";
    group = item.group;
    return `${head}<div class="finder-row" id="finder-${i}" role="option" data-finder="${i}" aria-selected="${i === FINDER.at}"><span class="what">${esc(item.label)}</span>${item.sub ? `<span class="where cap">${esc(item.sub)}</span>` : ""}${item.hint ? `<span class="hint${item.mono ? " mono" : ""}">${esc(item.hint)}</span>` : ""}</div>`;
  });
  box.innerHTML = rows.join("") || `<div class="finder-none">Nothing matches “${esc(input.value.trim())}”. Try a bot's name, a page, or a setting of the open bot.</div>`;
  if (FINDER.list.length) input.setAttribute("aria-activedescendant", "finder-" + FINDER.at); else input.removeAttribute("aria-activedescendant");
  const on = $("#finder-" + FINDER.at);
  if (on) on.scrollIntoView({ block: "nearest" });
}
function openFinder() {
  const input = finderBox();
  FINDER.list = finderResults(input.value); FINDER.at = 0; FINDER.open = true;
  drawFinder();
}
function closeFinder() { FINDER.open = false; drawFinder(); }
function runFinder(i) {
  const item = FINDER.list[i], input = finderBox();
  if (!item) return;
  input.value = ""; closeFinder(); input.blur();
  item.run();
}
function focusFinder() { const input = finderBox(); if (input) { input.focus(); input.select(); } }

document.addEventListener("keydown", e => {
  const input = finderBox();
  if (!input) return;
  const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName || "") || e.target.isContentEditable;
  if ((e.key === "k" || e.key === "K") && (e.ctrlKey || e.metaKey) && !e.altKey) { e.preventDefault(); if ($("#modal").hidden) focusFinder(); return; }
  if (e.key === "/" && !typing && !e.ctrlKey && !e.metaKey && !e.altKey && $("#modal").hidden) { e.preventDefault(); return focusFinder(); }
  if (e.target !== input) return;
  if (e.key === "ArrowDown" || e.key === "ArrowUp") {
    e.preventDefault();
    if (!FINDER.open) return openFinder();
    const n = FINDER.list.length;
    if (n) FINDER.at = (FINDER.at + (e.key === "ArrowDown" ? 1 : n - 1)) % n;
    return drawFinder();
  }
  if (e.key === "Enter") { e.preventDefault(); e.stopPropagation(); if (FINDER.open) runFinder(FINDER.at); else openFinder(); return; }
  if (e.key === "Escape") { e.stopPropagation(); if (FINDER.open && input.value) { input.value = ""; return openFinder(); } closeFinder(); input.blur(); }
}, true);
on("input", el => { if (el.id === "finder-q") openFinder(); });
document.addEventListener("focusin", e => { if (e.target.id === "finder-q") openFinder(); else if (FINDER.open) closeFinder(); });
// A press on a result runs it before the box loses the keyboard; a press anywhere else closes the list.
document.addEventListener("mousedown", e => {
  const row = e.target.closest && e.target.closest("[data-finder]");
  if (row) { e.preventDefault(); return runFinder(Number(row.dataset.finder)); }
  if (FINDER.open && !(e.target.closest && e.target.closest("#finder"))) closeFinder();
});
document.addEventListener("mousemove", e => {
  const row = e.target.closest && e.target.closest("[data-finder]");
  if (row && Number(row.dataset.finder) !== FINDER.at) { FINDER.at = Number(row.dataset.finder); document.querySelectorAll("[data-finder]").forEach(r => r.setAttribute("aria-selected", String(r === row))); finderBox().setAttribute("aria-activedescendant", row.id); }
});
