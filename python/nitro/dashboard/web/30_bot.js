// A bot's page: its header (number plate, name, state, the three main actions), its figures, the four sectors
// with the "next" board, the tabs' frame, opening and refreshing a bot, following a command started for it,
// and the dialog that changes its race number and colours.
function runningSeconds(b) { return b.running_since ? Math.max(0, Date.now() / 1000 - b.running_since) : 0; }
function shortCount(x) { return x >= 1e6 ? (x / 1e6).toFixed(x >= 1e7 ? 0 : 1) + "M" : x >= 1e3 ? Math.round(x / 1e3) + "k" : String(x); }
function networkWord(b) { return b.encoder === "mlp" && b.gco ? "GCO network" : b.encoder === "attention_communication" ? "Attention, talking" : NETWORK_WORDS[b.encoder] || b.encoder; }
function whole(x) { return Number(x || 0).toLocaleString("en-US"); }
function waitingChanges(b) { return Object.keys(b.pending || {}).length + (b.reward_changes || []).length + (b.start_changes || []).length; }

// The one plain fact beside the state tag.
function headFact(b) {
  if (b.stopping) return "Saving after this step, then it stops";
  if (b.applying) return b.applying;
  if (b.status === "training") return b.counted && b.steps_per_second ? num(b.steps_per_second) + " steps a second" : firstRoundWord();
  if (b.status === "copying") return b.copy && b.copy.limit ? `${whole(b.copy.learned)} of ${whole(b.copy.limit)} replays learned` : "Getting the replays ready";
  if (b.status === "starting") return "Starting: its numbers come after its first round of training";
  if (b.status === "problem") return "It cannot start until its settings are fixed";
  if (b.checking) return "The Rewards tab shows the result";
  return b.newest_save != null ? `Newest save at ${steps(b.newest_save)} steps` : "No save yet";
}
// The facts of a bot as small labelled figures: one row where they fit, else rows of equal length (the class
// n<cells> says how many cells they take, the Start fact two).
const OBSERVATION_WORDS = { nitro_v2: "Detailed", native: "Basic" };
function figuresHTML(b) {
  const running = b.status === "training" || b.status === "copying";
  const p = b.parameters;
  const parts = p && p.total ? [p.shared ? "shared " + shortCount(p.shared) : "", "policy " + shortCount(p.policy), "value " + shortCount(p.value)].filter(Boolean).join(", ") : "";
  const fig = (label, value, title) => value === "" || value == null ? "" : `<div${title ? ` title="${esc(title)}"` : ""}><dt>${label}</dt><dd>${value}</dd></div>`;
  const cells = [
    fig("Game size", b.team_size ? `${b.team_size}v${b.team_size}` : ""),
    fig("Network", b.encoder ? esc(networkWord(b).replace(" network", "")) : ""),
    fig("Parameters", p && p.total ? shortCount(p.total) : "", parts ? "Network weights in its newest save: " + parts : ""),
    fig("Observation", b.observation ? esc(OBSERVATION_WORDS[b.observation] || b.observation) : "", b.observation ? "What the bot sees: " + b.observation : ""),
    fig("Tick skip", b.tick_skip || "", b.tick_skip ? `It decides every ${b.tick_skip} ticks of the game` : ""),
    b.start ? `<div class="wide" title="${esc(b.start.line)}"><dt>Start</dt><dd data-start-chip>${esc(b.start.chip)}</dd></div>` : "",
    fig("Agent steps", steps(b.agent_steps)),
    fig("Speed", running && b.counted !== false && b.steps_per_second ? num(b.steps_per_second) + "/s" : "–"),
    running ? fig("This session", duration(runningSeconds(b))) : "",
    fig("Saves", b.saves),
  ].filter(Boolean);
  const n = cells.length + (b.start ? 1 : 0);
  return `<dl class="figures n${n}" style="--half:${Math.ceil(n / 2)};--third:${Math.max(3, Math.ceil(n / 3))};" aria-label="Facts">${cells.join("")}</dl>`;
}

// The four sectors of a bot's lap, each with its state worked out from what is on disk: "done", "now",
// "stopped" (trained some, not training now) or "todo". `t` is the bot's tools data (bot_tools). Playing it in
// RLBot is done once its RLBot bot is made and RLBot is found.
function sectors(b, t) {
  const size = b.team_size ? `${b.team_size}v${b.team_size}` : "";
  const net = b.parameters && b.parameters.total ? shortCount(b.parameters.total) + " parameters" : b.encoder ? networkWord(b).toLowerCase() : "";
  const going = b.status === "training" || b.status === "copying" || b.status === "starting";
  const trained = going || b.agent_steps > 0 || b.saves > 0;
  const made = !!t.rlbot, played = made && rlbotFound();
  const held = made && t.rlbot.steps != null ? `its RLBot bot holds ${steps(t.rlbot.steps)} steps` : "its RLBot bot is made";
  return [
    { state: "done", name: "Set up", sub: [size, net].filter(Boolean).join(" · ") || "made", tab: "settings" },
    { state: going ? "now" : trained ? "stopped" : "todo", name: "Training", sub: trained ? `${steps(b.agent_steps)} steps · ${b.saves} ${b.saves === 1 ? "save" : "saves"}` : "not started yet", tab: "overview" },
    { state: t.recorded ? "done" : "todo", name: "Watch it play", sub: t.recorded ? "last recorded " + ago(t.recorded.at) : "see what it learned", act: b.saves ? "watch" : "" },
    { state: played ? "done" : made ? "now" : "todo", name: "Play it in RLBot", sub: played ? held : made ? "made; RLBot is not found yet" : "a real match", act: made && !played ? "rlbot-open" : b.saves ? "export" : "" },
  ];
}
// The most useful next thing for a bot, in one sentence, with the button that does it (none when nothing
// needs the user). `hazard` marks what stops the bot until the user acts.
function nextStep(b, t) {
  const waiting = waitingChanges(b), training = b.status === "training";
  if (b.status === "problem") return { text: "Its settings file has a mistake, so it cannot start.", label: "Fix its settings", tab: "settings", hazard: true };
  if (b.stopping) return { text: "It saves when the step it is on ends, then it stops. That can take a minute or two; nothing it learned is lost." };
  if (b.applying) return { text: b.applying + "." };
  if (waiting) return { text: `${waiting} ${waiting === 1 ? "change waits" : "changes wait"} until you apply ${waiting === 1 ? "it" : "them"}.`, label: `Apply your ${waiting} ${waiting === 1 ? "change" : "changes"}`, act: "apply" };
  if (b.checking) return { text: "Rewards are being tried on it. The Rewards tab shows what each would pay.", label: "See the Rewards tab", tab: "rewards" };
  if (b.status === "starting") return { text: "Starting. Its first numbers show here after its first round of training" + (noCard() ? ", which can take several minutes on the processor." : ".") };
  if (b.status === "copying") return { text: b.copy && b.copy.limit ? `Learning from your replays: ${whole(b.copy.learned)} of ${whole(b.copy.limit)} so far.` : "Learning from your replays." };
  if (!training && b.start && !b.start.done && b.start.problem) return { text: b.start.problem + ".", label: "Fix its settings", tab: "settings", hazard: true };
  if (!training && (b.copy_first || b.copy_so_far)) return { text: b.copy_so_far ? "The copy of your replays is saved part-way. Carry it on." : "It learns from your replays first. Copy them into it.", label: startWord(b), act: "start" };
  if (!training && !b.saves) return { text: "It has not trained yet. Start training to see it learn.", label: startWord(b), act: "start" };
  if (!b.saves) return { text: "Its first save comes 15 minutes into training by default (Save every, in Settings). Then you can watch it play." };
  if (!t.recorded) return { text: "Watch a match to see what it has learned.", label: "Watch it play", act: "watch" };
  if (!t.rlbot) return { text: "Make its RLBot bot to play it in a real match.", label: "Make RLBot bot", act: "export" };
  if (!rlbotFound() && !training) return { text: "Its RLBot bot is made. To play it in a real match, NitroLearn needs RLBot v5: tell it where RLBot is, or install it.", label: "Open RLBot", act: "rlbot-open" };
  if (!training) return { text: `It is stopped at ${steps(b.agent_steps)} steps. Carry on whenever you like.`, label: startWord(b), act: "start" };
  const trend = t.trend;
  if (trend && trend.now != null && trend.before != null) {
    const now = fmt(trend.now), before = fmt(trend.before);
    const moved = trend.now > trend.before * 1.05 ? `up from ${before}` : trend.now < trend.before * 0.95 ? `down from ${before}` : "about the same as";
    return { text: `It touches the ball ${now} times a minute now, ${moved} an hour of training ago.` };
  }
  return { text: `Training: ${steps(b.agent_steps)} steps so far. Nothing needs you right now.` };
}
function lapHTML(b, t) {
  const next = nextStep(b, t);
  const state = { done: " · done", now: " · now", stopped: " · stopped", todo: " · to do" };
  const cells = sectors(b, t).map((s, i) => {
    const inner = `<span class="cap">Sector ${i + 1}${state[s.state]}</span><b>${s.name}</b><small>${esc(s.sub)}</small>`;
    const attrs = `class="sector ${s.state}" data-sector="${i + 1}" data-state="${s.state}"`;
    return s.tab ? `<button type="button" ${attrs} data-tab="${s.tab}">${inner}</button>` : s.act ? `<button type="button" ${attrs} data-act="${s.act}">${inner}</button>` : `<div ${attrs}>${inner}</div>`;
  }).join("");
  const button = next.label ? `<button type="button" class="btn small" ${next.tab ? `data-tab="${next.tab}"` : `data-act="${next.act}"`}>${esc(next.label)}</button>` : "";
  return `<div class="lap">${cells}<section class="board${next.hazard ? " waits" : ""}" aria-label="Next step"><span class="cap">Next</span><p>${esc(next.text)}</p>${button}</section></div>`;
}

// Why the head's buttons that cannot be pressed now cannot, said under them (not only on hover).
function offWhy(b, busy, why) {
  if (!busy && why) return why + ".";
  if (!b.saves) return "Watch it play and Make RLBot bot come with its first save, 15 minutes into training by default.";
  return "";
}
function headHTML(b) {
  const t = S.tools || {};
  const running = b.status === "training" || b.status === "copying";
  const busy = running || b.status === "starting";
  const why = b.status === "problem" ? "Fix the settings problem first (the Settings tab says what it is)" : b.checking ? "Rewards are being tried on it: wait until that is done, or stop it in the Rewards tab" : b.applying ? "It is starting again with the applied changes" : "";
  const tag = b.stopping ? "strong" : b.status === "training" ? "good" : b.status === "problem" ? "bad" : b.status === "starting" || b.status === "copying" || b.checking ? "strong" : "";
  const note = S.toolNote && S.toolNote.bot === b.name ? S.toolNote : null;
  return `<section class="card botface" aria-label="Bot">
      <div class="bothead">
        <button type="button" class="platebtn" data-act="livery" title="Change its race number and colours" aria-label="Change the race number and colours of ${esc(b.name)}">${plate(b, 64)}</button>
        <div class="who">
          <nav class="crumbs" aria-label="Where this page is"><button type="button" data-go="home">Garage</button> / <b>${esc(b.name)}</b></nav>
          <div class="row"><h1 class="display name">${esc(b.name)}</h1><span class="tag ${tag}" data-status="${esc(b.status)}">${esc(statusWord(b))}</span><span class="cap fact">${esc(headFact(b))}</span></div>
        </div>
        <div class="actions">
          ${busy ? `<button type="button" class="btn primary" data-act="stop" ${b.stopping ? 'disabled title="It saves after the step it is on, then stops"' : b.status === "starting" ? 'disabled title="Starting: Stop comes once it trains"' : ""}>${ICON.stop}${b.stopping ? "Stopping…" : "Stop and save"}</button>`
                 : `<button type="button" class="btn primary" data-act="start" ${why ? `disabled title="${esc(why)}"` : ""}>${ICON.play}${esc(startWord(b))}</button>`}
          <button type="button" class="btn strong" data-act="watch"${b.saves ? "" : 'disabled title="It needs a save first: the first comes 15 minutes into training by default"'}>${ICON.eye}Watch it play</button>
          <button type="button" class="btn" data-act="export" ${b.saves && !b.exporting ? "" : `disabled title="${b.exporting ? "Being made now" : "It needs a save first"}"`}>${b.exporting ? "Making RLBot bot…" : "Make RLBot bot"}</button>
          ${offWhy(b, busy, why) ? `<div class="cap muted why-off" data-why-off>${esc(offWhy(b, busy, why))}</div>` : ""}
        </div>
      </div>
      ${figuresHTML(b)}
    </section>
    ${lapHTML(b, t)}
    ${b.start && !b.start.done && b.start.kind === "from" ? `<div class="muted" data-start-line style="font-size:13px;">${esc(b.start.line)}</div>` : ""}
    ${b.applying ? `<div class="banner">${esc(b.applying)}…</div>` : ""}
    ${b.restart_problem ? `<div class="banner">${esc(b.restart_problem)}.</div>` : ""}
    ${note ? `<div class="banner ${note.bad ? "bad" : "warn"}" role="status"><div class="grow">${esc(note.text)}</div><button type="button" class="btn small" data-tool="${esc(note.kind)}">${note.bad ? "See why" : "See the result"}</button><button type="button" class="btn small ghost" data-act="note-close">Close</button></div>` : ""}
    ${S.failed && S.failed.bot === b.name ? `<div class="banner bad" role="alert"><div class="grow"><b>${esc(S.failed.what)} did not work.</b> Its last lines:<pre>${esc(S.failed.why)}</pre></div><button type="button" class="btn small" data-act="failed-close">Close</button></div>` : ""}
    ${b.problem ? `<div class="banner bad"><pre>${esc(b.problem)}</pre><button type="button" class="btn small" data-act="open-settings">Open the settings file</button></div>` : ""}
    ${!busy && b.start && !b.start.done && b.start.problem ? `<div class="banner bad"><span>${esc(b.start.problem)}. Pick another in the Settings tab (Transfer learning), or switch transfer learning off there.</span></div>` : ""}
    ${!busy && b.copy_so_far ? `<div class="banner"><span>The copy is saved part-way (${whole(b.copy_so_far.learned)} of ${whole(b.copy_so_far.limit)} replays): Watch it play, Make RLBot bot and Try these rewards (Rewards tab) use that save. <b>Copy replays</b> carries on the copy; training starts once it is finished.</span></div>` : ""}
    ${!busy && b.copy_first && !b.copy_so_far ? `<div class="banner"><span>${esc(b.name)} learns from replays first. <b>Copy replays</b> copies them into it and saves it, then stops: no training yet. Then try rewards on it (Rewards tab, Try these rewards) and press Start training from the copy.</span></div>` : ""}
    ${!busy && copiedOnly(b) ? `<div class="banner"><span>The replays are copied and saved; training has not started. Try rewards on the copied bot in the Rewards tab (Try these rewards), then press Start training from the copy.</span></div>` : ""}`;
}
function renderBot() {
  const b = S.data;
  if (!b) { $("#main").innerHTML = '<div class="empty">Loading…</div>'; return; }
  S.head = headHTML(b);
  $("#main").innerHTML = `
    <div id="bothead">${S.head}</div>
    <nav class="tabs" role="tablist" aria-label="Bot sections">${tabsHTML()}</nav>
    <div id="tab"></div>`;
  renderTab();
}
// The head (status, buttons, sectors, banners) drawn anew only when it changed: a click on it is never lost to
// a redraw.
function updateHead() {
  const box = $("#bothead");
  if (!box || !S.data) return;
  const html = headHTML(S.data);
  if (html !== S.head) { S.head = html; box.innerHTML = html; }
}

const TABS = { overview: "Overview", rewards: "Rewards", starts: "Starts", replays: "Replays in training", settings: "Settings", tools: "Tools", console: "Console", notes: "Notes" };
function tabsHTML() {
  const waiting = Object.keys((S.data && S.data.pending) || {}), mixed = waiting.filter(k => k.startsWith("replay_mix.")).length;
  const counts = { settings: waiting.length - mixed, replays: mixed, rewards: ((S.data && S.data.reward_changes) || []).length, starts: ((S.data && S.data.start_changes) || []).length };
  return Object.keys(TABS).map(t => `<button type="button" role="tab" data-tab="${t}" aria-selected="${S.tab === t}">${TABS[t]}${counts[t] ? `<span class="count">${counts[t]}</span>` : ""}</button>`).join("");
}
// In a short window a tab opened from the head is scrolled up to just under the top band, so its content shows.
function scrollToTabs() {
  const bar = $(".tabs"), band = $("#band");
  if (!bar || window.innerHeight >= 900) return;
  const top = bar.getBoundingClientRect().top, under = band ? band.getBoundingClientRect().bottom : 0;
  if (top > window.innerHeight * 0.35) window.scrollBy({ top: top - under - 8 });
}
function updateTabs() { put($(".tabs"), tabsHTML()); }
function renderTab() {
  const box = $("#tab");
  if (!box || !S.data) return;
  if (TAB_VIEWS[S.tab]) return TAB_VIEWS[S.tab](box);
}
VIEWS.bot = () => renderBot();
on("click", t => {
  if (t.dataset.tab) { S.tab = t.dataset.tab; renderBot(); renderPending(); if (S.tab === "console") pollConsole(); return scrollToTabs(); }
  return SKIP;
});

// A bot's data: what the Python side knows of it (`bot`) and what its sectors and tools need (`bot_tools`).
async function loadBot(name) {
  const [data, tools] = await Promise.all([call("bot", name), call("bot_tools", name)]);
  return { data, tools: tools.error ? {} : tools };
}
async function openBot(name, tab) {
  S.view = "bot"; S.bot = name; S.tab = tab || "overview"; S.data = null; S.tools = null; S.errors = {}; S.consoleText = ""; S.consoleNext = 0; S.showChanges = false; S.tfPick = null;
  S.rw = null; S.rwNotesOpen = {}; S.rwAbout = {}; S.st = null; S.rm = null; S.span = "all"; S.ch = null; S.chHover = null;
  render();
  const { data, tools } = await loadBot(name);
  if (S.view !== "bot" || S.bot !== name) return;  // another bot was opened meanwhile
  if (data.error) { toast(data.error, true); S.view = "home"; render(); return; }
  S.data = data; S.tools = tools; render(); followRunning();
  if (S.tab === "console") pollConsole();
}

// Every 5 s on a bot's page: its head, the tabs' counts and the waiting changes follow the bot, each drawn
// again only when it changed; the Overview's charts when the numbers changed; a form, the console or a note
// being written is never redrawn under the mouse or the keyboard.
async function refreshBot() {
  if (S.view !== "bot" || !S.bot || S.refreshing) return;
  const bot = S.bot;
  S.refreshing = true;
  let loaded;
  try { loaded = await loadBot(bot); } finally { S.refreshing = false; }
  const data = loaded.data;
  if (data.error || S.view !== "bot" || S.bot !== bot || data.name !== bot) return;
  S.data = data; S.tools = loaded.tools;
  renderPending(); updateTabs(); updateHead(); followRunning();
  if (TAB_REFRESH[S.tab]) TAB_REFRESH[S.tab](data);
}

// Follows a command started from here: `done` when it ends well; when it fails, a red banner on the bot's page
// with its last lines (a failed start must never look like nothing happened). A trainer is
// followed until it trains: from then on its page shows how it goes.
async function watchJob(kind, done, what, bot) {
  bot = bot || S.bot;
  for (let i = 0; i < 1200; i++) {
    await new Promise(r => setTimeout(r, 2000));
    const out = await call("job", bot, kind);
    if (out.state === "done") return done();
    if (out.state === "failed") {
      S.failed = { bot, what: what || "That", why: out.why || "" };
      toast((what || "That") + " did not work: the reason is on " + bot + "'s page.", true);
      if (S.view === "bot" && S.bot === bot) updateHead();
      return;
    }
    if (kind === "train" && (((S.home && S.home.bots) || []).find(b => b.name === bot) || {}).status === "training") return;
  }
}

ACTIONS.start = async () => {
  const bot = S.bot, what = startWord(S.data || {});
  const out = await call("start", bot);
  if (out.error) return toast(out.error, true);
  S.failed = null; toast("Starting " + bot + "…"); refreshBot();
  return watchJob("train", () => {}, what, bot);  // a start that fails says why, here
};
ACTIONS["failed-close"] = async () => { S.failed = null; return updateHead(); };
ACTIONS["note-close"] = async () => { S.toolNote = null; return updateHead(); };
ACTIONS["open-settings"] = async () => { const out = await call("open_settings", S.bot); if (out.error) toast(out.error, true); return; };
ACTIONS.stop = async () => {
  const bot = S.bot, out = await call("stop", bot);
  if (out.error) return toast(out.error, true);
  toast("Saving and stopping " + bot + " after this step…");
  if (S.view === "bot" && S.bot === bot) return refreshBot();
  if (S.view === "home") return refreshGarage();
};
ACTIONS.report = async () => { const out = await call("open_report", S.bot); if (out.error) toast(out.error, true); return; };
ACTIONS.wandb = async () => { call("open_url", S.data.wandb); return; };
ACTIONS.folder = async () => { const out = await call("open_folder", S.bot, "bot"); if (out.error) toast(out.error, true); return; };

// ---------- the race number and the two colours ----------
// The dialog shows the plate as it will look while its number and colours are chosen; Python checks each
// choice (check_livery) and says what a livery cannot take.
function liveryModal() {
  const b = S.data, l = b.livery || {};
  S.livery = { number: l.number || "", main: l.main || "", contrast: l.contrast || "", ink: l.ink || "", error: "", seq: 0 };
  const pairs = (S.tools && S.tools.liveries) || [];
  modal(`<h2 class="display">Number and colours of ${esc(b.name)}</h2>
    <div class="livery">
      <div class="livery-show" id="lv-show"></div>
      <div class="livery-form">
        <label class="f">Race number<input id="lv-number" class="field" type="text" inputmode="numeric" maxlength="2" autocomplete="off" data-lv="number" value="${esc(S.livery.number)}"><span class="h">One or two digits, 0 to 99.</span></label>
        ${pairs.length ? `<div class="f">Ready-made colours<div class="swatches">${pairs.map(([main, second]) => `<button type="button" class="swatch" data-lv-pair="${esc(main)},${esc(second)}" style="--a:${esc(main)};--b:${esc(second)};" aria-label="Main ${esc(main)}, second ${esc(second)}"></button>`).join("")}</div></div>` : ""}
        <div class="livery-two">
          <label class="f">Main colour<span class="colour"><input type="color" id="lv-main-pick" data-lv="main" value="${esc(S.livery.main)}" aria-label="Pick the main colour"><input id="lv-main" class="field mono" type="text" maxlength="7" autocomplete="off" data-lv="main" value="${esc(S.livery.main)}"></span></label>
          <label class="f">Second colour<span class="colour"><input type="color" id="lv-contrast-pick" data-lv="contrast" value="${esc(S.livery.contrast)}" aria-label="Pick the second colour"><input id="lv-contrast" class="field mono" type="text" maxlength="7" autocomplete="off" data-lv="contrast" value="${esc(S.livery.contrast)}"></span></label>
        </div>
      </div>
    </div>
    <div id="lv-error"></div>
    <div class="foot"><button type="button" class="btn" data-close="1">Cancel</button><button type="button" class="btn primary" data-act="livery-save">Save</button></div>`);
  drawLivery();
}
function drawLivery() {
  const l = S.livery, box = $("#lv-show");
  if (!box) return;
  put(box, `${plate({ livery: l }, 96)}<div><div class="cap muted">How it will look</div><div class="display name">${esc(S.data.name)}</div></div>`);
  put($("#lv-error"), l.error ? `<div class="banner bad">${esc(l.error)}</div>` : "");
}
// A choice in the dialog: shown at once, then checked, which also gives the text colour that reads on it.
async function liveryChanged() {
  const l = S.livery, seq = ++l.seq;
  drawLivery();
  const out = await call("check_livery", l.number, l.main, l.contrast);
  if (!S.livery || S.livery.seq !== seq) return;  // a newer choice was made meanwhile
  l.error = out.error || "";
  if (out.livery) l.ink = out.livery.ink;
  drawLivery();
}
ACTIONS.livery = () => liveryModal();
ACTIONS["livery-save"] = async () => {
  const l = S.livery, bot = S.bot;
  const out = await call("set_livery", bot, l.number, l.main, l.contrast);
  if (out.error) { l.error = out.error; return drawLivery(); }
  if (S.data && S.bot === bot) S.data.livery = out.livery;
  closeModal(); updateHead();
  return loadHome();
};
on("click", t => {
  if (!t.dataset.lvPair || !S.livery) return SKIP;
  const [main, second] = t.dataset.lvPair.split(",");
  S.livery.main = main; S.livery.contrast = second;
  ["main", "contrast"].forEach(key => { $("#lv-" + key).value = S.livery[key]; $("#lv-" + key + "-pick").value = S.livery[key]; });
  liveryChanged();
});
on("input", el => {
  if (!el.dataset.lv || !S.livery) return;
  const key = el.dataset.lv, value = el.value.trim();
  S.livery[key] = key === "number" ? value : value.toUpperCase();
  if (key !== "number" && /^#[0-9a-f]{6}$/i.test(value)) {  // the picker and the text field show the same colour
    const twin = $(el.type === "color" ? "#lv-" + key : "#lv-" + key + "-pick");
    if (twin && twin.value.toUpperCase() !== value.toUpperCase()) twin.value = el.type === "color" ? value.toUpperCase() : value.toLowerCase();
  }
  liveryChanged();
});
