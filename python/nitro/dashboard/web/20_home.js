// The Garage page: the bot that is training on top ("on track"), the getting-set-up checklist, and every other
// bot as one row (a bay) with its number plate, its state, its newest figures and the one action that fits.
// Before the first bot it is the welcome instead: what NitroLearn does and its three first steps.

// ---------- drawings ----------
const LIVERY_HEX = /^#[0-9a-f]{6}$/i;
// A generic race car from the side, painted in a bot's livery: its main colour, its second colour along the
// side, on the wing and in the wheels, and its number on the door.
function carSVG(livery, width = 300) {
  const l = livery || {};
  const main = LIVERY_HEX.test(l.main || "") ? l.main : "#8C8A84", second = LIVERY_HEX.test(l.contrast || "") ? l.contrast : "#141413";
  const number = esc(l.number || "");
  return `<svg class="livery-car" viewBox="0 0 320 120" width="${Number(width) || 300}" role="img" aria-label="A race car with number ${number} in the bot's colours">
    <g fill="${main}" opacity="0.6"><rect x="0" y="50" width="56" height="5"/><rect x="16" y="64" width="44" height="5"/><rect x="4" y="78" width="52" height="5"/></g>
    <path d="M66 88 L72 62 L124 55 L152 33 L216 33 L252 55 L298 63 L309 75 L307 88 Z" fill="${main}" stroke="var(--rule2)" stroke-width="1.5" stroke-linejoin="miter"/>
    <path d="M73 64 L124 57 L252 57 L297 65 L306 76 L69 76 Z" fill="${second}"/>
    <path d="M157 38 L213 38 L241 56 L135 56 Z" fill="#141413"/>
    <path d="M184 38 L190 38 L194 56 L186 56 Z" fill="${main}"/>
    <rect x="56" y="37" width="46" height="7" fill="${second}" stroke="var(--rule2)" stroke-width="1"/><rect x="82" y="44" width="6" height="14" fill="${second}"/>
    <rect x="288" y="66" width="12" height="5" fill="#F3F1EC"/>
    <rect x="300" y="86" width="14" height="4" fill="#141413"/>
    <rect x="164" y="59" width="38" height="24" fill="#F3F1EC"/>
    <text x="183" y="78.5" text-anchor="middle" font-size="20" fill="#141413" style="font-family:var(--sans);font-stretch:125%;font-weight:900;font-style:italic;">${number}</text>
    <g stroke="var(--rule2)" stroke-width="1.5"><circle cx="112" cy="88" r="21" fill="#141413"/><circle cx="258" cy="88" r="21" fill="#141413"/></g>
    <g fill="${second}" stroke="${main}" stroke-width="2.5"><circle cx="112" cy="88" r="9"/><circle cx="258" cy="88" r="9"/></g>
    <rect x="109" y="85" width="6" height="6" fill="#141413"/><rect x="255" y="85" width="6" height="6" fill="#141413"/>
    <rect x="34" y="109" width="286" height="2" fill="var(--rule2)"/>
  </svg>`;
}
// A bot's reward so far as one small line, its newest value marked.
function sparkSVG(points) {
  if (!points || points.length < 2) return "";
  const w = 132, h = 30, low = Math.min(...points), span = (Math.max(...points) - low) || 1;
  const x = i => 2 + i / (points.length - 1) * (w - 9), y = v => h - 5 - (v - low) / span * (h - 10);
  const line = points.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  const last = points.length - 1;
  return `<svg class="g-spark" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" role="img" aria-label="Reward so far"><line x1="0" x2="${w}" y1="${h - 1}" y2="${h - 1}" stroke="var(--rule)" stroke-width="1"/><polyline points="${line}" fill="none" stroke="var(--dim)" stroke-width="2" stroke-linejoin="miter"/><rect x="${(x(last) - 3).toFixed(1)}" y="${(y(points[last]) - 3).toFixed(1)}" width="6" height="6" fill="var(--text)"/></svg>`;
}

// ---------- words and numbers ----------
function compact(n) {
  n = Number(n || 0);
  if (n >= 1e6) return (n / 1e6).toFixed(n >= 1e7 ? 0 : 1) + "M";
  if (n >= 1e4) return Math.round(n / 1e3) + "k";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + "k";
  return String(Math.round(n));
}
const NETWORK_WORDS = { mlp: "Plain network", attention: "Attention network", attention_communication: "Attention network, talking", transformer: "Transformer network" };
// What a bot is, as small capitals: its game size, its network, its parameters once it has a save.
function botLine(b) {
  const size = b.parameters && b.parameters.total ? compact(b.parameters.total) + " parameters" : "";
  return [b.team_size ? `${b.team_size}v${b.team_size}` : "", b.encoder ? networkWord(b) : "", size, b.start && b.start.kind === "from" ? b.start.chip : ""].filter(Boolean).map(esc).join(" · ");
}
function sinceWord(t) {
  if (!t) return "–";
  const s = Math.max(0, Date.now() / 1000 - t);
  if (s < 60) return "under 1 min";
  if (s < 3600) return Math.round(s / 60) + " min";
  if (s < 86400 * 2) return Math.floor(s / 3600) + " h " + Math.round(s % 3600 / 60) + " min";
  return Math.round(s / 86400) + " days";
}
function homeBots() { return (S.home && S.home.bots) || []; }
function homeBot(name) { return homeBots().find(b => b.name === name); }
// What `garage` knows of a bot beside `home`: its newest save's time, its reward so far, its copy's progress.
function garageOf(name) { return ((S.garage && S.garage.bots) || {})[name] || {}; }
function copyShare(b) {
  const c = garageOf(b.name).copy || b.copy_so_far;
  return c && c.limit ? Math.min(1, c.learned / c.limit) : null;
}
function botRuns(b) { return b.status === "training" || b.status === "copying"; }
// What a running bot does, in plain words.
function doingWord(b) {
  if (b.stopping) return "Stopping";
  if (b.status === "copying") { const share = copyShare(b); return "Learning from replays" + (share != null ? " " + pct(share) : ""); }
  if (b.status === "starting") return "Starting";
  return b.checking ? "Trying rewards" : "Training";
}
// The bot on top: the one training (the one that started first, of several), else one learning from replays.
function heroBot(bots) {
  const order = b => [b.status === "training" ? 0 : 1, b.running_since || 0, b.name];
  return bots.filter(botRuns).sort((a, b) => { const x = order(a), y = order(b); return x[0] - y[0] || x[1] - y[1] || (x[2] < y[2] ? -1 : 1); })[0] || null;
}

// ---------- the bot on track ----------
function heroStats(b) {
  const g = garageOf(b.name), fig = (v, k) => `<span class="g-fig"><b class="num">${esc(v)}</b><span class="cap muted">${esc(k)}</span></span>`;
  const copy = b.status === "copying" ? g.copy || b.copy_so_far : null;
  const first = copy ? fig(Number(copy.learned || 0).toLocaleString("en-US"), "replays copied") + fig(pct(copyShare(b)), "of " + Number(copy.limit || 0).toLocaleString("en-US"))
    : fig(b.status === "copying" ? "–" : steps(b.agent_steps), "steps") + fig(b.counted !== false && b.steps_per_second ? compact(b.steps_per_second) : "–", "steps a second");
  return first + fig(b.saves, b.saves === 1 ? "save" : "saves") + fig(b.saves ? sinceWord(g.last_save) : "–", b.saves ? "since the last save" : "no save yet");
}
function heroActions(b) {
  const name = esc(b.name);
  return `<button type="button" class="btn primary" id="gh-stop" data-act="garage-stop" data-name="${name}" ${b.stopping ? 'disabled title="It saves after the step it is on, then stops"' : ""}>${ICON.stop}${b.stopping ? "Stopping…" : "Stop and save"}</button>
    <button type="button" class="btn strong" id="gh-watch" data-act="garage-watch" data-name="${name}" ${b.saves ? "" : 'disabled title="It needs a save first: the first comes 15 minutes in by defaultto training"'}>${ICON.eye}Watch it play</button>`;
}
// What else runs beside the bot on top, and what waits to.
function alsoHTML(bots, hero) {
  const others = bots.filter(b => b !== hero && (botRuns(b) || b.status === "starting" || b.checking));
  const parts = others.map(b => `<span><button type="button" class="btn link" data-bot="${esc(b.name)}">${esc(b.name)}</button> ${esc(botRuns(b) ? doingWord(b).toLowerCase() : b.status === "starting" ? "waits to start" : "tries rewards")}</span>`);
  if (S.garage && S.garage.converting) parts.push(`<span>Replays are being converted (<button type="button" class="btn link" data-go="resources">Resources</button>)</span>`);
  return parts.length ? `<span class="cap muted">${hero ? "Also at work" : "At work"}</span>${parts.join("")}` : "";
}
// The training bot's state tag: On track once its numbers since this start are in.
function heroTag(b) {
  if (b.stopping) return '<span class="tag strong">Stopping</span>';
  if (b.status === "training" && !b.counted) return `<span class="tag strong" title="${esc(firstRoundWord())}">Warming up</span>`;
  return '<span class="tag good">On track</span>';
}
function drawHero(bots, hero) {
  const box = $("#g-hero");
  if (!hero) {
    const also = alsoHTML(bots, null);
    return put(box, bots.length ? `<div class="pitlane"><span class="tag">Pit lane</span><span>Nothing is training right now.${also ? "" : " Start a bot from its row below."}</span>${also ? `<span class="g-also">${also}</span>` : ""}</div>` : "");
  }
  // The frame is drawn once a bot; its parts each only when they change, so no click on a button is lost.
  put(box, `<section class="ontrack" data-hero="${esc(hero.name)}" aria-label="The bot that is training">
      <div class="ontrack-main"><div class="ontrack-id" id="gh-id"></div><div class="ontrack-foot"><div class="ontrack-stats" id="gh-stats"></div><div class="ontrack-actions" id="gh-actions"></div></div></div>
      <div class="ontrack-car" id="gh-car"></div>
    </section><div class="g-also" id="gh-also"></div>`);
  const line = [doingWord(hero), botLine(hero)].filter(Boolean).join(" · ");
  put($("#gh-id"), `${plate(hero, 92)}<div class="ontrack-who">
      <div class="row">${heroTag(hero)}<span class="cap muted">${hero.stopping ? esc(STOPPING) : hero.applying ? esc(hero.applying) : line}</span></div>
      <button type="button" class="ontrack-name display" id="gh-name" data-bot="${esc(hero.name)}" title="Open ${esc(hero.name)}">${esc(hero.name)}</button>
    </div>`);
  put($("#gh-stats"), heroStats(hero));
  put($("#gh-actions"), heroActions(hero));
  put($("#gh-car"), carSVG(hero.livery));
  put($("#gh-also"), alsoHTML(bots, hero));
}

// ---------- getting set up ----------
// The four things a new user does once: each "done", "todo", "checking" or "problem".
function setupSteps(bots) {
  const d = S.doctor, checks = d && d.checks;
  const arena = checks && checks.find(c => /mesh/i.test(c.name));
  const state = list => !d || d.running ? "checking" : !checks ? "problem" : list.some(c => !c || c.state === "problem") ? "problem" : "done";
  return [
    { key: "pc", state: state(checks ? checks.filter(c => c !== arena) : []), done: "This PC is ready", todo: "This PC is not ready", checking: "Checking this PC…" },
    { key: "arena", state: state([arena]), done: "Arena files found", todo: "Arena files not found", checking: "Looking for the arena files…" },
    { key: "bot", state: bots.length ? "done" : "todo", done: "First bot made", todo: "Make your first bot" },
    { key: "rlbot", state: rlbotFound() ? "done" : "todo", done: "RLBot v5 found", todo: "Tell NitroLearn where RLBot is" },
  ];
}
const SETUP_TICK = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3.4" stroke-linecap="square" aria-hidden="true"><path d="M5 12l5 5 9-10"/></svg>';
function laneHTML(bots) {
  const prefs = (S.home && S.home.prefs) || {};
  if (prefs.setup_done || !bots.length) return "";  // before the first bot, the welcome has the steps
  const list = setupSteps(bots), done = list.filter(s => s.state === "done").length;
  if (done === list.length) return `<div class="setup" role="status"><span class="cap">All set</span><span class="setup-step done"><span class="setup-tick">${SETUP_TICK}</span>This PC, the arena files, your first bot and RLBot are ready.</span><span class="grow"></span><button type="button" class="btn small ghost" id="setup-dismiss" data-act="setup-dismiss" title="Hides this line for good">Dismiss</button></div>`;
  const item = s => s.state === "done" ? `<span class="setup-step done"><span class="setup-tick">${SETUP_TICK}</span>${esc(s.done)}</span>`
    : s.state === "checking" ? `<span class="setup-step muted"><span class="setup-tick todo"></span>${esc(s.checking)}</span>`
    : s.state === "problem" ? `<span class="setup-step"><span class="setup-tick hazard" aria-hidden="true"></span><button type="button" class="btn link" id="setup-${s.key}" data-go="doctor" title="Check this PC says what is wrong and how to fix it">${esc(s.todo)}</button></span>`
    : `<span class="setup-step"><span class="setup-tick todo"></span><b>${esc(s.todo)}</b></span>`;
  const next = list.find(s => s.state === "problem") || list.find(s => s.state === "todo");
  const button = !next ? "" : next.state === "problem" ? `<button type="button" class="btn small primary" id="setup-next" data-go="doctor">See how to fix it</button>`
    : next.key === "bot" ? ""  // the empty garage's own big button makes it
    : `<button type="button" class="btn small primary" id="setup-next" data-act="rlbot-choose">Choose RLBot…</button>`;
  return `<div class="setup" aria-label="Getting set up"><span class="cap">Getting set up · ${done} of ${list.length}</span>${list.map(item).join("")}<span class="grow"></span>${button}</div>`;
}
// The checks of Check this PC, once a session and in the background, while the checklist still shows. They
// start a moment after the Garage page is drawn, and wait for the next visit when it was left meanwhile.
function askSetup() {
  if (S.setupAsked || S.doctor || ((S.home && S.home.prefs) || {}).setup_done || !homeBots().length) return;
  S.setupAsked = true;
  setTimeout(() => {
    if (S.doctor) return;
    if (S.view !== "home") { S.setupAsked = false; return; }
    Promise.resolve(ACTIONS["doctor-run"]()).then(() => { if (S.view === "home") drawGarage(); });
    drawGarage();
  }, 1200);
}

// ---------- the welcome (no bot yet) ----------
// Check this PC's answer as the welcome's first step shows it: its state, a line, and each problem.
function welcomePC() {
  const d = S.doctor;
  if (!d) return { state: "todo", line: "Sees whether this PC can train: the graphics card, the arena files, room on the disk.", button: "Check this PC" };
  if (d.running) return { state: "checking", line: "Checking… a few seconds, sometimes a minute the first time.", button: "Checking…" };
  if (d.error) return { state: "problem", line: "The checks did not finish.", problems: [{ name: "Check this PC", message: d.error.split("\n")[0] }], button: "Check again" };
  const problems = (d.checks || []).filter(c => c.state === "problem");
  if (problems.length) return { state: "problem", line: `${problems.length === 1 ? "1 thing needs" : problems.length + " things need"} fixing before training:`, problems, button: "Check again" };
  const n = d.warnings || 0;
  return { state: "done", line: "This PC is ready." + (n ? ` ${n === 1 ? "1 warning" : n + " warnings"}: training works, but read ${n === 1 ? "it" : "them"} on Check this PC.` : ""), button: "Check again", warned: n > 0 };
}
function welcomeStep(n, key, state, title, line, button, extra, later) {
  const mark = state === "done" ? SETUP_TICK : state === "problem" ? "" : esc(n);
  return `<li class="wstep" data-step="${key}" data-state="${state}">
      <span class="wmark${state === "problem" ? " hazard" : ""}" aria-hidden="true">${mark}</span>
      <div class="wbody"><div class="wtitle">${esc(title)}${later ? '<span class="tag">Later</span>' : ""}</div><div class="wline" id="wel-${key}-line">${esc(line)}</div>${extra || ""}</div>
      ${button}
    </li>`;
}
function welcomeHTML() {
  const logo = $("#band .lockup svg"), fair = $("#foot span");
  const pc = welcomePC();
  const problems = (pc.problems || []).map(c => `<li><button type="button" class="btn link" data-go="doctor" title="Check this PC says how to fix it">${esc(c.name)}</button> <span class="muted">${esc(c.message)}</span></li>`).join("");
  const pcExtra = (problems ? `<ul class="wproblems">${problems}</ul>` : "") + (pc.warned ? '<button type="button" class="btn link" data-go="doctor">See the warnings</button>' : "");
  const rlbot = rlbotFound() ? "done" : "todo";
  return `<section class="welcome" aria-label="Welcome to NitroLearn">
      <div class="welcome-top"><div class="welcome-head">
        <div class="welcome-lockup">${logo ? logo.outerHTML : ""}<span class="wordmark"><b>Nitro</b>Learn</span></div>
        <p class="welcome-what">NitroLearn trains Rocket League bots on this PC: a bot plays thousands of games against itself and gets better at what you reward.</p>
        <p class="welcome-fair">${esc(fair ? fair.textContent : "")}</p>
      </div><div class="ontrack-car">${carSVG({ number: "01", main: "#FFD60A", contrast: "#141413" }, 260)}</div></div>
      <ol class="welcome-steps">
        ${welcomeStep(1, "pc", pc.state, "Check this PC", pc.line, `<button type="button" class="btn small" id="wel-pc" data-act="welcome-check" ${pc.state === "checking" ? "disabled" : ""}>${esc(pc.button)}</button>`, pcExtra)}
        ${welcomeStep(2, "bot", "todo", "Make your first bot", "A minute to make. Almost everything about it can be changed later.", '<button type="button" class="btn small primary" id="g-first" data-act="newbot">Make your first bot</button>')}
        ${rlbot === "done"
          ? welcomeStep(3, "rlbot", rlbot, "RLBot v5 is found", "RLBot plays your bot in a real Rocket League match, offline. Open RLBot (top right) starts it.", '<button type="button" class="btn small" id="wel-rlbot" data-act="rlbot-open">Open RLBot</button>', "", true)
          : welcomeStep(3, "rlbot", rlbot, "Tell NitroLearn where RLBot is", "RLBot plays your bot in a real Rocket League match, offline. Only needed once a bot has trained.", '<button type="button" class="btn small" id="wel-rlbot" data-act="rlbot-choose">Choose RLBot…</button>', `<div class="rli" data-where="welcome">${rlbotInstallHTML("welcome")}</div>`, true)}
      </ol>
      <div class="welcome-foot">
        <button type="button" class="btn link" id="g-unpack" data-act="unpack" title="A bot packed on another computer (its Tools tab: Pack for another computer), such as a rented cloud machine">Add a packed bot instead</button>
        <span class="grow"></span>
        <button type="button" class="btn small ghost" id="welcome-dismiss" data-act="setup-dismiss" title="Hides this welcome for good">Hide this welcome</button>
      </div>
    </section>`;
}
ACTIONS["welcome-check"] = () => {
  const done = ACTIONS["doctor-run"]();
  drawGarage();
  return Promise.resolve(done).then(() => { if (S.view === "home") drawGarage(); });
};

// ---------- the bays ----------
// The one button of a bot's row: what it says, what it does ("start", "settings" or "open") and its look.
function bayAction(b) {
  if (b.status === "problem") return ["Fix settings", "settings", "strong", "Its settings file has a mistake: its Settings tab says what it is"];
  if (botRuns(b) || b.status === "starting" || b.checking || b.applying) return ["See progress", "open", "", ""];
  if (b.pending_count) return ["See changes", "open", "", "Changes wait for Apply on its page"];
  const done = b.start && b.start.done;
  const label = done && b.agent_steps ? "Carry on" : !done && b.copy_first ? "Copy replays" : "Start training";
  return [label, "start", b.saves ? "" : "primary", startWord(b)];
}
function bayHTML(b) {
  const g = garageOf(b.name), name = esc(b.name);
  const look = b.status === "training" ? "good" : b.status === "problem" ? "bad" : botRuns(b) || b.status === "starting" || b.checking ? "strong" : "";
  const word = botRuns(b) ? doingWord(b) : statusWord(b);
  const [label, what, kind, title] = bayAction(b);
  const spark = sparkSVG(g.reward);
  return `<button type="button" class="bay" id="bay-${name}" data-bot="${name}" title="Open ${name}">
      ${plate(b)}
      <span class="who"><span class="name display">${name}</span><span class="cap muted"${b.start && b.start.kind === "from" ? ` title="${esc(b.start.line)}"` : ""}>${botLine(b) || (b.status === "problem" ? "Its settings file has a mistake" : "")}</span></span>
      <span class="state"><span class="tag ${look}">${esc(word)}</span>${b.pending_count ? `<span class="tag hot">${b.pending_count} waiting for Apply</span>` : ""}</span>
      <span class="g-fig"><b class="num">${steps(b.agent_steps)}</b><span class="cap muted">steps</span></span>
      <span class="g-fig"><b class="num">${b.saves}</b><span class="cap muted">${b.saves === 1 ? "save" : "saves"}</span></span>
      <span class="trend">${spark}<span class="cap muted">${b.saves ? "Last save " + esc(ago(g.last_save || b.updated)) : "No save yet"}</span></span>
    </button>
    <button type="button" class="btn small ${kind}" id="bay-act-${name}" data-act="bay-go" data-name="${name}" data-do="${what}" ${title ? `title="${esc(title)}"` : ""}>${esc(label)}</button>`;
}
function drawList(bots, hero) {
  const box = $("#g-list"), rows = bots.filter(b => b !== hero);
  const tools = `<button type="button" class="btn small" id="g-unpack" data-act="unpack" title="A bot packed on another computer (its Tools tab: Pack for another computer), such as a rented cloud machine">Add a packed bot…</button>`;
  if (!bots.length && !((S.home && S.home.prefs) || {}).setup_done) return put(box, welcomeHTML());
  if (!bots.length) {
    return put(box, `<section class="firstrun"><div class="firstrun-words">
        <h1 class="display">Your garage is empty</h1>
        <p>NitroLearn trains Rocket League bots on this PC: a bot plays thousands of games against itself and gets better at what you reward. When it is good enough, it plays a real match, offline, through RLBot.</p>
        <p class="muted">A first bot takes a minute to make. Almost everything about it can be changed later.</p>
        <div class="row"><button type="button" class="btn primary big" id="g-first" data-act="newbot">Make your first bot</button>${tools}</div>
      </div><div class="ontrack-car">${carSVG({ number: "01", main: "#FFD60A", contrast: "#141413" })}</div></section>`);
  }
  // The frame holds one box a bot; each row is drawn anew only when it changed.
  put(box, `<div class="heading"><h1 class="display">In the garage</h1><div class="row">${tools}<button type="button" class="btn small primary" id="g-new" data-act="newbot">+ New bot</button></div></div>
    ${rows.length ? `<div class="bays">${rows.map(b => `<div class="botcard" data-bot="${esc(b.name)}"></div>`).join("")}</div>` : `<div class="muted" style="padding:14px 4px;">${esc(hero.name)} is your only bot.</div>`}
    <div class="muted garage-note">Make a bot, train it, change its settings, watch it play and turn it into an RLBot bot, all from here. Changes you make wait until you press Apply.</div>`);
  const boxes = box.querySelectorAll(".botcard");
  rows.forEach((b, i) => put(boxes[i], bayHTML(b)));
}

// ---------- drawing and refreshing ----------
function drawGarage() {
  if (!$("#g-list")) return;
  keepFocus(() => {
    const bots = homeBots(), hero = heroBot(bots);
    drawHero(bots, hero);
    put($("#g-lane"), laneHTML(bots));
    drawList(bots, hero);
  });
}
async function loadGarage() {
  if (S.garageLoading) return;
  S.garageLoading = true;
  let out;
  try { out = await call("garage"); } finally { S.garageLoading = false; }
  if (!out.error) S.garage = out;
  if (S.view === "home") drawGarage();
}
// `changedOnly` (the 5-second refresh): only what changed is drawn again.
function renderHome(changedOnly) {
  if (!changedOnly || !$("#g-list")) $("#main").innerHTML = '<div id="g-hero"></div><div id="g-lane"></div><div id="g-list"></div>';
  drawGarage();
  loadGarage();
  askSetup();
}
VIEWS.home = () => renderHome();

// ---------- actions ----------
// The bot page's own actions (Stop and save, Watch it play, Start), aimed at a bot from the Garage: they read
// which bot from S.bot and S.data.
function garageAct(act, name) {
  const b = homeBot(name);
  if (!b) return;
  S.bot = name; S.data = b;
  return ACTIONS[act]();
}
async function refreshGarage() { await loadHome(); if (S.view === "home") renderHome(true); }
ACTIONS["garage-stop"] = el => garageAct("stop", el.dataset.name);
ACTIONS["garage-watch"] = el => garageAct("watch", el.dataset.name);
ACTIONS["bay-go"] = el => {
  const name = el.dataset.name, what = el.dataset.do;
  if (what !== "start") return openBot(name, what === "settings" ? "settings" : undefined);
  garageAct("start", name);  // it follows the start by itself; the row shows it meanwhile
  setTimeout(refreshGarage, 700);
};
ACTIONS["setup-dismiss"] = async () => {
  const out = await call("set_pref", "setup_done", true);
  if (out.error) return toast(out.error, true);
  if (S.home) S.home.prefs = out.prefs || Object.assign({}, S.home.prefs, { setup_done: true });
  if (S.view === "home") drawGarage();
};
