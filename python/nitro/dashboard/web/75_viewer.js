// Watching a bot play: the "Watch it play" dialog (live, record a game, the recorded games) and the 3D match
// viewer, which fills the window over the page. The viewer's files are not in this page (the window takes a
// page of at most 2 MB): they come through the bridge the first time a game is watched and are run once.
// Every game shown gets a viewer of its own (NV_MAKE), ended and taken out of the page when it is closed.
const WATCH = {
  files: null,   // a promise of the viewer's files being in the page
  markup: "",    // the viewer's markup, put into its box at every opening
  arenas: {},    // bot -> a promise of its arena's meshes (null: the viewer draws a generated arena)
  prefs: null,   // the viewer's remembered choices
  saving: 0,     // the timer that sends them to Python
  open: null,    // what shows now: { bot, live, nv, started, closed }
  pick: null,    // the dialog's choices: { bot, size, seconds }
  job: null,     // a recording under way: { bot, since, text, cancelled }
};
const WATCH_LENGTHS = [[30, "30 s"], [60, "1 min"], [120, "2 min"], [300, "5 min"]];
const WATCH_RECORD = '<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><circle cx="12" cy="12" r="7"/></svg>';
// The viewer that shows, for a look from outside (tests, the developer tools); null when none does.
window.NV = null;

// ---------- the viewer's files and data, through the bridge ----------
// A big text from Python comes in pieces: a few calls, not one large answer.
async function watchText(method, ...args) {
  const first = await call(method, ...args, 0);
  if (first.error) throw new Error(first.error);
  const rest = await Promise.all(Array.from({ length: Math.max(0, (first.pieces || 0) - 1) }, (_, i) => call(method, ...args, i + 1)));
  const failed = rest.find(piece => piece.error);
  if (failed) throw new Error(failed.error);
  return [first, ...rest].map(piece => piece.text || "").join("");
}
// Puts the viewer's style and its two scripts into the page, once: three.js first, the viewer's own uses it.
function watchFiles() {
  if (!WATCH.files) {
    WATCH.files = (async () => {
      const [three, script, style, markup] = await Promise.all(["three", "script", "style", "markup"].map(part => watchText("viewer_assets", part)));
      const sheet = document.createElement("style");
      sheet.textContent = style;
      document.head.appendChild(sheet);
      for (const text of [three, script]) {
        const tag = document.createElement("script");
        tag.textContent = text;
        document.head.appendChild(tag);
      }
      if (typeof THREE === "undefined" || typeof NV_MAKE !== "function") throw new Error("the match viewer could not be loaded");
      WATCH.markup = markup;
    })();
    WATCH.files.catch(() => { WATCH.files = null; });  // asked for again at the next try
  }
  return WATCH.files;
}
function watchArena(bot) {
  if (!WATCH.arenas[bot]) WATCH.arenas[bot] = watchText("viewer_arena", bot).then(text => text || null).catch(() => { delete WATCH.arenas[bot]; return null; });
  return WATCH.arenas[bot];
}
async function watchPrefs() {
  if (!WATCH.prefs) WATCH.prefs = (await call("viewer_prefs")).prefs || {};
}
// The viewer's choices (camera, speed, picture quality) are kept with the dashboard's own.
function watchKeepPrefs(prefs) {
  WATCH.prefs = { ...prefs };
  clearTimeout(WATCH.saving);
  WATCH.saving = setTimeout(() => call("set_viewer_prefs", WATCH.prefs), 400);
}

// ---------- the viewer over the window ----------
// The box the viewer fills. What happens inside stays there: the page underneath never takes the viewer's
// clicks and keys for its own.
function watchBox() {
  let box = $("#watching");
  if (box) return box;
  box = document.createElement("div");
  box.id = "watching";
  box.hidden = true;
  for (const kind of ["click", "change", "input", "keydown", "mousedown", "mouseup"]) box.addEventListener(kind, e => e.stopPropagation());
  box.addEventListener("click", e => { if (e.target.closest("[data-watch-close]")) closeViewer(); });
  box.addEventListener("keydown", e => { if (e.key === "Escape" && WATCH.open && !WATCH.open.started) closeViewer(); });
  document.body.appendChild(box);
  return box;
}
// Keys pressed while the focus is nowhere in the viewer (what held it went away) are still the viewer's.
document.addEventListener("keydown", e => {
  const box = $("#watching");
  if (!WATCH.open || !box || box.contains(e.target)) return;
  e.stopPropagation();
  if (e.key === "Escape") return closeViewer();
  const inside = box.querySelector(".watch-wait button, .nv-root");
  if (inside) inside.focus({ preventScroll: true });
}, true);
// While the viewer shows, the page underneath rests: it takes no click or key, and its timers ask for nothing
// and redraw nothing (S.resting, 90_boot.js).
function watchRest(on) {
  S.resting = on;
  document.documentElement.toggleAttribute("data-watching", on);
  for (const el of document.body.children) if (el.id !== "watching" && el.id !== "toasts") el.inert = on;
}

// Opens the viewer over the window with a live game of `bot` (`what.live`: its options) or one of its recorded
// games (`what.id`).
async function watchShow(bot, what) {
  closeViewer();
  closeModal();
  const session = { bot, live: !!what.live, nv: null, started: false, closed: false };
  WATCH.open = session;
  watchRest(true);
  const box = watchBox();
  box.innerHTML = `<div class="watch-wait" role="status"><div class="display">${what.live ? "Starting the live game" : "Opening the game"}</div>
    <div class="watch-note">${what.live ? `${esc(bot)}'s newest save is being loaded and its first game set up. This can take up to a minute.` : `A recorded game of ${esc(bot)}.`}</div>
    <button type="button" class="btn" data-watch-close="1">Cancel</button></div>`;
  box.hidden = false;
  box.querySelector("button").focus();
  try {
    const [data, arena] = await Promise.all([what.live ? watchLiveData(session, what.live) : watchRecorded(bot, what.id), watchArena(bot), watchFiles(), watchPrefs()]);
    if (session.closed) return;
    const host = document.createElement("div");
    host.className = "nv-root";
    host.tabIndex = -1;
    host.innerHTML = WATCH.markup;
    box.prepend(host);
    const nv = NV_MAKE();
    session.nv = window.NV = nv;
    nv.store = { load: () => ({ ...WATCH.prefs }), save: watchKeepPrefs };
    await nv.start(data, { host, arena, embedded: true, onClose: closeViewer, live: what.live ? cursor => watchPoll(session, cursor) : null });
    if (session.closed) return;
    session.started = true;
    box.querySelector(".watch-wait").remove();
    host.focus({ preventScroll: true });
  } catch (err) {
    if (session.closed) return;
    closeViewer();
    toast(String((err && err.message) || err), true);
  }
}
// Closes the viewer: its loop and graphics memory are freed (its canvas leaves the page), a live game is ended,
// and the page underneath wakes up.
function closeViewer() {
  const session = WATCH.open;
  if (!session) return;
  session.closed = true;
  WATCH.open = null;
  if (session.nv) session.nv.stop();
  window.NV = null;
  if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
  const box = watchBox();
  box.replaceChildren();
  box.hidden = true;
  if (session.live) call("watch_live_stop", session.bot);
  watchRest(false);
  const again = $('[data-act="watch"]');
  if (again) again.focus({ preventScroll: true });
  if (S.view === "bot") refreshBot();
}

async function watchRecorded(bot, id) {
  return JSON.parse(await watchText("recording", bot, id || "last"));
}
// Starts the live game and waits for its first line: the page data the viewer starts with.
async function watchLiveData(session, options) {
  const out = await call("watch_live_start", session.bot, options);
  if (session.closed) { if (!out.error) call("watch_live_stop", session.bot); return null; }
  if (out.error) throw new Error(out.error);
  for (;;) {
    await new Promise(r => setTimeout(r, 250));
    if (session.closed) return null;
    const now = await call("watch_live_state");
    if (now.state === "playing") return now.data;
    if (now.state !== "starting") {
      watchLiveFailed(session.bot, now.why);
      throw new Error(`The live game of ${session.bot} did not start: ${now.why ? "the reason is on its page" : (now.error || "its Console tab may say why")}.`);
    }
  }
}
// A live game that went by itself leaves its last lines on the bot's page, as every command that fails does.
function watchLiveFailed(bot, why) {
  if (!why) return;
  S.failed = { bot, what: "The live game", why };
  if (S.view === "bot" && S.bot === bot) updateHead();
}
// What the live game has played since `cursor`, for the viewer. A game that ended by itself is said so, once.
async function watchPoll(session, cursor) {
  const answer = await call("viewer_poll", cursor);
  if (answer.error) throw new Error(answer.error);  // the viewer asks again
  if (answer.closed && !session.closed && !session.ended) {
    session.ended = true;
    call("watch_live_state").then(now => {
      if (session.closed) return;
      const why = now.state === "ended" ? now.why : "";
      watchLiveFailed(session.bot, why);
      if (!session.started) closeViewer();
      toast(`The live game of ${session.bot} stopped by itself${why ? ": the reason is on its page" : ""}.`, true);
    });
  }
  return answer;
}

// ---------- the dialog ----------
function watchSeg(name, options, current) {
  return `<div class="seg" data-seg="${name}">${options.map(([v, label]) => `<button type="button" data-v="${v}" aria-pressed="${String(v) === String(current)}">${label}</button>`).join("")}</div>`;
}
function watchRecordingLine(r) {
  const save = r.steps != null ? `Save at ${steps(r.steps)} steps` : esc(r.title || "A recorded game");
  return `${save} · ${r.size}v${r.size} · ${r.seconds >= 60 && r.seconds % 60 === 0 ? r.seconds / 60 + " min" : r.seconds + " s"} · recorded ${ago(r.time)}`;
}
async function watchModal() {
  const bot = S.bot, b = S.data;
  if (WATCH.job && WATCH.job.bot === bot && !WATCH.job.cancelled) return watchRecordingModal();
  const [recorded, saved] = await Promise.all([call("recordings", bot), call("saves", bot)]);
  if (S.bot !== bot) return;
  const sizes = b.sizes || [b.team_size || 1], saves = saved.saves || [], games = recorded.recordings || [];
  const [last, ...earlier] = games;
  WATCH.pick = { bot, size: b.team_size || sizes[0], seconds: 60 };
  const sizeNote = sizes.length > 1 ? "Its padded observation lets it play these sizes." : "It plays the size it was made for.";
  modal(`<h2 class="display">Watch ${esc(b.name)} play</h2>
    <div class="watch-fields"><div><span class="cap">Game size</span>${watchSeg("watch-size", sizes.map(s => [s, s + "v" + s]), WATCH.pick.size)}</div><span class="muted watch-note">${sizeNote}</span></div>
    <section class="watch-part">
      <div class="watch-head"><h3>Watch now, live</h3><button type="button" class="btn primary" data-act="watch-live">${ICON.play}Watch live</button></div>
      <p class="muted">Its newest save plays on and on while you watch. While the bot trains, each new save takes over. The game is played on the processor, not the graphics card; on a busy PC it may slow training slightly.</p>
    </section>
    <section class="watch-part">
      <div class="watch-head"><h3>Record a game</h3><button type="button" class="btn strong" data-act="watch-record">${WATCH_RECORD}Record and watch</button></div>
      <div class="watch-fields"><div><span class="cap">Length</span>${watchSeg("watch-length", WATCH_LENGTHS, WATCH.pick.seconds)}</div>
        <label><span class="cap">Which save</span><select id="watch-at" class="field">${saves.map((s, i) => `<option value="${i ? s.steps : ""}">${i ? "" : "Newest: "}${steps(s.steps)} steps, ${ago(s.time)}</option>`).join("")}</select></label></div>
      <p class="muted">The game is played as fast as the processor can and opens here when it is ready.</p>
    </section>
    ${last ? `<section class="watch-part">
      <div class="watch-head"><h3>The last recording</h3><button type="button" class="btn strong" data-act="watch-recorded" data-rec="${esc(last.id)}">${ICON.eye}Watch it</button></div>
      <p>${watchRecordingLine(last)}</p>
      ${recorded.page ? `<p class="muted">It is also a page, <span class="mono">watch.html</span>, that can be passed on to someone else; opened there it draws a generated arena. <button type="button" class="btn link" data-act="watch-page">Open the page in the browser</button> · <button type="button" class="btn link" data-act="watch-file">Show the file</button></p>` : ""}
      ${earlier.length ? `<div class="watch-earlier"><span class="cap">Recorded before</span>${earlier.map(r => `<div class="watch-rec"><span>${watchRecordingLine(r)}</span><button type="button" class="btn small" data-act="watch-recorded" data-rec="${esc(r.id)}">Watch it</button></div>`).join("")}</div>` : ""}
    </section>` : ""}
    <div class="foot"><button type="button" class="btn" data-close="1">Close</button></div>`);
}
function watchRecordingModal() {
  const job = WATCH.job;
  modal(`<h2 class="display">Recording ${esc(job.bot)}</h2>
    <div class="watch-run"><div class="bar"><span></span></div><div class="watch-head"><span>${esc(job.text)}</span><span class="mono" id="watch-took"></span></div></div>
    <div class="muted watch-note">It opens here by itself when it is ready. Hiding this box does not stop the recording.</div>
    <div class="foot"><button type="button" class="btn" data-act="watch-cancel">Stop the recording</button><button type="button" class="btn" data-close="1">Hide</button></div>`);
  watchTook();
}
function watchTook() {
  const el = $("#watch-took");
  if (el && WATCH.job) el.textContent = Math.round((Date.now() - WATCH.job.since) / 1000) + " s";
  return !!el;
}
// Follows the recording to its end: the game opens by itself when the box still shows, a hidden one says it
// is ready; a failure goes to the bot's page with its last lines, as every command's does.
async function watchFollow(job) {
  for (;;) {
    await new Promise(r => setTimeout(r, 400));
    if (WATCH.job === job) watchTook();
    const out = await call("job", job.bot, "watch");
    if (out.state === "running") continue;
    const showing = WATCH.job === job && watchTook();
    if (WATCH.job === job) WATCH.job = null;
    if (job.cancelled) return;
    if (showing) closeModal();
    if (out.state === "failed") {
      S.failed = { bot: job.bot, what: "Recording a game", why: out.why || "" };
      toast("Recording a game did not work: the reason is on " + job.bot + "'s page.", true);
      if (S.view === "bot" && S.bot === job.bot) updateHead();
    } else if (out.state !== "done") toast(out.error || `The recording of ${job.bot} did not finish.`, true);
    else if (showing) watchShow(job.bot, { id: "last" });
    else toast(`The recorded game of ${job.bot} is ready: Watch it play, then The last recording.`);
    return;
  }
}

ACTIONS.watch = () => watchModal();
SEGS["watch-size"] = value => { WATCH.pick.size = Number(value); };
SEGS["watch-length"] = value => { WATCH.pick.seconds = Number(value); };
// A new try takes an earlier failure's lines off the bot's page.
function watchForgetFailure(bot) {
  if (S.failed && S.failed.bot === bot) { S.failed = null; updateHead(); }
}
ACTIONS["watch-live"] = () => { watchForgetFailure(WATCH.pick.bot); watchShow(WATCH.pick.bot, { live: { size: WATCH.pick.size, follow: true } }); };
ACTIONS["watch-recorded"] = el => { watchShow(WATCH.pick.bot, { id: el.dataset.rec }); };
ACTIONS["watch-record"] = async () => {
  const { bot, size, seconds } = WATCH.pick, at = $("#watch-at").value;
  const out = await call("watch_record", bot, { size, seconds, at: at ? Number(at) : null });
  if (out.error) return toast(out.error, true);
  watchForgetFailure(bot);
  const length = WATCH_LENGTHS.find(([s]) => s === seconds);
  WATCH.job = { bot, since: Date.now(), text: `A ${size}v${size} game of ${length ? length[1] : seconds + " s"} is being played and recorded.`, cancelled: false };
  watchRecordingModal();
  watchFollow(WATCH.job);
  return;
};
ACTIONS["watch-cancel"] = async () => {
  const job = WATCH.job;
  if (job) { job.cancelled = true; await call("stop_job", job.bot, "watch"); toast("The recording was stopped."); }
  return closeModal();
};
ACTIONS["watch-page"] = async () => { const out = await call("open_watch_page", WATCH.pick.bot); if (out.error) toast(out.error, true); return; };
ACTIONS["watch-file"] = async () => { const out = await call("show_watch_file", WATCH.pick.bot); if (out.error) toast(out.error, true); return; };

// ---------- the search box's commands, where the page has one ----------
function watchCommands() {
  if (typeof COMMANDS === "undefined") return;
  const ready = () => S.view === "bot" && !!S.data && S.data.saves > 0;
  // "Watch it play" itself is the search's own entry.
  COMMANDS.push(
    { id: "watch-live", label: "Watch live", keywords: "watch live viewer game now", about: "the bot's newest save plays on and on", command: "nitro watch {bot} --live", when: ready, run: () => watchShow(S.bot, { live: { follow: true } }) },
    { id: "watch-last", label: "Watch the last recording", keywords: "watch recording recorded game last replay", about: "the bot's newest recorded game", when: ready, run: () => watchShow(S.bot, { id: "last" }) },
  );
}
// A list declared by a later part of this script cannot be read yet: then it is filled once the script has run.
try { watchCommands(); } catch (err) { queueMicrotask(watchCommands); }
