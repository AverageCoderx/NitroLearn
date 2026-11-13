// The Console tab (what the bot's commands print, each under the command that printed it) and the Notes tab.
const CONSOLE_EMPTY = "Nothing yet. Start the bot to see its console here.";
// Lines a command prints for its terminal, as this tab says them: the window has no Ctrl+C (its Stop and save
// does that), and a setting changed by the user's own Apply is a note, not a warning. Copy all copies them
// as printed.
const CONSOLE_WORDS = [
  [/^WARNING: Setting changed since the save: (.*)$/, "Note: changed since the last save (by Apply or in the settings file): $1"],
  [/^Stopping after this iteration and saving\. Press Ctrl\+C again to quit at once without saving\.$/, "Stopping after this round of training and saving."],
  [/^(Training '.*) Ctrl\+C stops\.$/, "$1 Stop and save stops it."],
  [/ Ctrl\+C stops\.$/, ""],
];
function consoleLine(line) { return CONSOLE_WORDS.reduce((text, [find, put]) => text.replace(find, put), line); }
// The console's text with each command's line ("$ nitro ...") marked.
function consoleHTML(text) {
  return text.split("\n").map(line => line.startsWith("$ nitro ") ? `<b class="cmd">${esc(line)}</b>` : esc(consoleLine(line))).join("\n");
}
// The command the newest output is from, and when it started: its "$ nitro resume my-bot   (date time)" line.
function consoleCommand(text) {
  const at = text.lastIndexOf("$ nitro ");
  if (at < 0 || (at > 0 && text[at - 1] !== "\n")) return null;
  const m = /^\$ (nitro .*?)(?:\s{3}\((.*)\))?$/.exec(text.slice(at).split("\n", 1)[0]);
  return m ? { command: m[1], started: m[2] || "" } : null;
}
function drawConsole() {
  const pre = $("#consoleText");
  if (!pre) return;
  const atEnd = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 30;
  if (put(pre, S.consoleText ? consoleHTML(S.consoleText) : CONSOLE_EMPTY) && atEnd) pre.scrollTop = pre.scrollHeight;
  const from = consoleCommand(S.consoleText);
  put($("#consoleFrom"), from ? `<span class="cap muted">Newest command</span><span class="mono">${esc(from.command)}</span>${from.started ? `<span class="muted">started ${esc(from.started)}</span>` : ""}` : '<span class="muted">What the bot\'s commands print, newest at the bottom.</span>');
}
function renderConsole(box) {
  box.innerHTML = `<div class="consolebox"><div class="row"><div class="row grow" id="consoleFrom"></div><button type="button" class="btn small" data-act="console-copy">Copy all</button></div><pre class="console mono" id="consoleText" tabindex="0" aria-label="What the bot's commands printed"></pre></div>`;
  drawConsole();
  const pre = $("#consoleText"); pre.scrollTop = pre.scrollHeight;
}

async function pollConsole() {
  if (S.view !== "bot" || S.tab !== "console" || S.consoleBusy) return;
  const bot = S.bot;
  S.consoleBusy = true;  // one read at a time: two at once added the same text twice
  let out;
  try { out = await call("console_text", bot, S.consoleNext); } finally { S.consoleBusy = false; }
  if (S.bot !== bot) return;
  if (!out.error) {
    if (out.next < S.consoleNext) S.consoleText = "";
    S.consoleText = (S.consoleText + (out.text || "")).slice(-300000);
    S.consoleNext = out.next || 0;
    drawConsole();
  }
}
// Copies text to the clipboard; where the window gives the page no clipboard, through a selection.
async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch (err) {
    const area = document.createElement("textarea");
    area.value = text; area.setAttribute("readonly", ""); area.style.position = "fixed"; area.style.opacity = "0";
    document.body.appendChild(area); area.select();
    let done = false;
    try { done = document.execCommand("copy"); } catch (err2) { done = false; }
    area.remove();
    return done;
  }
}
ACTIONS["console-copy"] = async () => {
  if (!S.consoleText) return toast("There is nothing to copy yet.");
  return toast(await copyText(S.consoleText) ? "Copied the console's text." : "Copying is not allowed here: select the text and press Ctrl+C.", false);
};

// Notes save themselves a second after typing stops, as waiting changes are kept; Save note saves at once.
// The word beside a note ([data-note-state]) says how its saving goes.
const NOTE_TIMERS = {};
function noteState(key, word) { document.querySelectorAll(`[data-note-state="${CSS.escape(key)}"]`).forEach(el => { el.textContent = word; }); }
function autosaveNote(key, value) {
  const bot = S.bot;
  clearTimeout(NOTE_TIMERS[key]);
  noteState(key, "Saving…");
  NOTE_TIMERS[key] = setTimeout(async () => {
    delete NOTE_TIMERS[key];
    const out = await call("set_note", bot, key, value);
    if (out.error) { noteState(key, "Not saved"); return toast(out.error, true); }
    if (S.bot === bot && S.data) { S.data.notes = out.notes; if (S.drafts[key] === value) delete S.drafts[key]; }
    noteState(key, "Saved");
    if (S.bot !== bot || S.view !== "bot" || document.querySelector(`[data-note-state="${CSS.escape(key)}"]`)) return;
    if (key.startsWith("setting:")) rerenderSetting(key.slice(8));  // closed meanwhile: its row shows the note
    else if (key.startsWith("reward:") && S.tab === "rewards") renderRewardList();
  }, 1000);
}
function noteStateHTML(key) { return `<span class="muted note-state" data-note-state="${esc(key)}" aria-live="polite"></span>`; }
function renderNotes(box) {
  const note = (S.data.notes || {}).bot || "";
  box.innerHTML = `<section class="card pad notes"><div class="heading"><h2 class="display">Notes about ${esc(S.data.name)}</h2></div><div class="muted">Anything to remember about this bot. Notes on single settings are kept beside them in the Settings tab.</div><textarea id="botnote" class="field" aria-label="Notes about this bot">${esc(draftOr("bot", note))}</textarea><div class="row"><button type="button" class="btn primary" data-act="savenote">Save note</button>${noteStateHTML("bot")}<span class="muted">Notes save by themselves as you type.</span></div></section>`;
}

ACTIONS.savenote = async () => { clearTimeout(NOTE_TIMERS.bot); const out = await call("set_note", S.bot, "bot", $("#botnote").value); if (out.error) return toast(out.error, true); S.data.notes = out.notes; delete S.drafts.bot; noteState("bot", "Saved"); return toast("Note saved."); };
on("input", el => {
  if (el.id === "botnote") { S.drafts.bot = el.value; autosaveNote("bot", el.value); }
});
TAB_VIEWS.console = renderConsole;
TAB_VIEWS.notes = renderNotes;
