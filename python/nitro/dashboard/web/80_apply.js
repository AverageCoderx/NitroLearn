// Changes wait until Apply: the pending-changes bar, the Apply dialog, and staging one setting's change.
function changeLines() {
  const p = S.view === "bot" && S.data ? S.data.pending || {} : {};
  const lines = Object.keys(p).map(k => { const item = settingItem(k); return { path: k, text: (item ? showValue(item, p[k]) : show(p[k])) || "not set" }; });
  ((S.view === "bot" && S.data && S.data.reward_changes) || []).forEach(line => lines.push({ path: "", text: line }));
  ((S.view === "bot" && S.data && S.data.start_changes) || []).forEach(line => lines.push({ path: "", text: line, start: true }));
  return lines;
}
// A changed setting by its plain title, as the Settings tab names it.
function changeName(path) { const item = settingItem(path); return item ? settingName(item) : path; }
// What waits, in a few words: the changed settings by name, then the rewards and starts by count.
function changeWords(lines) {
  const names = lines.filter(l => l.path).map(l => changeName(l.path));
  const rewards = lines.filter(l => !l.path && !l.start).length, starts = lines.filter(l => l.start).length;
  const shown = names.slice(0, 3);
  if (names.length > 3) shown.push(`${names.length - 3} more ${names.length - 3 === 1 ? "setting" : "settings"}`);
  if (rewards) shown.push(`${rewards} reward ${rewards === 1 ? "change" : "changes"}`);
  if (starts) shown.push(`${starts} start ${starts === 1 ? "change" : "changes"}`);
  return shown.join(" · ");
}
function renderPending() {
  const box = $("#pending");
  const lines = changeLines();
  box.hidden = lines.length === 0;
  document.body.classList.toggle("has-pending", lines.length > 0);  // the toasts move up, off its buttons
  if (!lines.length) return;
  const running = S.data.status === "training" || S.data.status === "copying";
  const row = l => l.path
    ? `<div class="change"><span class="p"><b>${esc(changeName(l.path))}</b><span class="mono k">${esc(l.path)}</span></span><span class="mono v">${esc(l.text)}</span><button type="button" class="iconbtn" data-undo="${esc(l.path)}" title="Undo this change" aria-label="Undo ${esc(changeName(l.path))}">${ICON.reset}</button></div>`
    : `<div class="change"><span class="words"><b>${l.start ? "Start" : "Reward"}</b> · ${esc(l.text)}</span><button type="button" class="btn small link" data-tab="${l.start ? "starts" : "rewards"}">${l.start ? "Starts tab" : "Rewards tab"}</button></div>`;
  put(box, `<div class="row"><b class="display count">${lines.length} ${lines.length === 1 ? "change" : "changes"} waiting</b><span class="what cap" title="${running ? "Apply saves the bot, then carries on training with them (it takes about as long as a save)." : "Apply writes them into the bot's settings; they are used from its next start."}">${esc(changeWords(lines))}</span>
    <button type="button" class="btn small ghost" data-act="review" aria-expanded="${!!S.showChanges}">${S.showChanges ? "Hide the list" : "Review"}</button><button type="button" class="btn small" data-act="discard">Discard</button><button type="button" class="btn primary" data-act="apply">${running ? "Apply and restart" : "Apply"}</button></div>
    ${S.showChanges ? `<div class="list">${lines.map(row).join("")}</div>` : ""}`);
}

function applyModal() {
  const lines = changeLines();
  const running = S.data.status === "training" || S.data.status === "copying";
  const problems = S.rw && S.rw.bot === S.bot && (S.rw.rewards || []).some(e => (e.problems || []).length);
  modal(`<h2 class="display">Apply ${lines.length} ${lines.length === 1 ? "change" : "changes"}?</h2>
    <div>${running ? `<b>${esc(S.data.name)} is training.</b> It will save, stop and start again with the changes. Nothing it learned is lost.` : `They are written into ${esc(S.data.name)}'s settings and used from its next start.`}</div>
    ${problems ? '<div class="banner bad">A reward still has a problem (marked in red in the Rewards tab): Apply will refuse until it is fixed.</div>' : ""}
    <div class="apply-list">${lines.map(l => `<div>${l.path ? `<span class="p"><b>${esc(changeName(l.path))}</b><span class="mono k">${esc(l.path)}</span></span><span class="mono v">${esc(l.text)}</span>` : `<span class="words"><b>${l.start ? "Start" : "Reward"}</b> · ${esc(l.text)}</span>`}</div>`).join("")}</div>
    <div class="foot"><button type="button" class="btn" data-close="1">Not yet</button><button type="button" class="btn primary" data-act="apply-go">${running ? "Apply and restart" : "Apply"}</button></div>`);
}

// ---------- actions ----------
// What else shows a setting's change: the cards that lead a group, the Replays in training tab, the group list.
function stagedElsewhere(path) {
  if (path.startsWith("start_from.")) renderTransfer();
  if (path.startsWith("pretrain.") || path === "env.obs_team_size") renderCopySetup();
  if (path.startsWith("replay_mix.")) replayStaged();
  updateSettingsNav(); renderPending(); updateTabs();
}
async function stage(path, value) {
  const bot = S.bot, epoch = S.discards;
  const out = await call("stage", bot, path, value);
  if (S.bot !== bot || S.discards !== epoch) return;  // another bot, or Discard pressed meanwhile
  if (out.error) { S.errors[path] = out.error; S.typed[path] = value; } else { delete S.errors[path]; delete S.typed[path]; }
  if (out.pending) S.data.pending = out.pending;
  rerenderSetting(path);
  stagedElsewhere(path);
}

async function unstage(path) {
  const out = await call("unstage", S.bot, path);
  S.data.pending = out.pending || {}; delete S.errors[path]; delete S.typed[path];
  rerenderSetting(path);
  stagedElsewhere(path);
}

ACTIONS.review = async () => { S.showChanges = !S.showChanges; return renderPending(); };
// Discard asks first, listing what goes: nothing waiting can be got back afterwards.
function discardModal() {
  const lines = changeLines();
  modal(`<h2 class="display">Discard ${lines.length} ${lines.length === 1 ? "change" : "changes"}?</h2>
    <div>${lines.length === 1 ? "This change is" : "These changes are"} thrown away and ${esc(S.data.name)} stays as it is. This cannot be undone.</div>
    <div class="apply-list">${lines.map(l => `<div>${l.path ? `<span class="p"><b>${esc(changeName(l.path))}</b></span><span class="mono v">${esc(l.text)}</span>` : `<span class="words"><b>${l.start ? "Start" : "Reward"}</b> · ${esc(l.text)}</span>`}</div>`).join("")}</div>
    <div class="foot"><button type="button" class="btn" data-close="1">Keep them</button><button type="button" class="btn danger" data-act="discard-go">Discard ${lines.length === 1 ? "it" : "them"}</button></div>`);
}
ACTIONS.discard = () => discardModal();
ACTIONS["discard-go"] = async () => { closeModal(); S.discards++; rwSeq++; stSeq++; const out = await call("discard", S.bot); S.data.pending = out.pending || {}; S.data.reward_changes = []; S.data.start_changes = []; S.errors = {}; S.typed = {}; S.rw = null; S.st = null; S.rm = null; return render(); };
ACTIONS.apply = () => applyModal();
ACTIONS["apply-go"] = async () => {
  const out = await call("apply", S.bot); closeModal();
  if (out.error) return toast(out.error, true);
  toast(out.restarting ? "Applied. " + S.bot + " is saving and will start again with the changes." : "Applied " + out.applied + " " + (out.applied === 1 ? "change" : "changes") + ".");
  if (out.switched) toast(out.switched);
  const data = await call("bot", S.bot); if (!data.error) S.data = data; S.rw = null; S.st = null; S.rm = null; return render();
};
on("click", t => {
  if (t.dataset.undo) return unstage(t.dataset.undo);
  return SKIP;
});
on("change", el => {
  if (!el.dataset.set) return SKIP;
  stage(el.dataset.set, el.type === "checkbox" ? el.checked : el.value);
});
