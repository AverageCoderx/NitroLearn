// The bot's Coach tab: a plan in four stages (basics, shooting, air play, duels and reads) read from its skill checks, and
// hints when its training looks stalled, farmed, too certain or held back by its learning rate. It only
// suggests: nothing here changes a setting.
S.coach = null;
async function loadCoach() {
  const name = S.bot, out = await call("coach", name);
  if (S.view !== "bot" || S.bot !== name) return;
  S.coach = out;
  if (S.tab === "coach") put($("#coach-page"), coachHTML());
}
function coachValue(skill) {
  if (!skill.checked || skill.value == null) return "not checked yet";
  const unit = skill.unit ? " " + skill.unit : "";
  return `${num(skill.value, 2)}${unit} (bar ${skill.lower_is_better ? "at most" : "at least"} ${num(skill.bar, 2)}${unit})`;
}
function coachHTML() {
  const c = S.coach;
  if (!c || c.loading) return `<div class="muted">Reading its skill checks and training…</div>`;
  if (c.error) return `<div class="banner warn">${esc(c.error)}</div>`;
  const stage = s => {
    const state = s.passed ? "done" : s.key === c.current ? "now" : "";
    const word = s.passed ? "Done" : s.key === c.current ? "Now" : "Later";
    return `<div class="card coach-stage ${state}">
      <div class="row" style="justify-content:space-between;"><b>${esc(s.name)}</b><span class="cap muted">${word}</span></div>
      <div class="muted" style="font-size:13px;">${esc(s.about)}</div>
      ${s.skills.map(k => `<div class="coach-skill"><span>${esc(k.label)}</span><span class="${k.above ? "ok" : "muted"}">${k.above ? "above the bar" : esc(coachValue(k))}</span></div>`).join("")}
    </div>`;
  };
  const hints = (c.hints || []).map(h => `<div class="card coach-hint"><b>${esc(h.title)}</b><div class="muted" style="font-size:13px;">${codeMarks(esc(h.text))}</div><div style="font-size:13px;"><b>Try:</b> ${codeMarks(esc(h.try))}</div></div>`).join("");
  const quiet = c.too_early
    ? "Too early for hints: they need a few dozen learning updates of training."
    : "Nothing in its training looks wrong right now.";
  return `<div class="muted" style="font-size:13px;max-width:90ch;">Suggestions only: the coach never changes a setting. It reads the bot's skill checks (Tools, Skill check) and its training numbers.</div>
    <div class="banner" style="margin-top:12px;"><div class="grow">${esc(c.verdict)}${c.checked ? ` <span class="muted">(skill check at ${steps(c.checked_at)} steps)</span>` : ""}</div><button type="button" class="btn small" data-tool="skills">Run a skill check</button></div>
    <div class="coach-stages">${(c.stages || []).map(stage).join("")}</div>
    <div class="heading"><h2 class="display">Hints</h2><button type="button" class="btn small ghost" data-act="coach-refresh">Look again</button></div>
    <div style="margin-top:12px;">${hints || `<div class="muted" style="font-size:13px;">${quiet}</div>`}</div>
    <div class="muted" style="font-size:13px;margin-top:8px;">How the coach decides: <button type="button" class="btn link" data-act="guide-open" data-page="coach.md">The training coach</button> in the handbook.</div>`;
}
TAB_VIEWS.coach = box => { box.innerHTML = `<div id="coach-page">${coachHTML()}</div>`; if (!S.coach) { S.coach = { loading: true }; loadCoach(); } };
ACTIONS["coach-refresh"] = () => loadCoach();
