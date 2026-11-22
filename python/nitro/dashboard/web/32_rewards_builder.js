// Rewards built from blocks, without code: a built reward's card in the Rewards tab gets its title,
// variables and terms here. A term pays its measures (multiplied, or the smallest), or a formula, times its
// weight on the steps where all of its conditions (blocks or formulas) hold. The blocks, the formula words and
// their help come from Python (rewards.blocks, nitro/config/built.py); formulas are checked by the core as the
// person types (check_built), since only it can judge them.
const BUILT_NUMBERS = { ball_above: 300, ball_below: 300, car_above: 300, car_below: 300, ball_closer_than: 500, ball_farther_than: 500, facing_ball_within: 30, facing_goal_within: 30, boost_above: 50, boost_below: 50, speed_above: 80, speed_below: 80, ball_speed_above: 80, ball_speed_below: 80 };

function builtBlocks() { return (S.rw && S.rw.blocks) || { flags: [], numbered: [], measures: [], term_keys: {}, var_keys: {}, formula: { car: [], ball: [], names: [], functions: [] }, templates: [] }; }
function blockOf(list, name) { return (builtBlocks()[list] || []).find(b => b.name === name) || null; }
function asList(v) { return v == null || v === "" ? [] : Array.isArray(v) ? v : [v]; }
function wordsOf(name) { const s = String(name || "").replace(/_/g, " ").trim(); return s ? s[0].toUpperCase() + s.slice(1) : ""; }
// A metrics label (and layout key) of a reward: a built reward is named by its title.
function keyName(view) { return view.name === "built" ? (view.title || "built") : view.name; }

// A condition written as a formula ("me.z > 300"): any text that is not a block's name.
function condIsFormula(c) { return typeof c === "string" && !blockOf("flags", c); }
// A condition as written: "in_air", { ball_above = 300 } or { not = ... }; as its name, number and "not".
function condParts(c) {
  if (c && typeof c === "object" && !Array.isArray(c)) {
    const key = Object.keys(c)[0];
    if (key === "not") { const inner = condParts(c.not); return Object.assign(inner, { not: !inner.not }); }
    return { name: key, value: c[key], not: false };
  }
  return { name: String(c), value: null, not: false };
}
function condWritten(p) {
  const base = p.value == null ? p.name : { [p.name]: p.value };
  return p.not ? { not: base } : base;
}
function condIsEvent(c) { const p = condParts(c), f = blockOf("flags", p.name); return !!(f && f.event); }
function condWords(c) {
  if (condIsFormula(c)) return c;
  const p = condParts(c);
  return (p.not ? "not " : "") + wordsOf(p.name).toLowerCase() + (p.value != null ? " " + p.value : "");
}
// Paid every step (of the state alone): the same rule as nitro.config.built.term_every_step for terms of blocks;
// a term with a formula is the core's to judge (null: not known here).
function termEveryStep(t) {
  if (termIsFormula(t) || asList(t.when).some(condIsFormula)) return null;
  const measures = asList(t.measure), plain = !["once_per_touch", "once_per_episode", "cap_step", "cap_episode"].some(k => t[k] != null && t[k] !== "" && t[k] !== false);
  return plain && measures.length > 0 && measures.every(m => { const b = blockOf("measures", m); return !!(b && b.every_step); }) && !asList(t.when).some(condIsEvent);
}
function termIsFormula(t) { return t.value != null; }
// The core's answer for the whole reward when there is one (formulas), else the blocks' rule.
function builtEveryStep(view) {
  const checked = builtChecked(view);
  if (checked) return !!checked.every_step && !checked.problems.length;
  const terms = view.terms || [];
  return terms.length > 0 && terms.every(t => termEveryStep(t) === true);
}
// One plain sentence per term, as the card's "about".
function builtSummary(view) {
  const terms = view.terms || [];
  if (!terms.length) return "No terms yet: add one below.";
  return terms.map(t => {
    const ms = asList(t.measure).map(m => wordsOf(m).toLowerCase()), when = asList(t.when).map(condWords);
    if (termIsFormula(t)) ms.splice(0, ms.length, String(t.value || "…"));
    const what = ms.length > 1 ? (t.combine === "min" ? "the smaller of " : "") + ms.join(t.combine === "min" ? " and " : " × ") : (ms[0] || "nothing");
    const w = t.weight == null || t.weight === "" ? 1 : t.weight;
    const named = terms.length > 1 ? `Term ${terms.indexOf(t) + 1}: pays` : "Pays";
    return `${named} ${what}${String(w) !== "1" ? " × " + w : ""}${when.length ? " while " + when.join(" and ") : ""}${t.once_per_touch ? ", once per touch" : ""}${t.once_per_episode ? ", once an episode" : ""}.`;
  }).join(" ");
}
// The card's kind: a built reward's title, whether it is paid every step and what it pays, worked out from its terms.
function viewKind(view) {
  const k = kindOf(view.name);
  if (view.name !== "built") return k;
  return Object.assign({}, k, { title: wordsOf(view.title) || "Untitled built reward", every_step: builtEveryStep(view), about: builtSummary(view) });
}
// The shares list names a reward by its metrics label; a built reward's label is its title.
function builtTitleOf(label) {
  const e = ((S.rw && S.rw.rewards) || []).find(x => x.view.name === "built" && x.label === label);
  return e ? wordsOf(e.view.title) : "";
}
// The term a new built reward starts with, an example to change: 1 on each touch while in the air.
const BUILT_EXAMPLE = { when: ["in_air"], measure: "touch" };
function isBuiltExample(t) { return JSON.stringify(t) === JSON.stringify(BUILT_EXAMPLE); }
// A new built reward: a title no other built reward of the bot has, and one term to start from.
function newBuiltView(view) {
  const titles = new Set(((S.rw && S.rw.rewards) || []).filter(e => e.view.name === "built").map(e => e.view.title));
  let title = "my_reward", n = 2;
  while (titles.has(title)) title = "my_reward_" + n++;
  return Object.assign(view, { title, vars: [], terms: [JSON.parse(JSON.stringify(BUILT_EXAMPLE))] });
}
// A built reward started from a template: its variables and terms, a title of its own.
function templateView(tpl) {
  const view = newBuiltView({ name: "built", known: true, enabled: true, weight: 1, zero_sum: null, ramp: null, params: {} });
  const titles = new Set(((S.rw && S.rw.rewards) || []).filter(e => e.view.name === "built").map(e => e.view.title));
  let title = tpl.title, n = 2;
  while (titles.has(title)) title = tpl.title + "_" + n++;
  return Object.assign(view, { title, vars: JSON.parse(JSON.stringify(tpl.vars || [])), terms: JSON.parse(JSON.stringify(tpl.terms)) });
}

// Its lines in the nerd explanation's settings block.
function builtTomlValue(v) {
  if (Array.isArray(v)) return "[" + v.map(builtTomlValue).join(", ") + "]";
  if (v && typeof v === "object") return "{ " + Object.entries(v).map(([k, x]) => `${k} = ${builtTomlValue(x)}`).join(", ") + " }";
  if (typeof v === "boolean") return v ? "true" : "false";
  const n = builtNumber(v);
  return typeof n === "number" ? String(n) : JSON.stringify(String(v));
}
function builtBlockLines(view) {
  const lines = [];
  (view.vars || []).forEach(v => {
    lines.push("", "[[rewards.vars]]", `name = ${JSON.stringify(String(v.name || ""))}`);
    if (v.start != null && v.start !== "" && String(v.start) !== "0") lines.push(`start = ${builtTomlValue(v.start)}`);
    lines.push(`next = ${JSON.stringify(String(v.next || ""))}`);
  });
  (view.terms || []).forEach(t => {
    lines.push("", "[[rewards.terms]]");
    if (asList(t.when).length) lines.push(`when = ${builtTomlValue(asList(t.when))}`);
    if (termIsFormula(t)) lines.push(`value = ${JSON.stringify(String(t.value))}`);
    else { const ms = asList(t.measure); lines.push(`measure = ${builtTomlValue(ms.length === 1 ? ms[0] : ms)}`); }
    if (t.combine === "min" && !termIsFormula(t)) lines.push('combine = "min"');
    if (t.weight != null && t.weight !== "" && String(t.weight) !== "1") lines.push(`weight = ${builtTomlValue(t.weight)}`);
    ["once_per_touch", "once_per_episode", "every_step"].forEach(k => { if (t[k]) lines.push(`${k} = true`); });
    ["cap_step", "cap_episode"].forEach(k => { if (t[k] != null && t[k] !== "") lines.push(`${k} = ${builtTomlValue(t[k])}`); });
  });
  return lines;
}
// The editor under a built reward's options: its title, its variables, then each term's conditions, what it
// pays (blocks or a formula), its weight and options. Under each field, the core's message about it (bd-msg).
function builtEditorHTML(view, i) {
  const b = builtBlocks(), keys = b.term_keys || {}, vkeys = b.var_keys || {}, terms = view.terms || [], vars = view.vars || [];
  const id = s => `bd-${i}-${s}`;
  const msg = where => `<div class="bd-msg" data-bdmsg="${i}:${esc(where)}" role="status"></div>`;
  const formula = (key, value, label, help, where) => `<textarea class="field mono bd-formula" id="${id(key.replace(/\./g, "-"))}" rows="1" spellcheck="false" data-bd="${i}" data-bk="${esc(key)}" data-bdf="${esc(where)}" aria-label="${esc(label)}" title="${esc(help || "")}">${esc(value == null ? "" : value)}</textarea>`;
  const condPicker = j => `<select class="field" id="${id(j + "-cond")}" data-bd="${i}" data-bk="t.${j}.when.add" aria-label="Add a condition" title="${esc(keys.when || "")}">
      <option value="">+ add a condition…</option>
      <option value="__formula">A formula…</option>
      <optgroup label="Yes or no">${b.flags.map(f => `<option value="${esc(f.name)}" title="${esc(f.help)}">${esc(wordsOf(f.name))}</option>`).join("")}</optgroup>
      <optgroup label="With a number">${b.numbered.map(f => `<option value="${esc(f.name)}" title="${esc(f.help)}">${esc(wordsOf(f.name))} …</option>`).join("")}</optgroup>
    </select>`;
  const measurePicker = (j, used) => `<select class="field" id="${id(j + "-measure")}" data-bd="${i}" data-bk="t.${j}.measure.add" aria-label="Add a measure" title="${esc(keys.measure || "")}">
      <option value="">${used ? "+ multiply by…" : "+ pick a measure…"}</option>
      <optgroup label="Every step">${b.measures.filter(m => m.every_step).map(m => `<option value="${esc(m.name)}" title="${esc(m.help)}">${esc(wordsOf(m.name))}</option>`).join("")}</optgroup>
      <optgroup label="When it happens">${b.measures.filter(m => !m.every_step).map(m => `<option value="${esc(m.name)}" title="${esc(m.help)}">${esc(wordsOf(m.name))}</option>`).join("")}</optgroup>
    </select>`;
  const cond = (c, j, n) => {
    if (condIsFormula(c)) {
      return `<span class="chip bd-cond bd-fcond">${formula(`t.${j}.when.${n}`, c, "A condition as a formula", "Pays while this is true (not 0), e.g. me.z > 300 and ball.speed > 1000", `terms[${j}].when[${n}]`)}
        <button type="button" class="more" data-bdx="delcond:${i}:${j}:${n}" aria-label="Remove the formula condition">×</button></span>`;
    }
    const p = condParts(c), block = blockOf(p.value == null ? "flags" : "numbered", p.name);
    return `<span class="bd-condwrap"><button type="button" class="bd-notbtn" data-bdx="not:${i}:${j}:${n}" aria-pressed="${p.not}" title="${p.not ? "On: pays while this does NOT hold. Click to turn it off." : "Click to turn the condition around: pays while it does not hold."}">${p.not ? "not ✓" : "not"}</button><span class="chip bd-cond${p.not ? " bd-not" : ""}" title="${esc(block ? block.help : "Not a condition NitroLearn knows")}">
      <span>${p.not ? "Not: " : ""}${esc(wordsOf(p.name))}</span>
      ${p.value != null ? `<input class="field mono bd-num" id="${id(j + "-c" + n)}" type="text" inputmode="decimal" data-bd="${i}" data-bk="t.${j}.when.${n}" value="${esc(p.value)}" aria-label="${esc(wordsOf(p.name))}">` : ""}
      <button type="button" class="more" data-bdx="delcond:${i}:${j}:${n}" aria-label="Remove the condition ${esc(wordsOf(p.name))}">×</button></span></span>`;
  };
  const opt = (j, key, label) => `<label class="rw-opt" for="${id(j + "-" + key)}" title="${esc(keys[key] || "")}"><input type="checkbox" id="${id(j + "-" + key)}" data-bd="${i}" data-bk="t.${j}.${key}" ${t_(j)[key] ? "checked" : ""}>${label}</label>`;
  const t_ = j => terms[j] || {};
  const cap = (j, key, label) => `<label class="rw-field" for="${id(j + "-" + key)}">${label}<input class="field mono" id="${id(j + "-" + key)}" type="text" inputmode="decimal" placeholder="none" data-bd="${i}" data-bk="t.${j}.${key}" value="${esc(t_(j)[key] == null ? "" : t_(j)[key])}" title="${esc(keys[key] || "")}"></label>`;
  const pays = (t, j) => {
    const toggle = `<span class="seg bd-mode" role="group" aria-label="What term ${j + 1} pays"><button type="button" data-bdx="mode:${i}:${j}:blocks" aria-pressed="${!termIsFormula(t)}">Blocks</button><button type="button" data-bdx="mode:${i}:${j}:formula" aria-pressed="${termIsFormula(t)}">Formula</button></span>`;
    if (termIsFormula(t)) return `<div class="bd-row"><span class="bd-lead">Pays</span>${toggle}</div>${formula(`t.${j}.value`, t.value, `What term ${j + 1} pays`, keys.value, `terms[${j}].value`)}${msg(`terms[${j}].value`)}`;
    const ms = asList(t.measure);
    const helps = ms.map(m => { const x = blockOf("measures", m); return x ? `<span><b>${esc(wordsOf(m))}</b>: ${esc(x.help)}</span>` : `<span><b>${esc(m)}</b>: not a measure NitroLearn knows</span>`; }).join("");
    return `<div class="bd-row"><span class="bd-lead">Pays</span>${toggle}${ms.map((m, n) => `<span class="chip" title="${esc((blockOf("measures", m) || {}).help || "")}">${esc(wordsOf(m))}<button type="button" class="more" data-bdx="delmeasure:${i}:${j}:${n}" aria-label="Remove the measure ${esc(wordsOf(m))}">×</button></span>`).join(ms.length > 1 ? `<span class="muted">${t.combine === "min" ? "or" : "×"}</span>` : "")}${measurePicker(j, ms.length)}
        ${ms.length > 1 ? `<label class="rw-field" for="${id(j + "-combine")}">Combine<select class="field" id="${id(j + "-combine")}" data-bd="${i}" data-bk="t.${j}.combine" title="${esc(keys.combine || "")}"><option value="product" ${t.combine === "min" ? "" : "selected"}>multiply them</option><option value="min" ${t.combine === "min" ? "selected" : ""}>the smallest</option></select></label>` : ""}</div>
      ${helps ? `<div class="bd-help">${helps}</div>` : ""}${msg(`terms[${j}].measure`)}`;
  };
  const term = (t, j) => {
    const when = asList(t.when), every = termEveryStep(t);
    return `<div class="bd-term" data-bdterm="${j}">
      <div class="bd-term-head"><span class="cap">Term ${j + 1}</span>${every ? '<span class="badge" title="Paid on every step">every step</span>' : ""}
        ${terms.length > 1 ? `<button type="button" class="iconbtn" data-bdx="delterm:${i}:${j}" title="Remove this term" aria-label="Remove term ${j + 1}">${ICON.trash}</button>` : ""}</div>
      ${isBuiltExample(t) ? '<div class="bd-note">An example to start from: pays 1 on each touch while the car is in the air. Change it, or start from a template (Add reward).</div>' : ""}
      <div class="bd-row"><span class="bd-lead">When</span>${when.length ? when.map((c, n) => cond(c, j, n)).join("") : '<span class="muted">always</span>'}${condPicker(j)}</div>
      ${when.map((c, n) => msg(`terms[${j}].when[${n}]`)).join("")}
      ${pays(t, j)}
      <div class="rw-grid">
        <label class="rw-field" for="${id(j + "-weight")}">Term weight<input class="field mono" id="${id(j + "-weight")}" type="text" inputmode="decimal" data-bd="${i}" data-bk="t.${j}.weight" value="${esc(t.weight == null ? 1 : t.weight)}" title="${esc(keys.weight || "")}"></label>
        ${cap(j, "cap_step", "Most a step")}${cap(j, "cap_episode", "Most an episode")}
      </div>
      <div class="row bd-opts">${opt(j, "once_per_touch", "Once per touch")}${opt(j, "once_per_episode", "Once an episode")}${opt(j, "every_step", "Scale as every step")}</div>
      ${msg(`terms[${j}]`)}
    </div>`;
  };
  const variable = (v, k) => `<div class="bd-var" data-bdvar="${k}">
      <label class="rw-field" for="${id("v" + k + "-name")}">Name<input class="field mono" id="${id("v" + k + "-name")}" type="text" data-bd="${i}" data-bk="v.${k}.name" value="${esc(v.name || "")}" title="${esc(vkeys.name || "")}"></label>
      <label class="rw-field" for="${id("v" + k + "-start")}">Start<input class="field mono bd-num" id="${id("v" + k + "-start")}" type="text" inputmode="decimal" data-bd="${i}" data-bk="v.${k}.start" value="${esc(v.start == null ? 0 : v.start)}" title="${esc(vkeys.start || "")}"></label>
      <span class="bd-var-next"><span class="cap">Each step</span>${formula(`v.${k}.next`, v.next, `The variable ${v.name || k + 1}'s next value`, vkeys.next, `vars[${k}].next`)}</span>
      <button type="button" class="iconbtn" data-bdx="delvar:${i}:${k}" title="Remove this variable" aria-label="Remove the variable ${esc(v.name || "")}">${ICON.trash}</button>
      ${msg(`vars[${k}]`)}${msg(`vars[${k}].next`)}
    </div>`;
  return `<div class="rw-sub bd"><span class="lead">Built from blocks, no code: the reward pays the sum of its terms, times its weight above. A term pays on the steps where all of its conditions hold. Blocks: distances and heights in game units (the ball is about 92 across, the ceiling 2044 high), speeds in km/h, angles in degrees, boost 0 to 100. Formulas use game units per second for speeds; the Formula words list says what each word means.</span>
    <label class="rw-field" for="${id("title")}">Title<input class="field mono bd-title" id="${id("title")}" type="text" data-bd="${i}" data-bk="title" value="${esc(view.title || "")}" title="Its name in the training's numbers: small letters, digits and _"></label>
    ${msg("title")}${msg("")}
    <div class="bd-vars"><span class="cap">Variables</span>${vars.length ? vars.map(variable).join("") : '<span class="muted">None: a variable keeps a number from step to step (a counter, a time), for formulas to read.</span>'}
      <div class="row"><button type="button" class="btn small" data-bdx="addvar:${i}">+ Add a variable</button></div></div>
    ${terms.map(term).join("")}${msg("terms")}
    <div class="row"><button type="button" class="btn small" data-bdx="addterm:${i}">+ Add a term</button>
      <button type="button" class="btn small" data-bdx="score:${i}" title="Plays a recorded game you choose and shows what this reward alone would have paid its players, in the match viewer's reward chart">Test this reward on a recorded game</button>
      <button type="button" class="btn small ghost" data-act="open-builder-guide" title="docs/reward_builder.md: every block and formula word explained, with examples">How built rewards work</button></div>
    ${builtReferenceHTML()}
  </div>`;
}
// Every word a formula knows, with its help; a click puts it at the cursor of the formula last typed in.
function builtReferenceHTML() {
  const f = builtBlocks().formula || {}, item = (text, help) => `<button type="button" class="chip bd-word" data-bdins="${esc(text)}" title="${esc(help)}"><span class="mono">${esc(text)}</span></button>`;
  const group = (title, list, make) => list && list.length ? `<div class="bd-ref-group"><span class="cap">${esc(title)}</span><div class="bd-ref-items">${list.map(w => `<span class="bd-ref-item">${make(w)}<span class="muted">${esc(w.help)}</span></span>`).join("")}</div></div>` : "";
  const open = !!S.bdRefOpen;
  return `<div class="bd-ref"><button type="button" class="more" data-bdx="ref:0" aria-expanded="${open}">${open ? "Close the formula words" : "Formula words: what a formula can read and do"}</button>
    ${!open ? "" : `
    <p class="muted">Cars: <span class="mono">me</span> (this car), <span class="mono">mate</span> (the nearest teammate), <span class="mono">opp</span> (the nearest opponent), <span class="mono">it</span> (each car, inside sum_mates(...) and the like). Write <span class="mono">me.z</span>, <span class="mono">ball.speed</span>, <span class="mono">dist(me.pos, ball.pos)</span>. True is 1, false 0; and, or, not, &lt;, &lt;=, &gt;, &gt;=, ==, != and + - * / work as usual.</p>
    ${group("A car's fields (me., mate., opp., it.)", f.car, w => item("me." + w.name, w.help))}
    ${group("The ball's fields", f.ball, w => item("ball." + w.name, w.help))}
    ${group("Named values", f.names, w => item(w.name, w.help))}
    ${group("Functions", f.functions, w => item(w.name + "(", w.help))}`}
  </div>`;
}

function builtNumber(raw) { const s = String(raw).trim().replace(/_/g, ""); return s !== "" && isFinite(Number(s)) ? Number(s) : raw; }
// An edit of a built reward: the change made to its view (`stage` false while typing in a formula: the page
// is not drawn again, so the cursor stays), then staged like any other reward change.
function builtApply(v, key, raw) {
  v.terms = v.terms || []; v.vars = v.vars || [];
  if (key === "title") { v.title = String(raw).trim(); return true; }
  const vm = /^v\.(\d+)\.(name|start|next)$/.exec(key);
  if (vm) {
    const variable = v.vars[+vm[1]];
    if (!variable) return false;
    variable[vm[2]] = vm[2] === "start" ? builtNumber(raw) : String(raw);
    return true;
  }
  const m = /^t\.(\d+)\.(.+)$/.exec(key), t = m && v.terms[+m[1]];
  if (!t) return false;
  const what = m[2];
  if (what === "weight") t.weight = builtNumber(raw);
  else if (what === "value") t.value = String(raw);
  else if (["once_per_touch", "once_per_episode", "every_step", "once"].includes(what)) {
    const k = what === "once" ? "once_per_touch" : what;
    if (raw) t[k] = true; else delete t[k];
  } else if (what === "cap_step" || what === "cap_episode") {
    if (String(raw).trim() === "") delete t[what]; else t[what] = builtNumber(raw);
  } else if (what === "combine") { if (raw === "min") t.combine = "min"; else delete t.combine; }
  else if (what === "when.add") {
    if (!raw) return false;
    const numbered = !!blockOf("numbered", raw);
    const item = raw === "__formula" ? "me.z > 300" : numbered ? { [raw]: BUILT_NUMBERS[raw] != null ? BUILT_NUMBERS[raw] : 0 } : raw;
    if (raw !== "__formula" && asList(t.when).some(c => !condIsFormula(c) && condParts(c).name === raw)) {
      toast(`'${wordsOf(raw)}' is already a condition of term ${+m[1] + 1}.`);
      return true; // drawn again: the picker goes back to "+ add a condition"
    }
    t.when = asList(t.when).concat([item]);
  } else if (what === "measure.add") { if (!raw) return false; const ms = asList(t.measure).concat([raw]); t.measure = ms.length === 1 ? ms[0] : ms; }
  else if (what.startsWith("when.")) {
    const n = +what.slice(5), when = asList(t.when);
    if (condIsFormula(when[n])) when[n] = String(raw);
    else { const p = condParts(when[n]); p.value = builtNumber(raw); when[n] = condWritten(p); }
    t.when = when;
  } else return false;
  return true;
}
// `formula`: from a formula box, whose view typing already changed: the page is not drawn again on leaving it,
// so a click that left it (on a button of the card) still lands.
function builtEdit(i, key, raw, formula = false) {
  const e = S.rw.rewards[i];
  if (!e || !builtApply(e.view, key, raw)) return;
  if (!formula) renderRewardList();
  builtCheckSoon(i);
  return stageRewards();
}
function builtClick(how) {
  const [act, a, b, c] = how.split(":"), e = S.rw.rewards[+a];
  if (act === "ref") { S.bdRefOpen = !S.bdRefOpen; return renderRewardList(); }
  if (!e) return;
  if (act === "score") return builtScore(+a);
  const terms = e.view.terms = e.view.terms || [], t = terms[+b], vars = e.view.vars = e.view.vars || [];
  if (act === "addterm") terms.push({ measure: "one" });
  else if (act === "delterm") terms.splice(+b, 1);
  else if (act === "addvar") { const names = new Set(vars.map(v => v.name)); let name = "x", n = 2; while (names.has(name)) name = "x" + n++; vars.push({ name, start: 0, next: name + " + dt" }); }
  else if (act === "delvar") vars.splice(+b, 1);
  else if (!t) return;
  else if (act === "mode") {
    if (c === "formula" && !termIsFormula(t)) { t.value = "1"; delete t.measure; delete t.combine; }
    else if (c === "blocks" && termIsFormula(t)) { delete t.value; t.measure = "one"; }
  }
  else if (act === "delcond") { const when = asList(t.when); when.splice(+c, 1); if (when.length) t.when = when; else delete t.when; }
  else if (act === "not") { const when = asList(t.when), p = condParts(when[+c]); p.not = !p.not; when[+c] = condWritten(p); t.when = when; }
  else if (act === "delmeasure") { const ms = asList(t.measure); ms.splice(+c, 1); t.measure = ms.length === 1 ? ms[0] : ms; if (ms.length < 2) delete t.combine; }
  renderRewardList();
  builtCheckSoon(+a);
  return stageRewards();
}

// ---------- checking as the person types (the core's own check) ----------
// Per reward, by its place in the list: the newest answer and the text of the view it answered.
const BUILT_CHECKS = {};
let builtCheckTimers = {};
function builtChecked(view) {
  const i = ((S.rw && S.rw.rewards) || []).findIndex(e => e.view === view), got = BUILT_CHECKS[i];
  return got && got.of === JSON.stringify(view) ? got.out : null;
}
function builtCheckSoon(i, wait = 250) {
  clearTimeout(builtCheckTimers[i]);
  builtCheckTimers[i] = setTimeout(() => builtCheck(i), wait);
}
async function builtCheck(i) {
  const e = S.rw && S.rw.rewards[i];
  if (!e || e.view.name !== "built") return;
  const of = JSON.stringify(e.view), out = await call("check_built", e.view);
  if (out.error || !S.rw || S.rw.rewards[i] !== e || JSON.stringify(e.view) !== of) return;
  const was = BUILT_CHECKS[i] && BUILT_CHECKS[i].out;
  BUILT_CHECKS[i] = { of, out };
  builtPaint(i, out);
  // The every-step badge follows the core's answer; the card is drawn again only when it changes.
  if (!was || was.every_step !== out.every_step) { const active = document.activeElement && document.activeElement.id; renderRewardList(); if (active && $("#" + active)) $("#" + active).focus(); builtPaint(i, out); }
}
// A message that names a place in a formula ("at 7", "(at 7)", counted from 1) shown with the formula and a
// mark under that place, instead of the count.
function builtSpot(message, text) {
  const m = /\s*\(?\bat (\d+)\)?\s*$/.exec(message) || /\s*\(?\bat (\d+)\)?/.exec(message);
  if (!m || text == null) return null;
  const at = Math.max(0, Math.min(+m[1] - 1, String(text).length));
  const line = String(text).replace(/\n/g, " ");
  return { words: (message.slice(0, m.index) + message.slice(m.index + m[0].length)).trim(), line, mark: " ".repeat(at) + "^" };
}
// Two built rewards of the bot with the same title (the second is refused when staged): said under the title.
function builtTitleClash(i) {
  const rewards = (S.rw && S.rw.rewards) || [], view = rewards[i] && rewards[i].view;
  if (!view || !view.title) return "";
  const other = rewards.findIndex((e, k) => k !== i && e.view.name === "built" && e.view.title === view.title);
  return other < 0 ? "" : `Another built reward is called '${view.title}' too: give each its own title.`;
}
// A condition listed twice in a term (a file written by hand): said under the term.
function builtRepeats(view) {
  const out = [];
  (view.terms || []).forEach((t, j) => {
    const seen = new Set();
    asList(t.when).forEach(c => { const key = JSON.stringify(c); if (seen.has(key)) out.push({ where: `terms[${j}]`, message: `'${condWords(c)}' is listed twice in term ${j + 1}.` }); seen.add(key); });
  });
  return out;
}
// Each message under the field it is about (the nearest place shown: a term's value, else the term, else the
// reward's own box); a formula with no problem says so.
function builtPaint(i, out) {
  const boxes = [...document.querySelectorAll(`[data-bdmsg^="${i}:"]`)];
  boxes.forEach(box => { box.textContent = ""; box.classList.remove("bad", "ok"); });
  const e = S.rw && S.rw.rewards[i], clash = builtTitleClash(i);
  const problems = (out.problems || []).concat(clash ? [{ where: "title", message: clash }] : [], e ? builtRepeats(e.view) : []);
  const said = new Set();
  problems.forEach(p => {
    let where = p.where, box = null;
    while (!box) {
      box = boxes.find(x => x.dataset.bdmsg === `${i}:${where}`);
      if (box || !where) break;
      where = where.includes(".") ? where.slice(0, where.lastIndexOf(".")) : "";
    }
    box = box || boxes.find(x => x.dataset.bdmsg === `${i}:`);
    if (!box) return;
    said.add(box.dataset.bdmsg);
    const field = document.querySelector(`[data-bd="${i}"][data-bdf="${CSS.escape(where)}"]`), spot = builtSpot(p.message, field && field.value);
    box.classList.add("bad");
    if (spot) {
      box.insertAdjacentHTML("beforeend", `<span>${esc(spot.words)}, here:</span><pre class="bd-spot mono">${esc(spot.line)}\n${esc(spot.mark)}</pre>`);
    } else box.insertAdjacentText("beforeend", (box.textContent ? " " : "") + p.message);
  });
  if (out.problems && !out.problems.length) {
    document.querySelectorAll(`[data-bd="${i}"][data-bdf]`).forEach(f => {
      const box = boxes.find(x => x.dataset.bdmsg === `${i}:${f.dataset.bdf}`);
      if (box && !said.has(box.dataset.bdmsg) && f.value.trim()) { box.textContent = "Looks good."; box.classList.add("ok"); }
    });
  }
}
// After the list is drawn: every built reward's messages again, and a check for those not checked yet.
function builtAfterRender() {
  ((S.rw && S.rw.rewards) || []).forEach((e, i) => {
    if (e.view.name !== "built" || !document.querySelector(`[data-bdmsg^="${i}:"]`)) return;
    const got = BUILT_CHECKS[i];
    if (got && got.of === JSON.stringify(e.view)) builtPaint(i, got.out); else builtCheckSoon(i, 50);
  });
  document.querySelectorAll(".bd-formula").forEach(builtGrow);
}
function builtGrow(el) { el.style.height = "auto"; el.style.height = el.scrollHeight + 2 + "px"; }

// ---------- testing a reward on a recorded game (the scoring, the reward alone) ----------
async function builtScore(i) {
  const e = S.rw.rewards[i], bot = S.bot;
  if (!e) return;
  const chosen = await call("choose_game");
  if (chosen.cancelled) return;
  if (chosen.error) return toast(chosen.error, true);
  const out = await call("score_built", bot, e.view, chosen.game);
  if (out.error) return toast(out.error, true);
  const name = chosen.game.split(/[\\/]/).pop();
  WATCH.pick = Object.assign(WATCH.pick || {}, { bot });
  WATCH.job = { bot, since: Date.now(), text: `${name} is being scored with ${wordsOf(e.view.title) || "this reward"} alone.`, cancelled: false, open: "scored" };
  watchRecordingModal();
  watchFollow(WATCH.job);
}

ACTIONS["open-builder-guide"] = async () => { const out = await call("open_folder", "", "builder-guide"); if (out.error) toast(out.error, true); };
// The formula a word goes into: the one typed in last (its id).
let builtLastFormula = "";
on("click", t => {
  if (t.dataset.bdins != null) {
    const el = builtLastFormula && $("#" + builtLastFormula);
    if (!el) return toast("Click in a formula first, then on the word to put it there.");
    const at = el.selectionStart != null ? el.selectionStart : el.value.length, end = el.selectionEnd != null ? el.selectionEnd : at;
    el.setRangeText(t.dataset.bdins, at, end, "end");
    el.focus();
    el.dispatchEvent(new Event("input", { bubbles: true }));
    return el.dispatchEvent(new Event("change", { bubbles: true }));
  }
  return t.dataset.bdx ? builtClick(t.dataset.bdx) : SKIP;
});
on("change", el => el.dataset.bd != null ? builtEdit(+el.dataset.bd, el.dataset.bk, el.type === "checkbox" ? el.checked : el.value, el.dataset.bdf != null) : SKIP);
// Typing in a formula: the view changes and the core checks it, without drawing the page again.
on("input", el => {
  if (el.dataset.bdf == null || el.dataset.bd == null) return;
  builtLastFormula = el.id;
  builtGrow(el);
  const i = +el.dataset.bd, e = S.rw && S.rw.rewards[i];
  if (e && builtApply(e.view, el.dataset.bk, el.value)) builtCheckSoon(i);
});
document.addEventListener("focusin", ev => { if (ev.target && ev.target.dataset && ev.target.dataset.bdf != null) builtLastFormula = ev.target.id; });

// ---------- templates: built rewards to start from (python/nitro/config/built_templates.json) ----------
function builtTemplatesHTML(q, chosen) {
  if (chosen && chosen !== "builtin") return "";
  const low = String(q || "").toLowerCase();
  const list = (builtBlocks().templates || []).filter(t => !low || (t.title + " " + wordsOf(t.title) + " " + t.about + " " + t.copies).toLowerCase().includes(low));
  if (!list.length) return "";
  return `<div class="catalog-head">Start from a template <span class="muted">${list.length}</span></div>` + list.map(t => `<div class="catalog-item"><button type="button" class="catalog-pick" data-bdtpl="${esc(t.title)}" title="Add a built reward made from this template, to edit">
      <span class="row" style="gap:8px;"><b>${marked(wordsOf(t.title), q)}</b><span class="badge">template</span>${t.copies ? `<span class="badge new" title="Checked: it pays exactly what ${esc(t.copies)} pays">a copy of ${esc(t.copies)}</span>` : ""}</span>
      <span class="about">${marked(t.about, q)}</span></button></div>`).join("");
}
on("click", t => {
  if (t.dataset.bdtpl == null) return SKIP;
  const tpl = (builtBlocks().templates || []).find(x => x.title === t.dataset.bdtpl);
  if (!tpl) return;
  S.rw.rewards.push({ from: null, view: templateView(tpl) });
  closeModal(); S.rwQuery = "";
  return stageRewards().then(() => { renderRewards($("#tab")); jumpToReward(S.rw.rewards.length - 1); });
});
