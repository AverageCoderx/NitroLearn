// The new-bot wizard, a page of four steps: the kind of bot, its name and number, its brain size, and a review.
// It makes the bot with `create_bot`, from the starting settings and the changed
// values (or `copy_bot` for a bot started from another), not started.
const ATTENTION_ONLY = ["model.attention_width", "model.attention_blocks", "model.attention_heads"];
const TRANSFORMER_ONLY = ["model.transformer_layers", "model.transformer_width", "model.transformer_heads", "model.transformer_ff"];
const WIZ_STEPS = ["Kind of bot", "Name and number", "Brain size", "Review"];
// The plain brain sizes: the layers of the policy and of the value network each. A preset can offer its own
// instead (its "brains", from garage.PRESET_CARDS).
const WIZ_BRAINS = [
  { key: "small", title: "Small", layers: [256, 256, 256], about: "The starters' own size. The quickest to train." },
  { key: "medium", title: "Medium", layers: [512, 512, 512], about: "Bigger: more to learn with, slower to train." },
  { key: "large", title: "Large", layers: [1024, 1024, 1024, 1024], about: "For long runs on a strong graphics card.", card: true },
  { key: "xl", title: "Extra large", layers: [1280, 1280, 1280, 1280], about: "The biggest here: the slowest to train.", card: true },
];
const WIZ_KIND_ABOUT = {
  1: "One car a side. The quickest to train and the best first bot.",
  2: "Two cars a side that learn to play as a team.",
  3: "Three a side. Needs the most training time.",
};

// ---------- drawings ----------
// The arena from above with `cars` a side: the small picture on a kind of bot. With `trails`, each car draws
// the line it drove (a recorded game).
function fieldSVG(cars, trails) {
  const line = "var(--rule2)", blue = [[150, 150, -25], [120, 96, 10], [92, 170, -5], [70, 120, 20]], orange = [[262, 104, 160], [296, 160, 190], [318, 84, 175], [338, 130, 170]];
  const car = (p, c) => `<g transform="translate(${p[0]} ${p[1]}) rotate(${p[2]})">${trails ? `<line x1="-64" x2="-16" y1="0" y2="0" stroke="${c}" stroke-width="4" stroke-dasharray="8 7"/>` : ""}<rect x="-12" y="-7" width="24" height="14" fill="${c}"/></g>`;
  return `<svg class="mini" viewBox="0 0 400 260" preserveAspectRatio="xMidYMid slice" role="img" aria-label="The arena from above, ${cars} a side"><rect width="400" height="260" fill="var(--pitch)"/><path d="M46 16h308l30 30v168l-30 30H46l-30-30V46z" fill="none" stroke="${line}" stroke-width="3"/><line x1="200" y1="16" x2="200" y2="244" stroke="${line}" stroke-width="3"/><circle cx="200" cy="130" r="36" fill="none" stroke="${line}" stroke-width="3"/><rect x="6" y="98" width="10" height="64" fill="var(--blue)"/><rect x="384" y="98" width="10" height="64" fill="var(--orange)"/>${blue.slice(0, cars).map(p => car(p, "var(--blue)")).join("")}${orange.slice(0, cars).map(p => car(p, "var(--orange)")).join("")}<circle cx="216" cy="118" r="9" fill="var(--text)"/></svg>`;
}

// ---------- what can be chosen ----------
function sourceOf(id) { return ((S.wiz && S.wiz.sources) || []).find(s => s.id === id); }
// What the API says of a preset beside its name and file: a title, a few words, tags and its own brain sizes.
function presetCard(name) { return ((S.home && S.home.presets) || []).find(p => p.name === name) || {}; }
function wizPreset(size) {
  const presets = (S.home && S.home.presets) || [];
  const found = presets.find(p => p.name === `${size}v${size}_starter`) || presets[0];
  return found ? found.name : "";
}
// The kinds of bot: every preset the API lists, then the two that start from something of the user's.
function wizKinds() {
  const w = S.wiz, presets = (S.home && S.home.presets) || [];
  const kinds = presets.map(p => {
    const src = sourceOf(p.name), size = (src && src.team_size) || Number((/^(\d)v\d/.exec(p.name) || [])[1]) || 1;
    const starter = /^(\d)v\1_starter$/.test(p.name);
    const title = p.name.replace(/_/g, " ");
    const tags = (p.tags || []).map(word => ({ word, cls: "" })).concat(p.name === "1v1_starter" ? [{ word: "Recommended", cls: "hot" }] : [], cardTag(!!p.brains));
    const about = p.about || (starter && WIZ_KIND_ABOUT[size] ? WIZ_KIND_ABOUT[size] : src ? src.about : "");
    return { id: "preset:" + p.name, preset: p.name, size, title: p.title || title[0].toUpperCase() + title.slice(1), about, tags };
  });
  kinds.push({ id: "replays", size: 2, picture: fieldSVG(2, true), title: "Learn from my replays first", about: "Copies how the players in your replay files play, then trains from there." });
  if (homeBots().length) kinds.push({ id: "from-bot", size: 1, picture: plate(fromBot(), 54), title: "Start from one of my bots", about: "A new bot with another bot's settings, which can keep what that bot already learned." });
  if (w.kind === "file") { const src = sourceOf(w.source); kinds.push({ id: "file", size: (src && src.team_size) || 1, title: src ? src.title : "A settings file", about: src ? src.about : "" }); }
  return kinds;
}
// The settings file the new bot starts from: the kind's preset, the picked file, or the replay bot's starter.
function wizSource() {
  const w = S.wiz;
  if (w.kind === "file") return w.source;
  if (w.kind === "replays") return w.replayPreset || wizPreset(1);
  return w.kind.startsWith("preset:") ? w.kind.slice(7) : "";
}
function wizSize() { const w = S.wiz, src = sourceOf(wizSource()); return w.form ? w.form.team_size : src ? src.team_size : 1; }
function fromBot() { return homeBot(S.wiz.fromBot) || homeBots()[0] || null; }

// Said on the kind and brain steps when this PC has no graphics card training can use.
function noCardNote() {
  return noCard() ? `<div class="banner warn" id="wz-no-card" role="note"><span><b>This PC has no graphics card NitroLearn can use</b>, so its bots train on the processor, many times slower. A 1v1 bot with the Small brain suits it best; the bigger brains and the GCO bot are made for a graphics card and can take minutes for each round of training here. Check this PC says more.</span></div>` : "";
}
// A brain size or kind of bot made for a graphics card: marked on a PC without one.
function cardTag(made) { return made && noCard() ? [{ word: "Slow on this PC", cls: "" }] : []; }

// ---------- the detailed fields (Custom) ----------
function wizValue(item) { return Object.prototype.hasOwnProperty.call(S.wiz.values, item.path) ? S.wiz.values[item.path] : item.value; }
function wizItem(path) { return ((S.wiz.form && S.wiz.form.steps) || []).flatMap(s => s.items).find(i => i.path === path); }
function wizItems(step) {
  const encoder = String(wizValue({ path: "model.encoder", value: (wizItem("model.encoder") || {}).value }));
  const attention = encoder === "attention" || encoder === "attention_communication";
  return step.items.filter(i => (attention || !ATTENTION_ONLY.includes(i.path)) && (encoder === "transformer" || !TRANSFORMER_ONLY.includes(i.path)));
}
function wizField(item) {
  const v = wizValue(item), id = "wz-" + item.path.replace(/\W/g, "-");
  let control;
  if (item.kind === "bool") control = `<input id="${id}" class="sw" type="checkbox" data-wiz="${esc(item.path)}" ${v ? "checked" : ""}>`;
  else if (item.kind === "choice") control = `<select id="${id}" class="field" data-wiz="${esc(item.path)}">${item.options.map(o => `<option value="${esc(o)}" ${String(o) === show(v) ? "selected" : ""}>${esc(o)}</option>`).join("")}</select>`;
  else control = `<input id="${id}" class="field mono" type="text" data-wiz="${esc(item.path)}" value="${esc(show(v))}" ${item.kind === "numbers" ? 'placeholder="e.g. 256, 256, 256"' : ""}>`;
  const extra = item.path === "env.tick_skip" ? ` <span class="muted" id="wz-rate" style="font-weight:400;">${tickRate(v)}</span>` : "";
  const open = S.wiz.helpOpen[item.path], help = item.plain || item.help || "";
  // A plain first line, with the setting's full description under "Nerd explanation"; else the description.
  const toggle = item.plain && item.help ? (open ? "Close the nerd explanation" : "Nerd explanation") : help.length > 220 ? (open ? "less" : "more") : "";
  return `<div class="wiz-item"><label for="${id}" style="font-weight:700;">${esc(item.title || item.label)}${extra}</label>
    ${help ? `<div class="help${open && !item.plain ? " full" : ""}">${esc(help)}</div>${toggle ? `<button type="button" class="btn link small" style="padding:0;align-self:flex-start;" data-wizhelp="${esc(item.path)}" aria-expanded="${!!open}">${toggle}</button>` : ""}` : ""}
    ${open && item.plain && item.help ? `<div class="help full nerd-help" data-nerd-help><span class="cap">Nerd explanation</span> ${esc(item.help)} <span class="mono muted">${esc(item.path)}</span></div>` : ""}
    ${control}${S.wiz.errors[item.path] ? `<span class="err" role="alert">${esc(S.wiz.errors[item.path])}</span>` : ""}</div>`;
}
function wizNorm(item, v) {
  if (item.kind === "numbers" && typeof v === "string") v = v.replace(/[\[\]]/g, "").split(/[\s,]+/).filter(Boolean);
  if (item.kind === "bool") v = !!v;
  return JSON.stringify(canon(v));
}
function wizChanges() {
  if (!S.wiz.form) return [];
  return S.wiz.form.steps.flatMap(s => wizItems(s)).filter(i =>
    Object.prototype.hasOwnProperty.call(S.wiz.values, i.path) && wizNorm(i, S.wiz.values[i.path]) !== wizNorm(i, i.value));
}
function tickRate(v) {
  const n = Number(String(v).trim());
  return String(v).trim() !== "" && Number.isInteger(n) && n >= 1 ? `(${num(120 / n, 1)} decisions a second)` : "(a whole number, such as 4)";
}
function wizardEdit(path, value) {
  const w = S.wiz;
  w.values[path] = value;
  delete w.errors[path];
  if (path === "model.encoder") keepFocus(renderWizard);  // a choice: drawn anew at once, never under a click
  if (path === "env.tick_skip" && $("#wz-rate")) $("#wz-rate").textContent = tickRate(value);
  if (/^(model|env)\./.test(path)) countBrains();
}

// ---------- brain sizes ----------
function sameLayers(a, b) { return JSON.stringify(canon(a)) === JSON.stringify(canon(b)); }
function layersOf(path) { const item = wizItem(path); return item ? wizValue(item) : []; }
// The brain sizes offered, each with the settings it changes: the preset's own, or the plain ones.
function brainCards() {
  const own = presetCard(S.wiz.preset).brains;
  return own || WIZ_BRAINS.map(b => Object.assign({ values: { "model.policy": b.layers, "model.value": b.layers } }, b));
}
// The settings the brain sizes change; a size that leaves one out keeps the settings file's.
function brainPaths() { return [...new Set(brainCards().flatMap(b => Object.keys(b.values)))].filter(path => wizItem(path)); }
function brainLayers(card, path) { return Object.prototype.hasOwnProperty.call(card.values, path) ? card.values[path] : wizItem(path).value; }
// Which size the form's layers are now: one of brainCards(), or "file" (the settings file's own, none of them).
function brainNow() {
  const paths = brainPaths();
  const found = brainCards().find(b => paths.every(path => wizNorm({ kind: "numbers" }, layersOf(path)) === JSON.stringify(canon(brainLayers(b, path)))));
  return found ? found.key : "file";
}
function brainValues(key) {
  const values = Object.assign({}, S.wiz.values), size = brainCards().find(b => b.key === key);
  for (const path of brainPaths()) {
    const item = wizItem(path), want = size && brainLayers(size, path);
    if (!size || sameLayers(item.value, want)) delete values[path]; else values[path] = want;
  }
  return values;
}
function pickBrain(key) {
  const w = S.wiz;
  if (key === "custom") { w.custom = true; }
  else { w.custom = false; w.values = brainValues(key); brainPaths().forEach(path => { delete w.errors[path]; }); }
  keepFocus(renderWizard);
  countBrains();
}
function layersWord(layers) {
  const list = (Array.isArray(layers) ? layers : String(layers).replace(/[\[\]]/g, "").split(/[\s,]+/).filter(Boolean)).map(Number);
  if (!list.length) return "no layers";
  return list.every(n => n === list[0]) ? `${list.length} ${list.length === 1 ? "layer" : "layers"} of ${num(list[0])}` : list.map(n => num(n)).join(", ");
}
// The widths of the layers the sizes change, network by network: "trunk 1,152 · policy 768 · value 1,280".
const BRAIN_PARTS = { "model.shared": "trunk", "model.policy": "policy", "model.value": "value", "model.value_head_layers": "heads" };
function partsWord(layersAt) {
  return brainPaths().map(path => {
    const list = JSON.parse(wizNorm({ kind: "numbers" }, layersAt(path))).map(Number);
    const widths = !list.length ? "none" : list.every(n => n === list[0]) ? num(list[0]) : list.map(n => num(n)).join("/");
    return `${BRAIN_PARTS[path] || path} ${widths}`;
  }).join(" · ");
}
function countWord(size) {
  if (!size) return "counting…";
  if (size.error) return "";
  return compact(size.total) + " parameters";
}
// Asks for the parameters of each plain size and of the form as it stands; the answers fill the cards.
function countBrains() {
  const w = S.wiz;
  clearTimeout(w.countTimer);
  w.countTimer = setTimeout(async () => {
    if (S.wiz !== w || !w.form) return;
    const asked = JSON.stringify(w.values), variants = { now: JSON.parse(asked), file: brainValues("file") };
    brainCards().forEach(b => { variants[b.key] = brainValues(b.key); });
    const key = JSON.stringify([w.preset, variants]);
    if (w.countKey === key) return;
    w.countKey = key;
    const out = await call("brain_sizes", w.preset, variants);
    if (S.wiz !== w || w.countKey !== key) return;
    w.sizes = out.sizes || {}; w.sizesError = out.error || ""; w.sizesFor = asked;
    if (w.step === 3) keepFocus(renderWizard); else drawBrainCounts();
  }, 250);
}
// The count of the form as it stands; none while a newer one is still being counted.
function nowCount() { const w = S.wiz; return w.sizes && w.sizesFor === JSON.stringify(w.values) ? w.sizes.now : null; }
function drawBrainCounts() {
  const w = S.wiz;
  document.querySelectorAll("[data-count]").forEach(el => {
    const size = w.sizes && w.sizes[el.dataset.count];
    put(el, esc(w.sizesError ? "" : countWord(size)));
  });
  // The line keeps its place while counting, so nothing below it moves when the count comes.
  const now = nowCount();
  const why = w.sizesError || (now && now.error) || "";
  put($("#wz-count"), why ? `<span class="err">These networks cannot be counted yet: ${esc(why)}</span>` : !now ? "Counting the parameters…" : `These networks: <b>${compact(now.total)} parameters</b> (${now.shared ? "shared " + compact(now.shared) + ", " : ""}policy ${compact(now.policy)}, value ${compact(now.value)}).`);
}

// ---------- name, number and colours ----------
function nameProblem(name) {
  if (!name) return "Give the bot a name first.";
  if (name.length > 64) return `A bot's name is at most 64 characters; this one has ${name.length}.`;
  if (!/^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$/.test(name)) return "A bot's name uses letters, digits, '-', '_' and '.', and starts with a letter or digit.";
  if (/\.$/.test(name) || /^(con|prn|aux|nul|com[1-9]|lpt[1-9])$/i.test(name.split(".")[0])) return "'" + name + "' cannot name a folder on Windows: pick another name.";
  if (homeBots().some(b => b.name === name)) return "There is already a bot called " + name + ".";
  return "";
}
function previewHTML() {
  const w = S.wiz, l = w.livery || {};
  return `<span class="cap muted">${w.liveryTouched ? "Your number and colours" : "Picked from its name"}</span>
    <div class="row">${plate({ livery: l }, 92)}<span class="display wiz-preview-name">${esc(w.name || "my-bot")}</span></div>${carSVG(l, 280)}`;
}
function drawPreview() {
  const w = S.wiz, l = w.livery || {};
  put($("#wz-preview"), previewHTML());
  const set = (id, value) => { const el = $(id); if (el && document.activeElement !== el && value && el.value !== value) el.value = value; };
  set("#wz-number", l.number); set("#wz-main", (l.main || "").toLowerCase()); set("#wz-second", (l.contrast || "").toLowerCase());
  document.querySelectorAll("[data-swatch]").forEach(el => { const pair = w.palette[Number(el.dataset.swatch)] || []; el.setAttribute("aria-pressed", String(pair[0] === l.main && pair[1] === l.contrast)); });
  put($("#wz-livery-err"), w.liveryError ? esc(w.liveryError) : "");
  const auto = $("#wz-auto"); if (auto) auto.hidden = !w.liveryTouched;
}
// The automatic pick for the name as typed, until the owner chooses a number or a colour.
function pickLivery() {
  const w = S.wiz;
  clearTimeout(w.pickTimer);
  w.pickTimer = setTimeout(async () => {
    const name = w.name;
    const out = await call("livery_pick", name || "my-bot");
    if (S.wiz !== w || w.name !== name || out.error) return;
    w.auto = out.livery; w.palette = out.palette || [];
    if (!w.liveryTouched) w.livery = out.livery;
    if (w.step === 1) { if (!$("[data-swatch]") && w.palette.length) keepFocus(renderWizard); else drawPreview(); }
  }, 120);
}
// A number or colour the owner chose: checked as the bot's livery will be, the preview follows.
function editLivery(change) {
  const w = S.wiz;
  w.liveryTouched = true;
  w.asked = Object.assign({ number: (w.livery || {}).number, main: (w.livery || {}).main, contrast: (w.livery || {}).contrast }, w.asked, change);
  const seq = w.liverySeq = (w.liverySeq || 0) + 1;
  return w.liveryCheck = call("livery_check", w.asked.number, w.asked.main, w.asked.contrast).then(out => {
    if (S.wiz !== w || w.liverySeq !== seq) return;
    w.liveryError = out.error ? out.error[0].toUpperCase() + out.error.slice(1) + "." : "";
    if (out.livery) w.livery = out.livery;
    if (w.step === 1) drawPreview();
  });
}

// ---------- the steps ----------
function kindStep() {
  const w = S.wiz, kinds = wizKinds();
  const files = w.sources.filter(s => s.group !== "preset");
  const groups = [["my-settings", "My settings (my-settings folder)"], ["library", "Added on the Resources page"]];
  const starters = ((S.home && S.home.presets) || []).filter(p => /^(\d)v\1_starter$/.test(p.name));
  const from = fromBot();
  let extra = "";
  if (w.kind === "replays") {
    extra = `<div class="wiz-more"><div class="f"><span class="lab">Game size</span><div class="seg" data-seg="wzsize">${starters.map(p => `<button type="button" data-v="${esc(p.name)}" aria-pressed="${wizSource() === p.name}">${esc(p.name.split("_")[0])}</button>`).join("")}</div></div>
      <div class="muted">The bot is made from the starter of this size with <b>Learn from replays first</b> switched on. Its Settings tab opens next: add the folder your replays are in there, then press Apply.</div></div>`;
  } else if (w.kind === "from-bot" && from) {
    const newest = from.newest_save;
    extra = `<div class="wiz-more"><label class="f" for="wz-from"><span class="lab">Which bot</span><select id="wz-from" class="field">${homeBots().map(b => `<option value="${esc(b.name)}" ${b.name === from.name ? "selected" : ""}>${esc(b.name)}${b.team_size ? " · " + b.team_size + "v" + b.team_size : ""} · ${b.saves} ${b.saves === 1 ? "save" : "saves"}</option>`).join("")}</select></label>
      <label class="check"><input id="wz-transfer" type="checkbox" ${newest != null && w.transfer ? "checked" : ""} ${newest != null ? "" : "disabled"}><span>${newest != null ? `Start from ${esc(from.name)}'s trained networks (transfer learning): its newest save, at ${Number(newest).toLocaleString("en-US")} steps, taking everything it learned, since the new bot keeps its rewards.` : `Start from ${esc(from.name)}'s trained networks (transfer learning): not possible yet, it has no save.`} Unticked, it starts from scratch with the same settings.</span></label>
      ${from.status === "problem" ? `<span class="err" role="alert">${esc(from.name)}'s settings file has a mistake: fix it first, or pick another bot.</span>` : ""}</div>`;
  }
  return `<div><h1 class="display wiz-title">What kind of bot?</h1><div class="muted wiz-lede">You can change almost everything later. Pick the closest one.</div></div>
    ${noCardNote()}
    <div class="wiz-kinds">${kinds.map(k => `<button type="button" class="wiz-kind" id="wz-kind-${esc(k.id.replace(/\W/g, "-"))}" data-kind="${esc(k.id)}" aria-pressed="${w.kind === k.id}">
      <span class="top">${k.picture || fieldSVG(k.size)}<span class="tags">${(k.tags || []).map(t => `<span class="tag${t.cls ? " " + t.cls : ""}">${esc(t.word)}</span>`).join("")}</span></span><b>${esc(k.title)}</b><span class="about">${esc(k.about)}</span></button>`).join("")}</div>
    ${extra}
    ${files.length ? `<label class="f wiz-file" for="wz-source"><span class="lab">Or start from a settings file</span><span class="h">Its rewards, start states and every other setting are copied into the new bot; all can be changed later.</span>
      <select id="wz-source" class="field"><option value="">None: one of the kinds above</option>${groups.map(([g, label]) => { const list = files.filter(s => s.group === g); return list.length ? `<optgroup label="${esc(label)}">${list.map(s => `<option value="${esc(s.id)}" ${w.kind === "file" && s.id === w.source ? "selected" : ""}>${esc(s.title)} · ${esc(s.about)}</option>`).join("")}</optgroup>` : ""; }).join("")}</select></label>` : ""}
    ${w.formError ? `<div class="banner bad" role="alert">${esc(w.formError)}</div>` : ""}`;
}
function nameStep() {
  const w = S.wiz, l = w.livery || {};
  const hex = c => LIVERY_HEX.test(c || "") ? c.toLowerCase() : "#8c8a84";
  return `<div><h1 class="display wiz-title">Name and number</h1><div class="muted wiz-lede">The name is for good. The number and colours are how you tell your bots apart at a glance.</div></div>
    <div class="wiz-two">
      <div class="wiz-fields">
        <label class="f" for="wz-name"><span class="lab">Name</span><span class="h">Up to 64 letters, digits, -, _ and . (starting with a letter or digit). It names the bot's folder and its RLBot bot.</span>
          <input id="wz-name" class="field" type="text" autocomplete="off" spellcheck="false" value="${esc(w.name)}" ${w.nameError ? 'aria-invalid="true"' : ""}></label>
        <span class="err" id="wz-name-err" role="alert">${esc(w.nameError)}</span>
        <label class="f" for="wz-number"><span class="lab">Race number</span><span class="h">One or two digits, on its number plate and its car.</span>
          <input id="wz-number" class="field mono wiz-number" type="text" inputmode="numeric" maxlength="2" autocomplete="off" value="${esc(l.number || "")}"></label>
        <div class="f"><span class="lab">Colours</span><span class="h">Pick a pair, or set the main and the second colour yourself.</span>
          <div class="wiz-swatches">${w.palette.map((pair, i) => `<button type="button" class="wiz-swatch" data-swatch="${i}" aria-pressed="${pair[0] === l.main && pair[1] === l.contrast}" aria-label="Colour pair ${i + 1}" title="Colour pair ${i + 1}"><span style="background:${esc(pair[0])};"></span><span style="background:${esc(pair[1])};"></span></button>`).join("")}</div>
          <div class="row"><label class="wiz-colour" for="wz-main"><input id="wz-main" type="color" value="${hex(l.main)}"><span>Main colour</span></label><label class="wiz-colour" for="wz-second"><input id="wz-second" type="color" value="${hex(l.contrast)}"><span>Second colour</span></label></div></div>
        <span class="err" id="wz-livery-err" role="alert">${esc(w.liveryError)}</span>
        <button type="button" class="btn link" id="wz-auto" data-act="wiz-auto" ${w.liveryTouched ? "" : "hidden"}>Back to the number and colours picked from its name</button>
      </div>
      <div class="wiz-preview" id="wz-preview" aria-label="How its number plate and car look"></div>
    </div>`;
}
function brainStep() {
  const w = S.wiz;
  if (w.kind === "from-bot") {
    const from = fromBot() || {};
    return `<div><h1 class="display wiz-title">Brain size</h1><div class="muted wiz-lede">A bot started from another gets that bot's brain.</div></div>
      <div class="banner"><div><b>${esc(w.name)} gets ${esc(from.name)}'s networks' kind and sizes</b>${from.encoder ? ` (${esc(networkWord(from))}${from.parameters && from.parameters.total ? ", " + compact(from.parameters.total) + " parameters" : ""})` : ""}, with every other setting of it.<br>Until its first start everything can still change in its Settings tab, network sizes and transfer learning included.</div></div>`;
  }
  const now = brainNow(), picked = w.custom ? "custom" : now;
  const card = presetCard(w.preset), paths = brainPaths();
  const words = b => b.layers ? layersWord(b.layers) : b.layersWord || partsWord(path => brainLayers(b, path));
  const size = b => { const mine = picked === b.key; return `<button type="button" class="wiz-kind brain" id="wz-brain-${b.key}" data-brain="${b.key}" aria-pressed="${mine}">${cardTag(b.card || !!card.brains).map(t => `<span class="tag">${esc(t.word)}</span>`).join("")}<b>${esc(b.title)}</b><span class="mono layers">${esc(words(b))}</span><span class="cap count" data-count="${b.key}"></span><span class="about">${esc(b.about)}</span></button>`; };
  const file = path => (wizItem(path) || {}).value || [];
  const fileOwn = !brainCards().some(b => paths.every(path => sameLayers(file(path), brainLayers(b, path))));
  const cards = (fileOwn ? [{ key: "file", title: "As the starting settings have it", layersWord: "policy " + layersWord(file("model.policy")) + " · value " + layersWord(file("model.value")), about: "The sizes written in the settings file you start from." }] : []).concat(brainCards());
  const others = wizChanges().filter(i => !paths.includes(i.path));
  const custom = `<button type="button" class="wiz-kind brain" id="wz-brain-custom" data-brain="custom" aria-pressed="${picked === "custom"}"><b>Custom</b><span class="mono layers">every detail</span><span class="cap count"></span><span class="about">Layer sizes, the network's kind, what the bot sees, how often it decides and how it learns.</span></button>`;
  return `<div><h1 class="display wiz-title">Brain size</h1><div class="muted wiz-lede">${card.brains_about ? esc(card.brains_about) : "How big the bot's networks are: the policy (which action to take) and the value network (how good the situation is), each with these layers. Fixed once the bot has its first save."}</div></div>
    ${noCardNote()}
    <div class="wiz-kinds wiz-brains">${cards.map(size).join("")}${custom}</div>
    ${!w.custom && others.length ? `<div class="muted">Kept from Custom: ${others.map(i => `${esc(i.label)} ${esc(show(w.values[i.path]))}`).join(" · ")}.</div>` : ""}
    <div class="muted" id="wz-count"></div>
    ${w.custom ? w.form.steps.map(step => `<section class="wiz-group"><div class="heading"><h2 class="display">${esc(step.title)}</h2><span class="muted">${esc(step.about)}</span></div><div>${wizItems(step).map(wizField).join("")}</div></section>`).join("") : ""}`;
}
function reviewStep() {
  const w = S.wiz, l = w.livery || {};
  const head = `<div><h1 class="display wiz-title">Review</h1><div class="muted wiz-lede">Nothing is made until you press Make the bot.</div></div>`;
  const who = `<div class="wiz-made">${plate({ livery: l }, 72)}<div><div class="display wiz-preview-name">${esc(w.name)}</div><div class="cap muted">Number ${esc(l.number || "")} · ${w.liveryTouched ? "your number and colours" : "number and colours picked from its name"}</div></div><span class="grow"></span>${carSVG(l, 200)}</div>`;
  if (w.kind === "from-bot") {
    const from = fromBot() || {}, newest = from.newest_save;
    return `${head}${who}<div class="wiz-review"><span class="muted">Kind</span><span>A new bot from <b>${esc(from.name)}</b>: its rewards, its network sizes and every other setting.</span>
      <span class="muted">Starts from</span><span>${w.transfer && newest != null ? `${esc(from.name)}'s trained networks (transfer learning): its save at ${Number(newest).toLocaleString("en-US")} steps, taking everything it learned.` : "Scratch, with the same settings."}</span></div>
      <div class="banner">The bot is made but not started: its Settings tab opens next. Until its first start everything can still change there, network sizes and transfer learning included.</div>`;
  }
  const changed = wizChanges(), src = sourceOf(w.preset), size = nowCount();
  const brain = brainCards().find(b => b.key === brainNow());
  const layers = presetCard(w.preset).brains ? partsWord(layersOf) : `policy ${layersWord(layersOf("model.policy"))} · value ${layersWord(layersOf("model.value"))}`;
  const fixed = w.fixed || [];
  return `${head}${who}
    <div class="wiz-review"><span class="muted">Game</span><span>${wizSize()}v${wizSize()}</span>
      <span class="muted">Starting settings</span><span>${esc(src ? src.title : w.preset.replace(/_/g, " ") + " preset")}: its rewards, start states and every other setting are copied into the new bot.</span>
      ${w.kind === "replays" ? `<span class="muted">Learns first</span><span>From your replays: it copies how the players in them play, then trains from there. Its Settings tab opens next to add your replay folders.</span>` : ""}
      <span class="muted">Brain</span><span>${brain ? esc(brain.title) + ": " : ""}${esc(layers)}${size && !size.error ? ` · <b>${compact(size.total)} parameters</b>` : ""}</span>
      ${changed.map(i => `<span class="muted">${esc(i.label)}</span><span class="mono">${esc(show(w.values[i.path]))} <span class="muted">(starting settings: ${esc(show(i.value))})</span></span>`).join("")}</div>
    ${changed.length ? "" : '<div class="muted">Everything else as in the starting settings.</div>'}
    ${changed.some(i => i.path === "env.tick_skip" || i.path === "model.encoder") ? '<div class="muted" style="font-size:13px;">If they need it, the action delay is set one tick below the tick skip and an attention network gets the observation room it needs; you will be told.</div>' : ""}
    ${fixed.length ? `<section class="wiz-fixed"><div class="cap">${ICON.lock} Cannot be changed after its first save</div><div class="chips">${fixed.map(f => `<span class="chip" title="${esc(f.path)}">${esc(f.title || f.label)}: ${esc(f.words || show(f.value) || "not set")}</span>`).join("")}</div><div class="muted">These belong to the bot's trained networks. Until the first save (15 minutes into training by default) they can still be changed in its Settings tab; everything else can be changed any time.</div></section>` : ""}
    <div class="banner">${w.kind === "replays" ? "The bot is made but not started: its Settings tab opens next, with Learn from replays first waiting for Apply." : "The bot is made but not started: its Rewards tab opens next, so you can set up its rewards first. Then press Start training."}</div>`;
}
function renderWizard() {
  const w = S.wiz;
  if (!w) { S.view = "home"; return render(); }
  const garage = $('.nav [data-go="home"]');
  if (garage) garage.setAttribute("aria-current", "page");  // making a bot is part of the Garage
  const last = WIZ_STEPS.length - 1;
  const body = [kindStep, nameStep, brainStep, reviewStep][w.step]();
  const next = w.step < last ? `<button type="button" class="btn primary" id="wz-next" data-act="wiz-next">Next: ${esc(WIZ_STEPS[w.step + 1].toLowerCase())}${ICON.chev}</button>`
    : `<button type="button" class="btn primary" id="wz-make" data-act="wiz-make">Make the bot</button>`;
  $("#main").innerHTML = `<div class="wizard" id="wizard">
    <nav class="wiz-steps" aria-label="New bot: its steps"><span class="cap muted">New bot</span>${WIZ_STEPS.map((title, i) => `<button type="button" id="wz-step-${i}" data-wizstep="${i}" class="${i < w.step ? "done" : ""}" ${i === w.step ? 'aria-current="step"' : ""} ${i <= w.seen ? "" : "disabled"}><i class="num">${i + 1}</i><b>${esc(title)}</b></button>`).join("")}</nav>
    <div class="wiz-body">${body}
      <div class="wiz-nav"><button type="button" class="btn" id="wz-cancel" data-act="wiz-cancel">Cancel</button><span class="grow"></span>${w.step > 0 ? '<button type="button" class="btn" id="wz-back" data-act="wiz-back">Back</button>' : ""}${next}</div>
    </div></div>`;
  if (w.step === 1) drawPreview();
  if (w.step === 2) drawBrainCounts();
}
// A name no bot has yet: my-bot, else my-bot-2, my-bot-3 and so on.
function freeBotName() {
  for (let n = 1; ; n++) { const name = n === 1 ? "my-bot" : "my-bot-" + n; if (!nameProblem(name)) return name; }
}
// Shows a step and puts the keyboard on its first field. The name step starts with a real name, selected, so
// typing replaces it and Next keeps it.
function showStep(step) {
  const w = S.wiz;
  w.step = step; w.seen = Math.max(w.seen, step);
  if (step === 1 && !w.name) w.name = freeBotName();
  renderWizard();
  window.scrollTo(0, 0);
  const first = step === 0 ? $(".wiz-kind[aria-pressed=true]") : step === 1 ? $("#wz-name") : step === 2 ? $(".brain[aria-pressed=true]") : $("#wz-make");
  if (first) first.focus({ preventScroll: true });
  if (step === 1 && first) first.select();
  if (step === 1) pickLivery();
  if (step === 2 && w.form) countBrains();
}
// A mistake on the step: said beside its field, which gets the keyboard.
function wizStop(field) {
  renderWizard();
  const el = field && $(field);
  if (el) { el.focus(); el.scrollIntoView({ block: "center" }); }
  return false;
}
// Whether the step can be left for the next one; a mistake is shown on it.
async function leaveStep() {
  const w = S.wiz;
  if (w.step === 0) {
    w.formError = "";
    if (w.kind === "from-bot") {
      const from = fromBot();
      if (!from || from.status === "problem") return wizStop("#wz-from");
      w.fromBot = from.name;
      if (!w.name) w.name = nameProblem(from.name + "-2") ? "" : from.name + "-2";
      return true;
    }
    const preset = wizSource();
    if (!w.form || w.preset !== preset) {
      const form = await call("wizard", preset);
      if (S.wiz !== w) return false;
      if (form.error) { w.formError = form.error; return wizStop(); }
      w.form = form; w.preset = preset; w.values = {}; w.errors = {}; w.custom = false; w.sizes = null; w.countKey = ""; w.fixed = null;
    }
    return true;
  }
  if (w.step === 1) {
    w.name = w.name.trim();
    w.nameError = nameProblem(w.name);
    if (w.nameError) return wizStop("#wz-name");
    if (w.liveryTouched) {
      await w.liveryCheck;
      if (S.wiz !== w) return false;
      if (w.liveryError) return wizStop("#wz-number");
    }
    return true;
  }
  if (w.step === 2 && w.kind !== "from-bot") {
    const out = await call("check_bot", w.preset, w.values);
    if (S.wiz !== w) return false;
    if (out.error) { w.formError = out.error; toast(out.error, true); return false; }
    const found = Object.entries(out.errors || {});
    if (found.length) {
      found.forEach(([path, why]) => { w.errors[path] = why; });
      w.custom = true;
      const known = found.find(([path]) => wizItem(path));
      if (!known) toast(found.map(([, why]) => why).join(" "), true);  // not about one of the form's fields
      return wizStop(known ? "#wz-" + known[0].replace(/\W/g, "-") : "");
    }
    const fixed = await call("fixed_after_first_save", w.preset, w.values);
    if (S.wiz !== w) return false;
    w.fixed = fixed.fixed || [];
  }
  return true;
}
// Goes to `target`: back at once; forward one step at a time, each checked as Next checks it.
async function wizardGo(target) {
  const w = S.wiz;
  if (!w || w.moving) return;
  w.moving = true;
  try {
    while (S.wiz === w && w.step < target) {
      if (!await leaveStep()) return;
      if (S.wiz !== w || S.view !== "wizard") return;
      showStep(w.step + 1);
    }
    if (S.wiz === w && target < w.step) showStep(Math.max(0, target));
  } finally { w.moving = false; }
}
function pickKind(id) {
  const w = S.wiz;
  w.kind = id; w.formError = "";
  if (id !== "file") w.source = "";
  keepFocus(renderWizard);
}

async function newBotWizard() {
  const first = ((S.home && S.home.presets) || [])[0];
  const wiz = S.wiz = {
    step: 0, seen: 0, kind: "preset:" + (wizPreset(1) || (first && first.name) || ""), source: "", replayPreset: "", fromBot: "", transfer: true,
    sources: [], name: "", nameError: "", livery: null, auto: null, palette: [], liveryTouched: false, liveryError: "", asked: null,
    preset: "", form: null, values: {}, errors: {}, helpOpen: {}, custom: false, sizes: null, fixed: null, formError: "",
  };
  S.view = "wizard"; S.bot = null; S.data = null;
  render();
  showStep(0);
  const out = await call("settings_sources");
  if (out.error || S.wiz !== wiz || S.view !== "wizard") return;  // left (or opened again) meanwhile: it stays as it is
  wiz.sources = out.sources || [];
  if (wiz.step === 0) keepFocus(renderWizard);
}
function leaveWizard() { S.wiz = null; S.view = "home"; return loadHome().then(render); }
// The number and colours the owner chose are kept with the new bot; the automatic pick needs no file.
async function keepLivery(name) {
  const w = S.wiz, l = w.livery || {}, a = w.auto || {};
  if (!w.liveryTouched || (l.number === a.number && l.main === a.main && l.contrast === a.contrast)) return;
  const out = await call("set_livery", name, l.number, l.main, l.contrast);
  if (out.error) toast("Its number and colours were not kept (" + out.error + "): it has the ones picked from its name.", true);
}
async function wizardMake() {
  const w = S.wiz;
  if (!w || w.making) return;
  w.making = true;
  try {
    if (w.kind === "from-bot") {
      const out = await call("copy_bot", w.fromBot, w.name, !!(w.transfer && (fromBot() || {}).newest_save != null));
      if (out.error) { w.nameError = nameProblem(w.name); if (w.nameError) { showStep(1); return wizStop("#wz-name"); } return toast(out.error, true); }
      await keepLivery(out.made);
      S.wiz = null; await loadHome();
      toast("Made " + out.made + "."); (out.notes || []).forEach(n => toast(n));
      return openBot(out.made, "settings");
    }
    const out = await call("create_bot", w.name, w.preset, w.values);
    if (out.errors) {
      Object.assign(w.errors, out.errors); w.custom = true;
      showStep(2);
      const known = Object.keys(out.errors).find(path => wizItem(path));
      if (known) wizStop("#wz-" + known.replace(/\W/g, "-"));
      return toast(Object.values(out.errors).join(" "), true);
    }
    if (out.error) { w.nameError = nameProblem(w.name); if (w.nameError) { showStep(1); return wizStop("#wz-name"); } return toast(out.error, true); }
    await keepLivery(out.made);
    const replays = w.kind === "replays";
    if (replays) await replaysFirst(out.made);
    S.wiz = null;
    toast(replays ? "Made " + out.made + ". Add the folder your replays are in, then press Apply." : "Made " + out.made + ". Set up its rewards, then press Start training.");
    (out.adjusted || []).concat(out.advice || []).forEach(line => toast(line));
    await loadHome();
    if (!replays) return openBot(out.made, "rewards");
    S.searchTarget = "pretrain.replays";
    return openBot(out.made, "settings");
  } finally { w.making = false; }
}
// Learn from replays first, switched on as its switch in the Settings tab does it: a change waiting for Apply.
async function replaysFirst(name) {
  const data = await call("bot", name);
  const batch = allSections(data.settings || []).flatMap(s => s.items).find(i => i.path === "pretrain.batch_replays");
  const out = await call("stage", name, "pretrain.enabled", true);
  if (out.error) return toast(out.error, true);
  if (batch && !Number(batch.value)) await call("stage", name, "pretrain.batch_replays", 100);
}

VIEWS.wizard = () => renderWizard();
ACTIONS.newbot = () => newBotWizard();
ACTIONS["wiz-next"] = () => wizardGo(S.wiz.step + 1);
ACTIONS["wiz-back"] = () => wizardGo(S.wiz.step - 1);
ACTIONS["wiz-make"] = () => wizardMake();
ACTIONS["wiz-cancel"] = () => leaveWizard();
ACTIONS["wiz-auto"] = () => { const w = S.wiz; w.liveryTouched = false; w.liveryError = ""; w.asked = null; w.livery = w.auto || w.livery; w.liverySeq = (w.liverySeq || 0) + 1; keepFocus(renderWizard); };
on("click", t => {
  if (!S.wiz || S.view !== "wizard") return SKIP;
  if (t.dataset.wizhelp) { S.wiz.helpOpen[t.dataset.wizhelp] = !S.wiz.helpOpen[t.dataset.wizhelp]; return keepFocus(renderWizard); }
  if (t.dataset.kind) return pickKind(t.dataset.kind);
  if (t.dataset.brain) return pickBrain(t.dataset.brain);
  if (t.dataset.wizstep) return wizardGo(Number(t.dataset.wizstep));
  if (t.dataset.swatch) { const pair = S.wiz.palette[Number(t.dataset.swatch)]; if (pair) editLivery({ main: pair[0], contrast: pair[1] }); return; }
  return SKIP;
});
on("change", el => {
  if (!S.wiz || S.view !== "wizard") return SKIP;
  if (el.dataset.wiz) return wizardEdit(el.dataset.wiz, el.type === "checkbox" ? el.checked : el.value);
  if (el.id === "wz-source") { S.wiz.source = el.value; return pickKind(el.value ? "file" : "preset:" + wizPreset(1)); }
  if (el.id === "wz-from") { S.wiz.fromBot = el.value; return keepFocus(renderWizard); }
  if (el.id === "wz-transfer") { S.wiz.transfer = el.checked; return; }
  return SKIP;
});
on("input", el => {
  const w = S.wiz;
  if (!w || S.view !== "wizard") return;
  if (el.id === "wz-name") {
    w.name = el.value.trim();
    if (w.nameError) { w.nameError = ""; el.removeAttribute("aria-invalid"); put($("#wz-name-err"), ""); }
    put($("#wz-preview"), previewHTML());
    pickLivery();
  }
  if (el.id === "wz-number") editLivery({ number: el.value });
  if (el.id === "wz-main") editLivery({ main: el.value });
  if (el.id === "wz-second") editLivery({ contrast: el.value });
});
SEGS.wzsize = value => { S.wiz.replayPreset = value; };
// Enter moves on: from a field, a kind of bot or a brain size.
document.addEventListener("keydown", e => {
  const w = S.wiz, t = e.target;
  if (e.key !== "Enter" || !w || S.view !== "wizard" || !$("#modal").hidden || !t.closest || !t.closest("#wizard")) return;
  const field = t.tagName === "INPUT" && t.type !== "checkbox" && t.type !== "color";
  if (!field && !t.dataset.kind && !t.dataset.brain) return;
  if (w.step >= WIZ_STEPS.length - 1) return;
  e.preventDefault();
  if (field) t.dispatchEvent(new Event("change", { bubbles: true }));
  if (t.dataset.kind && w.kind !== t.dataset.kind) pickKind(t.dataset.kind);
  if (t.dataset.brain && t.dataset.brain !== "custom") pickBrain(t.dataset.brain);
  if (t.dataset.brain === "custom" && !w.custom) return pickBrain("custom");
  wizardGo(w.step + 1);
});
