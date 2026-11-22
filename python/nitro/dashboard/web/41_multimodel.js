// The Multi-model bot page: pick two or more of your bots as its models (each with its kind, size and any
// problem in plain words), the bot whose game and rewards it plays, the selector's size and settings, and make it.
// The Overview of a multi-model bot shows each model's share of the driving as it trains (multiModelHTML).
const MM_NUMBERS = ["min_hold_seconds", "switch_cost", "model_pull", "models_start_after"];
const MM_SWITCHES = ["train_models", "goal_heads", "standardise_values"];
function mmField(key) { return ((S.mm && S.mm.fields) || []).find(f => f.path === "selector." + key) || { label: key, help: "", more: "" }; }
async function openMultiModel() {
  S.mm = { loading: true, picked: [], base: "", name: "", size: "small", attention: false, shares: {}, values: { min_hold_seconds: 0.5, switch_cost: 0, model_pull: 0.1, models_start_after: 20000000 }, switches: {}, errors: [] };
  S.view = "multimodel"; S.bot = null; S.data = null;
  render();
  await loadMultiModel();
}
async function loadMultiModel() {
  const m = S.mm;
  const out = await call("multi_model_bots", m.base || "");
  if (S.mm !== m) return;
  if (out.error) { m.error = out.error; m.loading = false; return renderMultiModel(); }
  Object.assign(m, { bots: out.bots || [], sizes: out.sizes || {}, fields: out.fields || [], loading: false });
  if (out.warmup != null && !m.warmupSet) { m.values.models_start_after = out.warmup; m.warmupSet = true; }
  renderMultiModel();
}
function mmBot(name) { return ((S.mm && S.mm.bots) || []).find(b => b.name === name); }
function mmShareSum() { return S.mm.picked.reduce((sum, name) => sum + (Number(S.mm.shares[name]) || 0), 0); }
function mmProblems() {
  const m = S.mm, out = [];
  if (m.picked.length < 2) out.push("Pick at least two models.");
  if (m.picked.length > 8) out.push("Pick at most eight models.");
  for (const name of m.picked) { const b = mmBot(name); if (b && b.problems.length) out.push(`${name}: ${b.problems.join("; ")}.`); }
  const sum = mmShareSum();
  if (sum >= 100) out.push(`The minimum shares add up to ${sum}%: keep them below 100%, so the selector still has room to choose.`);
  if (!/^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$/.test(m.name || "")) out.push("Give it a name: letters, digits, -, _ and ., starting with a letter or digit.");
  return out;
}
function mmBotRow(b) {
  const m = S.mm, on = m.picked.includes(b.name), bad = b.problems.length > 0;
  const facts = [b.kind, b.parameters ? compact(b.parameters) + " parameters" : "", b.game, b.observation ? b.observation + " observation" : "", b.tick_skip ? "tick skip " + b.tick_skip : ""].filter(Boolean).join(" · ");
  return `<label class="mm-bot${on ? " on" : ""}${bad ? " bad" : ""}" for="mm-pick-${esc(b.name)}">
    <input id="mm-pick-${esc(b.name)}" type="checkbox" data-mmpick="${esc(b.name)}" ${on ? "checked" : ""}>
    <span class="grow"><b>${esc(b.name)}</b><span class="muted"> ${esc(facts)}</span>
      ${bad ? `<span class="err">${esc(b.problems.join("; "))}.</span>` : ""}</span>
    ${on ? `<span class="mm-share" title="${esc(mmField("min_share").help)}">at least <input class="field mono" type="text" inputmode="decimal" data-mmshare="${esc(b.name)}" value="${esc(String(m.shares[b.name] || 0))}" aria-label="Smallest share of the driving for ${esc(b.name)}, percent">%</span>` : ""}
  </label>`;
}
function mmInput(key) {
  const f = mmField(key), id = "mm-" + key;
  return `<label class="f" for="${id}"><span class="lab">${esc(f.label)}</span><span class="h">${esc(f.help)}</span>
    <input id="${id}" class="field mono" type="text" inputmode="decimal" data-mmval="${esc(key)}" value="${esc(String(S.mm.values[key]))}"></label>`;
}
function mmCheck(key) {
  const f = mmField(key);
  return `<label class="check" title="${esc(f.help)}"><input type="checkbox" data-mmswitch="${esc(key)}" ${S.mm.switches[key] ? "checked" : ""}><span><b>${esc(f.label)}.</b> ${esc(f.help)}</span></label>`;
}
function mmGuideHTML() {
  return `<div class="guide">${["specialists", "min_hold_seconds", "switch_cost", "min_share", "train_models", "models_start_after", "model_pull", "goal_heads", "standardise_values"].map(key => { const f = mmField(key); return f.more ? `<p><b>${esc(f.label)}.</b></p>${guideHTML(f.more)}` : ""; }).join("")}</div>`;
}
function renderMultiModel() {
  const box = $("#main"), m = S.mm;
  if (!box || !m) return;
  if (m.loading) { box.innerHTML = '<div class="card empty">Looking at your bots…</div>'; return; }
  if (m.error) { box.innerHTML = `<div class="banner bad"><pre>${esc(m.error)}</pre></div>`; return; }
  const bots = m.bots || [], sum = mmShareSum(), problems = mmProblems();
  const sizes = Object.entries(m.sizes || {}).map(([k, layers]) => `<option value="${esc(k)}" ${m.size === k ? "selected" : ""}>${esc(k[0].toUpperCase() + k.slice(1))}: ${layers.join(" × ")}</option>`).join("");
  const bases = m.picked.map(name => `<option value="${esc(name)}" ${m.base === name ? "selected" : ""}>${esc(name)}</option>`).join("");
  keepFocus(() => { box.innerHTML = `<div class="mm-page">
    <div class="heading"><h1 class="display">New multi-model bot</h1><button type="button" class="btn small ghost" data-act="mm-guide">How multi-model bots work</button></div>
    <div class="muted" style="font-size:13px;max-width:96ch;">A multi-model bot is several of your trained bots in one. At every moment a small network, the selector, picks which of them (its models) drives each car, so each can do what it is best at; it learns when to hand over from the games it plays. Each model keeps seeing the game the way it learned to. You can train the models on too, so they learn the moments they are handed. Nothing here changes the bots you pick: they are copied in. How well it plays is not measured: try it, and compare it with your best single bot (nitro eval).</div>
    <section class="card pad"><h3>1. Its models</h3><div class="muted" style="font-size:13px;">Tick two to eight. A problem shown in red would stop it from training. The share box is the smallest share of the driving to aim for (0 = none).</div>
      <div class="mm-bots">${bots.map(mmBotRow).join("") || '<div class="empty">You have no bots yet. Train a few first.</div>'}</div>
      <div class="${sum >= 100 ? "err" : "muted"}" style="font-size:13px;">Minimum shares together: ${sum}%${sum >= 100 ? " — keep them below 100%." : " (below 100%: good)."}</div></section>
    <section class="card pad"><h3>2. Its game and rewards</h3>
      <label class="f" for="mm-base"><span class="lab">Plays the game, rewards and starts of</span><span class="h">One of its models. Use a full bot (one that plays against opponents), not one trained alone.</span>
        <select id="mm-base" class="field">${bases || '<option value="">Pick its models first</option>'}</select></label></section>
    <section class="card pad"><h3>3. The selector</h3>
      <label class="f" for="mm-size"><span class="lab">Its size</span><span class="h">The selector only picks a model, so it can stay small.</span><select id="mm-size" class="field">${sizes}</select></label>
      <label class="check"><input type="checkbox" id="mm-attention" ${m.attention ? "checked" : ""}><span><b>Attention selector.</b> Reads each car as its own token, as attention bots do. Not measured against the plain selector in this version.</span></label>
      ${mmInput("min_hold_seconds")}${mmInput("switch_cost")}
      ${mmCheck("standardise_values")}${mmCheck("goal_heads")}</section>
    <section class="card pad"><h3>4. Train the models too</h3>
      ${mmCheck("train_models")}
      ${m.switches.train_models ? `<div class="muted" style="font-size:13px;">First the selector learns alone while the models stay as they are, so they never learn from its early, clumsy choices; then both learn. The suggested warm-up is a guess, not measured.</div>${mmInput("models_start_after")}${mmInput("model_pull")}` : ""}</section>
    <section class="card pad"><h3>5. Name</h3>
      <label class="f" for="mm-name"><span class="lab">Name</span><span class="h">It names the bot's folder and its RLBot bot.</span><input id="mm-name" class="field" type="text" autocomplete="off" spellcheck="false" placeholder="my-multi-bot" value="${esc(m.name)}"></label></section>
    ${m.guide ? mmGuideHTML() : ""}
    ${(m.errors || []).length ? `<div class="banner bad" role="alert"><pre>${esc(m.errors.join("\n"))}</pre></div>` : ""}
    ${problems.length ? `<div class="banner" role="status">${problems.map(p => `<div>${esc(p)}</div>`).join("")}</div>` : ""}
    <div class="row"><button type="button" class="btn primary" data-act="mm-make" ${problems.length ? "disabled" : ""}>Make the bot</button><button type="button" class="btn ghost" data-act="mm-cancel">Cancel</button></div>
  </div>`; });
}
async function makeMultiModel() {
  const m = S.mm;
  if (!m || m.making || mmProblems().length) return;
  m.making = true;
  try {
    const form = {
      models: m.picked, base: m.base || m.picked[0], selector_size: m.size, attention: m.attention,
      min_share: m.picked.map(name => Number(m.shares[name]) || 0),
      min_hold_seconds: Number(m.values.min_hold_seconds), switch_cost: Number(m.values.switch_cost), model_pull: Number(m.values.model_pull),
      models_start_after: Math.max(0, Math.round(Number(String(m.values.models_start_after).replace(/[,_ ]/g, "")) || 0)),
      train_models: !!m.switches.train_models, goal_heads: !!m.switches.goal_heads, standardise_values: !!m.switches.standardise_values,
    };
    const out = await call("create_multi_model", m.name, form);
    if (out.error || out.errors) { m.errors = out.errors || [out.error]; return renderMultiModel(); }
    S.mm = null; await loadHome();
    toast("Made " + out.made + ". Press Start on its page to train it.");
    (out.advice || []).forEach(n => toast(n));
    return openBot(out.made, "overview");
  } finally { m.making = false; }
}
VIEWS.multimodel = () => { if (!S.mm) return openMultiModel(); renderMultiModel(); };
ACTIONS["mm-make"] = () => makeMultiModel();
ACTIONS["mm-cancel"] = () => { S.mm = null; S.view = "home"; return loadHome().then(render); };
ACTIONS["mm-guide"] = () => { S.mm.guide = !S.mm.guide; renderMultiModel(); };
on("change", t => {
  const m = S.mm;
  if (!m || S.view !== "multimodel") return SKIP;
  if (t.dataset.mmpick != null) {
    const name = t.dataset.mmpick;
    m.picked = t.checked ? [...m.picked, name] : m.picked.filter(n => n !== name);
    if (!m.picked.includes(m.base)) { m.base = m.picked[0] || ""; return loadMultiModel(); }
    return renderMultiModel();
  }
  if (t.id === "mm-base") { m.base = t.value; return loadMultiModel(); }
  if (t.id === "mm-size") { m.size = t.value; return renderMultiModel(); }
  if (t.id === "mm-attention") { m.attention = t.checked; return renderMultiModel(); }
  if (t.dataset.mmswitch) { m.switches[t.dataset.mmswitch] = t.checked; return renderMultiModel(); }
  return SKIP;
});
on("input", t => {
  const m = S.mm;
  if (!m || S.view !== "multimodel") return SKIP;
  if (t.dataset.mmshare != null) { m.shares[t.dataset.mmshare] = t.value; return renderMultiModel(); }
  if (t.dataset.mmval) { m.values[t.dataset.mmval] = t.value; return; }
  if (t.id === "mm-name") { m.name = t.value; return renderMultiModel(); }
  return SKIP;
});

function mmTrainingNote(mm) {
  if (!mm.train_models) return "";
  const done = Number(mm.agent_steps) || 0;
  if (mm.models_start_after > done) return ` · its models are held while the selector warms up (${compact(done)} of ${compact(mm.models_start_after)} steps)`;
  return " · its models train too";
}
// On a multi-model bot's Overview: each model's share of the driving now, against its aim, and the hand-overs.
function multiModelHTML(d) {
  const mm = d.multi_model;
  if (!mm) return "";
  const rows = mm.models.map(x => {
    const share = x.share == null ? null : Math.max(0, Math.min(1, x.share)), low = x.target > 0 && share != null && share < x.target;
    return `<div class="mm-live-row"><span class="grow"><b>${esc(x.name)}</b>${x.target > 0 ? ` <span class="muted">aim: at least ${pct(x.target)}</span>` : ""}${low && x.bonus ? ` <span class="tag" title="It earns a bonus each step it drives until it reaches its share.">below its aim: bonus on</span>` : ""}</span>
      <div class="bar" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${share == null ? 0 : Math.round(100 * share)}" aria-label="${esc(x.name)}'s share of the driving"><span style="width:${share == null ? 0 : (100 * share).toFixed(1)}%"></span>${x.target > 0 ? `<i class="mm-aim" style="left:${(100 * x.target).toFixed(1)}%"></i>` : ""}</div>
      <span class="mono">${share == null ? "–" : pct(share)}</span></div>`;
  }).join("");
  const switches = mm.switches_per_minute == null ? "–" : num(mm.switches_per_minute, 1);
  return `<section class="card pad mm-live" aria-label="Multi-model bot">
    <div class="heading"><h3>Its models</h3><span class="muted" style="font-size:13px;">Hand-overs a car-minute: <b class="mono">${switches}</b>${mm.goal_accuracy != null ? ` · goal guesses right: <b class="mono">${pct(mm.goal_accuracy)}</b>` : ""}${mmTrainingNote(mm)}</span></div>
    <div class="muted" style="font-size:13px;">How much of the driving each model did in the newest iteration (the line marks its aim).</div>
    ${rows}</section>`;
}
