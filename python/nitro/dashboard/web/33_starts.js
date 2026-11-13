// The Starts tab (state setters): which start each training game picks, and how often.
function startKind(name) { return ((S.st && S.st.catalog) || []).find(k => k.name === name) || { name, title: name, about: "", params: [] }; }
async function loadStarts() {
  const bot = S.bot;
  const out = await call("starts", bot);
  if (S.bot !== bot) return;
  S.st = out.error ? { bot, error: out.error } : Object.assign(out, { bot });
  if (S.view === "bot" && S.tab === "starts") renderStarts($("#tab"));
}
function renderStarts(box) {
  if (!S.st || S.st.bot !== S.bot) { S.st = { bot: S.bot, loading: true }; loadStarts(); }
  if (S.st.loading) { box.innerHTML = '<div class="card empty">Loading the starts…</div>'; return; }
  if (S.st.error) { box.innerHTML = `<div class="banner bad"><pre>${esc(S.st.error)}</pre></div>`; return; }
  box.innerHTML = `<div class="rw-layout">
    <div class="rw-main">
      <div class="settings-tools">
        <select id="st-add" class="field" style="width:auto;" aria-label="Add a start"><option value="">+ Add a start…</option>${(S.st.catalog || []).map(k => `<option value="${esc(k.name)}">${esc(k.title)}</option>`).join("")}</select>
        <button type="button" class="btn small ghost" data-act="st-guide">How starts work</button>
      </div>
      <div class="muted" style="font-size:13px;max-width:96ch;">How each training game begins. Every game picks one start at random, in proportion to the weights: weights 70 and 30 make 70% and 30% of games. ${S.st.in_file ? "" : "This bot lists none, so every game starts with a kickoff; a change writes them into its settings. "}Changes wait until you press Apply.</div>
      <div id="st-general"></div>
      <div id="st-list" class="rw-list"></div>
    </div>
    <aside class="rw-aside" aria-label="Share of games"><div id="st-mix"></div></aside>
  </div>`;
  renderStartList();
}
function renderStartList() {
  const list = $("#st-list");
  if (!list || !S.st || !S.st.starts) return;
  keepFocus(() => put(list, S.st.starts.map(startHTML).join("") || '<div class="empty">No starts: add one, or every game has nothing to begin from.</div>'));
  put($("#st-general"), (S.st.general || []).length ? `<div class="banner bad"><pre>${esc(S.st.general.join("\n"))}</pre></div>` : "");
  put($("#st-mix"), `<section class="card pad rw-panel"><div class="heading"><h3>Share of games</h3></div><div class="shares">${S.st.starts.map(e => `<div class="share"><span class="n">${esc(startKind(e.view.name).title)}</span><span class="pct num">${pct(e.share)}</span><div class="bar"><span style="width:${Math.round((e.share || 0) * 100)}%"></span></div></div>`).join("")}</div></section>`);
}
function stInput(i, key, value, label, extra) {
  const id = `st-${i}-${key.replace(/\W/g, "-")}`;
  return `<label class="rw-field" for="${id}">${esc(label)}<input id="${id}" class="field mono" type="text" data-st="${i}" data-k="${esc(key)}" value="${esc(value == null ? "" : value)}" ${extra || ""}></label>`;
}
function startHTML(e, i) {
  const v = e.view, k = startKind(v.name);
  const o = e.from != null && S.st.original ? S.st.original[e.from] : null;
  const fresh = !o, changed = fresh || JSON.stringify(canon(v)) !== JSON.stringify(canon(o));
  return `<section class="reward${changed ? " changed" : ""}" data-start="${i}">
    <div class="head">
      <div class="who"><span class="title">${esc(k.title)}</span><span class="kind mono">${esc(v.name)}</span>${fresh ? '<span class="badge new">new</span>' : changed ? '<span class="badge hot">changed</span>' : ""}</div>
      <span class="sharechip" title="Share of games that start this way">${pct(e.share)}</span>
      ${changed && !fresh ? `<button type="button" class="iconbtn" data-stundo="${i}" title="Undo the changes to this start" aria-label="Undo the changes to ${esc(k.title)}">${ICON.reset}</button>` : ""}
      <button type="button" class="iconbtn" data-stdel="${i}" title="Remove this start" aria-label="Remove ${esc(k.title)}">${ICON.trash}</button>
    </div>
    ${k.about ? `<div class="about full">${codeMarks(esc(k.about))}</div>` : ""}
    ${!v.known ? `<div class="rw-problems">'${esc(v.name)}' is not a start NitroLearn knows.</div>` : ""}
    <div class="rw-grid">
      ${stInput(i, "weight", v.weight, "Weight", 'title="How often this start is picked, compared with the others."')}
      ${(k.params || []).map(p => { const val = (v.params || {})[p.path];
        if (p.kind === "bool") { const id = `st-${i}-${p.path}`; return `<label class="rw-opt" for="${id}" title="${esc(p.help)}"><input id="${id}" type="checkbox" data-st="${i}" data-k="params.${esc(p.path)}" ${val ? "checked" : ""}>${esc(p.label)}</label>`; }
        if (p.kind === "choice") { const id = `st-${i}-params-${p.path.replace(/\W/g, "-")}`; return `<label class="rw-field" for="${id}" title="${esc(p.help)}">${esc(p.label)}<select id="${id}" class="field" data-st="${i}" data-k="params.${esc(p.path)}">${p.options.map(o => `<option value="${esc(o)}" ${String(o) === String(val) ? "selected" : ""}>${esc(o)}</option>`).join("")}</select></label>`; }
        return stInput(i, "params." + p.path, Array.isArray(val) ? val.join(", ") : val, p.label, `title="${esc(p.help)}"`); }).join("")}
    </div>
    ${(e.problems || []).length ? `<div class="rw-problems" role="alert">${e.problems.map(p => `<span>${esc(p)}</span>`).join("")}</div>` : ""}
  </section>`;
}
function startEdit(i, key, raw) {
  const e = S.st && S.st.starts[i];
  if (!e) return;
  if (key === "weight") e.view.weight = raw;
  else if (key.startsWith("params.")) e.view.params[key.slice(7)] = raw;
  return stageStarts();
}
function addStart(name) {
  const k = startKind(name), params = {};
  (k.params || []).forEach(p => { params[p.path] = p.default; });
  S.st.starts.push({ from: null, view: { name, known: true, weight: 1, params } });
  return stageStarts().then(() => { const cards = document.querySelectorAll("[data-start]"); if (cards.length) cards[cards.length - 1].scrollIntoView({ block: "center" }); });
}
// As the rewards: edits go to Python one at a time with every start; an overtaken answer is dropped.
let stSeq = 0, stChain = Promise.resolve();
function stageStarts() {
  const bot = S.bot, seq = ++stSeq;
  stChain = stChain.then(async () => {
    if (seq !== stSeq || S.bot !== bot || !S.st || !S.st.starts) return;
    const out = await call("stage_starts", bot, S.st.starts.map(e => ({ from: e.from, view: e.view })));
    if (seq !== stSeq || S.bot !== bot) return;
    if (out.error) return toast(out.error, true);
    Object.assign(S.st, out);
    S.data.start_changes = out.changes || [];
    renderStartList(); renderPending(); updateTabs();
  });
  return stChain;
}

ACTIONS["st-guide"] = () => guideModal("How starts work", Object.entries((S.st && S.st.guide) || {}).map(([k, v]) => [GUIDE_NAMES[k] || startKind(k.slice(6)).title, v]));
on("click", t => {
  if (t.dataset.stdel != null) { S.st.starts.splice(+t.dataset.stdel, 1); return stageStarts(); }
  if (t.dataset.stundo != null) { const st = S.st.starts[+t.dataset.stundo], o = S.st.original[st.from]; if (o) st.view = JSON.parse(JSON.stringify(o)); return stageStarts(); }
  return SKIP;
});
on("change", el => {
  if (el.dataset.st != null) return startEdit(+el.dataset.st, el.dataset.k, el.type === "checkbox" ? el.checked : el.value);
  if (el.id === "st-add") { const name = el.value; el.value = ""; return name ? addStart(name) : undefined; }
  return SKIP;
});
TAB_VIEWS.starts = renderStarts;
