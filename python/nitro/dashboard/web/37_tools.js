// A bot's tools: the ruled list (beside the Overview's charts and as its own tab), and the dialog of each:
// what it does, its options, the terminal command for the same thing, and once started how it goes and what
// came of it. The commands run on the Python side (nitro/dashboard/tools.py); their output is in the Console tab.
const TOOL_ICON = (() => {
  const svg = body => `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="square" stroke-linejoin="miter" aria-hidden="true">${body}</svg>`;
  return {
    flag: svg('<path d="M5 21V4"/><path d="M5 4h13l-3 4.5 3 4.5H5"/>'),
    bars: svg('<path d="M5 20v-7M12 20V4M19 20v-11"/>'),
    versus: svg('<path d="m8 7-5 5 5 5M16 7l5 5-5 5"/>'),
    grow: svg('<rect x="8" y="8" width="8" height="8"/><path d="M3 9V3h6M21 9V3h-6M3 15v6h6M21 15v6h-6"/>'),
    rocket: svg('<path d="M5 19c0-6 4-13 14-14-1 10-8 14-14 14z"/><path d="M5 19l5-5"/>'),
    check: svg('<path d="M6 3h9l4 4v14H6z"/><path d="m9 14 2.5 2.5L16 12"/>'),
    copy: svg('<rect x="8" y="8" width="12" height="12"/><path d="M16 8V4H4v12h4"/>'),
    box: svg('<path d="M3 8l9-5 9 5v8l-9 5-9-5z"/><path d="M3 8l9 5 9-5M12 13v8"/>'),
    chart: svg('<path d="M4 4v16h16"/><path d="m7 15 4-5 3 3 5-7"/>'),
    link: svg('<path d="M10 14a4 4 0 0 0 6 0l4-4a4 4 0 0 0-6-6l-1 1"/><path d="M14 10a4 4 0 0 0-6 0l-4 4a4 4 0 0 0 6 6l1-1"/>'),
    folder: svg('<path d="M3 6h7l2 3h9v10H3z"/>'),
    play: svg('<path d="M7 4v16l13-8z"/>'),
    pen: svg('<path d="M4 20h4L19 9l-4-4L4 16z"/>'),
    bin: svg('<path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"/>'),
    out: '<svg class="chev" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="square" aria-hidden="true"><path d="M8 5h11v11"/><path d="M19 5 6 18"/></svg>',
  };
})();

// Every tool: one with `form` opens its dialog (TOOL_FORMS); the others `run` at once, in a dialog of their own
// (`dialog`) or outside the window.
const TOOLS = [
  { id: "eval", icon: "flag", name: "Test against older saves", line: "Is the newest save really better? It plays an earlier one.", form: true, hint: "nitro eval", keywords: ["eval", "match", "versus", "older", "save", "better", "play against"] },
  { id: "skills", icon: "bars", name: "Skill check", line: "Six short drills: touch, landing, boost, shot, save, aerial.", form: true, hint: "nitro skills", keywords: ["skills", "drills", "score", "practice"] },
  { id: "ab", icon: "versus", name: "Compare two settings", line: "Trains fresh bots with each and says if one is clearly better.", form: true, hint: "nitro ab", keywords: ["ab", "compare", "settings", "a/b", "speed", "experiment"] },
  { id: "surgery", icon: "grow", name: "Change the network's size", line: "A wider copy of the bot that plays the same and can learn more.", form: true, hint: "nitro surgery", keywords: ["surgery", "widen", "network", "size", "bigger", "parameters"] },
  { id: "export", icon: "rocket", name: "Make RLBot bot", line: "A folder RLBot can play in a real match on this PC.", form: true, hint: "nitro export", keywords: ["export", "rlbot", "match", "bot folder"] },
  { id: "settings", icon: "check", name: "Check a settings file", line: "Finds mistakes in a settings file and says how big its networks are.", form: true, hint: "nitro check-settings", keywords: ["check", "settings", "mistakes", "validate", "toml"] },
  { id: "copy", icon: "copy", name: "New bot from this one", line: "The same settings, from scratch or from what this one learned.", dialog: true, run: () => copyModal(), hint: "New bot", keywords: ["copy", "clone", "new bot", "transfer"] },
  { id: "pack", icon: "box", name: "Pack for another computer", line: "One file with the bot, to carry on training elsewhere.", form: true, hint: "nitro pack", keywords: ["pack", "move", "cloud", "another computer", "zip"] },
  { id: "report", icon: "chart", name: "Progress report", line: "A page with every chart of its training, in your browser.", run: () => ACTIONS.report(), hint: "Opens in the browser", keywords: ["report", "charts", "progress"] },
  { id: "wandb", icon: "link", name: "W&B charts", line: "Its page on Weights & Biases, in your browser.", show: b => !!b.wandb, run: () => ACTIONS.wandb(), hint: "Opens in the browser", keywords: ["wandb", "weights", "biases"] },
  { id: "folder", icon: "folder", name: "Open its folder", line: "Its saves, settings and logs on this PC.", run: () => ACTIONS.folder(), hint: "Opens the folder", keywords: ["folder", "files", "saves", "logs"] },
  { id: "rlbot", icon: "play", name: "Open RLBot", line: "Starts RLBot, to play the bot in a real match.", run: () => openRLBot(), hint: "Starts RLBot", keywords: ["rlbot", "match", "play"] },
  { id: "rename", icon: "pen", name: "Rename this bot", line: "Gives the bot another name; everything it learned stays.", dialog: true, run: () => renameModal(), hint: "Rename", keywords: ["rename", "name", "call"] },
  { id: "delete", icon: "bin", name: "Delete this bot", line: "Removes the bot and all its saves from this PC. Other bots are not touched.", dialog: true, run: () => deleteModal(), hint: "Delete", keywords: ["delete", "remove", "trash", "bin"] },
];
function toolKey(bot, id) { return bot + "/" + id; }
// By bot and tool: how its newest run goes (tool_state), whether its end was looked at, and whether it is
// being followed.
S.toolStates = {}; S.toolSeen = {}; S.toolFollowed = {};

// ---------- the list ----------
function toolsHTML() {
  const b = S.data, jobs = (S.tools && S.tools.jobs) || {};
  const row = x => {
    const job = jobs[x.id], seen = S.toolSeen[toolKey(b.name, x.id)];
    const tag = !job ? "" : job.state === "running" ? '<span class="tag strong">Running</span>' : seen ? "" : job.state === "done" ? '<span class="tag hot">Result</span>' : job.state === "failed" ? '<span class="tag bad">Did not work</span>' : "";
    return `<button type="button" class="tool" data-tool="${x.id}">${TOOL_ICON[x.icon]}<span class="grow"><b>${esc(x.name)}</b><small>${esc(x.line)}</small></span>${tag}${x.form || x.dialog ? ICON.chev : TOOL_ICON.out}</button>`;
  };
  const heavy = S.tools && S.tools.heavy;
  const waits = heavy && heavy.bot !== b.name ? `<div class="banner" data-heavy>${esc(heavy.bot)} is busy with ${esc(heavy.what)}. The tools that need the graphics card wait until it is done.</div>` : "";
  return `<div class="heading"><h2 class="display">Tools</h2></div>${waits}<div class="tools ruled">${TOOLS.filter(x => !x.show || x.show(b)).map(row).join("")}</div>`;
}
TAB_VIEWS.tools = box => { box.innerHTML = `<div class="tools-page" id="tools-page">${toolsHTML()}</div>`; };
TAB_REFRESH.tools = () => { put($("#tools-page"), toolsHTML()); };
function redrawTools() {
  if (S.view !== "bot" || !S.data) return;
  if ($("#ov-tools")) put($("#ov-tools"), toolsHTML());
  if ($("#tools-page")) put($("#tools-page"), toolsHTML());
}
function openTool(id) {
  const tool = TOOLS.find(x => x.id === id);
  if (!tool || S.view !== "bot" || !S.data) return;
  return tool.form ? toolModal(id) : tool.run();
}
on("click", t => {
  if (!t.dataset.tool) return SKIP;
  openTool(t.dataset.tool);
});
// The search box's commands (another part's COMMANDS list, when the page has one): asked for once the whole
// script has run, wherever that list is declared.
setTimeout(() => {
  let list = null;
  try { list = typeof COMMANDS !== "undefined" && Array.isArray(COMMANDS) ? COMMANDS : null; } catch (err) { list = null; }
  if (!list) return;
  // A tool the search already lists under its own name is not listed twice.
  TOOLS.filter(x => !list.some(c => c.id === x.id)).forEach(x => list.push({
    id: "tool-" + x.id, label: x.name, keywords: (x.keywords || []).join(" "),
    command: /^nitro /.test(x.hint || "") ? x.hint : "", about: x.hint,
    when: () => S.view === "bot" && !!S.data && (!x.show || x.show(S.data)),
    run: () => openTool(x.id),
  }));
}, 0);

// ---------- the pieces of a tool's form ----------
// An option's field: its change sets S.tool.options[key] (setOpt) and asks for the command anew.
function optSeg(key, pairs) {
  const now = S.tool.options[key];
  return `<div class="seg" data-seg="opt-${key}">${pairs.map(([v, label]) => `<button type="button" data-v="${esc(v)}" aria-pressed="${String(v) === String(now)}">${esc(label)}</button>`).join("")}</div>`;
}
function optSelect(key, pairs, label) {
  const now = S.tool.options[key];
  const option = ([v, text]) => `<option value="${esc(v)}" ${String(v) === String(now) ? "selected" : ""}>${esc(text)}</option>`;
  const body = pairs.map(p => Array.isArray(p) ? option(p) : `<optgroup label="${esc(p.group)}">${p.options.map(option).join("")}</optgroup>`).join("");
  return `<select class="field" id="opt-${key}" data-opt="${key}" aria-label="${esc(label)}">${body}</select>`;
}
function optCheck(key, text) { return `<label class="check"><input type="checkbox" id="opt-${key}" data-opt="${key}" ${S.tool.options[key] ? "checked" : ""}><span>${text}</span></label>`; }
function optText(key, placeholder, label) { return `<input class="field" type="text" autocomplete="off" id="opt-${key}" data-opt="${key}" value="${esc(S.tool.options[key] || "")}" placeholder="${esc(placeholder)}" aria-label="${esc(label)}">`; }
function optRow(label, field, help) { return `<div class="opt"><div class="k">${esc(label)}</div><div class="v">${field}${help ? `<div class="h">${help}</div>` : ""}</div></div>`; }
function savesOf(bot) {
  if (bot === S.tool.bot) return (S.tools && S.tools.saves) || [];
  const other = ((S.toolChoices || {}).bots || []).find(o => o.value === bot);
  return other ? other.saves : [];
}
// The settings files to choose from, grouped as the Python side names them; those with a mistake only
// `withBroken` (the check takes them, a comparison does not).
function fileGroups(withBroken) {
  const titles = { bot: "Bots", preset: "Presets", "my-settings": "My settings", library: "Library" };
  const files = ((S.toolChoices || {}).files || []).filter(f => withBroken || !f.broken);
  return Object.keys(titles).map(g => ({ group: titles[g], options: files.filter(f => f.group === g).map(f => [f.id, `${f.title} (${f.about})`]) })).filter(g => g.options.length);
}

// Each tool's dialog: its title, the options it starts with, what it does in plain words (`about`), its
// options' fields (`fields`), what its start button says, and how its result reads (`result`).
const TOOL_FORMS = {
  eval: {
    title: b => `Test ${b.name} against older saves`,
    go: "Play the match",
    needs: "choices",
    defaults: () => ({ opponent: "", save: 0, points: 512, as_bot: false }),
    about: b => `<p>Plays <b>${esc(b.name)}</b>'s newest save against an earlier save, or against another bot, many short games at once. Each side plays both colours from the same starts, so a lucky kickoff side cannot decide it.</p><p>It counts the goals and says who took more, by how much, and whether that could be luck. Nothing is changed in any bot.</p>`,
    fields: b => {
      const o = S.tool.options, other = o.opponent && o.opponent !== b.name ? o.opponent : "";
      const all = savesOf(other || b.name), saves = other ? all : all.slice(1);
      const first = other ? [0, `Its newest save (${whole(all[0])} steps)`] : [0, saves.length ? `Its oldest save (${whole(saves[saves.length - 1])} steps)` : "It has no earlier save yet"];
      const rest = (other ? all.slice(1) : saves.slice(0, -1)).map(s => [s, `Save at ${whole(s)} steps`]);
      return optRow("Against", optSelect("opponent", [["", `An earlier save of ${b.name}`], ...(((S.toolChoices || {}).bots || []).map(x => [x.value, `Another bot: ${x.label}`]))], "Against"))
        + optRow("Which save", optSelect("save", [first, ...rest], "Which save"))
        + optRow("Points", optSeg("points", [[128, "128"], [512, "512"], [2048, "2,048"]]), `Each way round, so ${whole(2 * Number(o.points))} in all. More points make the answer surer and take longer.`)
        + optRow("How they play", optCheck("as_bot", "As the RLBot bot does: always the likeliest action, one way round only"), "Left off, both pick as they do in training, with some randomness.");
    },
    result: r => {
      if (!r) return noResultHTML();
      const strip = r.share == null ? "" : `<div class="share" role="img" aria-label="Its share of the goals is ${r.share}%, between ${r.low}% and ${r.high}% allowing for luck"><span class="range" style="left:${r.low}%;width:${Math.max(1, r.high - r.low)}%;"></span><span class="half"></span><span class="at" style="left:${r.share}%;"></span></div><div class="share-ends"><span>0%</span><span>half the goals</span><span>100%</span></div>`;
      const figs = `<div class="figs"><div><dt>Scored</dt><dd>${whole(r.scored)}</dd></div><div><dt>Let in</dt><dd>${whole(r.let_in)}</dd></div><div><dt>No goal</dt><dd>${whole(r.no_goal)}</dd></div>${r.share == null ? "" : `<div><dt>Share of the goals</dt><dd>${r.share}%</dd></div>`}</div>`;
      return `<h3 class="verdict" data-winner="${esc(r.winner)}">${esc(r.headline)}</h3>${figs}${strip}${r.lines.map(l => `<p>${esc(l)}</p>`).join("")}<p class="muted">The command's own reading: ${esc(r.reading)}</p>`;
    },
  },
  skills: {
    title: b => `Skill check of ${b.name}`,
    go: "Run the drills",
    defaults: () => ({ save: 0, situations: 512, as_bot: false }),
    about: b => `<p>Puts a save of <b>${esc(b.name)}</b> through six short drills against an opponent that stands still: touch a ball at rest, land on its wheels, collect boost, score in an empty net, save a shot, touch a ball in the air.</p><p>Each score is set against a bar: what a fresh bot trained on that one drill alone reaches. It says whether a skill is there, not how good the bot is. The scores are also added to <span class="mono">skills.csv</span> in the bot's folder.</p>`,
    fields: b => {
      const saves = savesOf(b.name);
      return optRow("Which save", optSelect("save", [[0, saves.length ? `Its newest save (${whole(saves[0])} steps)` : "It has no save yet"], ...saves.slice(1).map(s => [s, `Save at ${whole(s)} steps`])], "Which save"))
        + optRow("Situations", optSeg("situations", [[128, "128"], [512, "512"], [2048, "2,048"]]), "For each drill. More make the scores steadier and take longer.")
        + optRow("How it plays", optCheck("as_bot", "As the RLBot bot does: always its likeliest action"), "Left off, it picks as it does in training, with some randomness.");
    },
    result: r => {
      if (!r) return noResultHTML();
      const rows = r.rows.map(x => `<tr data-skill="${esc(x.task)}"><td><b>${esc(x.skill)}</b><br><span class="muted">${esc(x.measure)}</span></td><td class="num">${esc(x.value)}${esc(x.unit)}</td><td class="num muted">${x.lower_is_better ? "under" : "over"} ${esc(x.bar)}${esc(x.unit)}</td><td>${x.has ? '<span class="tag good">Has it</span>' : '<span class="tag">Not yet</span>'}</td></tr>`).join("");
      return `<h3 class="verdict">${esc(r.headline)}</h3>${r.who ? `<p class="muted">${esc(r.who)}${r.situations ? `, ${whole(r.situations)} situations a drill` : ""}.</p>` : ""}
        <table class="skills"><thead><tr><th>Drill</th><th>Score</th><th>Bar</th><th></th></tr></thead><tbody>${rows}</tbody></table>${r.note ? `<p class="muted">${esc(r.note)}</p>` : ""}`;
    },
  },
  ab: {
    title: () => "Compare two settings",
    go: "Start the comparison",
    needs: "choices",
    defaults: b => ({ a: "bot:" + b.name, b: "", on: "ball_touch", seeds: 10, repeats: 6, with_rewards: false, name: "" }),
    about: b => {
      const o = S.tool.options, speed = o.on === "speed";
      const long = speed
        ? `<b>How long:</b> ${2 * Number(o.repeats)} short runs, ${whole(o.repeats)} for each file. Minutes on a graphics card. Close other programs first and train nothing meanwhile: anything else running changes the speed.`
        : `<b>How long:</b> ${2 * Number(o.seeds)} trainings of a fresh bot, ${whole(o.seeds)} for each file, each of 10 to 20 million steps with a test before and after. Expect hours on a graphics card and far longer without one. After the first training the Console tab says how many minutes are left.`;
      return `<p>Trains fresh bots with each of two settings files, from the same seeds, and says whether one is clearly better or whether the difference is within the luck between runs. ${speed ? "Here it measures how fast each trains." : "Each bot learns one small practice task and is scored on that task, never by its reward. Only how a bot learns is compared (its networks, learning settings and observation); the files' rewards count only with the box ticked."} No bot of yours is changed.</p>
        <p>${long}</p>
        <p>While it runs ${esc(b.name)} cannot train. Stopping it is safe: start it again under the same name and it carries on where it was.</p>`;
    },
    fields: () => {
      const o = S.tool.options, c = S.toolChoices;
      if (!c) return '<div class="muted">Loading the settings files…</div>';
      const speed = o.on === "speed";
      const tasks = [{ group: "A practice task", options: c.tasks.map(t => [t.name, `${t.name.replace(/_/g, " ")}: ${t.about}`]) }, { group: "Or", options: [["speed", "Training speed"]] }];
      const done = (c.comparisons || []).slice(0, 5).map(x => `<li><span class="mono">${esc(x.name)}</span> · ${esc(x.headline)} <span class="muted">${ago(x.at)}</span></li>`).join("");
      return optRow("Settings A", optSelect("a", fileGroups(), "Settings A"))
        + optRow("Settings B", optSelect("b", [["", "Choose a file…"], ...fileGroups()], "Settings B"))
        + optRow("Compare on", optSelect("on", tasks, "Compare on"))
        + (speed ? optRow("Runs", optSeg("repeats", [[4, "4"], [6, "6"], [10, "10"]]), "For each file.")
                 : optRow("Seeds", optSeg("seeds", [[6, "6"], [10, "10"], [20, "20"]]), "Bots trained for each file. 10 is the number to decide by; fewer is quicker and less sure.")
                   + optRow("Rewards", optCheck("with_rewards", "Use each file's own rewards instead of the task's, to compare two reward designs")))
        + optRow("Name", optText("name", "A-vs-B-task, made from the two files", "Name of the comparison"), "Its results are kept in the <span class=\"mono\">ab</span> folder under this name.")
        + (done ? `<div class="opt"><div class="k">Finished before</div><div class="v"><ul class="plain">${done}</ul></div></div>` : "");
    },
    result: (r, st) => {
      if (!r) return noResultHTML();
      return `<h3 class="verdict" data-better="${esc(r.better)}">${esc(r.headline)}</h3>${r.meaning ? `<p>${esc(r.verdict)}: ${esc(r.meaning)}</p>` : ""}<p class="muted">A is ${esc(st.a)}, B is ${esc(st.b)}. Every run and the full result are kept in <span class="mono">${esc(st.folder)}</span>.</p><pre class="mono toolout tall">${esc(r.text)}</pre>`;
    },
  },
  surgery: {
    title: b => `Change the size of ${b.name}'s networks`,
    go: "Make the wider bot",
    defaults: b => ({ widen: 2, new_name: "" }),
    about: b => {
      const net = (S.tools && S.tools.network) || {}, k = Number(S.tool.options.widen) || 2;
      const made = (S.tool.options.new_name || "").trim() || `${b.name}-x${k}`;
      const part = (label, sizes) => sizes && sizes.length ? `<div><dt>${label}</dt><dd><span class="mono">${sizes.join(", ")}</span> <span class="muted">becomes</span> <span class="mono">${sizes.map(n => n * k).join(", ")}</span></dd></div>` : "";
      return `<p>Makes a new bot, <b>${esc(made)}</b>, from <b>${esc(b.name)}</b>'s newest save, with every hidden layer ${k} times as wide. The new bot plays exactly as ${esc(b.name)} does now (checked before anything is written) and has more room to learn.</p>
        <div class="figs sizes">${part("Shared layers", net.shared)}${part("Policy layers", net.policy)}${part("Value layers", net.value)}</div>
        <p><b>${esc(b.name)} itself is not changed:</b> it keeps its settings and every save, and you can carry on with either bot. The new bot has about ${k * k} times the weights in its hidden layers, so each step of training takes longer. Its learning's running averages start afresh; everything else it learned carries over.</p>
        <p class="muted">Only widening is offered, and only for plain networks: deeper networks or other inputs cannot be made to play the same.</p>`;
    },
    fields: b => optRow("How much wider", optSeg("widen", [[2, "2 times"], [3, "3 times"], [4, "4 times"]]))
      + optRow("Name of the new bot", optText("new_name", `${b.name}-x${Number(S.tool.options.widen) || 2}`, "Name of the new bot")),
    result: r => {
      if (!r) return noResultHTML();
      return `<h3 class="verdict">${esc(r.made)} is made.</h3><p>It holds everything ${esc(S.tool.bot)} had learned and plays the same; ${esc(S.tool.bot)} is as it was. Open ${esc(r.made)} and carry on training there.</p><pre class="mono toolout">${esc(r.lines.join("\n"))}</pre><div class="row"><button type="button" class="btn small strong" data-act="tool-made" data-made="${esc(r.made)}">Open ${esc(r.made)}</button></div>`;
    },
  },
  settings: {
    title: () => "Check a settings file",
    go: "Check it",
    needs: "choices",
    defaults: b => ({ file: "bot:" + b.name }),
    about: b => `<p>Reads a settings file the way training would and lists every mistake with its line, the size its networks would have, and notes on settings worth a second look. Nothing is changed.</p><p class="muted">${esc(b.name)}'s own settings are checked all the time: a mistake in them shows at the top of this page. This also checks any other file, before a bot is made from it or a comparison run with it.</p>`,
    fields: () => S.toolChoices ? optRow("Settings file", optSelect("file", fileGroups(true), "Settings file")) : '<div class="muted">Loading the settings files…</div>',
    result: r => {
      if (!r) return noResultHTML();
      return `<h3 class="verdict" data-fine="${r.fine}">${esc(r.headline)}</h3>${r.problems.length ? `<pre class="mono toolout">${esc(r.problems.join("\n"))}</pre>` : ""}${r.networks ? `<p>${esc(r.networks)}</p>` : ""}${r.notes.length ? `<ul class="plain">${r.notes.map(n => `<li>${esc(n)}</li>`).join("")}</ul>` : ""}`;
    },
  },
  export: {
    title: b => `Make ${b.name}'s RLBot bot`,
    go: "Make it",
    defaults: b => ({ bot_name: (S.tools && S.tools.rlbot && S.tools.rlbot.name) || b.name }),
    about: b => {
      const made = S.tools && S.tools.rlbot;
      return `<p>Turns <b>${esc(b.name)}</b>'s newest save${b.newest_save != null ? ` (${whole(b.newest_save)} steps)` : ""} into a folder RLBot can play in a real match: the trained policy, the bot program that plays it and a short README. It takes a few seconds.</p>
        ${made ? `<p>Its RLBot bot is already made${made.steps != null ? `, from the save at ${whole(made.steps)} steps` : ""}. Making it again puts the newest save in; the car's look (<span class="mono">loadout.toml</span>) is kept.</p><div class="row"><span class="mono path">${esc(made.folder)}</span><button type="button" class="btn small" data-act="tool-folder">Open folder</button><button type="button" class="btn small" data-act="tool-copy-path" data-path="${esc(made.folder)}">Copy folder path</button></div>` : ""}`;
    },
    fields: b => optRow("Name RLBot shows", optText("bot_name", b.name, "Name RLBot shows"), "Up to 40 characters. Left as the bot's own name unless you change it."),
    result: (r, st) => {
      if (!r || !r.written) return noResultHTML();
      return `<h3 class="verdict">${r.program ? "The RLBot bot is ready." : "The folder is written, but the bot program is missing."}</h3>
        ${r.program ? "<p>Add this folder to RLBot as a bot folder, then start a match with the bot. The README in it says how. RLBot's folder picker may not show hidden folders such as AppData: copy the folder's path and paste it there.</p>" : `<p>${esc(r.missing)}</p><p>The README in the folder says how to build the program.</p>`}
        <div class="row"><span class="mono path">${esc(r.folder)}</span></div>
        <div class="row"><button type="button" class="btn small strong" data-act="tool-folder">Open folder</button><button type="button" class="btn small" data-act="tool-copy-path" data-path="${esc(r.folder)}">Copy folder path</button><button type="button" class="btn small" data-go="rlbot">Open RLBot</button></div>`;
    },
  },
  pack: {
    title: b => `Pack ${b.name} for another computer`,
    go: "Pack it",
    defaults: () => ({ all_saves: false, meshes: false }),
    about: b => `<p>Puts <b>${esc(b.name)}</b> into one file in the <span class="mono">packs</span> folder: its settings, its newest save and what it needs to carry on training on another computer. There, in NitroLearn's Garage, <b>Add a packed bot…</b> adds it; to bring it back, pack it there and add it here the same way.</p><p class="muted">The logs of what ran on this PC stay here: they name its folders.</p>`,
    fields: () => optRow("Saves", optCheck("all_saves", "Every save, not only the newest (a bigger file)"))
      + optRow("Arena", optCheck("meshes", "Include the arena files (collision meshes)"), "Left out, the other computer uses its own, which NitroLearn's installer downloads. They are Rocket League's game files: include them only for your own computers, and never share a pack that has them."),
    result: r => {
      if (!r) return noResultHTML();
      return `<h3 class="verdict">Packed.</h3><div class="row"><span class="mono path">${esc(r.file)}</span></div>${(r.notes || []).map(n => `<p class="muted">${esc(n)}</p>`).join("")}<p>Copy this file to the other computer. There, in NitroLearn's Garage, press <b>Add a packed bot…</b> and pick it.</p><div class="row"><button type="button" class="btn small strong" data-act="tool-folder">Open the packs folder</button><button type="button" class="btn small" data-act="tool-copy-path" data-path="${esc(r.file)}">Copy file path</button></div>`;
    },
  },
};
function noResultHTML() { return '<p>It ended without a result this page can read. The Console tab has everything it printed.</p>'; }
// Which options change what the form offers (it is drawn anew), not only the command.
const TOOL_REDRAWS = ["opponent", "on", "widen", "points", "seeds", "repeats", "new_name"];

// ---------- the dialog ----------
// S.tool: the open tool's id, bot, options, and what Python said of them (command, blocked, error); `view` is
// "form" or "run" (how its newest run goes, or went).
function toolModal(id) {
  const b = S.data, form = TOOL_FORMS[id], key = toolKey(b.name, id);
  S.tool = { id, bot: b.name, options: form.defaults(b), view: "form", command: "", blocked: "", error: "", seq: 0 };
  S.toolChoices = null;
  modal(`<h2 class="display">${esc(form.title(b))}</h2><div id="tool-body"></div><div class="foot" id="tool-foot"></div>`);
  $(".modal").classList.add("wide");
  drawTool();
  (async () => {
    const [state, choices] = await Promise.all([call("tool_state", b.name, id), form.needs ? call("tool_choices", b.name) : null]);
    if (!toolOpen(b.name, id)) return;
    if (choices && !choices.error) {
      S.toolChoices = choices;
      if (id === "ab" && !S.tool.options.b) S.tool.options.b = (choices.files.find(f => !f.broken && f.id !== S.tool.options.a) || {}).id || "";
    }
    if (!state.error && state.state !== "none") {
      S.toolStates[key] = state; S.tool.view = "run";
      if (state.state === "running") followTool(b.name, id);
    }
    drawTool(); askCommand();
  })();
}
function toolOpen(bot, id) { return !!S.tool && S.tool.bot === bot && S.tool.id === id && !!$("#tool-body"); }
function drawTool() {
  const t = S.tool;
  if (!t || !$("#tool-body") || !S.data) return;
  const b = S.data, form = TOOL_FORMS[t.id], st = S.toolStates[toolKey(t.bot, t.id)];
  let body, foot;
  if (t.view === "run" && st) {
    body = toolRunHTML(form, st);
    const ended = st.state !== "running";
    if (ended && !S.toolSeen[toolKey(t.bot, t.id)]) {  // looked at: its "Result" tag and its note go
      S.toolSeen[toolKey(t.bot, t.id)] = true;
      if (S.toolNote && S.toolNote.bot === t.bot && S.toolNote.kind === t.id) { S.toolNote = null; updateHead(); }
      redrawTools();
    }
    foot = `<button type="button" class="btn" data-close="1">Close</button>${ended ? `<button type="button" class="btn" data-act="tool-again">${st.state === "stopped" && t.id === "ab" ? "Carry it on or change it" : "Run it again"}</button>` : ["export", "pack", "surgery", "settings"].includes(t.id) ? "" : '<button type="button" class="btn danger" data-act="tool-stop">Stop it</button>'}`;
  } else {
    const cannot = t.error || t.blocked;
    body = `<div class="tool-about">${form.about(b)}</div><div class="opts">${form.fields(b)}</div>
      ${cannot ? `<div class="banner ${t.error ? "bad" : "warn"}" id="tool-cannot" role="status">${esc(cannot)}${t.error ? "" : "."}</div>` : ""}
      ${t.command ? `<div class="cmdline"><span class="cap">The same in a terminal</span><pre class="mono" id="tool-command">${esc(t.command)}</pre></div>` : ""}`;
    foot = `<button type="button" class="btn" data-close="1">Cancel</button><button type="button" class="btn primary" data-act="tool-go" ${cannot || !t.command ? "disabled" : ""}>${esc(form.go)}</button>`;
  }
  keepFocus(() => put($("#tool-body"), body));
  put($("#tool-foot"), foot);
}
function toolRunHTML(form, st) {
  const cmd = `<div class="cmdline"><span class="cap">The command</span><pre class="mono" id="tool-command">${esc(st.command || "")}</pre></div>`;
  const took = duration(st.seconds) === "0 min" ? Math.max(1, Math.round(st.seconds)) + " s" : duration(st.seconds);
  if (st.state === "running") {
    const p = st.progress;
    return `<div class="row"><span class="tag strong">Running</span><b>${took}</b>${p ? `<span>${p.done} of ${p.of} runs done, about ${duration(p.minutes_left * 60)} left</span>` : ""}</div>
      ${p ? `<div class="bar" role="progressbar" aria-valuemin="0" aria-valuemax="${p.of}" aria-valuenow="${p.done}"><span style="width:${(100 * p.done / Math.max(1, p.of)).toFixed(1)}%"></span></div>` : ""}
      <pre class="mono toolout" id="tool-tail">${esc(st.tail || "Starting…")}</pre>
      <p class="muted">You can close this: it keeps running and the result waits here. Everything it prints is in the Console tab.</p>${cmd}`;
  }
  if (st.state === "failed" && !(st.result && ((st.result.problems || []).length || st.result.written))) {
    return `<div class="banner bad" role="alert"><div class="grow"><b>It did not work.</b> Its last lines:<pre>${esc(st.why || "")}</pre></div></div>${cmd}`;
  }
  if (st.state === "stopped") {
    return `<div class="row"><span class="tag">Stopped</span><span>after ${took}</span></div><p>${S.tool.id === "ab" ? "Start it again under the same name and it carries on where it was." : "Nothing was changed."}</p>${cmd}`;
  }
  return `<div class="tool-result" id="tool-result">${form.result(st.result, st)}</div><p class="muted">Took ${took}.</p>${cmd}`;
}
// The command as Python would run it with the options now, and why it cannot start (asked after each change).
async function askCommand() {
  const t = S.tool;
  if (!t || t.view !== "form") return;
  const seq = ++t.seq;
  const out = await call("tool_command", t.bot, t.id, t.options);
  if (S.tool !== t || t.seq !== seq) return;
  t.error = out.error || ""; t.blocked = out.blocked || "";
  if (!out.error) t.command = out.command || "";
  drawTool();
}
function setOpt(key, value) {
  const t = S.tool;
  if (!t) return;
  t.options[key] = value;
  if (key === "opponent") t.options.save = 0;
  if (TOOL_REDRAWS.includes(key)) drawTool();
  askCommand();
}
on("change", el => {
  if (!el.dataset.opt || !S.tool) return SKIP;
  setOpt(el.dataset.opt, el.type === "checkbox" ? el.checked : el.value);
});
on("input", el => {
  if (el.dataset && el.dataset.opt && S.tool && el.type === "text") { S.tool.options[el.dataset.opt] = el.value; if (TOOL_REDRAWS.includes(el.dataset.opt)) drawTool(); askCommand(); }
});
["points", "situations", "widen", "seeds", "repeats"].forEach(key => { SEGS["opt-" + key] = value => setOpt(key, Number(value)); });

// Follows a tool's run until it ends: the open dialog shows it; with the dialog closed, a note on the bot's
// page and a toast say it is done.
async function followTool(bot, id) {
  const key = toolKey(bot, id);
  if (S.toolFollowed[key]) return;
  S.toolFollowed[key] = true;
  try {
    for (;;) {
      const st = await call("tool_state", bot, id);
      if (st.error || st.state === "none") return;
      S.toolStates[key] = st;
      if (toolOpen(bot, id)) drawTool();
      if (st.state !== "running") return toolEnded(bot, id, st);
      await new Promise(r => setTimeout(r, 2000));
    }
  } finally { delete S.toolFollowed[key]; }
}
// The open bot's tools that run (started before the page was opened, or with their dialog closed) are followed
// too, so their end is told.
function followRunning() {
  const jobs = (S.tools && S.tools.jobs) || {};
  Object.keys(jobs).forEach(id => { if (jobs[id].state === "running" && TOOL_FORMS[id]) followTool(S.bot, id); });
}
async function toolEnded(bot, id, st) {
  const name = TOOLS.find(x => x.id === id).name, bad = st.state === "failed";
  if (!toolOpen(bot, id) && st.state !== "stopped") {
    S.toolSeen[toolKey(bot, id)] = false;
    S.toolNote = { bot, kind: id, bad, text: bad ? `${name} did not work.` : `${name} is done${st.result && st.result.headline ? ": " + st.result.headline : "."}` };
    toast(bad ? `${name} of ${bot} did not work: the reason is on its page.` : `${name} of ${bot} is done.`, bad);
  }
  await loadHome();
  if (S.view === "bot" && S.bot === bot) { await refreshBot(); updateHead(); redrawTools(); }
}
ACTIONS["tool-go"] = async () => {
  const t = S.tool, key = toolKey(t.bot, t.id);
  const out = await call("tool_start", t.bot, t.id, t.options);
  if (S.tool !== t) return;
  if (out.error) { t.error = out.error; return drawTool(); }
  S.toolSeen[key] = true;
  S.toolStates[key] = { state: "running", command: out.command, seconds: 0, tail: "" };
  t.view = "run";
  drawTool();
  followTool(t.bot, t.id);
  return refreshBot();
};
ACTIONS["tool-again"] = async () => { S.tool.view = "form"; S.tool.error = ""; drawTool(); return askCommand(); };
ACTIONS["tool-stop"] = async () => { const out = await call("tool_stop", S.tool.bot, S.tool.id); if (out.error) toast(out.error, true); return; };
ACTIONS["tool-copy-path"] = async el => toast(await copyText(el.dataset.path) ? "Copied: " + el.dataset.path : "Copying is not allowed here: select the path and press Ctrl+C.", false);
ACTIONS["tool-made"] = async el => { closeModal(); await loadHome(); return openBot(el.dataset.made); };
ACTIONS["tool-folder"] = async () => {
  const out = await call("open_folder", S.tool.bot, S.tool.id === "pack" ? "packs" : "export");
  if (out.error) toast(out.error, true);
};
