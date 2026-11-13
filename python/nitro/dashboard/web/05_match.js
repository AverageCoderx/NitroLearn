// How well a setting fits what was typed, for both searches over the settings (the Settings tab's own box and
// the top band's). Plain words a newcomer types ("games at once", "how often it saves") reach the settings
// through SETTING_WORDS; then come the plain title, the name in the file and the help text.

// Plain phrases -> the settings they mean, the likeliest first. "group:<id>" adds a group's settings and
// "<section>." a section's, after the ones named.
const SETTING_WORDS = [
  [["learning rate", "lr", "learn faster", "learning speed", "step size"], ["ppo.policy_lr", "ppo.value_lr"]],
  [["replay learning rate", "replays learning rate", "learning rate replays", "learning rate for replays", "copy learning rate", "learning rate copy"], ["pretrain.lr", "replay_mix.lr"]],
  [["how often it saves", "how often to save", "save every", "save often", "saving", "saves", "save", "checkpoint", "checkpoints", "autosave"], ["run.checkpoint_minutes", "run.checkpoint_steps", "run.keep_checkpoints", "run.milestone_steps"]],
  [["keep saves", "old saves", "how many saves", "delete saves", "disk space"], ["run.keep_checkpoints", "run.milestone_steps", "selfplay.pool_disk_mb"]],
  [["games at once", "how many games", "number of games", "more games", "parallel games", "arenas", "arena", "games"], ["env.num_arenas", "env.threads"]],
  [["speed", "faster", "fast", "slow", "slower", "training speed", "train faster", "steps per second", "performance"], ["env.num_arenas", "env.threads", "run.precision", "ppo.steps_per_iteration", "run.device", "group:speed"]],
  [["decisions per second", "how often it decides", "decisions", "reaction", "reaction time", "react", "tick skip", "ticks"], ["env.tick_skip", "env.action_delay"]],
  [["self play", "self-play", "past versions", "old versions", "against itself", "play itself", "past selves"], ["selfplay.enabled", "selfplay.past_share", "selfplay."]],
  [["randomness", "random choices", "exploration", "explore", "entropy", "try new things", "curiosity"], ["ppo.entropy", "ppo.entropy_target", "ppo.entropy_normalization", "ppo.entropy_target_rate", "ppo.entropy_target_min"]],
  [["network size", "network", "brain", "brain size", "bigger network", "bigger brain", "model size", "neurons", "layers"], ["model.policy", "model.value", "model.shared", "model.encoder", "model."]],
  [["replays", "replay", "learn from replays", "copy players", "copy replays", "pretrain", "pretraining", "human games"], ["pretrain.enabled", "pretrain.replays", "replay_mix.enabled", "replay_mix.share", "pretrain.", "replay_mix."]],
  [["team size", "game size", "players", "how many players", "1v1", "2v2", "3v3"], ["env.team_size", "env.team_sizes", "env.obs_team_size"]],
  [["game length", "how long a game", "episode", "episode length", "restart", "reset", "timeout", "no touch"], ["terminals.no_touch_seconds", "terminals.max_episode_seconds", "terminals."]],
  [["graphics card", "gpu", "cuda", "cpu", "device"], ["run.device", "run.precision", "run.cards"]],
  [["precision", "fp16", "fp32", "half precision", "memory"], ["run.precision", "run.fp16_sums"]],
  [["threads", "cores", "processor", "cpu cores"], ["env.threads"]],
  [["batch size", "batch", "steps per update", "update size", "rollout"], ["ppo.steps_per_iteration", "ppo.minibatches", "ppo.epochs"]],
  [["discount", "gamma", "horizon", "plan ahead", "how far ahead", "future rewards"], ["ppo.discount_half_life", "ppo.gae_lambda"]],
  [["wandb", "weights and biases", "weights & biases", "online charts", "tracking"], ["wandb.enabled", "wandb.project", "wandb."]],
  [["transfer", "transfer learning", "start from", "another bot", "start from a bot"], ["start_from.enabled", "start_from.run", "start_from."]],
  [["seed", "random seed", "repeatable", "deterministic"], ["run.seed", "run.deterministic"]],
  [["console", "log", "output", "printout"], ["run.console", "run.console_seconds"]],
  [["observation", "what it sees", "inputs", "obs"], ["env.observation", "env.obs_team_size"]],
  [["meshes", "collision meshes", "arena files"], ["paths.collision_meshes"]],
  [["report", "progress report"], ["run.report_minutes"]],
];
// Words that say nothing about which setting is meant.
const MATCH_FILLER = new Set(["a", "an", "the", "it", "its", "is", "of", "to", "at", "in", "on", "for", "how", "do", "does", "i", "my", "bot", "bots", "what", "much", "many"]);
// Words that say a replay setting is meant: without one, the main training's setting comes first.
const MATCH_REPLAYS = /\b(replays?|copy|copies|copying|pretrain\w*)\b/;

function matchText(text) { return String(text || "").toLowerCase().replace(/[?!,;:'’"()]/g, " ").replace(/\s+/g, " ").trim(); }
function matchWords(q) { const all = q.split(" ").filter(Boolean), some = all.filter(w => !MATCH_FILLER.has(w)); return some.length ? some : all; }
// Whether `phrase` stands in `q` as whole words, or `q` is the start of it being typed.
function phraseIn(q, phrase) {
  if (` ${q} `.includes(` ${phrase} `)) return phrase.split(" ").length;
  return q.length >= 4 && phrase.startsWith(q) ? q.split(" ").length - 0.5 : 0;
}
// The synonyms' answer for `q`: path -> its score (the more specific the phrase, the higher), read once a query.
const MATCH_CACHE = { q: null, items: null, out: null };
function synonymScores(q, items) {
  if (MATCH_CACHE.q === q && MATCH_CACHE.items === items) return MATCH_CACHE.out;
  const out = new Map();
  for (const [phrases, targets] of SETTING_WORDS) {
    const fit = Math.max(0, ...phrases.map(p => phraseIn(q, p)));
    if (!fit) continue;
    const paths = [];
    for (const t of targets) {
      const more = t.startsWith("group:") ? items.filter(i => i.group === t.slice(6)) : t.endsWith(".") ? items.filter(i => i.path.startsWith(t)) : items.filter(i => i.path === t);
      more.forEach(i => { if (!paths.includes(i.path)) paths.push(i.path); });
    }
    paths.forEach((path, at) => {
      const score = 3000 + 200 * fit - 5 * Math.min(at, 30);
      if (score > (out.get(path) || 0)) out.set(path, score);
    });
  }
  MATCH_CACHE.q = q; MATCH_CACHE.items = items; MATCH_CACHE.out = out;
  return out;
}
// How well the setting `item` ({ path, title, label, help, level, group }) fits `query`, among `items` (all
// the settings searched): 0 when it does not. A synonym or the exact plain title first, then words of the
// title, then the name in the file, then the help; a Basic setting before an Advanced one on a tie.
function settingMatch(item, query, items) {
  const q = matchText(query);
  if (!q) return 0;
  const title = matchText(item.title || item.label), label = matchText(item.label), path = item.path.toLowerCase();
  const words = matchWords(q), plain = words.join(" ");
  const hasAll = text => words.every(w => text.includes(w));
  const said = synonymScores(q, items || [item]);
  let score = said.get(item.path) || 0;
  if (title === q) score = Math.max(score, 3500);
  else if (!score) {
    if (title.includes(q) || title.includes(plain)) score = title.startsWith(plain) ? 2100 : 2000;
    else if (words.every(w => title.split(" ").some(t => t.startsWith(w)))) score = 1800;
    else if (path.includes(q) || label.includes(q) || label.includes(plain)) score = 1000;
    else if (hasAll(title + " " + label + " " + path)) score = 800;
    else if (said.size === 0) {
      // The help counts only when no plain word was recognised: it names many settings in passing.
      const help = matchText(item.help);
      if (help.includes(q) || help.includes(plain)) score = 500;
      else if (words.length > 1 && hasAll(help)) score = 300;
    }
  }
  if (!score) return 0;
  const side = /^(pretrain|replay_mix)\./.test(item.path);
  if (side && !MATCH_REPLAYS.test(q)) score -= 20;
  return score + (item.level === "advanced" ? 0 : 3);
}
// `items` that fit `query`, the best first: [{ item, score }].
function rankSettings(items, query) {
  return items.map(item => ({ item, score: settingMatch(item, query, items) })).filter(x => x.score > 0)
    .sort((a, b) => b.score - a.score || (a.item.order || 0) - (b.item.order || 0));
}
