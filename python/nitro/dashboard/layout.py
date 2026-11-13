"""Where each setting sits in the dashboard's Settings tab: its group (a plain-named topic), its level (basic
is shown always, advanced only with "Show advanced" on) and, where the name made from its key is jargon, a
plain title. The key in the settings file stays visible under the title.

To move a setting, change its line. One line per setting, grouped as the tab shows them.
tests/test_dashboard_settings.py fails when a setting the editor shows is missing here, or a line here names
a setting that does not exist. NOT_SHOWN lists the settings the editor leaves out."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Level = Literal["basic", "advanced"]


@dataclass(frozen=True)
class Group:
    id: str
    title: str
    about: str  # one line under the group's title


@dataclass(frozen=True)
class Place:
    group: str
    level: Level
    title: str = ""  # "" = the name made from the setting's key


def basic(group: str, title: str = "") -> Place:
    """What a newcomer may reasonably change."""
    return Place(group, "basic", title)


def advanced(group: str, title: str = "") -> Place:
    """What needs some knowledge of the method, or is for fine-tuning."""
    return Place(group, "advanced", title)


# The groups, in the order the tab lists them. "basics" holds no setting of its own: it shows the settings
# named in BASICS, each of which keeps its home group too.
GROUPS: tuple[Group, ...] = (
    Group("basics", "Basics", "The few settings most people change."),
    Group("learning", "Learning", "How the bot learns from the games it plays."),
    Group("games", "Games", "The simulated games: their size, how many at once, how often it decides."),
    Group("endings", "Game endings", "When a game is restarted. A goal always restarts it."),
    Group("network", "Network", "The size and shape of the bot's networks."),
    Group("selfplay", "Self-play", "Playing against saved past versions of the bot."),
    Group("start_from", "Start from another bot", "Start this bot from another bot's trained networks."),
    Group("replays_first", "Learn from replays", "Copy how the players in your replays play, first."),
    Group("replay_mix", "Replays in training", "Keep copying replays in a share of the learning steps."),
    Group("speed", "Speed and precision", "How fast training runs on this PC. The learning stays the same."),
    Group("saving", "Saving and the run", "How often the bot is saved, what is kept, the console."),
    Group("tracking", "Tracking", "Sending the run's numbers to Weights & Biases."),
    Group("files", "Files", "Where NitroLearn finds files on this PC."),
)  # fmt: skip

# The Basics group, in the order shown.
BASICS: tuple[str, ...] = (
    "ppo.policy_lr",
    "env.num_arenas",
    "env.tick_skip",
    "env.team_size",
    "terminals.no_touch_seconds",
    "selfplay.enabled",
    "selfplay.past_share",
    "run.checkpoint_minutes",
    "run.keep_checkpoints",
)

SETTINGS: dict[str, Place] = {
    # ---- Learning: how the bot learns from the games it plays ----
    "ppo.policy_lr": basic("learning", "Learning rate"),
    "ppo.value_lr": advanced("learning", "Value network's learning rate"),
    "ppo.entropy": basic("learning", "Exploration bonus (entropy)"),
    "ppo.steps_per_iteration": basic("learning", "Steps collected before each update"),
    "ppo.epochs": advanced("learning", "Passes over each round of games"),
    "ppo.minibatches": advanced("learning", "Learning steps per pass"),
    "ppo.discount_half_life": advanced("learning", "How far ahead rewards count (seconds)"),
    "ppo.gae_lambda": advanced("learning", "GAE lambda"),
    "ppo.goal_ends": advanced("learning", "What a goal ends for the learner"),
    "ppo.clip_range": advanced("learning", "How far one update may move the policy"),
    "ppo.encoder_lr_scale": advanced("learning", "Encoder's learning rate, as a multiple"),
    "ppo.entropy_normalization": advanced("learning", "How randomness is measured"),
    "ppo.entropy_target": advanced("learning", "Randomness to hold (entropy target)"),
    "ppo.entropy_target_rate": advanced("learning", "Entropy target: how fast the bonus moves"),
    "ppo.entropy_target_min": advanced("learning", "Entropy target: lowest bonus"),
    "ppo.advantage_keep": advanced("learning", "Share of samples the policy learns from"),
    "ppo.advantage_keep_rest": advanced("learning", "Share of the other samples still run"),
    "ppo.value_share": advanced("learning", "Share of samples the value side learns from"),
    "ppo.goal_critic_weight": advanced("learning", "Goal critic: weight"),
    "ppo.goal_critic_half_life": advanced("learning", "Goal critic: horizon (seconds)"),
    "ppo.vdag_beta": advanced("learning", "Twin optimistic critics: weight"),
    "ppo.vdag_expectile": advanced("learning", "Twin optimistic critics: optimism"),
    "ppo.vdag_reinit_steps": advanced("learning", "Twin optimistic critics: start afresh every (steps)"),
    "ppo.head_loss_scale": advanced("learning", "How the extra critics' losses are weighed"),
    "ppo.optimizer": advanced("learning"),
    "ppo.muon_lr_scale": advanced("learning", "Muon's learning rate, as a multiple"),
    "ppo.weight_decay": advanced("learning"),
    "ppo.adam_eps": advanced("learning", "Adam epsilon"),
    "ppo.max_grad_norm": advanced("learning", "Largest gradient size"),
    "ppo.reward_clip": advanced("learning"),
    "ppo.normalize_advantages": advanced("learning", "Normalise advantages"),
    "ppo.popart": advanced("learning", "Normalise value targets (PopArt)"),
    # ---- Games: the simulated games ----
    "env.team_size": basic("games", "Players per team"),
    "env.num_arenas": basic("games", "Games played at once"),
    "env.tick_skip": basic("games", "Physics ticks per decision"),
    "env.observation": advanced("games", "What the bot sees (observation)"),
    "env.obs_team_size": advanced("games", "Observation room (players per team)"),
    "env.action_delay": advanced("games", "Delay before an action acts (ticks)"),
    "env.team_sizes": advanced("games", "Several game sizes in one bot"),
    "env.team_size_shares": advanced("games", "Share of games for each size"),
    "env.shuffle_slots": advanced("games", "Shuffle the cars' places"),
    "env.opponents": advanced("games", "Play against an opposing team"),
    # ---- Game endings: when a game is restarted ----
    "terminals.no_touch_seconds": basic("endings", "Restart after no touch for (seconds)"),
    "terminals.max_episode_seconds": advanced("endings", "Longest game (seconds)"),
    "terminals.end_on_touch": advanced("endings", "End the game at the first touch"),
    "terminals.end_on_landing": advanced("endings", "End the game once every car has landed"),
    "terminals.truncate_on_ball_landing": advanced("endings", "Chance to cut the game when the ball lands"),
    # ---- Network: the size and shape of the networks (fixed once the bot has a save) ----
    "model.policy": basic("network", "Policy network layers"),
    "model.value": basic("network", "Value network layers"),
    "model.shared": advanced("network", "Shared layers"),
    "model.encoder": advanced("network", "Kind of network (encoder)"),
    "model.activation": advanced("network", "Activation function"),
    "model.layer_norm": advanced("network", "Layer normalisation"),
    "model.residual": advanced("network", "Residual layers"),
    "model.init": advanced("network", "Starting weights"),
    "model.value_heads": advanced("network"),
    "model.value_head_layers": advanced("network", "Layers of each value head"),
    "model.attention_width": advanced("network", "Attention: detail per car"),
    "model.attention_blocks": advanced("network", "Attention: looks per decision"),
    "model.attention_heads": advanced("network", "Attention: things looked for at once"),
    "model.attention_skip": advanced("network", "Attention: also read the observation directly"),
    "model.transformer_layers": advanced("network", "Transformer: blocks"),
    "model.transformer_width": advanced("network", "Transformer: detail per token"),
    "model.transformer_heads": advanced("network", "Transformer: things looked for at once"),
    "model.transformer_ff": advanced("network", "Transformer: feed-forward width"),
    "model.transformer_pads": advanced("network", "Transformer: boost pads as tokens"),
    "model.transformer_readout": advanced("network", "Transformer: what the layers read"),
    # ---- Self-play: playing against past versions ----
    "selfplay.enabled": basic("selfplay", "Play against its past versions"),
    "selfplay.past_share": basic("selfplay", "Share of games against past versions"),
    "selfplay.snapshot_steps": advanced("selfplay", "Keep a past version every (steps)"),
    "selfplay.active_opponents": advanced("selfplay", "Past versions playing at once"),
    "selfplay.pool_size": advanced("selfplay", "Past versions kept"),
    "selfplay.pool_disk_mb": advanced("selfplay", "Disk space for past versions (MB)"),
    "selfplay.matchmaking": advanced("selfplay", "How past versions are picked"),
    "selfplay.flat_rating_steps": advanced("selfplay", "Warn when its rating stops rising for (steps)"),
    # ---- Start from another bot: transfer learning, read at the bot's first start ----
    "start_from.enabled": basic("start_from"),
    "start_from.run": basic("start_from"),
    "start_from.at": basic("start_from"),
    "start_from.parts": basic("start_from"),
    # ---- Learn from replays: copying your replays before training ----
    "pretrain.enabled": basic("replays_first"),
    "pretrain.replays": basic("replays_first"),
    "pretrain.team_sizes": advanced("replays_first"),
    "pretrain.limit": basic("replays_first"),
    "pretrain.skill_every": advanced("replays_first"),
    "pretrain.skill_points": advanced("replays_first"),
    "pretrain.passes": advanced("replays_first"),
    "pretrain.batch_replays": advanced("replays_first"),
    "pretrain.keep_batches": advanced("replays_first", "Rounds learned from together"),
    "pretrain.workers": advanced("replays_first"),
    "pretrain.converter": advanced("replays_first", "What converts the replays"),
    "pretrain.keep_converted": advanced("replays_first"),
    "pretrain.continue_from": advanced("replays_first", "Carry on another bot's copy"),
    "pretrain.new_only": advanced("replays_first", "Only replays not converted before"),
    "pretrain.value_warmup_steps": advanced("replays_first"),
    "pretrain.lr": advanced("replays_first", "The copy's learning rate"),
    "pretrain.lr_end_share": advanced("replays_first", "Learning rate at the end, as a share"),
    "pretrain.prev_action_dropout": advanced("replays_first"),
    "pretrain.minibatch": advanced("replays_first", "Decisions per learning step"),
    "pretrain.measure_every": advanced("replays_first", "Measure the held-back replays every (rounds)"),
    "pretrain.save_every": advanced("replays_first"),
    # ---- Replays in training: copying replays while the bot trains ----
    "replay_mix.enabled": basic("replay_mix"),
    "replay_mix.share": basic("replay_mix"),
    "replay_mix.lr": advanced("replay_mix"),
    "replay_mix.minibatch": advanced("replay_mix"),
    "replay_mix.kickoff_only_seconds": advanced("replay_mix", "Copy only kickoffs (seconds after the touch)"),
    "replay_mix.replays": advanced("replay_mix"),
    # ---- Speed and precision: how fast training runs; the learning stays the same ----
    "run.precision": advanced("speed"),
    "env.threads": advanced("speed", "Processor threads for the games"),
    "run.device": advanced("speed", "Where the networks run"),
    "run.fp16_sums": advanced("speed", "FP16 running sums"),
    "model.fused_norm": advanced("speed", "LayerNorm and ReLU as one routine"),
    "ppo.graphed_steps": advanced("speed", "Recorded learning steps (CUDA graphs)"),
    "ppo.microbatch": advanced("speed", "Learning steps worked out in parts"),
    "env.groups": advanced("speed", "Groups the games are split into"),
    "env.pipelined_inference": advanced("speed", "Overlap the network passes"),
    "env.concurrent_passes": advanced("speed", "Run the networks' passes side by side"),
    "env.card_starts_games": advanced("speed", "Let the graphics card start the games"),
    "model.attention_fast": advanced("speed", "Attention in fewer, bigger steps"),
    "model.lean_attention": advanced("speed"),
    "model.small_attention": advanced("speed", "Transformer attention by NitroLearn's own routine"),
    "ppo.lean_heads_pass": advanced("speed", "Lean pass for the extra critics"),
    "ppo.muon_batched": advanced("speed", "Muon steps in batches"),
    "run.cards": advanced("speed", "Graphics cards training together"),
    "run.cards_sum_precision": advanced("speed", "Precision of the sums between cards"),
    "run.cards_graphed_steps": advanced("speed", "Recorded learning steps on several cards"),
    "run.cards_pin_threads": advanced("speed", "Own processor cores for each card"),
    "run.cards_check_every": advanced("speed", "Check the cards agree every (iterations)"),
    # ---- Saving and the run: saves, the report and the console ----
    "run.checkpoint_minutes": basic("saving", "Save every (minutes)"),
    "run.checkpoint_steps": advanced("saving", "Save every (steps) instead"),
    "run.keep_checkpoints": basic("saving", "Newest saves kept"),
    "run.milestone_steps": advanced("saving", "Also keep a save every (steps)"),
    "run.console": basic("saving", "What the console shows"),
    "run.console_seconds": advanced("saving", "Console update every (seconds)"),
    "run.report_minutes": advanced("saving", "Rewrite the progress report every (minutes)"),
    "run.on_training_problem": advanced("saving", "When training goes wrong"),
    "run.missing_data": advanced("saving", "When training data is missing"),
    "run.seed": advanced("saving", "Random seed"),
    "run.deterministic": advanced("saving", "Exactly repeatable runs"),
    # ---- Tracking: Weights & Biases ----
    "wandb.enabled": basic("tracking", "Send the numbers to Weights & Biases"),
    "wandb.project": basic("tracking", "W&B project"),
    "wandb.entity": advanced("tracking", "W&B user or team"),
    "wandb.name": advanced("tracking", "Run name on W&B"),
    "wandb.group": advanced("tracking", "W&B group"),
    "wandb.tags": advanced("tracking", "W&B labels"),
    "wandb.mode": advanced("tracking", "Send while training or later"),
    # ---- Files ----
    "paths.collision_meshes": basic("files", "Collision meshes folder"),
}

# Settings the editor leaves out, each with why. A settings file may still write them.
NOT_SHOWN: dict[str, str] = {}

_ORDER = {path: index for index, path in enumerate(SETTINGS)}
# A setting not listed above yet (the test names it) still shows, as advanced, in its section's group.
_SECTION_GROUPS = {
    "ppo": "learning",
    "env": "games",
    "terminals": "endings",
    "model": "network",
    "selfplay": "selfplay",
    "start_from": "start_from",
    "pretrain": "replays_first",
    "replay_mix": "replay_mix",
    "run": "saving",
    "wandb": "tracking",
    "paths": "files",
}


def place(path: str) -> Place:
    """Where the setting at `path` sits."""
    found = SETTINGS.get(path)
    if found is not None:
        return found
    return advanced(_SECTION_GROUPS.get(path.split(".", 1)[0], "saving"))


def order(path: str) -> int:
    """The setting's place among all of them, as listed above (one not listed yet comes last)."""
    return _ORDER.get(path, len(_ORDER))


def groups() -> list[dict[str, str]]:
    """The groups as the page lists them."""
    return [{"id": group.id, "title": group.title, "about": group.about} for group in GROUPS]


# The first line a newcomer reads about a setting, in plain words, where the setting's own description is
# written for those who know the terms: the new-bot form and the Settings tab show this, and the full
# description under "Nerd explanation".
PLAIN: dict[str, str] = {
    "env.tick_skip": "How often the bot decides: every this many physics ticks (the game runs 120 a second). "
    "4 = 30 decisions a second; 8 = 15.",
    "env.observation": "What the bot sees of the game. The detailed one ('nitro_v2', the default) tested "
    "clearly better than the basic one ('native').",
    "env.obs_team_size": "Room in what the bot sees for bigger teams, so one bot can later play 1v1, 2v2 and "
    "3v3. 0 = room for its own game size only.",
    "model.encoder": "The kind of network. 'mlp' (plain layers, the default) suits 1v1 and trains fastest; "
    "the attention kinds play any team size and suit 2v2 and 3v3, but train slower.",
    "model.shared": "Layers the policy and the value network share. Empty (the usual) = two separate "
    "networks.",
    "model.policy": "Layer sizes of the policy: the network that picks each action.",
    "model.value": "Layer sizes of the value network: the one that judges how good the situation is, to help "
    "the policy learn.",
    "model.value_head_layers": "Layers each value head has to itself (the GCO design's extra judges). Empty "
    "= none of their own.",
    "model.attention_width": "How much detail the attention network keeps about each car and the ball. 128 "
    "is the tested size.",
    "model.attention_blocks": "How many times the bot looks over the other cars before it decides. 2 is the "
    "tested number.",
    "model.attention_heads": "How many things it looks for at once in each look. 4 is the tested number.",
    "model.transformer_layers": "How many blocks the transformer has. More can follow longer chains of play "
    "but train slower; 2 is a good start.",
    "model.transformer_width": "How much detail the transformer keeps about each car and the ball. 128 is a "
    "good start.",
    "model.transformer_heads": "How many things the transformer looks for at once. 4 is a good start.",
    "model.transformer_ff": "How much each transformer block can work on what it saw: usually 4 times its "
    "width.",
    "ppo.policy_lr": "How big a step the policy takes each time it learns. Bigger learns faster but less "
    "steadily; lower it if the console warns that its updates are too big.",
    "ppo.value_lr": "How big a step the value network takes each time it learns.",
    "ppo.entropy": "How much the bot is rewarded for trying different things instead of always the same "
    "one. More keeps it exploring for longer.",
    "ppo.discount_half_life": "How far ahead the bot looks: a reward this many seconds away counts half. "
    "Longer plans further ahead but learns more slowly.",
    "ppo.steps_per_iteration": "How much play is collected before each round of learning. Bigger rounds are "
    "steadier but come less often.",
    "ppo.epochs": "How many times the bot learns from each round of play. 2 tested clearly better than 1.",
    "ppo.minibatches": "Into how many parts each round of play is split; the bot takes one learning step per "
    "part. 4 is the tested choice.",
}
