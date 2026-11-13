"""The settings a NitroLearn settings file may contain."""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from nitro.config.loader import section, setting
from nitro.config.rewards import DEFAULT_REWARDS, Reward
from nitro.config.state_setters import DEFAULT_STATE_SETTERS, StateSetter

# NitroLearn's own folder (python/nitro/config/schema.py is three folders below it).
PROJECT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class PathsConfig:
    collision_meshes: Path = setting(
        Path("collision_meshes"),
        "Folder with the Rocket League arena collision meshes; it must contain a 'soccar' subfolder. "
        "Relative paths are relative to the settings file. Not written: the collision_meshes folder next "
        "to the settings file when it holds the meshes, else NitroLearn's own collision_meshes folder (where "
        "the installer puts them), so a settings file works wherever it is kept.",
    )

    @staticmethod
    def default_path(name: str, beside: Path) -> Path:
        """Where a path setting that is not written points: `beside` (its default next to the settings
        file), or for the meshes NitroLearn's own folder when `beside` holds none."""
        own = PROJECT / "collision_meshes"
        project = (PROJECT / "configs" / "presets").is_dir()  # an installed NitroLearn folder
        if name == "collision_meshes" and project and not (beside / "soccar").is_dir():
            return own
        return beside


@dataclass(frozen=True)
class RunConfig:
    seed: int = setting(
        0,
        "Starting point for all randomness. The same seed and settings replay the same games.",
        minimum=0,
        maximum=2**63 - 1,
    )
    device: Literal["auto", "cuda", "cpu"] = setting(
        "auto", "Where the networks run: 'auto' uses the graphics card if PyTorch can see one."
    )
    cards: int = setting(
        1,
        "How many graphics cards train this bot together. Each card plays its share of the games "
        "(env.num_arenas and env.threads are split between them) and learns on it; after every learning step"
        " the cards add up what they learned, so the bot learns as it would on one card with all the games, "
        "only faster. 1 = one card. Linux only (the cards talk through NVIDIA's NCCL library, which Windows "
        "lacks); num_arenas and steps_per_iteration must divide by it. Measured with two cards: about 1.55x "
        "as fast for 1v1 bots whose networks keep the cards busy (10 and 51 million weights in FP16), about "
        "1.7x for a large 2v2 transformer, 1.23x for a small, fast bot with cards_graphed_steps on (without "
        "it two cards were slower than one), and no faster for a bot whose games hold training up (the cards"
        " share one processor).",
        minimum=1,
        maximum=16,
    )
    cards_sum_precision: Literal["fp32", "bf16"] = setting(
        "fp32",
        "With run.cards: how exactly the cards send each other what they learned every learning step. 'fp32'"
        " sends the numbers as computed; 'bf16' sends them at half the size, each rounded to about 3 "
        "significant digits: faster where the connection between the cards is slow, but not yet measured for "
        "learning.",
    )
    cards_graphed_steps: bool = setting(
        False,
        "With run.cards and ppo.graphed_steps: several cards also replay their learning steps as recordings "
        "(see ppo.graphed_steps) instead of being handed each small job one by one. Without it, small and "
        "medium networks lose most of what a second card gains. Off by default: not yet checked on two real "
        "cards.",
    )
    cards_pin_threads: bool = setting(
        False,
        "With run.cards on Linux: each card's games and its share of the work run on their own group of "
        "processor cores (cores that share a memory cache), instead of every card's work spread over every "
        "core. For machines with many cores. Speed only; not measured on a machine with several cards.",
    )
    cards_check_every: int = setting(
        1,
        "With run.cards: how often the cards check that their networks are still identical, in iterations"
        ". 1 = after every iteration (the default). Higher checks less often, and always right "
        "before a save, so a drift (never seen so far) is still caught within that many iterations and "
        "before it is saved. Speed only; not measured on a machine with several cards.",
        minimum=1,
        maximum=1000,
    )
    precision: Literal["fp32", "fp16"] = setting(
        "fp32",
        "'fp32' uses full precision everywhere. 'fp16' runs the networks' hidden layers in half precision "
        "on the graphics card (faster for big networks; the sensitive maths stays in full precision). "
        "FP16 is opt-in until it has been shown to learn as well as FP32 on Rocket League.",
    )
    fp16_sums: Literal["off", "playing", "forward", "all"] = setting(
        "off",
        "With precision = 'fp16' on a graphics card: the card also keeps the running totals inside its "
        "multiplications in half precision, which many GeForce cards do faster (on a recent card 1.3-1.6x "
        "for big networks). 'playing': only when choosing actions (the bot's and its past versions'); "
        "'forward': also the first half of each learning step; 'all': all of learning. The answers move "
        "slightly (each total keeps about 3 significant digits). Not yet measured for learning; needs "
        "PyTorch 2.7 or"
        " newer.",
    )
    deterministic: bool = setting(
        False,
        "Make runs exactly repeatable: the same seed and settings give the same games and the same "
        "learning, bit for bit. Somewhat slower; meant for testing and comparisons.",
    )
    on_training_problem: Literal["stop", "warn"] = setting(
        "stop",
        "What to do when training goes wrong (NaN in losses or weights, learning updates being skipped "
        "repeatedly): 'stop' with a clear error, or 'warn' and carry on.",
    )
    missing_data: Literal["warn", "stop"] = setting(
        "warn",
        "What to do when some of what a bot trains with is not here, as after a move to another computer"
        ". 'warn' trains on: past versions its save rates that are not in its pool folder "
        "are named at the start and in the report, replay_mix folders without converted replays at the "
        "first iteration, at every save and in the report. 'stop' refuses to start training (a new run too) "
        "when any of those is found, and says what is missing, so it can be copied over first. Only "
        "training is refused: nitro reward-check and a copy of replays' save run either way.",
    )
    checkpoint_minutes: float = setting(
        15.0,
        "Save the run this often, in minutes. Every save also starts all games afresh, so a stopped run "
        "continues exactly where it left off.",
        minimum=0.5,
    )
    checkpoint_steps: int = setting(
        0,
        "Save the run every this many agent steps instead of every checkpoint_minutes (e.g. "
        "200_000_000: with keep_checkpoints = 5 the newest billion steps stay). 0 = by time. A stop "
        "(Ctrl+C) still saves at once, whichever is set.",
        minimum=0,
    )
    keep_checkpoints: int = setting(5, "How many of the newest saves to keep.", minimum=1)
    milestone_steps: int = setting(
        1_000_000_000,
        "Also keep the first save of every this many agent steps, so older versions of the bot stay "
        "available. 0 = keep only the newest saves.",
        minimum=0,
    )
    report_minutes: float = setting(2.0, "Rewrite the run's report.html this often, in minutes.", minimum=0.1)
    console: Literal["line", "report"] = setting(
        "line",
        "What the console shows while training: 'line' = one line (steps, speed, reward, entropy, value "
        "prediction); 'report' = a full report (speed and where the time "
        "goes, the rewards and each reward's part, how the policy and value learn, how the bots play, "
        "self-play, the bot the run started from, the next save, new warnings), and a "
        "report for every round of copying replays. Training is the same either way.",
    )
    console_seconds: float = setting(
        10.0, "How often the console line or report is printed while training, in seconds.", minimum=1.0
    )


def large_networks(shared: Sequence[int], policy: Sequence[int], value: Sequence[int]) -> bool:
    """Networks with a layer of about a million weights or more (two hidden layers 1024 wide in a row,
    counting the shared trunk into each head): the sizes where FP16 trained about twice as fast as FP32 and
    learned as well."""
    chains = ((*shared, *policy), (*shared, *value))
    return any(min(a, b) >= 1024 for chain in chains for a, b in itertools.pairwise(chain))


@dataclass(frozen=True)
class ModelConfig:
    shared: tuple[int, ...] = setting(
        (),
        "Layer sizes of a trunk shared by the policy and the value network, e.g. [256, 256]. "
        "Empty = two separate networks. Measured on the ball-touch task: a [256, 256] trunk "
        "with one layer each on top made no measurable difference (2.18 s against 2.16 s to the first "
        "touch). With an attention encoder, the trunk starts with the one encoder both networks read "
        "through, a shared head (not measured yet).",
        minimum=1,
    )
    policy: tuple[int, ...] = setting((256, 256, 256), "Layer sizes of the policy network.", minimum=1)
    value: tuple[int, ...] = setting(
        (256, 256, 256),
        "Layer sizes of the value network. With value_heads or value_head_layers set, these are the layers "
        "every value head shares (the critic trunk, on top of any model.shared).",
        minimum=1,
    )
    value_head_layers: tuple[int, ...] = setting(
        (),
        "Layer sizes each value head has to itself, after the layers of model.value, e.g. [1280, 1280]"
        ". Empty = no layers of their own: each head reads model.value's last layer. Fixed once the bot "
        "has saves.",
        minimum=1,
    )
    value_heads: tuple[Literal["main", "goal", "vdag1", "vdag2"], ...] = setting(
        ("main",),
        "The value heads on top of model.value, 'main' first: 'main' is the value the bot learns from as "
        "always; 'goal', 'vdag1' and 'vdag2' are extra heads (a goal critic and twin value critics) "
        "for ppo.goal_critic_weight and ppo.vdag_beta: without those they are kept and saved but "
        "learn nothing. They live only in training; the exported bot never has them. Fixed once the bot has "
        "saves.",
    )
    residual: bool = setting(
        False,
        "Make the hidden layers add to what they are given instead of replacing it (x + layer(x)), wherever "
        "a layer is as wide as the one before it, so a deep network passes on what earlier layers worked out"
        " and each layer only learns a change; deep, wide networks stay easier to train. A layer that "
        "changes the width (usually the first of each network) stays plain. Needs layer_norm = true. "
        "Not measured yet. Fixed once the bot has saves.",
    )
    layer_norm: bool = setting(
        True,
        "Rescale each layer's outputs to a steady range (layer normalisation), "
        "which keeps learning steady however the weights grow. Needed by "
        "residual, fused_norm and weight_decay.",
    )
    fused_norm: bool = setting(
        False,
        "Speed: run each hidden layer's normalisation and activation on the graphics card as one step "
        "instead of several. The same maths (to rounding), so it can be switched between stopping and "
        "resuming. Built the first time it is used, with nothing extra to install. Needs layer_norm = true "
        "and activation 'relu' or 'leaky_relu'. On the processor, or if it cannot be built, the usual way "
        "runs (with a warning). Measured: with large networks in FP16 (4x1024) 11% faster (8% for a large "
        "1v1 network), so recommended there; with the starters' [256, 256, 256] no faster in FP32 and 7% "
        "slower in FP16, so leave it off with small networks.",
    )

    @property
    def large(self) -> bool:
        return large_networks(self.shared, self.policy, (*self.value, *self.value_head_layers))

    init: Literal["orthogonal", "pytorch"] = setting(
        "orthogonal",
        "How the networks' weights start: 'orthogonal' (a careful random start in which the policy begins "
        "undecided between its actions) or 'pytorch' (PyTorch's plain random start). Measured on ball touch:"
        " no measurable difference (2.10 s with 'pytorch' against 2.16 s; a small gain is not ruled out).",
    )
    activation: Literal["relu", "leaky_relu", "gelu", "tanh"] = setting(
        "leaky_relu",
        "The small bending step between layers that lets the networks learn more than straight lines. "
        "'leaky_relu' (the default): negative numbers keep a hundredth of their size instead of becoming 0, "
        "so no neuron can stop learning. 'relu' turns them into 0; 'gelu' and 'tanh' are other curves "
        "('gelu' cannot be exported to RLBot). Fixed once the bot has saves.",
    )
    encoder: Literal["mlp", "attention", "attention_communication", "transformer"] = setting(
        "mlp",
        "The kind of network. 'mlp' (the default): plain layers read the whole observation. 'attention': "
        "first an attention encoder, in which the bot's own car looks at every other car through one shared "
        "set of weights, so what it learns about one opponent carries to the others and their order does not"
        " matter; the network's size does not depend on the number of cars, so the bot plays any game size "
        "(how well at sizes it never trained at is not measured) and its training can move to another size. "
        "For 2v2 and 3v3; a 1v1 bot learns better with 'mlp'. Trains at 41% of the speed of a plain network "
        "of its size; the bot stays fast. 'attention_communication': the cars, the ball and the own car "
        "first look at each other, so the network can work out how they relate (who is nearer the ball, who "
        "covers whom), all inside one bot; 31% of attention's training speed, checked for working only. "
        "'transformer': a standard transformer over the own car, the ball and every other car (and, with "
        "transformer_pads, every boost pad), with its own sizes (transformer_*); checked for working only, "
        "its speed and learning not measured. Each attention kind reads each car from a place of its own: "
        "env.obs_team_size = 0 gives room for team_size a side. Fixed once the bot has saves.",
    )
    attention_width: int = setting(
        128,
        "Only with encoder = 'attention' or 'attention_communication' (otherwise ignored). How many numbers "
        "the encoder uses to describe each car and the ball: its 'detail'. Bigger can notice more but trains "
        "slower; 128 is what was measured and the suggested value; 256 for a much bigger network. "
        "Must divide evenly by attention_heads. Fixed once the bot has saves.",
        minimum=8,
    )
    attention_blocks: int = setting(
        2,
        "Only with an attention encoder (otherwise ignored). How many times your car 'looks over' the other "
        "cars before deciding: each look can build on the last (first who is near, then who is a threat). "
        "With attention_communication each block is a round where the cars compare notes, then a look. More "
        "blocks = more reasoning but slower; 2 is what was measured and suggested, 3 at most for bigger "
        "networks. Fixed once the bot has saves.",
        minimum=1,
        maximum=8,
    )
    attention_heads: int = setting(
        4,
        "Only with an attention encoder (otherwise ignored). How many separate things your car looks for "
        "in each look, at once (e.g. one head watches the nearest opponent, another your teammate). "
        "attention_width is split between them, so it must divide evenly (128 / 4 = 32 each). 4 is what was "
        "measured and suggested; 8 with a width of 256. Fixed once the bot has saves.",
        minimum=1,
    )
    attention_skip: bool = setting(
        False,
        "With an attention encoder (any of the three): also give the layers after the encoder the "
        "observation's own part (the ball, the own car, the pads and the last action) directly, as a plain "
        "network sees it. Not measured yet.",
    )
    attention_fast: bool = setting(
        False,
        "Speed, with encoder = 'attention': the same maths in fewer, bigger steps on the graphics card. The "
        "same answers to rounding, so saves, resumes and the exported bot are the same with it on or off. "
        "Measured: 10% faster training with the 2v2 attention settings, so recommended whenever encoder = "
        "'attention' (the new-bot form turns it on). Not for attention_communication or the transformer.",
    )
    lean_attention: bool = setting(
        False,
        "Speed, with any attention encoder: the same maths with less work on the graphics card (parts of the"
        " attention that cannot change the answer are skipped or folded together). The same answers to "
        "rounding, so saves, resumes and the exported bot are the same with it on or off. With encoder = "
        "'attention' it replaces attention_fast. Measured: the transformer learns and chooses actions 27-40%"
        " faster, attention_communication 1-2% and 6-9% faster, so recommended with those two; with "
        "'attention' no faster than attention_fast.",
    )
    small_attention: bool = setting(
        True,
        "Speed, with encoder = 'transformer': NitroLearn's own graphics-card routine for the attention over "
        "a handful of cars and the ball, instead of PyTorch's, which is made for long texts and took a "
        "quarter of a 2v2 transformer's training time. The same maths to rounding. Up to 8 tokens (cars, "
        "ball) and 32 numbers a head (more falls back to PyTorch's). A big 2v2 transformer trained 1.21x "
        "faster with it. On by default.",
    )
    transformer_layers: int = setting(
        2,
        "Only with encoder = 'transformer' (otherwise ignored). How many transformer blocks the encoder has: "
        "in each, every token (the bot's own car, the ball, every other car and, with transformer_pads, "
        "every boost pad) looks at all the others and then works on what it saw. More blocks can follow "
        "longer chains of play (who covers whom while the ball is where) but train slower. 2 is the "
        "suggested start, as the attention encoders' 2 blocks; not measured. Fixed once the bot has saves.",
        minimum=1,
        maximum=12,
    )
    transformer_width: int = setting(
        128,
        "Only with encoder = 'transformer' (otherwise ignored). How many numbers describe each token: its "
        "'detail'. Bigger can notice more but trains slower; 128 is suggested (the attention encoders' "
        "measured size), 256 for a much bigger network. Must divide evenly by transformer_heads. Not "
        "measured. Fixed once the bot has saves.",
        minimum=8,
    )
    transformer_heads: int = setting(
        4,
        "Only with encoder = 'transformer' (otherwise ignored). How many separate things each token looks "
        "for at once (one head may follow the nearest opponent, another the teammate). transformer_width "
        "is split between them, so it must divide evenly (128 / 4 = 32 each). 4 suggested, 8 with a width "
        "of 256. Not measured. Fixed once the bot has saves.",
        minimum=1,
    )
    transformer_ff: int = setting(
        512,
        "Only with encoder = 'transformer' (otherwise ignored). How wide each block's feed-forward part is, "
        "where every token works on what it saw; usually 4 x transformer_width (512 for 128). Bigger thinks "
        "more per block and trains slower. Not measured. Fixed once the bot has saves.",
        minimum=8,
    )
    transformer_pads: bool = setting(
        False,
        "Only with encoder = 'transformer' (otherwise ignored). Also make each of the 34 boost pads a token "
        "of its own (whether it is up, plus a learned place on the field), so the bot can look at pads as it "
        "looks at cars. Off: the pads' numbers go into the bot's own car token instead. On: 34 more tokens "
        "in every block, so training is much slower (not measured). Not measured yet. Fixed once the bot "
        "has saves.",
    )
    transformer_readout: Literal["own_car", "summary"] = setting(
        "own_car",
        "Only with encoder = 'transformer' (otherwise ignored). Which token the policy and value layers "
        "read after the last block: 'own_car' (the bot's own car token, as bots that attend from their own "
        "player do) or 'summary' (one more learned token that belongs to nothing and looks at everything). "
        "Not measured either way. Fixed once the bot has saves.",
    )

    @property
    def value_heads_apart(self) -> bool:
        """The value side is a critic trunk (`value`) with named heads on top, not one network."""
        return self.value_heads != ("main",) or bool(self.value_head_layers)

    @property
    def uses_attention(self) -> bool:
        """Any attention encoder (the transformer): each reads one place per car, with its
        "present" flag, from a padded observation (env.obs_team_size, or the run's own team size)."""
        return self.encoder in ("attention", "attention_communication", "transformer")


@dataclass(frozen=True)
class PPOConfig:
    steps_per_iteration: int = setting(
        50_000,
        "Agent steps collected before each learning update (rounded up to a whole number of steps of "
        "every arena).",
        minimum=64,
    )
    epochs: int = setting(
        2,
        "Passes over each iteration's data. Measured on the ball-touch task (seconds to the first touch, "
        "lower is better): one pass learned clearly slower (2.38 s against 2.16 s), so keep 2.",
        minimum=1,
        maximum=32,
    )
    minibatches: int = setting(
        4,
        "Learning steps per pass; the data is split into this many parts. Measured on the ball-touch task "
        "(seconds "
        "to the first touch, lower is better): 1 (with one pass) learned far slower (7.6 s against 2.16 s); "
        "8 was only slightly better (2.05 s), too little to count, and its training speed was not measured, "
        "so 4 stays.",
        minimum=1,
    )
    microbatch: int = setting(
        0,
        "Work each learning step out in parts of at most this many samples, one after the other, and add the"
        " parts up: the same learning (to rounding) in less graphics-card memory, a little slower. For a big"
        " network whose learning steps do not fit on the card. 0 = each step at once.",
        minimum=0,
    )
    graphed_steps: bool = setting(
        True,
        "Speed, on a graphics card: record each learning step once and replay the recording, instead of "
        "handing the card hundreds of small jobs one by one (a CUDA graph). Helps small and medium networks "
        "on fast cards, which otherwise wait for those jobs (a 1v1 bot with 4x512 networks trained 1.24x "
        "faster); nothing for networks that keep the card busy anyway. The same learning (a smaller last "
        "step is padded: the same maths, rounded differently). Uses some more graphics memory. On several "
        "cards only with run.cards_graphed_steps. On by default.",
    )
    policy_lr: float = setting(
        4.5e-4,
        "Learning rate of the policy network: how big a step it takes each time it learns. Measured: "
        "4.5e-4 learned boost pickup faster than 1.5e-4 (88 against 59 boost) and recovery the same. "
        "Lower it if warnings say updates are too big.",
        minimum=0.0,
        maximum=1.0,
    )
    value_lr: float = setting(1.5e-4, "Learning rate of the value network.", minimum=0.0, maximum=1.0)
    encoder_lr_scale: float = setting(
        1.0,
        "With an attention encoder (model.encoder = 'attention', 'attention_communication' or "
        "'transformer'): the encoders' learning rate as a multiple of their "
        "network's (policy_lr for the policy's encoder, value_lr for the value's). Not measured yet: meant "
        "for "
        "attention learning slower than the plain network, as it did on ball touch.",
        minimum=0.01,
        maximum=100.0,
    )
    discount_half_life: float = setting(
        4.6,
        "Seconds after which a future reward counts half as much. 4.6 s equals a gamma of 0.995 at "
        "tick skip 4 and 0.99 at tick skip 8; longer makes the bot plan further ahead but learn more slowly.",
        minimum=0.01,
    )
    gae_lambda: float = setting(
        0.95,
        "How far the bot looks at what really happened before trusting the value network's guess, when it "
        "judges each action (0 = trust the guess after one step, 1 = only what really happened). Counted in "
        "decisions, so its reach in game time shrinks with the tick skip: the same seconds at another tick "
        "skip is lambda^(new/old) (0.95 at tick skip 8 = 0.9747 at 4). The goal critic uses it too.",
        minimum=0.0,
        maximum=1.0,
    )
    goal_ends: Literal["terminal", "continue"] = setting(
        "terminal",
        "How learning treats a goal. 'terminal' (the default): nothing after a goal counts. 'continue': the "
        "moment of the goal is also worth what the next kickoff is worth, as a real match goes on. A goal "
        "then no longer cuts off the other rewards' stream. The goal's own reward is unchanged. For bots "
        "with rewards beside the goal; it changes what the bot learns, so use it on a new bot or as a "
        "deliberate switch. Not measured yet.",
    )
    clip_range: float = setting(
        0.2,
        "How far one learning update may change how likely an action is: 0.2 = by about 20%. Smaller is"
        " steadier and slower.",
        minimum=0.0,
        maximum=1.0,
    )
    entropy: float = setting(
        0.035,
        "Bonus for keeping options open: a reward for some randomness in the bot's choices, so it keeps "
        "trying things. The randomness it is measured on runs from 0 (always the same action) to 1 (every "
        "allowed action equally likely); see ppo.entropy_normalization.",
        minimum=0.0,
    )
    entropy_normalization: Literal["allowed", "all"] = setting(
        "allowed",
        "How the randomness is measured: 'allowed' among the actions allowed at that moment; 'all' against "
        "all 90 actions, as some other frameworks do (so the same ppo.entropy is a weaker bonus there).",
    )
    entropy_target: float = setting(
        0.0,
        "Hold the policy's randomness near this entropy (0 = certain, 1 = uniform, as the entropy metric) by "
        "moving the entropy bonus after every update: up while the entropy is below it, down while "
        "above. ppo.entropy is where the bonus starts. 0 = off: the bonus stays ppo.entropy. Not measured "
        "yet.",
        minimum=0.0,
        maximum=1.0,
    )
    entropy_target_rate: float = setting(
        1.0,
        "With entropy_target: how fast the bonus moves. After each update it is multiplied by "
        "exp(rate x (target - entropy)), and kept between entropy_target_min x ppo.entropy and 10 x "
        "ppo.entropy.",
        minimum=0.0,
        maximum=100.0,
    )
    entropy_target_min: float = setting(
        0.01,
        "With entropy_target: the lowest the bonus may go, as a share of ppo.entropy. 0.01 = "
        "ppo.entropy / 100; 1 = never below ppo.entropy, so the bonus only rises when the "
        "randomness falls under the target and exploration stays open. Above 0, at most 10.",
        minimum=0.0,
        maximum=10.0,
    )
    advantage_keep: float = setting(
        1.0,
        "Of each learning step's samples, the policy learns only from this share whose actions turned out "
        "best or worst compared with what was expected (0.5 = the half that says most). The value side still"
        " learns from every sample. 1 (or 0) = off. Not measured yet.",
        minimum=0.0,
        maximum=1.0,
    )
    advantage_keep_rest: float = setting(
        1.0,
        "Speed, with advantage_keep: the share of the other samples each learning step still uses, drawn at "
        "random each update, only to teach the value side and the exploring bonus; each counts that much "
        "more, so their averages stay right. 1 = every sample (no time saved). 0.2 at keep 0.5: the steps "
        "use about 60% of the samples. Not measured yet.",
        minimum=0.0,
        maximum=1.0,
    )
    value_share: float = setting(
        1.0,
        "Speed: the share of each learning step's samples the value side learns from (a random share); the "
        "policy still learns from all of them. 1 = all. For bots whose value side is most of the learning "
        "work: for the GCO starter, a learning step's arithmetic falls to 61% at 0.5 and 41% at 0.25. "
        "Changes learning; not measured yet.",
        minimum=0.05,
        maximum=1.0,
    )
    goal_critic_weight: float = setting(
        0.0,
        "The goal critic: the value head 'goal' judges moments over a much longer horizon "
        "(goal_critic_half_life), and its opinion of each action is added to the usual one at this share of "
        "its size: 0.25 = a quarter. Needs 'goal' in model.value_heads. 0 = off. Not measured yet.",
        minimum=0.0,
        maximum=10.0,
    )
    goal_critic_half_life: float = setting(
        77.0,
        "With goal_critic_weight: the goal critic's horizon, the seconds after which a future reward counts "
        "half. 77 s = gamma 0.9997 at tick skip 4 (0.9994 at 8). Its advantages use ppo.gae_lambda.",
        minimum=0.01,
    )
    vdag_beta: float = setting(
        0.0,
        "The twin optimistic critics: the value heads 'vdag1' and 'vdag2' learn what the bot could get from "
        "where it is, judged hopefully, and actions that lead toward such places get a push of this share of"
        " the usual opinion's size: 0.30 = 30%. Needs 'vdag1' and 'vdag2' in model.value_heads. 0 = off. "
        "Not measured yet.",
        minimum=0.0,
        maximum=10.0,
    )
    vdag_expectile: float = setting(
        0.75,
        "With vdag_beta: how hopeful the twin critics are, from 0.5 (the average outcome) to near 1 (the "
        "best outcomes). 0.75 as in the design they follow.",
        minimum=0.5,
        maximum=0.99,
    )
    vdag_reinit_steps: float = setting(
        0.0,
        "With vdag_beta: every this many agent steps the twin critics start again from fresh random weights, "
        "so their optimism does not set in. 0 = never. Nobody has published a good interval.",
        minimum=0.0,
    )
    head_loss_scale: Literal["raw", "popart"] = setting(
        "raw",
        "How much the goal critic's and the twin critics' mistakes count, in the layers they share with the "
        "main value. 'raw' (the default): in the rewards' own units, which on a long goal-only run made them"
        " about 20 times louder than the main value's for the same relative mistake. 'popart': in the main "
        "value's units, so they count about as much. Their own outputs and charts are unchanged. Not "
        "measured yet.",
    )
    lean_heads_pass: bool = setting(
        False,
        "Speed, with goal_critic_weight or vdag_beta: the extra critics are worked out only for the cars the"
        " bot learns from (not its past versions in self-play), and with run.fp16_sums 'forward' or 'all' in"
        " the same half precision as learning. The same learning to rounding; an estimated 2-3% faster for a"
        " big GCO bot, not yet timed.",
    )
    optimizer: Literal["adam", "muon"] = setting(
        "adam",
        "The method that moves the networks' weights each learning step. 'adam' (the default) or 'muon': "
        "Muon moves the hidden layers' weight grids in a direction with every part's size evened out, so no "
        "few large directions dominate; everything else (and the first and last layers, and attention "
        "encoders) stays with Adam at the same rates. At one learning rate Muon moves the shared trunk and "
        "the value side 3-5x as far as Adam, the policy about as far. Not measured yet. A bot may "
        "switch on a resume: the new optimizer then starts afresh.",
    )
    muon_batched: bool = setting(
        False,
        "Speed, with optimizer = 'muon': the weight grids of one shape take their Muon steps together, as a "
        "few big jobs on the graphics card instead of about 30 small ones each. The same learning to "
        "rounding. Not measured on a graphics card.",
    )
    muon_lr_scale: float = setting(
        1.0,
        "With optimizer = 'muon': a multiplier on the Muon weight grids' learning rate only (Adam's part "
        "keeps policy_lr and value_lr). 1 = no change. 0.2 makes Muon's steps about as big as Adam's usually"
        " are. For new bots: on a running bot any other value is a large change of rate. Not measured yet.",
        minimum=0.0,
        maximum=10.0,
    )
    weight_decay: float = setting(
        0.0,
        "Shrink the hidden layers' weights a little after every learning step. The network computes the same"
        " (each layer is normalised), but later steps then turn the layers further: without it the weights "
        "grow over a long run (Muon's 10-16x), and each step changes them less and less. 0.1 at a rate of "
        "1.5e-4 holds the weights near 3x their start; 0.7-1 near their start. 0 = off. Needs "
        "model.layer_norm. Not measured yet.",
        minimum=0.0,
        maximum=10.0,
    )
    adam_eps: float = setting(
        1e-5,
        "A tiny number Adam adds so it never divides by 0; larger damps the "
        "steps of weights that rarely move. 1e-5 is usual for this kind of "
        "training (PyTorch's own default is 1e-8).",
        minimum=0.0,
    )
    max_grad_norm: float = setting(
        0.5,
        "The largest step size allowed per network: a learning step whose direction is longer than this"
        " is shortened to it, keeping its direction, so one odd batch cannot throw the network off.",
        minimum=0.0,
    )
    reward_clip: float = setting(
        10.0,
        "Rewards are first rescaled to a steady size; any rescaled reward beyond plus or minus this is"
        " cut to it, so one huge reward cannot swamp a learning step. 0 = no cutting.",
        minimum=0.0,
    )
    normalize_advantages: bool = setting(
        True,
        "Rescale the actions' scores (how much better or worse each turned out than expected) to a steady "
        "size in each learning step, so the step size does not depend on the rewards' size. Measured on ball"
        " touch: off learned clearly slower (2.96 s against 2.16 s), so keep it on.",
    )
    popart: bool = setting(
        True,
        "Let the value network learn rescaled values that it turns back into real ones (PopArt), instead of "
        "the raw sums of rewards, so its job stays the same size as rewards grow. Measured on ball touch: "
        "off made no measurable difference (2.17 s against 2.16 s); it stays on: no task measured a cost, "
        "and it is meant for rewards that grow during a long run (untested).",
    )

    def problems(self) -> list[tuple[str, str]]:
        # 0 would zero every gradient (clipping) or divide 0 by 0 (Adam) and quietly stop learning.
        found = [
            (name, "must be above 0") for name in ("max_grad_norm", "adam_eps") if getattr(self, name) <= 0
        ]
        if self.entropy_target > 0 and self.entropy <= 0:
            found.append(("entropy_target", "needs ppo.entropy above 0 (where the bonus starts)"))
        if self.entropy_target_min <= 0:
            found.append(("entropy_target_min", "must be above 0 (a share of ppo.entropy, at most 10)"))
        found += self._rest_problems()
        if self.head_loss_scale == "popart":
            if not self.popart:
                found.append(("head_loss_scale", "'popart' needs ppo.popart = true (PopArt's statistics)"))
            if self.goal_critic_weight <= 0 and self.vdag_beta <= 0:
                found.append(
                    (
                        "head_loss_scale",
                        "'popart' does nothing without the goal critic or the twins (ppo.goal_critic_weight "
                        "or ppo.vdag_beta above 0)",
                    )
                )
        if self.muon_lr_scale <= 0:
            found.append(("muon_lr_scale", "must be above 0"))
        elif self.muon_lr_scale != 1.0 and self.optimizer != "muon":
            found.append(("muon_lr_scale", "only scales Muon's matrices: needs optimizer = 'muon'"))
        if self.weight_decay * max(self.policy_lr, self.value_lr) >= 1e-3:
            found.append(
                (
                    "weight_decay",
                    "x the larger of policy_lr and value_lr must be below 0.001 (a learning step's shrink)",
                )
            )
        return found

    def _rest_problems(self) -> list[tuple[str, str]]:
        """What advantage_keep_rest below 1 cannot be used with yet, in this section."""
        name = "advantage_keep_rest"
        if self.advantage_keep_rest <= 0:
            return [
                (name, "must be above 0 (the value heads and the entropy need some of the other samples)")
            ]
        if self.advantage_keep_rest >= 1:
            return []
        refused = [
            (
                "needs advantage_keep between 0 and 1 (it runs on the samples that filter leaves out)",
                not 0 < self.advantage_keep < 1,
            ),
            ("needs normalize_advantages = true", not self.normalize_advantages),
            ("does not work with graphed_steps yet: set graphed_steps = false", self.graphed_steps),
            ("does not work with value_share below 1 yet", self.value_share < 1),
        ]
        return [(name, f"below 1 {why}") for why, used in refused if used]

    def signals_on(self) -> list[str]:
        """The GCO learning signals switched on, by setting name."""
        on = {
            "goal_critic_weight": self.goal_critic_weight > 0,
            "vdag_beta": self.vdag_beta > 0,
            "advantage_keep": 0 < self.advantage_keep < 1,
            "entropy_target": self.entropy_target > 0,
        }
        return [name for name, used in on.items() if used]


@dataclass(frozen=True)
class EnvConfig:
    # [env.physics] held physics constants; an empty table is accepted and ignored, any value refused.

    team_size: int = setting(1, "Players per team: 1 = 1v1, 2 = 2v2, 3 = 3v3.", minimum=1, maximum=3)
    team_sizes: tuple[int, ...] = setting(
        (),
        "Several game sizes in one run, e.g. [1, 2, 3]: each arena plays one of these sizes for the whole "
        "run, the sizes spread evenly over the arenas, so every batch holds all of them. Empty = every game "
        "is team_size (the normal way). When set, team_size must be the largest of them, and the observation "
        "needs room for that size: env.obs_team_size at least the largest, or an attention or transformer "
        "encoder. Past versions (self-play) play the Orange cars in every size; one rating covers all sizes. "
        "Not yet with replay starts. Advice: train 1v1 "
        "first (with obs_team_size = 3), then add the bigger sizes on a resume; training every size from the "
        "start spreads the first touches thin (Pulsar's authors saw such a run stall). Not measured yet.",
        minimum=1,
        maximum=3,
    )
    team_size_shares: tuple[float, ...] = setting(
        (),
        "With team_sizes: each size's share of the arenas, in the same order, e.g. [0.5, 0.25, 0.25] (they "
        "need not add up to 1). Empty = equal shares. A 3v3 arena has 6 cars to a 1v1's 2, so equal arena "
        "shares give 3v3 half of the training data; the log says each size's share when the run starts.",
        minimum=0.0,
    )
    shuffle_slots: bool = setting(
        False,
        "With a padded observation (env.obs_team_size): every game, each car takes a random place among its "
        "team's places, so a plain network sees teammates and opponents in every place instead of always the "
        "first ones (an idea from buta-ppo and Pulsar for networks that play several sizes). The games "
        "themselves do not change. Attention and transformer encoders do not need it. Off by default. "
        "Not measured yet.",
    )
    opponents: bool = setting(
        True,
        "false = only the Blue team plays, with nobody to play against. For practice tasks such as "
        "learning to touch the ball or to land on the wheels.",
    )
    num_arenas: int = setting(
        1024,
        "Games simulated at the same time. More uses more processor time and memory. 1024, the default, "
        "was chosen by measurement for 1v1: 2048 is about 5% faster but gives each game only half as many "
        "steps per learning update.",
        minimum=1,
        maximum=65536,
    )
    tick_skip: int = setting(
        4,
        "Physics ticks (1/120 s each) per decision. 4 = 30 decisions per second (the default); 8 = 15.",
        minimum=1,
        maximum=32,
    )
    action_delay: int = setting(
        1,
        "Ticks between choosing an action and the car using it; must be smaller than tick_skip. 1, the "
        "lowest the bot program can keep up with: in a real game a decision then has "
        "about two ticks (16.7 ms) to reach the game. Many bots use tick_skip - 1 (3 with tick_skip 4).",
        minimum=0,
        maximum=31,
    )
    observation: Literal["native", "nitro_v2"] = setting(
        "nitro_v2",
        "What the bot sees: 'nitro_v2' (the default: NitroLearn's own observation plus the relative, "
        "goal-aware and speed features strong bots use, 148 numbers in 1v1) or 'native' (the smaller one, "
        "112 numbers in 1v1). Measured: with nitro_v2 bots scored on 88% of shots against 53%, and did the "
        "same on ball touch and on saves (0.88 against 0.87). Fixed once the bot has saves.",
    )
    obs_team_size: int = setting(
        0,
        "Players per team the observation has room for. 0 = exactly team_size. A larger number (up to 4) "
        "gives the bot an empty, marked place for every car a smaller game lacks, so one bot fits every "
        "game up to that size: a bot trained with 3 here plays 1v1, 2v2 and 3v3 with the same network, and "
        "its training can move from 1v1 to 3v3. Costs a few more numbers per car. With an attention encoder "
        "(model.encoder), 0 means room for team_size players a side, as such an encoder reads each car from "
        "a place of its own; its networks do not depend on the room, so with one this setting and team_size "
        "may change between stop and resume. Not measured yet.",
        minimum=0,
        maximum=4,
    )
    groups: int = setting(
        2,
        "Arenas are split into this many groups so the CPU steps one group while the GPU thinks for "
        "another. 2 is right for almost everyone.",
        minimum=1,
        maximum=16,
    )
    pipelined_inference: bool = setting(
        True,
        "Speed: ask the graphics card for the next group's actions before reading the previous group's, so "
        "the time spent asking overlaps the card's work. The games and the learning are exactly the same; "
        "only speed can change, and only on a graphics card with 2 or more groups. Measured: no faster with "
        "the starters' small networks; with large FP16 networks and self-play, 5% faster for a large 1v1 "
        "network and 2% for a 2v2 attention bot. On by default.",
    )
    concurrent_passes: bool = setting(
        True,
        "Speed: let the graphics card work out the value network's answers and each past version's (in "
        "self-play) at the same time as the policy's, instead of one after another. The games and the "
        "learning are exactly the same; only speed can change. Small networks leave most of a big card idle,"
        " so they overlap well: a big 2v2 transformer trained 1.10x faster; a 1v1 bot with plain networks no"
        " faster (its games hold it up). On by default.",
    )
    card_starts_games: bool = setting(
        True,
        "Speed: on a graphics card, the card itself starts each group's games the moment their actions are "
        "ready, instead of the main program doing it after waiting for the card, which then prepares the "
        "other group meanwhile. The games and the learning are exactly the same. A 1v1 bot with 4x512 "
        "networks trained 1.17x faster with it on a fast card. On by default.",
    )
    threads: int = setting(
        0,
        "CPU threads that run the games. 0 = automatic (all but two of the CPU's threads).",
        minimum=0,
        maximum=256,
    )

    def problems(self) -> list[tuple[str, str]]:
        found: list[tuple[str, str]] = []
        if self.action_delay >= self.tick_skip:
            found.append(("action_delay", f"must be smaller than tick_skip ({self.tick_skip})"))
        if self.num_arenas < self.groups:
            found.append(("num_arenas", f"must be at least groups ({self.groups})"))
        if 0 < self.obs_team_size < self.team_size:
            found.append(("obs_team_size", f"must be 0 or at least team_size ({self.team_size})"))
        if self.team_size_shares and len(self.team_size_shares) != len(self.team_sizes):
            found.append(
                ("team_size_shares", f"needs one share for each of team_sizes ({len(self.team_sizes)} sizes)")
            )
        if any(share <= 0 for share in self.team_size_shares):
            found.append(("team_size_shares", "each share must be above 0"))
        if not self.team_sizes:
            return found
        largest = max(self.team_sizes)
        if len(self.team_sizes) < 2:
            found.append(
                ("team_sizes", "needs two or more game sizes, e.g. [1, 2, 3]; one size: use team_size")
            )
        elif len(set(self.team_sizes)) < len(self.team_sizes):
            found.append(("team_sizes", "names a game size twice"))
        if self.team_size != largest:
            found.append(("team_size", f"must be {largest}, the largest of env.team_sizes"))
        if not self.opponents:
            found.append(("team_sizes", "needs opponents (env.opponents = true)"))
        places = -(-self.num_arenas // self.groups)
        if places < len(self.team_sizes):
            found.append(
                (
                    "num_arenas",
                    f"is too few for {len(self.team_sizes)} game sizes in {self.groups} groups: every group "
                    f"needs at least one arena of each size (at least {len(self.team_sizes) * self.groups})",
                )
            )
        return found

    @property
    def mixed_sizes(self) -> bool:
        """Several game sizes in one run."""
        return bool(self.team_sizes)

    def one_size(self) -> EnvConfig:
        """These games at team_size alone, cars in order: for a look at the bot (watch, eval, skills), which
        plays one game size."""
        return dataclasses.replace(self, team_sizes=(), team_size_shares=(), shuffle_slots=False)

    @property
    def sizes_label(self) -> str:
        """The game sizes played, as "1v1" or "1v1, 2v2 and 3v3"."""
        names = [f"{n}v{n}" for n in sorted(self.team_sizes or (self.team_size,))]
        return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


@dataclass(frozen=True)
class TerminalsConfig:
    no_touch_seconds: float = setting(
        10.0,
        "Restart a game after this many seconds without anyone touching the ball. 0 = never.",
        minimum=0.0,
    )
    max_episode_seconds: float = setting(
        0.0, "Restart a game after this many seconds no matter what. 0 = never.", minimum=0.0
    )
    end_on_touch: bool = setting(
        False, "Practice tasks: the episode is over (task done) as soon as a car touches the ball."
    )
    end_on_landing: bool = setting(
        False, "Practice tasks: the episode is over (task done) once every car is on its wheels."
    )
    truncate_on_ball_landing: float = setting(
        0.0,
        "Chance (0 to 1) that an episode is cut off (not done) when the ball comes down to the floor after "
        "being in the air. 1 in the aerial-touch task, so only an aerial touch can finish it; a small chance "
        "such as 0.1 ends match episodes at a natural pause now and then, so more episodes start from the "
        "start states. 0 = never. Not measured yet.",
        minimum=0.0,
        maximum=1.0,
    )


@dataclass(frozen=True)
class SelfPlayConfig:
    """Training against frozen past versions of the bot."""

    enabled: bool = setting(
        False,
        "Also play against saved past versions of the bot, not only against its current self. Helps "
        "against forgetting how to beat earlier play styles. Not measured yet (only full games could show "
        "it), "
        "so off until you try it.",
    )
    past_share: float = setting(
        0.2,
        "Share of the games whose Orange team is a past version (the rest are the current bot against "
        "itself). 0.2 = one game in five.",
        minimum=0.01,
        maximum=0.9,
    )
    active_opponents: int = setting(
        4,
        "How many different past versions play at the same time; each gets an equal share of those games. "
        "Changed at every save.",
        minimum=1,
        maximum=16,
    )
    snapshot_steps: int = setting(
        20_000_000,
        "Save a copy of the bot into the pool of past versions every this many agent steps.",
        minimum=1_000_000,
    )
    pool_size: int = setting(
        64, "Most past versions kept; the ones rated closest to their neighbours go first.", minimum=2
    )
    pool_disk_mb: int = setting(2048, "Most disk space (MB) the pool may use.", minimum=16)
    matchmaking: Literal["variance", "uniform", "hardest", "squared", "linear_capped"] = setting(
        "variance",
        "How past versions are picked to play: 'variance' favours even matches (a chance of winning near "
        "50%), 'uniform' picks any, 'hardest' favours the ones the bot loses to, 'squared' favours those "
        "much more, 'linear_capped' favours them but treats every version the bot beats at most half the "
        "time alike. The last two are not measured yet.",
    )
    flat_rating_steps: int = setting(
        500_000_000,
        "Warn when the bot's rating against its past versions has not risen for this many agent steps "
        "(0 = never warn).",
        minimum=0,
    )


@dataclass(frozen=True)
class StartFromConfig:
    """Transfer learning: a new run starts from another run's trained networks."""

    enabled: bool = setting(
        True,
        "Transfer learning on or off. On, the bot's first start takes the networks of the bot named in run "
        "(nothing is taken while run is empty); off, it starts from scratch and run is kept for later. Off "
        "also wins over a bot named by nitro train --from. Read only at the first start: after its first "
        "save a bot carries on from its own networks.",
    )
    run: str = setting(
        "",
        "The bot to start from: its name (its folder in the runs folder), or the path of a run folder or of "
        "one of its saves. Empty = from scratch. Only a new run reads it; a "
        "resumed run continues from its own save. The other run must have the same networks (sizes, "
        "layer_norm, activation) and observation; with env.obs_team_size the same in both, its team size "
        "may differ (a 1v1 bot can start a 3v3 run).",
    )
    at: int = setting(
        0,
        "Which save of that bot, by its agent steps (a folder name in its checkpoints); 0 = its newest when "
        "this bot starts. New bot from this one writes the newest save's steps, so the copy starts from "
        "that save even if the other bot trains on.",
        minimum=0,
    )
    parts: Literal["all", "policy", "matching"] = setting(
        "all",
        "'all' takes the policy, the value network and the value and reward scaling: for going on with the "
        "same rewards (a new game size, new start states, other learning settings). 'policy' takes only the "
        "policy (and a shared trunk) and starts the value network afresh: for new rewards, whose values the "
        "old value network does not know (the dashboard switches to it when a not-yet-started bot's rewards "
        "change). 'matching' takes the policy and shared trunk whole (their sizes must be the same), every "
        "part of the value side whose layers are the same size (the critic trunk, any value head of the "
        "same layers), the value and reward scaling as 'all' does, and starts the rest afresh: the same bot "
        "with another value side, e.g. a lighter critic or fewer value heads.",
    )

    @property
    def source(self) -> str:
        """The bot this run starts from: `run` while transfer learning is on, else "" (from scratch)."""
        return self.run if self.enabled else ""


@dataclass(frozen=True)
class PretrainConfig:
    """Copying your own replays before training (off by default)."""

    enabled: bool = setting(
        False,
        "Before a new bot trains, copy how the players in your own Rocket League replays play: the replays "
        "of its team size are converted (once; kept in the project's replays folder), the policy learns to "
        "choose as the players did, and the copy is saved as the bot's first save, without training (try "
        "rewards on it with nitro reward-check); the next start trains from it, with a fresh value network. "
        "Only a new bot reads it. Shown to learn the players' choices; whether it makes a better bot than "
        "training from scratch is not measured yet.",
    )
    replays: tuple[Path, ...] = setting(
        (),
        "A folder of .replay files, or a list of folders (such as one per game size). Left out = the game's "
        "own replay folder (Documents\\My Games\\Rocket League\\TAGame\\DemosEpic for Epic, Demos for "
        "Steam). With batch_replays, several folders are taken in turn, each newest first, so every batch "
        "mixes them. Relative paths are relative to the settings file.",
    )
    team_sizes: tuple[int, ...] = setting(
        (),
        "With batch_replays: the game sizes to copy, e.g. [1, 2, 3] for one bot of every size; each must fit "
        "the observation's room (env.obs_team_size). Empty = this run's team size only.",
        minimum=1,
        maximum=4,
    )
    limit: int = setting(
        30,
        "Copy at most this many replays (the newest of this run's team size, or of team_sizes, in all). "
        "More copy better but take longer to convert. Without batch_replays every one is held in memory at "
        "once, about 300 to 400 replays at most on 32 GB, and at most five times this many are converted "
        "per start, so a rare team size does not take hours; with batch_replays, any number (10,000 and "
        "more), converted until enough are found.",
        minimum=1,
    )
    passes: int = setting(
        10,
        "Passes over the replays while copying (as nitro pretrain --epochs). With batch_replays: passes over "
        "each round's replays (the newest batch and the ones kept with it); 1 or 2 is plenty there.",
        minimum=1,
        maximum=100,
    )
    batch_replays: int = setting(
        0,
        "0 = every replay is converted, then all are learned from at once (up to a few hundred). More = the "
        "large replay mode: replays are converted this many at a time, several at once, "
        "into a compact form held in memory only (no recordings are written; per 5-minute replay about 2 MB "
        "for 1v1, 6 MB for 2v2, 13 MB for 3v3 with the standard observation, about 12 MB on average for a "
        "nitro_v2 one with room for 3 a side), each batch is learned from and then dropped, so memory and "
        "disk stay small however many replays limit asks for; at most about 4 batches are in memory at "
        "once. Progress is saved after every batch: stopped (or the PC turned off), `nitro resume` carries "
        "on where it was. 200 suits 32 GB for 2v2.",
        minimum=0,
        maximum=2000,
    )
    keep_batches: int = setting(
        2,
        "With batch_replays: how many batches are learned from together, the newest and the ones before it, "
        "so each replay is learned from in this many rounds; memory holds this many batches.",
        minimum=1,
        maximum=10,
    )
    workers: int = setting(
        0,
        "With batch_replays: replays converted at once. 0 = half the threads of this computer (at most 16; "
        "10 on a 10-core, 20-thread PC, where it was faster than 18).",
        minimum=0,
        maximum=64,
    )
    converter: Literal["nitro", "replicar"] = setting(
        "nitro",
        "With batch_replays: what converts the replays. nitro = NitroLearn's own build of replicar "
        "(crates/nitro_replicar, built on first use): the same states and controls as replicar's tool, so "
        "the same copy, but it writes only what the copy reads instead of a compressed file of every column; "
        "measured: 14% more replays a minute, the same bit for bit on 20 of 20. "
        "replicar = replicar's own tool.",
    )
    keep_converted: bool = setting(
        True,
        "With batch_replays and converter = nitro: keep every replay converted (any game size) in the "
        "project's replays\\store folder, so no later copy, by this bot or another, converts it again; "
        "a copy uses the kept ones first. About 8-10 MB a 5-minute 2v2 replay; nothing is kept "
        "while the disk has under 20 GB free.",
    )
    continue_from: str = setting(
        "",
        "With batch_replays: carry on another run's copy instead of starting from a fresh network: the name "
        "of a run in the same runs folder whose copy of the replays (its pretrained folder, made before any "
        "training) this copy starts from; its networks and settings must match this run's. The replays here "
        "are learned on top. Empty = a fresh network.",
    )
    new_only: bool = setting(
        False,
        "With batch_replays: leave out every replay converted on this PC before, by a copy or the converter "
        "(the project's replays\\sizes.txt notes each one), so carrying on a copy learns from new replays "
        "only.",
    )
    value_warmup_steps: int = setting(
        0,
        "Agent steps at the start of the run during which only the fresh value network learns and the "
        "copied policy is held as it came, so the value network knows what situations are worth before it "
        "steers the policy. 0 = none. With a shared trunk (model.shared), the trunk is held too and only "
        "the value network's own layers learn. It can change while the bot trains (the Overview "
        "shows how far it is): the steps already trained count, so a lower number frees the policy sooner, "
        "and one above them holds it still again until then.",
        minimum=0,
    )
    lr: float = setting(
        3e-4,
        "The copy's learning rate (Adam). 3e-4 is what every copy so far used. Bigger networks usually want "
        "less: 1e-4 to 2e-4 for 10M+ parameters.",
        minimum=1e-7,
        maximum=1e-2,
    )
    lr_end_share: float = setting(
        1.0,
        "The copy's learning rate at its end, as a share of lr, reached along a cosine curve over the "
        "replays learned (limit), so the last replays fine-tune what the first ones taught. 1 = the same "
        "rate throughout; 0.1 = a tenth at the end. In a small check (6 replays) the "
        "falling rate copied worse than a fixed one; not measured for long copies.",
        minimum=0.0,
        maximum=1.0,
    )
    prev_action_dropout: float = setting(
        0.0,
        "The share of each learning step's decisions whose previous action (part of the observation) is "
        "blanked while copying, so the copy learns to choose from the game, not only from what it did last"
        ". A copy leans hard on its previous action: repeating it matches the players on about 71% of "
        "decisions, and one copy of a big 2v2 bot, always taking its likeliest action, kept one "
        "action 117 decisions in a row. In play the bot always sees its real previous action. 0 = never; "
        "0.3-0.5 is what the research used. Not measured yet: judged only by the copy check.",
        minimum=0.0,
        maximum=0.9,
    )
    minibatch: int = setting(
        4096,
        "Decisions per learning step of the copy. Bigger is faster on a big graphics card (fewer, larger "
        "steps); with it, a somewhat larger lr is usual.",
        minimum=16,
        maximum=262_144,
    )
    measure_every: int = setting(
        1,
        "With batch_replays: measure the held-back replays every this many rounds (and after the last one). "
        "Measuring 50 replays after each round of 100 costs about a sixth of the copy's learning time; 10 "
        "saves nearly all of it. 1 = every round.",
        minimum=1,
        maximum=1000,
    )
    skill_every: int = setting(
        0,
        "With batch_replays: every this many rounds (and after the last), the copy so far plays earlier "
        "versions of itself, the copy where it started and the one of the last check, from kickoffs in the "
        "simulator, and gets a rating from the goals: how good it is getting, not only how closely it "
        "copies. Needs a game with opponents. 0 = never. A check takes about a minute on a PC; 10 "
        "with 100 replays a round checks about every 12 minutes there.",
        minimum=0,
    )
    skill_points: int = setting(
        200,
        "With skill_every: points played against each earlier version in a check; each runs from a kickoff "
        "to a goal, or ends without one (a draw) after the bot's no-touch time or 60 seconds. More points "
        "give a surer rating and take longer.",
        minimum=10,
        maximum=5000,
    )
    save_every: int = setting(
        2000,
        "With batch_replays: every this many replays learned, the copy as it is so far is saved as the bot's "
        "save (at 0 agent steps), so it can be watched, made into an RLBot bot and have rewards tried on it "
        "while the copy goes on. Starting the bot again carries on the copy; training starts from the "
        "finished copy only. A save takes about as long as a training save; the copy waits for it. 0 = only "
        "the finished copy is saved.",
        minimum=0,
    )


@dataclass(frozen=True)
class ReplayMixConfig:
    """Copying converted replays while the bot trains (off by default)."""

    enabled: bool = setting(
        False,
        "While the bot trains, mix in learning steps that copy the players in your converted replays (the "
        "replay copy's learning) with a learning rate of their own, so a trained bot keeps "
        "learning from real play: after each update, share of the steps that move the policy are copy "
        "steps. They move the policy and any shared trunk, never the value network, with an optimizer of "
        "their own kept in the saves. It can be switched on or off at any time, also for a bot that has "
        "trained for billions of steps: nothing else is reset. Needs replays converted into the project's "
        "replays\\store (nitro convert-replays, or the Converted replays card); without any of the bot's "
        "game size it says so at the first iteration and at every save, keeps it in the report and trains "
        "as usual. Whether it makes a better bot is not measured yet.",
    )
    share: float = setting(
        0.2,
        "The copy steps' share of all the learning steps that move the policy: 0.2 = one in five (after an "
        "update of 8 policy steps, 2 copy steps). 0 = none, the same as enabled = false. Each copy step "
        "takes minibatch decisions, so with the presets (50,000 steps an update, 8 policy steps of 12,500) "
        "0.2 copies about 8,000 decisions an update, roughly one 5-minute 1v1 replay.",
        minimum=0.0,
        maximum=0.9,
    )
    lr: float = setting(
        3e-5,
        "The copy steps' learning rate (Adam, their own), kept below the training rate (ppo.policy_lr: "
        "4.5e-4 by default, 1e-4 to 2e-4 for big bots) so the copying nudges the trained policy "
        "instead of pulling it back to a plain copy. 3e-5 is a tenth of the replay copy's rate (pretrain.lr "
        "3e-4) and under a third of the lowest usual training rate. Not measured yet.",
        minimum=1e-8,
        maximum=1e-2,
    )
    minibatch: int = setting(
        4096,
        "Decisions per copy step, drawn at random from 16 replays at a time, as pretrain.minibatch.",
        minimum=16,
        maximum=262_144,
    )
    replays: tuple[Path, ...] = setting(
        (),
        "Folders of converted replays (.nrpl.zst files, as the project's replays\\store keeps them; "
        "subfolders included). Empty = the store's folder of this bot's game size (of each of "
        "pretrain.team_sizes when set). The replays are taken in a random order fixed by run.seed, each "
        "once a cycle, and the cycle starts again when all are used. Relative paths are relative to the "
        "settings file.",
    )
    kickoff_only_seconds: float = setting(
        0.0,
        "Above 0, the copy steps learn only from the replays' kickoffs: from each kickoff to this many "
        "seconds of play after the ball's first touch (a kickoff nothing touches within 10 s ends there), "
        "and nothing else in the replays. 3 = the kickoff and the 3 s after the first touch. 0 = "
        "every decision of the replays (off). The console's Replays line and metrics.csv "
        "(replay_mix_rows_kept, replay_mix_rows_left_out) show how much of the replays it keeps.",
        minimum=0.0,
        maximum=30.0,
    )

    @property
    def active(self) -> bool:
        """Whether copy steps are taken: enabled with a share above 0."""
        return self.enabled and self.share > 0


@dataclass(frozen=True)
class WandbConfig:
    """Weights & Biases tracking."""

    enabled: bool = setting(
        False,
        "Also send the run's numbers (every metrics.csv column, plus all settings and this PC's processor "
        "and graphics-card use) to Weights & Biases (wandb.ai): charts in the browser, from any device, and "
        "runs side by side. Needs a free W&B account; log in once with `wandb login` (your key stays with "
        "W&B on this PC, never in the settings). Training is the same with or without it. While a run sends,"
        " W&B's helper program talks to it through a port on this PC that only this PC can reach.",
    )
    project: str = setting("nitrolearn", "The W&B project the run goes into.")
    entity: str = setting("", "Your W&B user or team name; empty = your default.")
    name: str = setting("", "The run's name on W&B; empty = the run folder's name.")
    group: str = setting("", "Optional W&B group, to show related runs together; empty = none.")
    tags: tuple[str, ...] = setting((), 'Optional labels for the run on W&B, e.g. ["1v1", "nitro_v2"].')
    mode: Literal["online", "offline"] = setting(
        "online",
        "'online' sends while training; 'offline' only writes W&B's files into the run folder, to upload "
        "later with `wandb sync`.",
    )

    def problems(self) -> list[tuple[str, str]]:
        return [("project", "needs a name")] if self.enabled and not self.project.strip() else []


@dataclass(frozen=True)
class NitroConfig:
    paths: PathsConfig = section(PathsConfig, "Where NitroLearn finds files on this PC.")
    run: RunConfig = section(RunConfig, "Settings for the run as a whole.")
    env: EnvConfig = section(EnvConfig, "The simulated games.")
    terminals: TerminalsConfig = section(
        TerminalsConfig, "When a game is restarted. A goal always restarts it."
    )
    rewards: tuple[Reward, ...] = setting(
        DEFAULT_REWARDS, "What the bot is rewarded for; see nitro/config/rewards.py."
    )
    state_setters: tuple[StateSetter, ...] = setting(
        DEFAULT_STATE_SETTERS, "How games start; see nitro/config/state_setters.py."
    )
    model: ModelConfig = section(ModelConfig, "The size and shape of the networks.")
    ppo: PPOConfig = section(PPOConfig, "How the bot learns from the games it plays.")
    selfplay: SelfPlayConfig = section(SelfPlayConfig, "Playing against past versions of the bot.")
    start_from: StartFromConfig = section(
        StartFromConfig, "Starting a new run from another run's trained networks (transfer learning)."
    )
    pretrain: PretrainConfig = section(
        PretrainConfig, "Copying your own replays before training (off by default)."
    )
    replay_mix: ReplayMixConfig = section(
        ReplayMixConfig,
        "Copying converted replays while the bot trains, in a share of its learning steps (off by default).",
    )
    wandb: WandbConfig = section(
        WandbConfig, "Sending the run's numbers to Weights & Biases (off by default)."
    )

    @property
    def active_rewards(self) -> tuple[Reward, ...]:
        """The rewards a run pays and measures: every one not switched off with `enabled = false`."""
        return tuple(reward for reward in self.rewards if reward.enabled)

    def advice(self) -> list[str]:
        """Settings that are allowed but usually a mistake, in plain English (shown by `nitro check-settings`
        and when a run starts)."""
        notes = []
        if not any(reward.enabled and reward.weight != 0 for reward in self.rewards):
            notes.append(
                "No reward is switched on, so the bot has nothing to learn from: switch a reward on (or give "
                "one a weight other than 0)."
            )
        if self.model.uses_attention and self.env.team_size == 1:
            notes.append(
                f"model.encoder = '{self.model.encoder}' with team_size = 1: attention is for 2v2 and 3v3 "
                "bots; a 1v1 bot learns better with the plain network. Fine if this run is a 1v1 "
                "stage of a bot that will play 2v2 or 3v3 (its team size can change on a resume)."
            )
        if self.model.attention_fast and self.model.encoder in ("attention_communication", "transformer"):
            notes.append(
                f"model.attention_fast does nothing with encoder = '{self.model.encoder}' (its tokens "
                "change every block); it applies to encoder = 'attention' only."
            )
        if self.model.large and self.run.precision == "fp32":
            notes.append(
                "Networks this big train about twice as fast on the graphics card with run.precision = "
                "'fp16' (and model.fused_norm = true), and learned as well as FP32 in every check. "
                "Precision is fixed once a bot has saves: for a bot already training, make a new bot "
                "from it."
            )
        if self.replay_mix.active and self.replay_mix.lr >= self.ppo.policy_lr:
            notes.append(
                f"replay_mix.lr ({self.replay_mix.lr:g}) is not below ppo.policy_lr "
                f"({self.ppo.policy_lr:g}): the copy steps are meant to nudge the trained policy, so their "
                "rate is usually well below the training rate (3e-5 by default)."
            )
        return notes

    def _goal_ends_problems(self) -> list[tuple[str, str]]:
        """What ppo.goal_ends = 'continue' cannot be used with."""
        found = []
        ends = [name for name in ("end_on_touch", "end_on_landing") if getattr(self.terminals, name)]
        if ends:
            named = " and ".join(f"terminals.{name}" for name in ends)
            found.append(
                (
                    "ppo.goal_ends",
                    f"'continue' cannot be used with {named}: these end games for reasons the learner cannot "
                    "tell from a goal",
                )
            )
        signals = [name for name in ("goal_critic_weight", "vdag_beta") if getattr(self.ppo, name) > 0]
        if signals:
            named = " and ".join(f"ppo.{name}" for name in signals)
            found.append(
                (
                    "ppo.goal_ends",
                    f"'continue' cannot be used with {named}: their targets keep goals as ends; in zero-sum "
                    "runs it changes nothing",
                )
            )
        return found

    def _mixed_size_problems(self) -> list[tuple[str, str]]:
        """What several game sizes in one run (env.team_sizes) need, and what they cannot be used with
        yet; and env.shuffle_slots's padded observation."""
        env = self.env
        room = env.obs_team_size or (env.team_size if self.model.uses_attention else 0)
        found = []
        if env.shuffle_slots and room == 0:
            found.append(("env.shuffle_slots", "needs a padded observation: set env.obs_team_size"))
        if not env.team_sizes:
            return found
        largest = max(env.team_sizes)
        if room < largest:
            found.append(
                (
                    "env.team_sizes",
                    f"needs an observation with room for {largest}v{largest}: set env.obs_team_size = "
                    f"{largest} (or more), or use an attention or transformer encoder (model.encoder)",
                )
            )
        if any(setter.name == "replay" for setter in self.state_setters):
            found.append(
                (
                    "state_setters",
                    "cannot be used with several game sizes (env.team_sizes) yet: replay start states (a "
                    "replay state setter)",
                )
            )
        return found

    def problems(self) -> list[tuple[str, str]]:
        if not self.state_setters:
            return [("state_setters", "needs at least one state setter")]
        if sum(setter.weight for setter in self.state_setters) <= 0:
            return [("state_setters", "at least one state setter needs a weight above 0")]
        found = []
        if self.run.fp16_sums != "off" and self.run.precision != "fp16":
            found.append(("run.fp16_sums", "needs run.precision = 'fp16'"))
        if self.rewards and not self.active_rewards:
            found.append(("rewards", "every reward is switched off (enabled = false): switch one on"))
        if self.pretrain.enabled and self.start_from.source:
            found.append(
                (
                    "pretrain.enabled",
                    "cannot be used together with start_from.run (transfer learning, start_from.enabled): "
                    "both choose what the run starts from",
                )
            )
        if self.pretrain.skill_every and not self.env.opponents:
            found.append(("pretrain.skill_every", "needs a game with opponents (env.opponents) to play"))
        for name in ("continue_from", "new_only", "skill_every"):
            if getattr(self.pretrain, name) and not self.pretrain.batch_replays:
                found.append((f"pretrain.{name}", "needs pretrain.batch_replays (the large replay mode)"))
        if any(c in self.pretrain.continue_from for c in "/\\:"):
            found.append(("pretrain.continue_from", "is the name of a run in the runs folder, not a path"))
        if self.pretrain.team_sizes:
            room = self.env.obs_team_size
            if not self.pretrain.batch_replays:
                found.append(("pretrain.team_sizes", "needs pretrain.batch_replays (the large replay mode)"))
            elif room == 0 and set(self.pretrain.team_sizes) != {self.env.team_size}:
                found.append(
                    (
                        "pretrain.team_sizes",
                        "other game sizes than this run's need a padded observation: set env.obs_team_size "
                        "to the largest",
                    )
                )
            elif room and max(self.pretrain.team_sizes) > room:
                found.append(
                    (
                        "pretrain.team_sizes",
                        f"{max(self.pretrain.team_sizes)} a side does not fit the observation's room for "
                        f"{room} (env.obs_team_size)",
                    )
                )
        if self.model.fused_norm:
            if not self.model.layer_norm:
                found.append(("model.fused_norm", "needs model.layer_norm = true (it fuses the LayerNorm)"))
            if self.model.activation not in ("relu", "leaky_relu"):
                found.append(
                    (
                        "model.fused_norm",
                        'needs model.activation = "relu" or "leaky_relu" (it fuses the activation)',
                    )
                )
        if self.ppo.weight_decay > 0 and not self.model.layer_norm:
            found.append(
                (
                    "ppo.weight_decay",
                    "needs model.layer_norm = true (only a layer feeding a LayerNorm shrinks unchanged)",
                )
            )
        if self.model.residual and not self.model.layer_norm:
            found.append(
                ("model.residual", "needs model.layer_norm = true (each block normalises its layer)")
            )
        if self.model.value_heads[:1] != ("main",):
            found.append(("model.value_heads", 'must start with "main", the value the bot learns from'))
        if len(set(self.model.value_heads)) < len(self.model.value_heads):
            found.append(("model.value_heads", "names a head twice"))
        if self.ppo.goal_critic_weight > 0 and "goal" not in self.model.value_heads:
            found.append(("ppo.goal_critic_weight", 'needs "goal" in model.value_heads'))
        if self.ppo.vdag_beta > 0 and not {"vdag1", "vdag2"} <= set(self.model.value_heads):
            found.append(("ppo.vdag_beta", 'needs "vdag1" and "vdag2" in model.value_heads'))
        if self.ppo.goal_ends == "continue":
            found += self._goal_ends_problems()
        if self.ppo.advantage_keep_rest < 1 and self.run.cards > 1:
            found.append(
                ("ppo.advantage_keep_rest", "below 1 does not work with several cards (run.cards) yet")
            )
        if self.model.uses_attention:
            if self.model.encoder == "transformer":
                if self.model.transformer_width % self.model.transformer_heads:
                    found.append(("model.transformer_heads", "must divide model.transformer_width"))
            elif self.model.attention_width % self.model.attention_heads:
                found.append(("model.attention_heads", "must divide model.attention_width"))
        if self.selfplay.enabled and not self.env.opponents:
            found.append(("selfplay.enabled", "needs an opposing team (env.opponents = true)"))
        found += self._mixed_size_problems()
        if self.selfplay.enabled:
            past = round(self.env.num_arenas * self.selfplay.past_share)
            needed = self.env.groups * self.selfplay.active_opponents
            if past < needed:
                found.append(
                    (
                        "selfplay.past_share",
                        f"gives {past} games to past versions but {needed} are needed (env.groups x "
                        "selfplay.active_opponents, so every past version plays in every group); raise "
                        "past_share or env.num_arenas, or lower active_opponents",
                    )
                )
        return found
