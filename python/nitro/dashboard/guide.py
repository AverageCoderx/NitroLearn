# ruff: noqa: E501  (long explanation texts, kept one paragraph line each)
"""The dashboard's "See more" explanations: what a setting does, when and how to change it, what to watch
out for and what was measured. Keys are setting paths; "reward.<option>" and "start.<kind>" are the Rewards
and Starts tabs'. Each part starts with its label; "- " lines are lists."""

from __future__ import annotations

MORE: dict[str, str] = {
    "paths.collision_meshes": (
        "What it does: Tells Nitro where the Rocket League arena shapes are on this PC. The simulator needs them to know where the floor, walls, ramps and goals are. Without them no game can be built, so nothing trains.\n"
        "\n"
        "When to change it: Only when the meshes live somewhere else. Left out (as in the presets), it is NitroLearn's own collision_meshes folder, or a collision_meshes folder next to the settings file when there is one, so a settings file works wherever you copy it. nitro check-settings says when the folder it points at has no meshes.\n"
        "\n"
        'How to use it: Point it at the folder that holds a "soccar" subfolder, not at the soccar folder itself. A relative path counts from the settings file\'s folder; a run remembers where its original file was, so its own copy keeps the same meaning. The installer downloads them for you (nitro fetch-meshes does it again any time). Run nitro doctor to check them: it counts the soccar meshes it recognises and names the fix if any are missing.\n'
        "\n"
        "Watch out for: The meshes come from RLGym's public package on PyPI and are checked by their fingerprints; meshes dumped from your own game work too. Files that are not genuine soccar meshes are flagged by nitro doctor. On a machine without meshes the tests that need them are skipped and counted, never passed."
    ),
    "run.seed": (
        "What it does: The starting number for every random choice in the run: how each arena's games start, the network's starting weights, the order the learner works through its data, and the random picks of actions. Every arena has its own random generator built from it, so the number of CPU threads never changes the games.\n"
        "\n"
        "When to change it: To train the same settings again and see a different version of the bot. Two runs that differ only in their seed show how much of a result is luck.\n"
        "\n"
        "How to use it: Any whole number from 0 up. The presets use 0. nitro ab already runs seeds 0 to 9 for each side of a comparison, so you do not set seeds there.\n"
        "\n"
        "Watch out for: The same seed alone gives the same games, but bit-for-bit identical learning on the graphics card also needs run.deterministic = true. One seed is one sample: a run that looks better on one seed may simply be lucky, which is why comparisons here always use ten."
    ),
    "run.device": (
        'What it does: Picks where the networks run, both while playing and while learning. "auto" uses the graphics card when PyTorch can see one and falls back to the processor otherwise. "cuda" asks for the graphics card; "cpu" uses the processor.\n'
        "\n"
        'When to change it: Almost never. "cpu" is for a machine without a suitable graphics card, or for a quick test that must not touch the card while something else uses it.\n'
        "\n"
        'How to use it: Leave "auto". Run nitro doctor first on a new machine: it reports whether PyTorch can use the graphics card.\n'
        "\n"
        "Watch out for: Training on the processor works but is many times slower; every speed figure in these explanations was measured on a graphics card. Some speed options only act on the graphics card: fp16 precision, model.fused_norm (falls back with a warning on the CPU) and env.pipelined_inference (nothing to overlap on the CPU). The games themselves always run on the processor (run.threads, env.threads)."
    ),
    "run.cards": (
        "What it does: Trains this one bot on several graphics cards at once. Each card plays its share of the games (env.num_arenas and env.threads are split between the cards) and learns on its share; every learning step the cards average what they learned, so the bot learns exactly as it would on one card from the whole batch, only faster. Card 0 writes the saves, metrics and report; the other cards' logs are in the bot's cards folder.\n"
        "\n"
        "When to change it: On a Linux machine with two or more cards, after the speed test there showed it is faster (benchmarks/profile/cards_speed.py). Most useful for big networks whose time goes to the card; a bot held back by the processor gains less.\n"
        "\n"
        "How to use it: Set it to the number of cards, e.g. 2. env.num_arenas and ppo.steps_per_iteration must divide by it. It can change between stop and resume.\n"
        "\n"
        "Watch out for: Linux only: the cards add up their learning through NVIDIA's NCCL library, which Windows does not have. [replay_mix] works, each card copying its share. The metrics' game columns and self-play ratings count card 0's games only. Running out of card memory stops the run and suggests ppo.microbatch. The cards' processes talk to each other only inside this computer (127.0.0.1), and only while training.\n"
        "\n"
        "Measured: two cards learning from the two halves of a batch end with the same networks as one card learning from the whole batch (tests/test_multigpu.py). Speed with two cards against one: about 1.55 times as fast for 1v1 bots whose networks keep the cards busy (10 and 51 million weights in FP16); about 1.7 times for a large 2v2 transformer; 1.23 times for a small, fast bot with run.cards_graphed_steps on (without it, two cards were slower than one); no faster for a bot whose games hold training up."
    ),
    "run.cards_sum_precision": (
        'What it does: With run.cards above 1, chooses how the cards\' gradients travel when the cards add them up every learning step. "fp32" sends them exactly as computed. "bf16" sends them at half the size, each rounded to about 3 significant digits.\n'
        "\n"
        "When to change it: Only where the cards' sums are a large part of each learning step: a slow link between the cards (no NVLink, an older PCIe) and a big network. On our two-card test machine a 51M-weight bot's 8 sums an iteration took about 0.24 s of its 1.42 s of learning.\n"
        "\n"
        'How to use it: Leave "fp32". Try "bf16" with the speed test (benchmarks/profile/cards_speed.py) on the machine; keep it only if it is clearly faster.\n'
        "\n"
        "Watch out for: Not yet measured for learning: the rounding changes each step a little. The measured losses and the counts of samples always travel in full precision. It also applies to graphed steps (run.cards_graphed_steps). Has no effect with run.cards = 1.\n"
        "\n"
        "Measured: in tests/test_multigpu.py two processes summing in bf16 end within 0.1% of one learning from the whole batch over two updates. Speed: not measured yet."
    ),
    "run.cards_pin_threads": (
        "What it does: With run.cards on Linux, gives each card's process its own share of the processor: whole groups of cores that share a memory cache. That card's games and its share of the work then run only there, instead of every card's work moving over every core.\n"
        "\n"
        "When to change it: On a machine with several cards and many cores (a rented machine with two big cards and a server processor, say).\n"
        "\n"
        "How to use it: Set it to true next to run.cards. Each card's games use about its share of env.threads.\n"
        "\n"
        "Watch out for: Has no effect with one card or on Windows. Speed only; the games and the learning stay the same.\n"
        "\n"
        "Measured: not yet."
    ),
    "run.cards_check_every": (
        "What it does: With run.cards above 1, sets how often the cards check that their networks are still exactly the same, in iterations. The check makes every card wait for the slowest one. 1, the default, checks after every iteration. A higher number checks after every that many iterations, and always right before a save, so a save never holds networks that drifted apart.\n"
        "\n"
        "When to change it: Only on a machine with several cards, after the speed test there (benchmarks/profile/cards_speed.py) showed it is faster. The cards have never drifted apart; the check is a safety net.\n"
        "\n"
        "How to use it: Leave 1. Try 10 with the speed test; keep it only if it is clearly faster.\n"
        "\n"
        "Watch out for: A drift, if it ever happened, would be found up to that many iterations later (still before the next save), and the run stops with the same message as now. Has no effect with one card. Speed only; the games and the learning stay the same.\n"
        "\n"
        "Measured: not yet."
    ),
    "run.cards_graphed_steps": (
        "What it does: With run.cards above 1 and ppo.graphed_steps on, each learning step replays as one recorded CUDA graph on every card, the cards' gradient sums recorded inside it, as one card's steps do. Without it, several cards learn step by step: for small and medium networks most of a step's time is then Python handing the card its small jobs, which a second card does not shorten.\n"
        "\n"
        "When to change it: On a machine with two or more cards, for any bot whose learning is not dominated by one huge network. Turn it on once the card checks pass on that machine (tests/test_multigpu.py), then compare speeds with benchmarks/profile/cards_speed.py.\n"
        "\n"
        "How to use it: Set it to true next to run.cards. Nothing else changes.\n"
        "\n"
        "Watch out for: Has no effect with run.cards = 1 or ppo.graphed_steps off.\n"
        "\n"
        "Measured: a first version (the cards' sums inside the graphs) hung on two cards; this one keeps the sums outside the graphs. Not checked on two cards."
    ),
    "run.precision": (
        'What it does: Chooses the number format for the networks\' hidden layers on the graphics card. "fp32" is full precision everywhere. "fp16" runs the hidden layers in half precision, while the sensitive parts of learning (output layers, probability ratios, losses) stay in full precision and a loss scaler guards against overflow.\n'
        "\n"
        "When to change it: For large networks, where it saves real time. With the presets' [256, 256, 256] networks it gains nothing.\n"
        "\n"
        'How to use it: Keep "fp32" for normal-sized networks. For large ones (two layers 1024 wide in a row or bigger) "fp16" is worth it, together with model.fused_norm. The New bot form picks "fp16" and fused_norm by itself when you choose networks that big, and the settings check suggests it for a big bot in "fp32".\n'
        "\n"
        "Watch out for: Fixed once the bot has saves: change it by making a new bot from this one. FP16 is opt-in; the default stays FP32. The console report shows the loss scale and any skipped updates.\n"
        "\n"
        "Measured:\n"
        "- Speed, presets' networks: no measurable difference, 404,525 against 400,733 agent steps per second.\n"
        "- Speed, large networks: learning about twice as fast; in the final benchmark, 1v1 with large networks trained about 2.1 times as fast in FP16 as in FP32.\n"
        "- Learning: no measurable difference from FP32 on ball touch, recovery, boost pickup and kickoff, and on large networks."
    ),
    "run.fp16_sums": (
        'What it does: With run.precision = "fp16" on a graphics card, the half-precision matrix products keep their running sums in half precision too. GeForce cards compute those products faster then than with the sums in full precision, PyTorch\'s default: on a recent card the big products ran 1.3-1.6x faster (older GeForce cards run full-precision sums at half speed, so they may gain more). "playing" does it only in the passes that choose the bot\'s actions (and its past versions\'); "forward" also in learning\'s forward passes; "all" also in its backward pass, where the gradients are summed over many samples.\n'
        "\n"
        "When to change it: For a big network that keeps the card busy, such as a big 2v2 transformer, where the card's arithmetic is the limit. Small networks wait on other things and gain little.\n"
        "\n"
        'How to use it: Measure first with benchmarks/profile/fp16_accumulation.py: it shows the speed of each way and how far the bot\'s choices and gradients move compared with full precision. Then set "playing", the safest.\n'
        "\n"
        "Watch out for: Each sum keeps about 3 significant digits, so the answers move a little. Needs PyTorch 2.7 or newer.\n"
        "\n"
        "Measured: on a recent card, the playing passes ran 1.26-1.40x faster for a big 2v2 transformer and up to 1.37x for a 1v1 MLP bot, and the bot's choices moved no more than FP16 itself moves them. Learning with the sums in FP16 (\"all\") moved the transformer's gradients far more (cosine 0.945 against FP32's) and made the 1v1 bot's learning slower: not recommended. A touch-ball learning check of \"playing\" (10 seeds each, big networks) found no measurable difference: 1.95 s to the ball against 1.97 s. Against the bot's play today, 0.0-0.4% of its sampled actions change."
    ),
    "run.deterministic": (
        "What it does: Makes a run exactly repeatable. With the same seed and settings, two runs give the same games, the same actions, the same rewards and the same learning, bit for bit. It switches PyTorch to its repeatable maths and turns off the graphics card's automatic tuning. A run stopped and resumed then matches one that never stopped.\n"
        "\n"
        "When to change it: For tests and careful comparisons, or to chase a problem you want to see happen again exactly. Not needed for normal training.\n"
        "\n"
        "How to use it: Set true together with a fixed run.seed. A different seed still gives different games.\n"
        "\n"
        "Watch out for: It is somewhat slower; how much has not been measured. The games are always repeatable whatever this says (each arena has its own random generator); this setting is about the learning on the graphics card matching too.\n"
        "\n"
        "Measured: Tested on the processor and on the graphics card in FP16: two runs gave identical games, actions, rewards and losses, and a different seed gave different games."
    ),
    "run.on_training_problem": (
        'What it does: Decides what happens when learning breaks: numbers that are not numbers (NaN) in the losses or weights, or learning updates skipped again and again. "stop" ends the run with a plain-English error; "warn" writes the warning and carries on.\n'
        "\n"
        'When to change it: Rarely. "warn" can help when you want to see how a broken run develops, for example while looking for the cause of a problem, knowing its saves may be useless.\n'
        "\n"
        'How to use it: Keep "stop". When it stops, the last good save is kept, never overwritten by the broken state, so you can change a setting (often a lower learning rate) and resume.\n'
        "\n"
        'Watch out for: With "warn", a run whose weights have turned into NaN keeps playing and saving, but it learns nothing, and its saves may hold the broken weights.'
    ),
    "run.missing_data": (
        'What it does: Decides what happens when some of what a bot trains with is not on this computer, usually after a move: past versions its save rates that are not in its pool folder (they cannot play against it), or replay_mix folders without converted replays (no replays are copied). "warn" trains on: the past versions are named at the start and in the report, and replay_mix says so at its first iteration, at every save and in the report. "stop" refuses to start training (a new bot too) and lists what is missing.\n'
        "\n"
        'When to change it: Set "stop" before moving a bot, so a forgotten folder is noticed at once instead of billions of steps later.\n'
        "\n"
        'How to use it: Keep "warn" normally. With "stop", copy what it lists from the other computer (the pool folder, the replays folder), or set "warn" to train on without it.\n'
        "\n"
        'Watch out for: A save that rates versions lost long ago (a long run may rate dozens of versions it no longer keeps) is always reported, so "stop" would never let such a bot start. Only training is refused: trying rewards (nitro reward-check) and saving a copy of replays work either way. Neither choice changes how the bot learns.'
    ),
    "run.checkpoint_minutes": (
        "What it does: How often, in minutes, the run saves itself. A save holds the networks, the learner's state, the reward scaling and every random generator. Every save also ends all games in progress and builds the arenas afresh, which is what lets a stopped run continue exactly where it left off.\n"
        "\n"
        "When to change it: Lower it if your PC may switch off and losing 15 minutes matters; raise it if saves fill the disk too fast. Use run.checkpoint_steps instead if you think in steps rather than minutes.\n"
        "\n"
        "How to use it: The presets use 15. The smallest allowed is 0.5. It is ignored when checkpoint_steps is above 0. A stop with Ctrl+C always saves at once.\n"
        "\n"
        "Watch out for: Each save cuts every game short (the learner treats it like a time limit, so nothing is learned wrongly) and the first update after a save is slower while everything warms up, about 2.4 times slower. Saving every few minutes therefore costs some speed; how much has not been measured. The soak test saved every 5 minutes for 3 hours without trouble."
    ),
    "run.checkpoint_steps": (
        "What it does: Saves the run every this many agent steps instead of every checkpoint_minutes. 0 means saves go by time.\n"
        "\n"
        "When to change it: When you want saves at round step counts, so their folder names line up and you can compare saves from different runs at the same point. It is also steadier than minutes if your PC's speed changes from day to day.\n"
        "\n"
        "How to use it: a big bot can save every 200,000,000 steps with keep_checkpoints = 5, so the newest billion steps stay. Pair it with run.milestone_steps (default one billion) to keep older versions too. A stop with Ctrl+C still saves at once, whatever is set.\n"
        "\n"
        "Watch out for: A large value means more lost work if the PC stops without Ctrl+C. The time between saves depends on speed: at the 1v1 preset's measured 394k agent steps a second, 200 million steps is about 8 to 9 minutes; a 6.2-million-parameter bot ran at about 88k steps a second in its first iterations, so the same count is nearer 40 minutes. Each save also restarts every game, as with checkpoint_minutes."
    ),
    "run.keep_checkpoints": (
        "What it does: How many of the newest saves stay in the run's checkpoints folder. Older ones are deleted as new ones arrive, except the milestone saves kept by run.milestone_steps.\n"
        "\n"
        "When to change it: Raise it to be able to go back further, for example if you want to compare a few recent versions of the bot with nitro eval or watch them with nitro watch --at. Lower it to save disk space.\n"
        "\n"
        "How to use it: The default and the presets keep 5. The smallest allowed is 1. With checkpoint_steps = 200,000,000 and 5 kept, the newest billion steps stay.\n"
        "\n"
        "Watch out for: Saves of big networks are large: one save of a 6.2-million-parameter bot was 93 MB. A resume always starts from the newest save, so deleting is never needed to continue. The self-play pool of past versions is kept separately, under its own size limits (selfplay.pool_size, pool_disk_mb).\n"
        "\n"
        "Measured: The 3-hour soak test checked that no more saves are kept than this number plus the milestones, and that no half-written save is left behind."
    ),
    "run.milestone_steps": (
        "What it does: Also keeps the first save after every this many agent steps, for good. These milestone saves are never deleted by keep_checkpoints, so older versions of the bot stay available at a small, bounded disk cost.\n"
        "\n"
        "When to change it: Lower it on a long run if you want more older versions to compare against, watch, export or start new runs from. Set 0 to keep only the newest saves.\n"
        "\n"
        "How to use it: The default and the presets keep one per billion steps (1,000,000,000). An old save is picked by its step count: nitro watch --at, start_from.at, or the save's folder name in checkpoints.\n"
        "\n"
        "Watch out for: Each milestone is a full save, so a small value on a very long run adds up on disk (93 MB a save for a 6.2-million-parameter bot). Milestones are the first save at or after each multiple, so the exact step count depends on how often you save."
    ),
    "run.report_minutes": (
        "What it does: How often, in minutes, the run rewrites its report.html: a single page in the run folder with charts of speed, reward, how the bot's choices and predictions develop, and any warnings. It opens in any browser without a server.\n"
        "\n"
        "When to change it: Lower it if you watch the report closely early in a run; raise it if you rarely look.\n"
        "\n"
        "How to use it: The default is 2 minutes; the smallest allowed is 0.1. Reload the page in the browser to see the newest charts; it is written in one go, so a reload never shows half a file. The numbers behind it are in metrics.csv, written every iteration whatever this says.\n"
        "\n"
        "Watch out for: Nothing much. Writing the report costs milliseconds, so even a short interval does not slow training noticeably; this has not been measured as a speed comparison. The dashboard and W&B (if switched on) have their own views of the same numbers."
    ),
    "run.console": (
        'What it does: Chooses what the console window shows while training. "line" prints one short line: steps, speed, reward, entropy and value prediction. "report" prints a full block: speed and where the time goes, the rewards and each one\'s share, how the policy and value learn, how the bots play (touches, air time, boost, demos), self-play, the bot the run started from, the next save and new warnings. A large copy of replays also prints a block every round.\n'
        "\n"
        'When to change it: Purely a matter of taste. "report" is for watching closely; "line" for a calm window.\n'
        "\n"
        'How to use it: The presets use "report". It can be changed between stop and resume; training is the same either way. run.console_seconds sets how often it prints.\n'
        "\n"
        "Watch out for: The report is plain text so any Windows console can show it. It averages the iterations since the last print, so a single odd iteration is smoothed. Everything it shows is also in metrics.csv and report.html."
    ),
    "run.console_seconds": (
        "What it does: How often, in seconds, the console prints its line or report while training.\n"
        "\n"
        "When to change it: Raise it if the report scrolls past too fast to read, or if a log of the console grows too long; lower it if you want quicker feedback early in a run.\n"
        "\n"
        'How to use it: The default is 10 seconds; the smallest allowed is 1. With run.console = "report", each block averages the iterations finished since the previous one, and the speed is measured over that time, so a longer interval gives steadier numbers.\n'
        "\n"
        "Watch out for: The check happens after each learning update, so if one update takes longer than this interval (large networks, or a slow PC), prints come once per update instead. It only affects what is printed: the metrics file, the report page and training itself are unchanged."
    ),
    "env.team_size": (
        "What it does: How many players each team has: 1 for 1v1, 2 for 2v2, 3 for 3v3. In 2v2 and 3v3 the exported bot drives every car on its team (hivemind).\n"
        "\n"
        "When to change it: When starting a bot for another game mode, or, with a padded observation, to move a bot from 1v1 to 2v2 or 3v3.\n"
        "\n"
        "How to use it: Start from the matching preset. The presets keep about 2,048 cars in play: 1024 arenas for 1v1, 512 for 2v2 and 342 for 3v3. The 2v2 and 3v3 arena counts are guesses, not measured. The attention networks (model.encoder) are only for 2v2 and 3v3.\n"
        "\n"
        "Watch out for: Fixed once the bot has saves: change it by making a new bot from this one. The exception is a padded observation: with env.obs_team_size set (for example 3) and kept the same, team_size may change between stop and resume, so 1v1 first, then 2v2 and 3v3 with the same networks. With an attention network (model.encoder) it may change between stop and resume too, room or not: a 2v2 bot trains on at 3v3. Whether a padded bot learns 1v1 as well as a 1v1-only bot is not measured."
    ),
    "env.team_sizes": (
        "What it does: Plays several game sizes in the same run, for example [1, 2, 3]: each arena keeps one of these sizes for the whole run, and the sizes are spread evenly over the arenas, so every batch the bot learns from holds all of them. Empty (the default) plays every game at team_size.\n"
        "\n"
        "When to change it: When one bot should learn 1v1, 2v2 and 3v3 at once, usually after it has learned 1v1 on its own.\n"
        "\n"
        "How to use it: Set team_size to the largest size, give the observation room for it (env.obs_team_size = 3, or an attention or transformer network) and resume. Start the bot in 1v1 with obs_team_size = 3 already set, then add [1, 2, 3] on a resume. Self-play works: past versions play Orange in every size, with one rating for all. metrics.csv gains columns per size (1v1_episode_seconds, 2v2_goals_per_minute, 3v3_touches_per_minute and others).\n"
        "\n"
        "Watch out for: Not yet with replay starts; the settings check says so. Training every size from scratch spread the first touches thin and stalled in Pulsar's runs. Whether a mixed run learns as well as single sizes is not measured yet."
    ),
    "env.team_size_shares": (
        "What it does: With env.team_sizes, each size's share of the arenas, in the same order, for example [0.5, 0.25, 0.25]. The numbers need not add up to 1. Empty means equal shares.\n"
        "\n"
        "When to change it: When one size should get more of the training, for example more 1v1 while the bot is still learning to touch the ball.\n"
        "\n"
        "How to use it: Give one positive number per size. The log says, when the run starts, each size's arenas and its share of the player-steps.\n"
        "\n"
        "Watch out for: A 3v3 arena has six cars to a 1v1's two, so equal arena shares give 3v3 half of the steps. A share too small for the arena count gives that size no arena, and the run refuses to start: raise num_arenas or the share."
    ),
    "env.shuffle_slots": (
        "What it does: With a padded observation (env.obs_team_size), every game each car takes a random place among its team's places in the view, so the network sees teammates and opponents in every place, not always in the first ones. The games themselves are the same.\n"
        "\n"
        "When to change it: For a plain network that plays several game sizes (env.team_sizes, or a run that moves from 1v1 to 3v3), so the places a smaller game leaves empty are not always the same ones. An idea from buta-ppo and Pulsar.\n"
        "\n"
        "How to use it: Set it to true; it can be switched on or off between stop and resume. Attention and transformer networks do not need it: they read every car the same way wherever it sits.\n"
        "\n"
        "Watch out for: It needs env.obs_team_size above 0. Off by default and not measured yet: whether it helps learning has not been measured."
    ),
    "env.opponents": (
        "What it does: Whether the Orange team exists. true is a normal game. false leaves only the Blue team on the field, with nobody to play against.\n"
        "\n"
        "When to change it: Only for practice tasks that train one skill alone, such as touching the ball, landing on the wheels or collecting boost. The built-in practice tasks use it that way.\n"
        "\n"
        "How to use it: Keep true for any bot meant to play matches. Self-play (selfplay.enabled) needs it true.\n"
        "\n"
        "Watch out for: Fixed once the bot has saves: change it by making a new bot from this one. Like team_size, it may change between stop and resume when the observation is padded (env.obs_team_size set and unchanged) or the bot has an attention network. A bot trained without opponents has never seen another car, so it will not know what to do around one."
    ),
    "env.num_arenas": (
        "What it does: How many games are simulated at the same time. More arenas give the graphics card bigger batches, which is faster up to a point, and use more processor time and memory.\n"
        "\n"
        "When to change it: To tune speed on your own PC, or to make room for features that share out arenas (self-play).\n"
        "\n"
        "How to use it: The default, 1024, suits 1v1. The 2v2 preset uses 512 and the 3v3 preset 342, keeping about 2,048 cars; those two counts are guesses. Try others with nitro ab --speed. It interacts with ppo.steps_per_iteration: more arenas mean fewer steps per game in each update.\n"
        "\n"
        "Watch out for: It must be at least env.groups, and with self-play it must leave each past version a game in every group.\n"
        "\n"
        "Measured (1v1): 256 arenas 244k, 512 317k, 1024 394k, 2048 412k, 4096 418k agent steps a second. 2048 is within 3% of the fastest, but at 50,000 steps per update each game then gets only 13 steps per update against 25 at 1024, so 1024 was chosen."
    ),
    "env.tick_skip": (
        "What it does: How many physics ticks (each 1/120 of a second) pass between the bot's decisions. 8 means 15 decisions a second; 4 means 30.\n"
        "\n"
        "When to change it: For mechanics that need quicker reactions or finer timing (flicks, resets, air dribbles), where a smaller tick skip lets the bot act more often.\n"
        "\n"
        "How to use it: 4, the default, gives 30 decisions a second; 8 is the other common value. A common pair is 4 with action_delay = 3. Keep action_delay smaller than tick_skip; a common rule is tick_skip - 1. Two things adjust on their own: the discount is set in seconds (ppo.discount_half_life), so the bot plans the same time ahead, and rewards paid every step are scaled so the same weights mean the same thing. Others do not: ppo.gae_lambda counts decisions (the same seconds at another tick skip is lambda^(new/old): 0.95 at 8 = 0.9747 at 4), as do the learning settings counted in agent steps (ppo.steps_per_iteration, selfplay.snapshot_steps and flat_rating_steps, ppo.vdag_reinit_steps and pretrain.value_warmup_steps). A bot started from another at a different tick skip lists in its log those it uses, with the other bot's number where it had one.\n"
        "\n"
        "Watch out for: A smaller tick skip means more decisions per second of game, so more network work for the same play time, and each update (counted in agent steps) covers less game time; the speed cost is not measured. The exported bot carries its tick skip. A carried-on replay copy and nitro eval between two saves need the same tick skip. Copying replays at tick skip 4 gives twice the decisions per replay."
    ),
    "env.action_delay": (
        "What it does: How many ticks pass between the bot choosing an action and its car using it. It copies the delay a real bot lives with, so the bot learns to act a little ahead.\n"
        "\n"
        "When to change it: When you change tick_skip, because it must stay smaller than tick_skip. Otherwise leave it.\n"
        "\n"
        "How to use it: 1, the default, is the lowest the exported bot can keep up with: in a real game a decision then has about two ticks (16.7 ms) to reach the car. Many bots use tick_skip - 1 instead (3 at tick skip 4). The exported bot is told this delay and times its decisions to match.\n"
        "\n"
        "Watch out for: It is not one of the fixed settings, but the bot learns its timing from it, so changing it on a running bot changes what its learned timing means. nitro eval between two saves, nitro skills and a carried-on replay copy need the same value.\n"
        "\n"
        "Measured: in a real RLBot match the game's input delay was exactly 1 tick, with no variation, and a check of a logged match found no extra tick in the game's physics, so the exported bot sends its controls on the training rhythm."
    ),
    "env.observation": (
        'What it does: What the bot sees each step, as a list of numbers. "native" is NitroLearn\'s own (112 numbers in 1v1). "nitro_v2" adds what strong bots use: things relative to the bot\'s car, the goals and the ball, speeds, the demo respawn timer and the time left to flip (148 numbers in 1v1).\n'
        "\n"
        'When to change it: Choose it when you start a new bot. The default, "nitro_v2", measured better than "native" (below); "native" is the smaller one.\n'
        "\n"
        'How to use it: Write it before the first start. A padded observation (env.obs_team_size) needs "native" or "nitro_v2".\n'
        "\n"
        "Watch out for: Fixed once the bot has saves: change it by making a new bot from this one. The exported bot reads the observation's name from its file, so it needs no change. Transfer learning (start_from) needs the same observation in both runs.\n"
        "\n"
        "Measured:\n"
        "- Shot task: nitro_v2 scored 88% of shots against 53%, better on every seed.\n"
        "- Ball touch: no measurable difference, 2.07 s against 2.16 s.\n"
        "- Saves: no measurable difference, 0.883 against 0.867.\n"
        "Better on one task of three and worse on none, so it is the default."
    ),
    "env.obs_team_size": (
        "What it does: Gives the observation room for this many players a side, whatever the game size. Places for missing cars are left empty and marked as empty, so one network fits every game up to that size. 0 means exactly team_size, with no padding.\n"
        "\n"
        "When to change it: When you want one bot for 1v1, 2v2 and 3v3, or a bot whose training moves from 1v1 to 3v3. With an attention network (model.encoder), 0 means room for team_size, which such a network needs; it plays bigger games anyway.\n"
        "\n"
        "How to use it: Set it when you start a new bot, for example 3 for every mode. The presets show 3 as a commented line. Then team_size (and opponents) may change between stop and resume. A 1v1 bot with room for 3 can start a 3v3 run (start_from), and pretrain.team_sizes can copy replays of several sizes. An exported bot plays any match up to this size and sends no input in bigger ones.\n"
        "\n"
        'Watch out for: Fixed once the bot has saves (but not with an attention network, whose networks do not depend on the room): change it by making a new bot from this one. It must be 0 or at least team_size, and it needs "native" or "nitro_v2". It costs a few numbers per car: the native observation grows from 112 (plain 1v1) to 173 numbers for room for 2 and 233 for room for 3. Not measured yet: how well a padded bot learns has not been compared.'
    ),
    "env.groups": (
        "What it does: Splits the arenas into this many groups, so the processor steps one group's games while the graphics card thinks for another. This hides the waiting on each side.\n"
        "\n"
        "When to change it: Almost never. 2 is right for almost everyone.\n"
        "\n"
        "How to use it: Keep 2. num_arenas must be at least this. With self-play, every past version must have a game in every group, so more groups need more arenas or a larger selfplay.past_share. It pairs with env.pipelined_inference, which only acts with 2 or more groups.\n"
        "\n"
        "Watch out for: More groups made training slower on our test machine. Changing it does not change the bot, only the speed.\n"
        "\n"
        "Measured (our test machine, 1v1 starter settings, agent steps a second):\n"
        "- 2 groups: 392,144\n"
        "- 3 groups: 371,186\n"
        "- 4 groups: 344,440\n"
        "With pipelined inference on, 3 groups reached 393,570, only 0.4% above 2 groups without it, far under the 3% that counts. So 2 groups stay."
    ),
    "env.pipelined_inference": (
        "What it does: Starts the next group's network pass on the graphics card before waiting for the previous group's, so the work of launching overlaps the card's own work. The games and the data collected are identical; only speed can change.\n"
        "\n"
        "When to change it: On by default: the games and the bot are the same either way, and it was never slower (5% faster for a large 1v1 bot, 2% for a 2v2 attention bot, no change for the presets' small networks). With env.card_starts_games on (the default) the card starts the games and this has nothing left to do.\n"
        "\n"
        "How to use it: true or false. It only acts on a graphics card with env.groups of 2 or more; on the processor it does nothing. It can be switched freely, even between a stop and a resume, since the games and the bot are the same either way.\n"
        "\n"
        "Watch out for: Nothing breaks with it on; with small networks it just gains nothing.\n"
        "\n"
        "Measured (our test machine, agent steps a second): at 2 groups, off 392,144 and on 378,382 (borderline, slower); at 3 groups, off 371,186 and on 393,570; at 4 groups, off 344,440 and on 375,510. The fastest combination was only 0.4% above the default, under the 3% that counts, so the setting stays off. Launching took 9.7% of the loop's time, the most this could ever hide.\n"
        "\n"
        "Measured again (self-play on): with large FP16 networks the card's work per group is long enough to hide the launching behind. A 1v1 network (shared [1024], policy and value [1024, 512, 512]): 287,116 to 301,400 agent steps a second, 5.0% faster (runs within 2.7%). A 2v2 attention bot's networks: 93,718 to 95,456, 1.9% faster (runs within 0.6%)."
    ),
    "env.card_starts_games": (
        "What it does: On a graphics card, the card starts each group of games itself as soon as their actions are back on the processor (a small call recorded into the same CUDA graph as the networks' pass), instead of Python starting them after waiting for the card. Python then prepares the other group while these games run.\n"
        "\n"
        "When to change it: On by default. It pays most on a fast card with a small or medium network, where the game threads and Python otherwise take turns. The training is the same to the bit either way; turn it off only to compare, or if a card or driver misbehaves with it.\n"
        "\n"
        "How to use it: true or false. It can be switched between a stop and a resume. It does nothing on the processor or without CUDA graphs.\n"
        "\n"
        "Watch out for: nothing known; it changes when the games start, not what they do.\n"
        "\n"
        "Measured: on a fast card the same training to the bit with it on and off, FP32 and FP16 (tests/test_card_starts_games.py); a 1v1 bot with 4x512 networks trained 1.17x faster with it, both with ppo.graphed_steps (655,809 to 769,902 steps a second)."
    ),
    "env.concurrent_passes": (
        "What it does: In each step's pass on the graphics card, the value network and every past version the bot plays against (self-play) run at the same time as the policy, on their own lanes of the card, instead of one after another. The same work, the same random numbers and the same results to the bit; the games and the bot are identical either way.\n"
        "\n"
        "When to change it: On by default. It pays most on a big card with self-play, where each past version's pass is small. The games and the learning are the same either way, so there is no reason to turn it off except to compare.\n"
        "\n"
        "How to use it: true or false. It only acts on a graphics card; on the processor it does nothing. It can be switched freely, even between a stop and a resume.\n"
        "\n"
        "Watch out for: It uses a little more graphics memory (a working area per lane).\n"
        "\n"
        "Measured: a big 2v2 transformer trained 1.10x faster on a fast card (65,169 to 71,517 steps a second); a 1v1 MLP bot there no faster, as its games hold collection up."
    ),
    "env.threads": (
        "What it does: How many processor threads run the simulated games. 0 means automatic: all of the processor's threads except two.\n"
        "\n"
        "When to change it: When you want to keep part of the processor free for other things while training (lower it), or on a machine where the automatic choice is wrong.\n"
        "\n"
        "How to use it: Leave 0. On a 10-core, 20-thread processor that gives 18. The games are the same whatever the thread count, because every arena has its own random generator; only speed changes.\n"
        "\n"
        "Watch out for: Using every thread is not faster: the Python main thread and the graphics card driver then compete with the game threads.\n"
        "\n"
        "Measured (random play, millions of agent steps a second): 9 threads about 0.89 to 1.04; 18 threads 1.48 to 1.63; 20 threads 1.48 to 1.59. At 20 threads the main thread's overhead rose to 1.5 to 7.1%, against 1.1 to 2.4% at 18. So two threads stay free."
    ),
    "terminals.no_touch_seconds": (
        "What it does: Restarts a game after this many seconds in which nobody touched the ball. 0 means never. The game is cut off rather than finished: the learner estimates what would have come next, so the bot is not taught that a long wait was a loss or a win.\n"
        "\n"
        "When to change it: Raise it if long, patient plays (shadow defence, boost runs) keep being cut off; lower it if games stall too long with nobody going for the ball.\n"
        "\n"
        "How to use it: 10 seconds is a common value, the default and the presets' value. The practice tasks set 0 and use max_episode_seconds instead. A goal always restarts the game.\n"
        "\n"
        "Watch out for: Shorter limits mean more restarts, so more games begin from your state setters (kickoffs, by default). The setting can be changed between stop and resume."
    ),
    "terminals.max_episode_seconds": (
        "What it does: Restarts a game after this many seconds whatever happens. 0 means never. Like the no-touch timer it cuts the game off rather than finishing it, so the learner bootstraps from the state reached.\n"
        "\n"
        "When to change it: Mostly for practice tasks, where each attempt should have a time limit. For normal training there is little need: data is collected in fixed-length chunks that bootstrap at every boundary anyway, which is why it is off by default.\n"
        "\n"
        "How to use it: The practice tasks use it as their attempt length: 6 seconds for kickoff, recovery, save and aerial touch, 8 for shot, 10 for ball touch and the two-skill task, 20 for boost pickup. A goal still restarts the game earlier.\n"
        "\n"
        "Watch out for: A limit in normal training makes more games start from your state setters, which changes the mix of situations the bot sees. In a practice task a short limit with end_on_touch makes a miss cheap; the aerial task's decline after 10M steps is still an open question."
    ),
    "terminals.end_on_touch": (
        "What it does: For practice tasks: ends the episode as done (the task was completed) as soon as any car touches the ball. Unlike a time limit, the learner treats it as a real end, with nothing more to come.\n"
        "\n"
        "When to change it: Only when building a practice drill whose goal is a touch, such as reaching the ball as fast as possible. Never for match training: every touch would end the game.\n"
        "\n"
        'How to use it: The built-in tasks ball touch, aerial touch and the two-skill touch task use it, with no_touch_seconds = 0 and a max_episode_seconds limit for misses. A reward for the touch then teaches "get there fast", and the time to touch is the task\'s measure.\n'
        "\n"
        "Watch out for: Done means the learner expects nothing after the touch, so any reward you want for the touch itself must be paid on that step. It combines with truncate_on_ball_landing in the aerial tasks, so only an aerial touch can finish them."
    ),
    "terminals.end_on_landing": (
        "What it does: For practice tasks: ends the episode as done (the task was completed) once every car is on its wheels.\n"
        "\n"
        "When to change it: Only when building a drill about landing or recovering, such as the recovery task where the car starts tumbling in the air. Never for match training: every landing would end the game.\n"
        "\n"
        "How to use it: The built-in recovery task uses it with no_touch_seconds = 0 and a 6-second limit (max_episode_seconds). It is measured by the time to land; a trained bot landed in about 1.20 seconds against 2.84 seconds untrained.\n"
        "\n"
        "Watch out for: Done means the learner expects nothing after the landing, so a reward for landing must be paid on that step. It only makes sense with a state setter that starts the car off its wheels."
    ),
    "terminals.truncate_on_ball_landing": (
        "What it does: The chance (0 to 1) that an episode is cut off when the ball comes down to the floor after being in the air. It is a cut-off, not a finish: the learner bootstraps from the state reached.\n"
        "\n"
        "When to change it: Two uses. At 1, in a drill where the ball starts in the air, only an aerial touch can finish the task (the aerial-touch tasks do this). At a small chance in match training, such as 0.1, games end at a natural pause now and then, so more of them start from your start states.\n"
        "\n"
        "How to use it: 0 means never. The presets show 0.1 as a commented example. Floor contact is checked every tick, so a quick bounce is never missed. The random draw only happens on a landing and only for a chance between 0 and 1, so runs without it keep their random sequence.\n"
        "\n"
        "Watch out for: Not measured yet: no comparison was run for match training; it is for your own full runs, judged with nitro eval. In the aerial pilot the bot found aerial touches and then lost them after 10M steps; this cut-off making misses cheap is the first suspect."
    ),
    "model.shared": (
        "What it does: puts a set of layers in front of both networks, so the policy (which action to take) and the value network (how good the moment is) read the game through the same first layers before each goes its own way. Empty means two fully separate networks, the default in every preset.\n"
        "\n"
        'When to change it: when you want a "shared head" layout, for example for a big bot with an attention network.\n'
        "\n"
        "How to use it: list the layer sizes, e.g. [768, 768]. A big bot can use [768, 768] under an attention encoder, with policy [1024, 1024, 1024] and value [1024, 768]: about 6.2M parameters, the trunk a third of the policy. Each 1024-to-1024 layer is about 1 million parameters, and a [1024, 1024] trunk with [1024, 1024, 256] policy and value comes to about 6.0M in 1v1 and 2v2 (worked out from the code, not a training result). With attention, the one encoder sits at the start of the trunk.\n"
        "\n"
        "Watch out for: the value warm-up (pretrain.value_warmup_steps) holds the trunk still, so only the value network's own layers learn. Fixed once the bot has saves: change it by making a new bot from this one (settings only; transfer learning needs the same networks).\n"
        "\n"
        "Measured: on ball touch a [256, 256] trunk with one layer each on top made no measurable difference. With attention it is not measured yet."
    ),
    "model.policy": (
        "What it does: sets how many layers the policy network has and how wide each is. The policy is the part that picks the bot's action, and it is the part that becomes the RLBot bot when you export.\n"
        "\n"
        "When to change it: for a bigger brain on a long run, usually together with model.value.\n"
        "\n"
        'How to use it: one number per layer, e.g. [256, 256, 256] (every preset) or [1024, 1024, 1024] (a big bot). Each 1024-to-1024 layer is about 1 million parameters; [512, 512, 512] is about 694k with a padded 3-a-side observation. The bot\'s page on the dashboard shows the parameter count from the newest save. Large networks pair well with run.precision = "fp16" and model.fused_norm.\n'
        "\n"
        "Watch out for: bigger networks train slower. On our test machine the presets' networks ran at about 406,000 agent steps a second in FP32, while 4x1024 networks in FP16 ran at about 127,000 to 140,000. Fixed once the bot has saves: change it by making a new bot from this one (settings only; transfer learning needs the same networks). nitro surgery can widen a saved bot's layers by a whole factor with the same play, but not add layers, and not for attention networks.\n"
        "\n"
        "Measured: whether a bigger policy makes a stronger bot is not measured here."
    ),
    "model.value": (
        "What it does: sets the layers of the value network, the part that guesses how good the current moment is so the policy knows which actions paid off. It is only used while training; the exported RLBot bot leaves it out.\n"
        "\n"
        "When to change it: when you make the policy much bigger, it is common to grow this too, though it does not have to match.\n"
        "\n"
        "How to use it: one number per layer. Every preset uses [256, 256, 256]; a big bot can use [1024, 768] beside a [1024, 1024, 1024] policy and a shared trunk, about 1.58M parameters for the value part. Each 1024-to-1024 layer is about 1 million parameters.\n"
        "\n"
        'Watch out for: a bigger value network costs training speed like any other layer, but adds nothing to the bot\'s time in a match. Fixed once the bot has saves: change it by making a new bot from this one (settings only; transfer learning needs the same networks). If you later start a bot from this one with start_from.parts = "policy", the value network starts afresh anyway.\n'
        "\n"
        "Measured: not measured on its own."
    ),
    "model.value_heads": (
        'What it does: splits the value network into a shared part (model.value, the "critic trunk") and named heads on top of it, each with its own layers (model.value_head_layers). "main" is the usual value the bot learns from. "goal", "vdag1" and "vdag2" are extra heads for the goal/concede-only bot: a goal critic and twin value critics, for the learning signals ppo.goal_critic_weight and ppo.vdag_beta.\n'
        "\n"
        'When to change it: only for a goal-only bot, as in the GCO starter (configs/presets/gco_1v1_starter.toml). Other bots keep the default, ["main"].\n'
        "\n"
        'How to use it: a list of names, "main" first, each at most once. Without the signals that use them, the extra heads are built, saved and restored, but learn nothing.\n'
        "\n"
        "Watch out for: each head costs as many parameters as its layers (the GCO starter: 0.7M a head, 3.3M at the design's full size), only in training: the exported bot never has them. Fixed once the bot has saves.\n"
        "\n"
        "Measured: nothing yet."
    ),
    "model.value_head_layers": (
        'What it does: the layers each value head has to itself, on top of the layers of model.value, which every head shares. With it empty and model.value_heads left at ["main"], the value network is one network (the default).\n'
        "\n"
        "When to change it: for a goal-only bot (the GCO starter, configs/presets/gco_1v1_starter.toml, uses [600, 600], or [1280, 1280] at the design's full size), where several value heads each learn something of their own.\n"
        "\n"
        "How to use it: one number per layer, as model.value. With model.residual, layers as wide as the one before them are residual blocks.\n"
        "\n"
        "Watch out for: every head gets these layers, so their cost multiplies by the number of heads; all of it is training-only, the bot's time in a match is unchanged. Fixed once the bot has saves.\n"
        "\n"
        "Measured: nothing yet."
    ),
    "model.residual": (
        "What it does: makes the hidden layers residual blocks: a layer as wide as the one before it adds its result to what it was given (x + layer(x)) instead of replacing it. A deep network can then pass on what earlier layers worked out and each layer only learns a change, which keeps deep, wide networks easier to train. A layer that changes the width (usually each network's first) stays a plain layer.\n"
        "\n"
        "When to change it: for deep, wide networks such as the GCO starter's (configs/presets/gco_1v1_starter.toml: three layers of 520, 352 and 600; 1152, 768 and 1280 at the design's full size). With the presets' small networks it is not needed.\n"
        "\n"
        "How to use it: true or false. It needs model.layer_norm = true. It works with FP16 and fused_norm.\n"
        "\n"
        "Watch out for: exported bots with it use bot file format 8, which the nitro_bot that comes with NitroLearn reads. Fixed once the bot has saves: a bot cannot switch it, and transfer learning needs both bots alike.\n"
        "\n"
        "Measured: the exported bot gives the same answers as training's networks (within 1e-4); how it learns is not measured."
    ),
    "model.layer_norm": (
        "What it does: after each hidden layer, rescales that layer's numbers to a steady range before the next layer reads them. This keeps learning stable as the networks change.\n"
        "\n"
        "When to change it: almost never. Leave it on unless you are deliberately copying a setup that has none.\n"
        "\n"
        "How to use it: true or false. It affects the policy, the value network and any shared trunk.\n"
        "\n"
        "Watch out for: model.fused_norm needs it on, so turning it off loses that speed-up for large FP16 networks. It costs some speed, most of all in FP16 (it was behind most of the gap to the graphics card's top FP16 speed); model.fused_norm exists to win that back. Transfer learning (start_from) needs both bots to have the same value. Fixed once the bot has saves: change it by making a new bot from this one (settings only; transfer learning needs the same networks).\n"
        "\n"
        "Measured: how a bot learns without it is not measured; only its speed cost was."
    ),
    "model.fused_norm": (
        "What it does: does each hidden layer's normalisation and ReLU in one step on the graphics card instead of several. The bot learns and plays the same; only the speed changes.\n"
        "\n"
        "When to change it: turn it on for large networks trained in FP16, such as a 6.2M-parameter bot or a large 1v1 bot (8% faster). Leave it off for the presets' [256, 256, 256] networks.\n"
        "\n"
        'How to use it: true or false. It needs model.layer_norm = true and model.activation = "relu" or "leaky_relu". It is not fixed: saves are the same with or without it, so you can switch it between a stop and a resume.\n'
        "\n"
        "Watch out for: on the CPU, or if the routine cannot be built, training carries on the normal way with a warning, so check the log if you expect the speed-up. It relies on parts of PyTorch that may move, which is why the project keeps PyTorch at version 2.14.1.\n"
        "\n"
        "Measured (on our test machine):\n"
        "- 4x1024 networks in FP16: 11% faster than an earlier FP16 LayerNorm (126,552 to 140,327 agent steps a second), which was itself 8% faster than the usual one there.\n"
        "- Presets' networks in FP32: no measurable difference.\n"
        "- Presets' networks in FP16: 7% slower.\n"
        "- Its graphics-card tests matched normal LayerNorm and ReLU in every shape tested.\n"
        "- A 1v1 network in FP16 (shared [1024], policy and value [1024, 512, 512]), self-play on: 8.0% faster than that earlier FP16 LayerNorm (287,116 to 310,119 agent steps a second; runs within 2.7%)."
    ),
    "model.init": (
        'What it does: chooses the random numbers a brand-new bot\'s networks start with. "orthogonal" starts the policy with very small outputs, so a new bot is unsure about every action and explores evenly. "pytorch" uses PyTorch\'s own starting numbers.\n'
        "\n"
        "When to change it: if you want to match another framework that starts from PyTorch's numbers. Otherwise the default is fine.\n"
        "\n"
        'How to use it: "orthogonal" or "pytorch". It only matters when networks are made from scratch: a resumed bot loads its saved weights, and a bot started from another (start_from) takes that bot\'s weights instead (with parts = "policy", only the fresh value network uses it).\n'
        "\n"
        "Watch out for: changing it on a bot that already has saves does nothing to the saved networks.\n"
        "\n"
        'Measured: on ball touch, "pytorch" reached the ball in 2.10 s against 2.16 s for "orthogonal": no measurable difference, though the range of the result leaves a small gain possible. Nothing changed because of it.'
    ),
    "model.activation": (
        'What it does: picks the small bending step between layers that lets the networks learn more than straight-line rules. "leaky_relu", the default, keeps a hundredth of negative numbers, so no neuron can go dead and stop learning; "relu" turns them into 0. "gelu" and "tanh" are alternatives.\n'
        "\n"
        "When to change it: rarely. Only if you are copying a setup that uses a different one, or starting a new bot from one that uses another (transfer learning needs the same).\n"
        "\n"
        'How to use it: one of "relu", "leaky_relu", "gelu" or "tanh".\n'
        "\n"
        "Watch out for:\n"
        '- "gelu" cannot be exported: nitro export only supports "relu", "leaky_relu" and "tanh", so a "gelu" bot cannot become an RLBot bot.\n'
        '- model.fused_norm needs "relu" or "leaky_relu".\n'
        "- Transfer learning (start_from) needs both bots to use the same one.\n"
        "- Fixed once the bot has saves: change it by making a new bot from this one (settings only; transfer learning needs the same networks).\n"
        "\n"
        'Measured: none of the other choices has been compared with "relu" in this project; not measured.'
    ),
    "model.encoder": (
        'What it does: decides how the bot reads the other cars. "mlp" is the plain network every preset uses. "attention" adds a front end where the bot\'s car looks at each other car through one shared set of weights, so a lesson about one opponent carries over to the others. "attention_communication" first lets the cars, the ball and the own car look at each other, inside one bot\'s network. "transformer": the own car, the ball, every other car (and optionally every pad) are tokens that all look at each other, block after block. All three read any number of cars, so the exported bot, nitro watch and nitro eval play any game size, even far beyond the trained one (how well there is not measured).\n'
        "\n"
        "When to change it: for 2v2 and 3v3 bots. Not 1v1: the ball-touch result says plain network there.\n"
        "\n"
        'How to use it: env.obs_team_size can stay 0 (room for the run\'s own team size), or 2 or 3 for one bot over several sizes. Attention: 128 width, 2 blocks, 4 heads suggested, plus model.attention_fast = true with "attention" (10% faster). The transformer has its own sizes (model.transformer_...). A big bot can put "attention" in a shared trunk.\n'
        "\n"
        "Watch out for: training is much slower. Exported transformer bots use bot file format 6, which the nitro_bot that comes with NitroLearn reads. Fixed once the bot has saves: change it by making a new bot from this one (settings only; transfer learning needs the same networks).\n"
        "\n"
        "Measured: attention trained at 41% of the speed of a plain network of its size and learned ball touch slower (2.38 s against 2.08 s); kickoffs no difference. The talking version trained at 31% of attention's speed. None was compared for learning in 2v2 or 3v3; the transformer is not measured at all (checked for working only)."
    ),
    "model.attention_width": (
        "What it does: sets how many numbers the attention encoder uses to describe each car and the ball. More numbers can capture more detail about each car, at the cost of speed.\n"
        "\n"
        'When to change it: only with an attention encoder, and only when building a much bigger network than usual. A plain network (encoder = "mlp") ignores it.\n'
        "\n"
        "How to use it: 128 is the default, the size that was measured and the size community advice suggests. The setting's help suggests 256 for a much bigger network, paired with 8 heads. It must divide evenly by model.attention_heads (128 / 4 = 32 per head). For scale, the design estimate for a 128-wide, 2-block, 4-head encoder was about 450k parameters with its heads.\n"
        "\n"
        "Watch out for: wider means slower training, and the attention network is already the slow option (41% of a plain network of its size). For an attention bot it is fixed once the bot has saves: change it by making a new bot from this one (settings only; transfer learning needs the same networks). For a plain bot it is ignored and free to change.\n"
        "\n"
        "Measured: only 128 has been run; other widths are not measured."
    ),
    "model.attention_blocks": (
        'What it does: sets how many rounds the attention encoder takes over the other cars before the bot decides. Each round can build on the one before, for example first noticing who is close, then who is a threat. With "attention_communication", each round also lets the cars compare notes first.\n'
        "\n"
        "When to change it: only with an attention encoder, and only for a much bigger network. A plain network ignores it.\n"
        "\n"
        'How to use it: 2 is the default and the measured size, also what community advice suggests ("2 attention blocks, 4 heads and size 128"). The setting\'s help suggests 3 at most for bigger networks; the limit is 8.\n'
        "\n"
        "Watch out for: each extra block adds training time, and in the talking network every block is a full round of every car looking at every other, which is why it trains at 31% of plain attention's speed. For an attention bot it is fixed once the bot has saves: change it by making a new bot from this one (settings only; transfer learning needs the same networks). For a plain bot it is ignored and free to change.\n"
        "\n"
        "Measured: only 2 blocks have been run; other counts are not measured."
    ),
    "model.attention_heads": (
        'What it does: splits each attention round into several separate "looks" taken at the same time, so one part of the network can follow the nearest opponent while another follows a teammate.\n'
        "\n"
        "When to change it: only with an attention encoder, and mainly together with model.attention_width. A plain network ignores it.\n"
        "\n"
        "How to use it: 4 is the default and the measured value. The setting's help suggests 8 with a width of 256. The width is shared out between the heads, so the width must divide evenly by this number: 128 with 4 gives 32 each. The settings check refuses a pair that does not divide.\n"
        "\n"
        "Watch out for: more heads with the same width means each head has fewer numbers to work with. For an attention bot it is fixed once the bot has saves: change it by making a new bot from this one (settings only; transfer learning needs the same networks). For a plain bot it is ignored and free to change.\n"
        "\n"
        "Measured: only 4 heads have been run; other counts are not measured."
    ),
    "model.attention_skip": (
        "What it does: gives the layers after the attention encoder a direct copy of the bot's own numbers (ball, own car, boost pads, last action), next to what the encoder says about the other cars. The bot then sees its own situation the way a plain network does, and the encoder only has to add what the other cars mean.\n"
        "\n"
        "When to change it: it was built as a possible fix for attention learning ball touch slower than a plain network. Try it if you use attention and want that direct path.\n"
        "\n"
        'How to use it: true or false, with an attention encoder only. It works with both "attention" and "attention_communication". The 2v2 and 3v3 presets carry it commented out.\n'
        "\n"
        "Watch out for: bots exported with it are bot file format 3, which the nitro_bot shipped with NitroLearn runs. For an attention bot it is fixed once the bot has saves: change it by making a new bot from this one (settings only; transfer learning needs the same networks).\n"
        "\n"
        "Measured: not yet. Its ball-touch check was cancelled, because attention is judged in 2v2 and up, for working only. With it, a 3-a-side attention policy had about 656k parameters, against 694k for a plain [512, 512, 512]."
    ),
    "model.attention_fast": (
        "What it does: runs the same attention maths in fewer, bigger steps on the graphics card. The bot it trains and the files it saves are the same as without it; only training speed changes.\n"
        "\n"
        'When to change it: turn it on whenever model.encoder = "attention". The 2v2 and 3v3 presets suggest it next to the attention lines.\n'
        "\n"
        "How to use it: true or false. It is not fixed, so you can switch it between a stop and a resume, and the exported bot is unchanged.\n"
        "\n"
        'Watch out for: it does nothing with encoder = "attention_communication", because that network\'s car descriptions change every round, so the shortcut does not apply; the settings check says so. It does nothing with a plain network either.\n'
        "\n"
        "Measured: 10% faster training with the 2v2 attention settings on our test machine, 80,715 to 88,876 agent steps a second, faster on every repeat. Its tests give the same answers to rounding in every game size."
    ),
    "model.small_attention": (
        "What it does: With the transformer encoder, its attention over the cars and the ball runs as NitroLearn's own routine on the graphics card instead of PyTorch's. PyTorch's kernel is made for long sequences; with 5 tokens most of its work is wasted, and it took about a quarter of a big 2v2 transformer's training time. The new routine does the same maths: one small group of the card's threads per sample and head, the softmax in full precision, nothing stored between the forward and backward passes.\n"
        "\n"
        "When to change it: On by default (it has no drawback). Turn it off only to compare with PyTorch's attention; its answers are the same to rounding.\n"
        "\n"
        'How to use it: true or false, with encoder = "transformer" only. It takes up to 8 tokens (4v4 without boost pads) and heads of up to 32 numbers; anything bigger, the processor, or a card where the routine cannot be built uses PyTorch\'s attention instead (with one warning). It can be switched between a stop and a resume: the bot and its saves are the same either way.\n'
        "\n"
        "Watch out for: The answers are the same to rounding, not to the bit, so two runs with it on and off drift apart slowly, as with any precision detail.\n"
        "\n"
        "Measured: a big 2v2 transformer trained 1.21x faster on a fast card (65,822 to 79,389 steps a second), 1.34x together with env.concurrent_passes; our routine alone is 4.9x faster than PyTorch's attention at a learning step's size."
    ),
    "model.lean_attention": (
        "What it does: runs the same attention maths in fewer steps on the graphics card. Where the bot's own car (or the transformer's read-out token) looks at the other cars, it no longer works out every car's keys and values first: those layers are folded into the look itself, and the transformer's last block works out only the one token the policy and value layers read. The bot it trains and the files it saves are the same as without it; only training speed changes.\n"
        "\n"
        'When to change it: turn it on with model.encoder = "transformer" (27-40% faster) and with "attention_communication" (a little faster). With "attention", keep model.attention_fast instead: lean attention made it no faster overall. A plain network ignores it.\n'
        "\n"
        'How to use it: true or false. It is not fixed, so you can switch it between a stop and a resume, and the exported bot is unchanged. With "attention" it takes the place of model.attention_fast.\n'
        "\n"
        "Watch out for: the answers equal the usual ones to rounding, not bit for bit, so a run switched on a resume carries on with numbers a hair different. Each look's key bias stops moving; it never changed what the bot does.\n"
        "\n"
        "Measured (on our test machine, the same weights with it on and off):\n"
        "- transformer: learning 27-40% faster, choosing actions 27-40% faster, about 20% less memory.\n"
        "- attention_communication: learning 1-2% faster, choosing actions 6-9% faster, 12-21% less memory.\n"
        "- attention: learning 3-12% slower than attention_fast, choosing actions 3-17% faster; a 2v2 attention bot trained at 94,018 against 93,718 agent steps a second, inside the noise.\n"
        "- Checked exactly on the card (FP64, every setup): the same answers and gradients as the usual network. In FP32 a few gradients differ by up to 0.5% from rounding alone."
    ),
    "model.transformer_layers": (
        "What it does: sets how many transformer blocks the bot's encoder has. In each block every token (the bot's own car, the ball, every other car and, with model.transformer_pads, every boost pad) looks at all the others, then thinks on what it saw. Each block builds on the one before, so more blocks can follow longer chains of play, like who covers whom while the ball is where.\n"
        "\n"
        'When to change it: only with model.encoder = "transformer", and only for a bigger network. Other networks ignore it.\n'
        "\n"
        "How to use it: 2 is the default and the suggested start, like the attention encoders' 2 blocks. The limit is 12. Each block adds about 200,000 parameters at width 128.\n"
        "\n"
        "Watch out for: every block costs training time, as every token looks at every other one. For a transformer bot it is fixed once the bot has saves: change it by making a new bot from this one (settings only; transfer learning needs the same networks). For other bots it is ignored and free to change.\n"
        "\n"
        "Measured: not measured; the transformer is checked for working only."
    ),
    "model.transformer_width": (
        "What it does: sets how many numbers the transformer uses to describe each token: each car, the ball, the bot's own car and, with pads as tokens, each pad. More numbers can capture more detail, at the cost of speed.\n"
        "\n"
        'When to change it: only with model.encoder = "transformer", and only when building a much bigger network than usual. Other networks ignore it.\n'
        "\n"
        "How to use it: 128 is the default, the size the attention encoders were measured at; 256 for a much bigger network, with 8 heads. It must divide evenly by model.transformer_heads (128 / 4 = 32 per head); the settings check refuses a pair that does not. Change model.transformer_ff with it, usually to 4 times the width.\n"
        "\n"
        "Watch out for: wider means slower training. For a transformer bot it is fixed once the bot has saves: change it by making a new bot from this one (settings only; transfer learning needs the same networks). For other bots it is ignored and free to change.\n"
        "\n"
        "Measured: not measured; the transformer is checked for working only."
    ),
    "model.transformer_heads": (
        'What it does: splits each block\'s attention into several separate "looks" taken at the same time, so one part of the network can follow the nearest opponent while another follows a teammate or the ball.\n'
        "\n"
        'When to change it: only with model.encoder = "transformer", mainly together with model.transformer_width. Other networks ignore it.\n'
        "\n"
        "How to use it: 4 is the default and the suggested value, as for the attention encoders; 8 with a width of 256. The width is shared out between the heads, so it must divide evenly: 128 with 4 gives 32 each.\n"
        "\n"
        "Watch out for: more heads with the same width means each head has fewer numbers to work with. For a transformer bot it is fixed once the bot has saves: change it by making a new bot from this one (settings only; transfer learning needs the same networks). For other bots it is ignored and free to change.\n"
        "\n"
        "Measured: not measured; the transformer is checked for working only."
    ),
    "model.transformer_ff": (
        "What it does: sets how wide each transformer block's feed-forward part is. After the tokens have looked at each other, each one works on what it saw through this layer, on its own.\n"
        "\n"
        'When to change it: only with model.encoder = "transformer", together with model.transformer_width. Other networks ignore it.\n'
        "\n"
        "How to use it: the usual choice is 4 times model.transformer_width: 512 for the default width of 128, 1024 for 256. Smaller is faster and thinks less per block.\n"
        "\n"
        "Watch out for: it holds most of each block's parameters (about two thirds at 4 times the width), so it is the main lever on the transformer's size and speed. For a transformer bot it is fixed once the bot has saves: change it by making a new bot from this one (settings only; transfer learning needs the same networks). For other bots it is ignored and free to change.\n"
        "\n"
        "Measured: not measured; the transformer is checked for working only."
    ),
    "model.transformer_pads": (
        "What it does: makes each of the 34 boost pads a token of its own in the transformer: whether it is up (and how soon it comes back), plus a learned place on the field. The bot can then look at the pads the way it looks at cars. When off, the pads' numbers go into the bot's own car token instead, as the other networks read them.\n"
        "\n"
        'When to change it: only with model.encoder = "transformer", when you want to try whether the bot plans its boost better with the pads as tokens. Other networks ignore it.\n'
        "\n"
        'How to use it: true or false; false is the default. nitro watch then shows "the boost pads" in the bot\'s "Looking at" list, all pads together.\n'
        "\n"
        "Watch out for: 34 more tokens in every block, so training is much slower (a 3v3 game goes from 7 tokens to 41, and every token looks at every other). For a transformer bot it is fixed once the bot has saves: change it by making a new bot from this one (settings only; transfer learning needs the same networks).\n"
        "\n"
        "Measured: not measured; checked for working only."
    ),
    "model.transformer_readout": (
        'What it does: decides which token the policy and value layers read after the transformer\'s last block. "own_car" reads the bot\'s own car token, as bots that attend from their own player do. "summary" adds one more learned token that belongs to nothing on the field and looks at everything, and reads that one.\n'
        "\n"
        'When to change it: only with model.encoder = "transformer", to try the summary token. Other networks ignore it.\n'
        "\n"
        'How to use it: "own_car" (the default) or "summary". In nitro watch the summary token\'s own share shows as "nothing".\n'
        "\n"
        "Watch out for: the summary token is one more token in every block, so it costs a little speed. For a transformer bot it is fixed once the bot has saves: change it by making a new bot from this one (settings only; transfer learning needs the same networks).\n"
        "\n"
        "Measured: not measured; neither choice has been compared."
    ),
    "start_from.enabled": (
        "What it does: switches transfer learning on or off. On, the bot's first start copies the trained networks of the bot named in start_from.run and trains on from there. Off, it starts from scratch, and the bot named in run stays written down for later.\n"
        "\n"
        "When to change it: before the bot's first start, whenever you change your mind: for example to compare a bot started from your trained one with one started from scratch on the same settings.\n"
        "\n"
        "How to use it: the switch at the top of the Settings tab (Transfer learning). New bot from this one switches it on when its box is ticked and off when not, and names the bot it was made from either way. The start button says what will happen: Start from my-bot, or Start training from scratch.\n"
        "\n"
        "Watch out for: the transfer happens once, at the first start. After the first save the bot has its own networks and every later start carries on from them (Carry on training), so this can no longer change. Off also wins over a bot named in the run's run.json by nitro train --from.\n"
        "\n"
        "Measured: nothing to measure; tested for working: on, the run starts from the other bot's networks; off, from scratch."
    ),
    "start_from.run": (
        "What it does: names the bot a new bot starts from, taking its trained networks instead of random ones (transfer learning, with start_from.enabled on). The new bot is its own run with its own saves; the old one is left untouched.\n"
        "\n"
        'When to change it: to carry a trained bot into a new game size, new start states, new learning settings, or new rewards, without starting from zero. The dashboard\'s "New bot from this one" fills it in, and nitro train --from does it from the command line.\n'
        "\n"
        "How to use it: in the Settings tab, pick another bot with a save from the list. In the file: the other run's name (its folder in runs), or the path of a run or of one of its saves. Pick the save with start_from.at and what to take with start_from.parts. With env.obs_team_size the same in both, a 1v1 bot can start a 3v3 run; with an attention network (the same encoder in both) any size can start any other, whatever the rooms.\n"
        "\n"
        "Watch out for: both bots must have the same network sizes, layer_norm, activation and observation, or the run is refused before training with every difference named. Only a brand-new run reads this; a resumed bot keeps its own networks. It cannot be used with [pretrain]. The optimizer's memory, the self-play pool and the step count start afresh.\n"
        "\n"
        "Measured: tested for working. Whether starting from a trained bot reaches further than training from scratch is not measured."
    ),
    "start_from.at": (
        "What it does: picks which save of the other bot (start_from.run) the new bot starts from, by the agent steps that save was made at.\n"
        "\n"
        "When to change it: when the newest save is not the one you want, for example because the bot got worse later in its run, or you want to branch off from an earlier point.\n"
        "\n"
        "How to use it: in the Settings tab, pick one of the other bot's saves from the list. In the file: the save's agent-step count, which is the folder name in that run's checkpoints folder. 0 (the default) means its newest save at the moment this bot starts. New bot from this one writes the newest save's steps, so the copy starts from the save you saw even if the other bot trains on. From the command line, nitro train --from my-bot --from-at STEPS does the same. If the number does not match a save, the run is refused and names the saves that exist.\n"
        "\n"
        "Watch out for: a run keeps only its newest few saves plus milestones (run.keep_checkpoints, run.milestone_steps), so an older step count may already have been deleted. Like the rest of [start_from], it is only read when a new bot starts.\n"
        "\n"
        "Measured: nothing to measure; tested for working."
    ),
    "start_from.parts": (
        'What it does: chooses how much of the other bot to take. "all": the policy, the value network and the reward and value scaling. "policy": only the policy (and any shared trunk), with a fresh value network. "matching": the policy and shared trunk whole, every part of the value network whose layers are the same size, and the scaling; the rest of the value network starts fresh.\n'
        "\n"
        'When to change it: "all" when the rewards stay the same and something else changes, such as the game size. "policy" when you change the rewards: the old value network learned what the old rewards are worth and would mislead the new run. "matching" for the same bot and rewards with another value network, such as a lighter critic or fewer value heads.\n'
        "\n"
        'How to use it: "all" (the default), "policy" or "matching". New bot from this one sets "all", since the copy keeps the rewards. When you apply new rewards to a bot that has not started yet, the dashboard switches it to "policy" and says so (a "matching" bot whose value network is another size than the other bot\'s keeps "matching"); you can change it back.\n'
        "\n"
        'Watch out for: "all" and "matching" need the same ppo.popart setting in both bots. With "policy" and a shared trunk, the trunk is taken too. "matching" never takes part of a policy: the policy and shared sizes must be the other bot\'s, so the new bot decides exactly as the other one. The entropy bonus ppo.entropy_target moved is not taken: it restarts at ppo.entropy.\n'
        "\n"
        "Measured: tested for working: what is copied and what starts fresh. Which choice gives a better bot is not measured."
    ),
    "ppo.steps_per_iteration": (
        "What it does: the bot plays this many agent steps across all arenas, then stops to learn from them in one update. The number is rounded up so every arena plays the same whole number of steps, so the real batch is a little larger (between about 50,176 and 65,536).\n"
        "\n"
        "When to change it: rarely. Bigger batches give each update steadier, less noisy data but fewer updates per hour; smaller ones update more often on noisier data.\n"
        "\n"
        "How to use it: the default and all three presets use 50,000, a common size. It works together with epochs and minibatches: each update cuts the batch into minibatches parts and goes over it epochs times. The batch must hold at least twice as many agent steps as there are minibatches, or the run refuses to start.\n"
        "\n"
        "Watch out for: the whole batch is kept on the graphics card. If the card runs out of memory during learning, the run halves the minibatch size for the rest of the run and says so; lowering this setting or env.num_arenas is the other fix.\n"
        "\n"
        "Measured: never compared."
    ),
    "ppo.epochs": (
        "What it does: how many times each update goes over the batch it just collected. More passes squeeze more learning out of the same games, but each pass moves the bot further from the version that played them, which PPO's clip has to hold back.\n"
        "\n"
        "When to change it: normally never. The default of 2 is the measured choice.\n"
        "\n"
        "How to use it: the total learning steps per update is epochs times minibatches (2 x 4 = 8 by default). Some frameworks use 1 pass of 1 minibatch. Allowed range 1 to 32. More passes cost learning time in proportion.\n"
        "\n"
        "Watch out for: lowering it looks attractive because updates get quicker, but on the practice task the time saved (about 18% per run, read on a busy PC, so only a hint) did not come close to making up for slower learning per step.\n"
        "\n"
        "Measured: on ball touch, 1 pass of 4 minibatches took 2.38 s to touch against 2.16 s for the default (worse on 9 of 10 seeds). 1 pass of 1 minibatch took 7.61 s, worse on every seed. So 2 stays."
    ),
    "ppo.minibatches": (
        "What it does: each pass over the batch is split into this many parts, and the networks take one learning step per part. More parts means more, smaller learning steps per update.\n"
        "\n"
        "When to change it: rarely. The run may also double it by itself if the graphics card runs out of memory during learning; it then tells you, and you can write the new value into the settings file to keep it.\n"
        "\n"
        "How to use it: the default and presets use 4. Total learning steps per update are epochs x minibatches (2 x 4 = 8). The batch (steps_per_iteration) must hold at least twice as many agent steps as this number.\n"
        "\n"
        "Watch out for: 8 learned a little faster per step in the test, but it took about 29% more time per run on a busy PC (not a real speed measurement), so it is not clearly better per hour.\n"
        "\n"
        "Measured: on ball touch, 1 minibatch with 1 pass was far slower: 7.61 s against 2.16 s to touch. 8 minibatches with 2 passes was borderline: 2.05 s (better on 5 seeds, the same on 5), which by the rules changes nothing. So 4 stays."
    ),
    "ppo.graphed_steps": (
        "What it does: On a graphics card, each learning step (the network's forward and backward passes, gradient clipping and the optimizer) is recorded once as a CUDA graph and then replayed as one job, instead of Python handing the card several hundred small jobs one by one. On a fast card with a small or medium network the card spends most of a learning step waiting for those jobs.\n"
        "\n"
        "When to change it: On by default. It pays most for 1v1 bots of a few million weights on a fast card, where learning waits on the processor; for a big network that keeps the card busy (a big transformer) it changes little. Turn it off if a card runs short of memory with it (a recorded step keeps its own working memory).\n"
        "\n"
        'How to use it: true or false. It can be switched between a stop and a resume. On several graphics cards it needs run.cards_graphed_steps. It also works with ppo.optimizer = "muon" (ppo.muon_batched too) and the GCO learning signals (goal_critic_weight, vdag_beta, advantage_keep, entropy_target); their card checks are tests/test_graphed_gco.py.\n'
        "\n"
        "Watch out for: It uses more graphics memory (a recorded step keeps its own working memory). A full minibatch gives the usual step to the bit.\n"
        "\n"
        "Measured: one learning step of a 1v1 bot's 4x512 network on a fast card, 15.9 ms one job at a time against 8.3 ms replayed (1.93x, the same result to the bit); in real training a 4x512 network ran 1.24x faster there (449k to 557k steps a second)."
    ),
    "ppo.microbatch": (
        "What it does: works each learning step out in smaller parts, one after the other, and adds their results up into the same step. The bot learns the same (up to tiny rounding); only the graphics card's memory use drops, and learning gets a little slower.\n"
        "\n"
        "When to change it: when a big network's learning steps do not fit in the card's memory, such as a 31M-parameter transformer on an 8 GB card. Unlike raising minibatches, it does not change how the bot learns.\n"
        "\n"
        "How to use it: 0 = each step at once. Otherwise the most samples worked out at once, such as 4096 on an 8 GB card for the 2s transformer, or 16384 on a 24 GB card. A step is steps_per_iteration / minibatches samples.\n"
        "\n"
        "Watch out for: smaller parts are slower; pick the biggest that fits.\n"
        "\n"
        "Measured: a test checks that a step in parts gives the same networks as the whole step. Memory, measured on the processor at full precision for the 2s transformer: about 0.6 GB per 1000 samples of a step."
    ),
    "ppo.policy_lr": (
        "What it does: how big a step the policy network (the part that picks actions) takes each time it learns. Higher learns faster but risks updates that overshoot and undo what the bot knew.\n"
        "\n"
        "When to change it: if warnings say an update moved the policy much more than usual, lower it. The default, 4.5e-4, measured better than 1.5e-4 (below).\n"
        "\n"
        "How to use it: 4.5e-4 by default; big networks often use 1e-4 to 2e-4 (the GCO starter uses 1.5e-4 with Muon).\n"
        "\n"
        "Watch out for: the value network has its own rate (value_lr), and an attention encoder can have a multiple of it (encoder_lr_scale).\n"
        "\n"
        "Measured:\n"
        "- Boost pickup: 4.5e-4 collected 88.1 boost against 59.4 (better on 9 of 10 seeds).\n"
        "- Recovery: no measurable difference (1.20 s against 1.21 s to land).\n"
        "- Ball touch: borderline faster (2.06 s against 2.16 s), a hint only.\n"
        "Better on one task and worse on none, so it is the default."
    ),
    "ppo.value_lr": (
        "What it does: how big a step the value network takes each time it learns. The value network predicts how much reward is coming from a situation; those predictions are what tell the policy whether an action was better or worse than expected.\n"
        "\n"
        "When to change it: if warnings say the value loss jumped, or that the value network has predicted worse than guessing the average for many iterations, this is one of the things to look at (lowering it for jumps).\n"
        "\n"
        "How to use it: default 1.5e-4, the same as policy_lr, and all presets keep it.\n"
        "\n"
        "Watch out for: with a shared trunk (model.shared not empty), the shared layers follow the policy's rate, not this one. With an attention network, encoder_lr_scale multiplies this for the value network's encoder.\n"
        "\n"
        "Measured: not measured on its own (the learning-rate comparison changed only the policy's rate)."
    ),
    "ppo.encoder_lr_scale": (
        'What it does: only with an attention network (model.encoder = "attention" or "attention_communication"). It gives the attention encoders their own learning rate, as a multiple of their network\'s: the policy\'s encoder gets policy_lr times this, the value\'s gets value_lr times this. The rest of each network keeps the plain rate.\n'
        "\n"
        "When to change it: only when using attention and the encoder seems to learn slowly. With the plain network (mlp) it does nothing.\n"
        "\n"
        "How to use it: 1.0 means the same rate as the rest (no change). Allowed range 0.01 to 100. The 2v2 and 3v3 presets mention 3.0 in a comment (three times faster for the encoder). Gradient clipping still works per whole network.\n"
        "\n"
        "Watch out for: a saved run can be resumed whichever way it was set; changing it mid-run is allowed.\n"
        "\n"
        "Measured: not measured. It was built because attention learned ball touch slower than the plain network (2.38 s against 2.08 s), and the comparison meant to test this rate was cancelled before it started."
    ),
    "ppo.discount_half_life": (
        "What it does: sets the bot's time horizon. A reward this many seconds in the future counts half as much as one right now; twice as far counts a quarter. It is written in seconds rather than as a gamma, so it means the same whatever the tick skip.\n"
        "\n"
        "When to change it: when the bot should care about longer plays (setting up shots, rotating) or, the other way, when you want it to focus on the immediate (early training, simple drills).\n"
        "\n"
        "How to use it: default 4.6 s, which equals a gamma of 0.995 at tick skip 4 (0.99 at tick skip 8), and all presets use it. Longer plans further ahead but learns more slowly, since far-off rewards are harder to predict.\n"
        "\n"
        "Watch out for: it is used in two places, the learner's advantages and value targets and the reward scaling. Changing it on a running bot means the value network must re-learn its predictions, so expect the value loss to rise for a while.\n"
        "\n"
        "Measured: not compared here; a widely used value."
    ),
    "ppo.gae_lambda": (
        "What it does: controls how advantages (how much better an action turned out than expected) are estimated. Near 1 it trusts the rewards that actually came, which is accurate on average but noisy. Lower values lean more on the value network's predictions, which is steadier but only as good as those predictions.\n"
        "\n"
        "When to change it: rarely; there is no sign here that the default is a problem.\n"
        "\n"
        "How to use it: default 0.95, a common value at tick skip 8, where with the presets' 4.6 s half-life the advantages look about 1.1 s ahead. The presets keep 0.95 at every tick skip, so at tick skip 4 they look about 0.6 s ahead; 0.975 there keeps the 1.1 s. Allowed range 0 to 1. It works together with the discount half-life: the half-life sets how far ahead rewards count, lambda sets how the estimate within that horizon is made.\n"
        "\n"
        "Watch out for: at 0 the advantages rest entirely on the value network, which is poor early in training. Lambda counts decisions, not seconds: a bot started from one at another tick skip gets a note in its log with the same-seconds value.\n"
        "\n"
        "Measured: not compared here; a widely used value, left out of the comparisons on purpose."
    ),
    "ppo.goal_ends": (
        'What it does: chooses how the learner treats a goal. "terminal" (the default): a goal ends the game for the learner, so nothing after it counts. "continue": the goal\'s moment also counts what the next game\'s first moment is worth, as a real match goes on to a kickoff. Today a goal cuts off all the small rewards the bot would have kept earning afterwards, so in a run with many such rewards scoring looks worth less than its weight and conceding costs more; "continue" takes that away. The goal\'s own pay is unchanged.\n'
        "\n"
        "When to change it: only for a new bot with many small rewards, or as a deliberate switch you choose. Not for bots paid only for goals and concedes: there the next game's start is worth about 0 to both sides, so it changes nothing.\n"
        "\n"
        'How to use it: set "continue" in [ppo]. Training refuses it with terminals.end_on_touch or end_on_landing (they end games for other reasons than a goal), and with ppo.goal_critic_weight or ppo.vdag_beta.\n'
        "\n"
        "Watch out for: it changes what the bot learns, so a running bot would drift from what it learned so far. After a goal the next game starts from your start mix, not always a kickoff. The charts start_value and start_value_share (every run, whatever this setting) show what a fresh start is worth on average (every start the bot learns from): about what a goal cuts off today.\n"
        "\n"
        'Measured: matches hand-worked values in tests, and "terminal" learns exactly as a plain goal ending; whether "continue" learns better is not measured.'
    ),
    "ppo.clip_range": (
        "What it does: PPO's safety limit. In one update, the chance of picking any action may only move by about this fraction (0.2 = 20%) before further learning on that sample stops counting. It keeps one batch of games from pushing the bot too far at once.\n"
        "\n"
        "When to change it: rarely. There is no sign here that the default is a problem.\n"
        "\n"
        "How to use it: default 0.2, a common value; the presets leave it at that. Allowed range 0 to 1. Smaller is more cautious, larger allows bigger jumps. It matters more with more passes per update (epochs) and higher learning rates, since those push further from the version that played.\n"
        "\n"
        "Watch out for: 0 would stop the policy from learning anything. The metrics column policy_change shows how far each update actually moved the policy.\n"
        "\n"
        "Measured: not compared here; a widely used value, left out of the comparisons on purpose."
    ),
    "ppo.entropy": (
        "What it does: a bonus for keeping the bot's choices varied, so it keeps trying things instead of locking in early on one habit. Higher means more exploring and slower settling.\n"
        "\n"
        "When to change it: if a warning says the bot has become almost certain of its choices (entropy below 0.1) and learning stalls, raise it a little. If the bot stays too random late in a run, a lower value is the lever.\n"
        "\n"
        'How to use it: default 0.035, a common value, used by all presets. The randomness it multiplies runs from 0 (always the same choice) to 1 (all choices equally likely); entropy_normalization decides what counts as all choices, and with "all" the same number is a weaker bonus.\n'
        "\n"
        "Watch out for: its strength is relative to the size of the advantages. A practice-task comparison (nitro ab) scores the bot sampled, which by itself favours the less random setting: add --also-likeliest, or --likeliest-decides, to score it as the RLBot bot plays.\n"
        "\n"
        "Measured: not compared here; a widely used value."
    ),
    "ppo.entropy_normalization": (
        'What it does: decides what the entropy bonus measures randomness against. "allowed" compares it with the actions the bot may take at that moment (some actions are not allowed in some situations), so the bonus means the same in every situation. "all" divides by the full action list (90 actions), as some other frameworks do.\n'
        "\n"
        "When to change it: mainly to match another framework that counts it this way, for example when copying its settings.\n"
        "\n"
        'How to use it: default "allowed". With "all", the same entropy number gives a weaker bonus, so 0.035 under "all" is not the same push as 0.035 here.\n'
        "\n"
        "Measured: not measured as a learning comparison."
    ),
    "ppo.optimizer": (
        'What it does: chooses how the learning steps move the weights. "adam" is what every bot has used. "muon" moves the weight matrices of the hidden layers with Muon: it takes the direction of the gradient (with momentum) and evens out its sizes, so every direction in the matrix moves about as much, instead of a few large ones dominating. Everything else (biases, normalisation, the first layer reading the observation, the output layers, attention encoders) stays with Adam.\n'
        "\n"
        "When to change it: for the GCO starter (configs/presets/gco_1v1_starter.toml) and other big networks you want to try it on. Muon has done well for large networks elsewhere; here it is not measured yet.\n"
        "\n"
        'How to use it: "adam" or "muon". It uses ppo.policy_lr and ppo.value_lr as they are: a Muon matrix\'s rate is scaled by the square root of its larger side (and by ppo.muon_lr_scale), so its numbers move about 0.6-0.9 times the rate each step. On the shared trunk and the value side that is 3-5 times what Adam\'s would move at the same rate on PPO\'s gradients; on the policy the two move roughly alike. It works with ppo.graphed_steps. It can be switched between a stop and a resume; the optimizer then starts afresh.\n'
        "\n"
        "Watch out for: each Muon step costs 15 matrix products per layer (for the GCO bot's networks about a fifth more arithmetic per learning step, counted from the sizes, not measured). On graphics cards that support BF16 (NVIDIA's from 2020 on) it works in BF16, elsewhere in FP32. Without weight decay the Muon matrices keep growing (on a long run 10-16 times their starting size), and since each feeds a normalisation, every step then turns them less: the turn_ and growth_ charts show it (ppo.weight_decay holds it).\n"
        "\n"
        "Measured: matches its reference implementations in tests and resumes exactly; how it learns and its speed are not measured."
    ),
    "ppo.muon_batched": (
        'What it does: with ppo.optimizer = "muon", the weight matrices of one shape take their Muon steps together: the momentum and the moves as a few list operations, the five Newton-Schulz steps as batched products. The same maths; one learning step makes a few dozen GPU jobs for Muon instead of about 30 per matrix.\n'
        "\n"
        "When to change it: where learning waits for Python to start GPU jobs, not for the card: a small or medium network on a fast card (a profile there showed Muon's step taking most of each learning step's processor time).\n"
        "\n"
        "How to use it: true or false; it can be switched between a stop and a resume (the optimizer's saves are the same).\n"
        "\n"
        "Watch out for: the same learning to rounding, not to the bit: a batched product may add its numbers up in another order.\n"
        "\n"
        "Measured: not yet."
    ),
    "ppo.muon_lr_scale": (
        'What it does: with ppo.optimizer = "muon", multiplies the rate of the Muon matrices only; everything Adam moves keeps ppo.policy_lr and ppo.value_lr.\n'
        "\n"
        "When to change it: for a new run or a narrow test, not a bot that is training. 0.2 gives Muon the step size Moonlight matched to Adam's. A value of 1/r makes a fresh network's Muon matrices learn as if they had grown r times bigger: each feeds a normalisation, so only its step relative to its size matters. 1/15 imitates the inner matrices of a bot trained for billions of steps on a short practice task, but not all of that bot: their biases (moved by Adam) keep their full step, where in a grown network they grew too, and the first layer (Adam, 4.5 times its start there) is not imitated.\n"
        "\n"
        "How to use it: a number above 0, at most 10; default 1 (no change). The settings check refuses anything but 1 without Muon, where it would do nothing.\n"
        "\n"
        "Watch out for: on a running bot any other value is a large change of its learning rate in one go.\n"
        "\n"
        "Measured: the maths in tests (0.2 moves each matrix 0.2 times as far, Adam's part unchanged); how it learns is not measured."
    ),
    "ppo.weight_decay": (
        "What it does: after every update, shrinks each hidden layer that feeds a normalisation (its weights and bias together; not the output layers, the normalisations or attention encoders) by (1 - rate x weight_decay) for each learning step that moved it, at its network's current rate. The normalisation undoes the shrink, so the network computes the same; what changes is that later steps turn the layer by more, because they are no longer small against ever bigger weights.\n"
        "\n"
        "When to change it: only for a new run after the narrow test shows that the slow-down from growing weights costs learning. Muon's matrices grow without it (10-16 times their start on a long run, so they turn 10-16 times less per step than at first).\n"
        "\n"
        "How to use it: 0 = off (default). It settles where weight_decay = 30 x turn^2 / (2 x rate): at rate 1.5e-4, 0.1 holds about 1e-3 turn a step (weights about 3 times their start), 0.7-1 about the starting turn. The turn_ charts show the turn per update, about the turn a step times the update's learning steps (epochs x minibatches): divide by those before using the rule. The growth_ charts show the size against the start, weight_norm_ each network's size. Needs model.layer_norm.\n"
        "\n"
        "Watch out for: on an already grown bot it raises the effective rate gradually (about 1 / (2 x rate x weight_decay) steps); it also removes the slow-down that growth gives late in a run, which may have been helping.\n"
        "\n"
        "Measured: the maths in tests (outputs unchanged, exact resume); how it learns is not measured."
    ),
    "ppo.entropy_target": (
        "What it does: holds the bot's randomness near this level by moving the entropy bonus after every learning update: when the measured randomness (the entropy chart, 0 = certain, 1 = picking at random among the allowed actions) is below the target the bonus grows, when above it shrinks. ppo.entropy is where the bonus starts.\n\n"
        "When to change it: for a goal-only bot (the GCO starter uses 0.5), where nothing else stops the bot from becoming certain of its choices before it has found its first goals. A goal-only bot that beat Nexto was credited to such a controller; the design gave no target, so 0.5 is a guess (a shaped-reward run with the usual bonus settled near 0.55).\n\n"
        "How to use it: a number between 0 and 1; 0 = off (the bonus stays ppo.entropy). The bonus moves at ppo.entropy_target_rate and stays between ppo.entropy_target_min x ppo.entropy (ppo.entropy / 100 by default) and 10 x ppo.entropy. The entropy_coef chart shows it.\n\n"
        "Watch out for: needs ppo.entropy above 0. A resumed run carries on from the bonus it had reached. "
        "A practice-task comparison (nitro ab) scores the bot sampled, which by itself favours the less random setting: add --also-likeliest, or --likeliest-decides, to score it as the RLBot bot plays.\n\n"
        "Measured: moves the bonus the right way in tests; how it learns is not measured."
    ),
    "ppo.entropy_target_rate": (
        "What it does: with ppo.entropy_target, how fast the entropy bonus moves. After each update the bonus is multiplied by exp(rate x (target - entropy)): with rate 1 and the randomness 0.05 below the target, it grows by about 5% an update.\n\n"
        "When to change it: lower it if the entropy chart swings up and down around the target (the bonus moves faster than the bot answers); raise it if the randomness stays far from the target for many updates while the bonus creeps.\n\n"
        "How to use it: a number from 0 to 100; 1 is the default. 0 holds the bonus where it is.\n\n"
        "Watch out for: the bonus never leaves ppo.entropy_target_min x ppo.entropy to 10 x ppo.entropy, whatever the rate.\n\n"
        "Measured: not measured."
    ),
    "ppo.entropy_target_min": (
        "What it does: with ppo.entropy_target, the lowest the entropy bonus may go, as a share of ppo.entropy. 0.01, the default, lets it fall to ppo.entropy / 100. 1 keeps it at ppo.entropy or above: the controller can then only raise the bonus, when the randomness falls under the target, and never take it away.\n\n"
        "When to change it: when the entropy_coef chart shows the bonus sinking to its floor while the bot is still learning, so it stops exploring. The GCO starter uses 1: a falling randomness at the start is normal learning, not something to punish, so the bonus stays at least the design's 0.035.\n\n"
        "How to use it: a number above 0, at most 10. It can change between a stop and a resume: a saved bonus below the new floor is raised to it at once.\n\n"
        "Watch out for: set at 10 the bonus can no longer move at all (10 x ppo.entropy is also its ceiling).\n\n"
        "Measured: keeps the bonus within its range in tests; how it learns is not measured."
    ),
    "ppo.advantage_keep": (
        "What it does: in each learning step, the policy learns only from this share of the samples, those whose advantage (how much better or worse a choice went than expected) is largest in size, good or bad. 0.5 keeps the half that says most about which choices to make more or less often. The value heads still learn from every sample.\n\n"
        "When to change it: for a goal-only bot (the GCO starter uses 0.5, as the design it follows), where most samples carry almost no signal.\n\n"
        "How to use it: a number between 0 and 1; 1 (or 0) = off, every sample. The charts advantage_kept and advantage_kept_size show the share of samples kept and the share of the advantages' total size they hold.\n\n"
        "Watch out for: it keeps both tails (by size), never only the best: keeping only the best choices turns learning into copying itself.\n\n"
        "Measured: keeps the right samples in tests; how it learns is not measured."
    ),
    "ppo.advantage_keep_rest": (
        "What it does: with ppo.advantage_keep, lets the learning steps skip most of the samples the policy ignores anyway. The steps run on the kept samples plus this share of the others, drawn at random each update. Those others only teach the value heads and the entropy bonus, and each counts 1/share, so their averages stay those over every sample. The policy still learns from each step's kept samples, but which are kept is decided once per update, not per step. A value warm-up keeps every sample. Fewer samples means less work.\n\n"
        "When to change it: only to test it, on a bot that uses advantage_keep. 0.2 at advantage_keep 0.5 runs the steps on about 60% of the samples; by a count of the multiply-adds that could make a big GCO bot's learning up to about 1.3x faster, not measured.\n\n"
        "How to use it: a number above 0, at most 1; 1, the default, = every sample. It needs advantage_keep between 0 and 1, graphed_steps = false and normalize_advantages, and does not work yet with several cards or ppo.value_share below 1. The chart advantage_rows shows the share of samples the steps run on.\n\n"
        "Watch out for: the value heads see the low-advantage samples only through a small random share, so their learning there is noisier (not biased); near 0 the few drawn samples count very heavily. 0 itself is refused: the kept samples' randomness is lower than the rest's (0.42 against 0.63 on one long run's save), so the entropy controller would read the wrong number.\n\n"
        "Measured: in tests, with one learning step per update the policy learns exactly as with advantage_keep alone (with more, only the choice per update differs), and the value and entropy means are unbiased. How it learns is not measured."
    ),
    "ppo.value_share": (
        "What it does: in each learning step, the value side (the value network, or the critic trunk and every value head) learns from only this share of the step's samples, while the policy still learns from all of them. 0.5 = half. The samples of a step come in a random order, so the share is a random half of them. Passes that train the value side alone (a value warm-up) still use every sample.\n\n"
        "When to change it: a speed setting for bots whose value side is most of the learning work, such as the GCO starter's at the design's full size, where a joint step's work falls to about 61% at 0.5 and 41% at 0.25 (counted, not timed). Leave it at 1 for every bot until it is checked.\n\n"
        "How to use it: a number from 0.05 to 1; 1 = every sample. It can change between a stop and a resume. With ppo.microbatch the share is taken of each part.\n\n"
        "Watch out for: the value loss and the value heads' losses are then means over the share learned from, so they are a little noisier; explained variance is still measured on every sample. The value side's steps get noisier too, which could slow how well it predicts.\n\n"
        "Measured: the same networks to the bit at 1, and the hand-computed losses at 0.5, in tests; its speed and how it learns are not measured."
    ),
    "ppo.goal_critic_weight": (
        "What it does: switches on the goal critic: the value head \"goal\" learns the bot's reward over a much longer horizon than the main value (ppo.goal_critic_half_life, 77 seconds), with advantages of its own, and those advantages, centred and scaled to this share of the usual advantages' spread, are added to them. A bot that only gets a reward for goals then also hears, a little, how its play changes its chances of a goal much later.\n\n"
        'When to change it: for a goal-only bot (the GCO starter uses 0.25, the design\'s value). It needs "goal" in model.value_heads.\n\n'
        "How to use it: a number; 0 = off. 0.25 adds a quarter of the usual advantages' spread. The charts goal_explained_variance and goal_loss show how well the goal critic predicts.\n\n"
        "Watch out for: in a run with other rewards the goal critic learns all of them, not only goals. Works on several graphics cards (run.cards).\n\n"
        "Measured: its advantages and blend match hand-computed values in tests; how it learns is not measured."
    ),
    "ppo.goal_critic_half_life": (
        "What it does: the goal critic's horizon: the seconds after which a future reward counts half as much to it. 77 s is gamma 0.9994 at tick skip 8, the design's value, against the main value's ppo.discount_half_life. Its advantages use ppo.gae_lambda, as the main value's.\n\n"
        "When to change it: only with ppo.goal_critic_weight above 0. Longer reaches further ahead but its predictions get noisier.\n\n"
        "How to use it: seconds, above 0.\n\n"
        "Watch out for: the advantages themselves look only about 1.3 s ahead at tick skip 8 with gae_lambda 0.95 (0.65 s at tick skip 4 unless lambda is raised to 0.975), whatever the horizon; the long horizon lives in the critic's values.\n\n"
        "Measured: not measured."
    ),
    "ppo.vdag_beta": (
        'What it does: switches on the twin optimistic critics (V-dagger). The value heads "vdag1" and "vdag2" learn what the bot could get from where it is: each step\'s reward plus the lower of the two critics\' values of the next state, learned optimistically (an upper expectile, ppo.vdag_expectile), so a move that worked somewhere counts even if the bot rarely strings such moves together. How that value changes along each step is added to the advantages (potential-based shaping, which cannot change which play is best in the end), centred, scaled to this share of the advantages\' spread and cut at 3 times it. It pushes the bot toward states it could do well from but rarely visits.\n\n'
        'When to change it: for a goal-only bot (the GCO starter uses 0.30; 0.30 to 0.45 worked best in the design this follows, and 1 or more worse than none). It needs "vdag1" and "vdag2" in model.value_heads.\n\n'
        "How to use it: a number; 0 = off. The charts vdag_dose (the push's spread as a share of the advantages', about this value), vdag_headroom, vdag_loss and vdag_clamped show it at work.\n\n"
        "Watch out for: works on several graphics cards (run.cards). Read the dose from vdag_dose, not from this number.\n\n"
        "Measured: its targets, shaping and clamp match hand-computed values in tests; how it learns is not measured."
    ),
    "ppo.vdag_expectile": (
        "What it does: how optimistic the twin critics are. Their loss counts a prediction that is too low this much, and one that is too high 1 minus this much, so they learn an upper part of what happened rather than the average: 0.5 = the average, nearer 1 = the best outcomes.\n\n"
        "When to change it: rarely. The design uses 0.75; a single critic at 0.9 kept raising its own estimates without the bot playing any better.\n\n"
        "How to use it: a number from 0.5 to 0.99, with ppo.vdag_beta above 0.\n\n"
        "Watch out for: higher is more optimistic and less stable; the two critics and the lower of their values in the target keep that in check.\n\n"
        "Measured: the loss matches hand-computed values in tests; not measured in training."
    ),
    "ppo.vdag_reinit_steps": (
        "What it does: every this many agent steps the twin optimistic critics (ppo.vdag_beta) start again from fresh random weights and a fresh optimizer state, so their optimism cannot set in and they keep pointing at states the bot rarely visits. The rest of the networks are untouched.\n\n"
        "When to change it: periodic restarts keep the critics helping exploration, but no interval is published; the GCO starter uses 500,000,000 steps (untested).\n\n"
        "How to use it: agent steps; 0 = never. A Stop and Start carries on counting from the last restart.\n\n"
        "Watch out for: for a while after each restart the critics know little and their push is mostly noise (it stays scaled to vdag_beta of the advantages' spread).\n\n"
        "Measured: not measured."
    ),
    "ppo.head_loss_scale": (
        "What it does: sets how loudly the goal critic and the twin optimistic critics (ppo.goal_critic_weight, ppo.vdag_beta) pull on the layers they share with the main value: the critic trunk and the shared trunk. With 'raw' (the default) their losses are in the scaled reward's units, while the main value learns PopArt's normalised values; on a long GCO run that makes them about 21 times louder for the same relative mistake, so they, mostly the goal critic, set the direction of those layers' steps. 'popart' divides their losses by PopArt's current variance, the main value's units: for the trunks the same as a PopArt on each head using the main value's spread, not each head's own (on that run the goal critic's spread is about 4.8-5.1, the main value's 4.6, so it still weighs 9-23% more than with its own PopArt). Their predictions, targets, what they add to the advantages and the goal_loss and vdag_loss charts stay as they are.\n\n"
        "When to change it: only for a new GCO bot (with the goal critic or the twins), never in the middle of a bot's run. It needs ppo.popart = true.\n\n"
        "How to use it: 'raw' or 'popart'. With 'popart' the chart head_loss_weight shows the factor (1 / PopArt's variance, about 0.05 on a long GCO run).\n\n"
        "Watch out for: the critics then pull far less on the shared trunk, so the policy leads it; if their features helped the bot's representation, that help shrinks.\n\n"
        "Measured: a check on a long GCO run's save without training, two seeds: the main critic's progress that carries to other games rose 1.5 to 4 times per step in the critic trunk. In the shared trunk the critics' progress mostly fell (10 of 12 measures, x-0.07 to x1.54) and the policy's did not improve (less on one seed, slightly less harm on the other). How a bot plays with it is not measured."
    ),
    "ppo.lean_heads_pass": (
        "What it does: before each update the bot values its states with its extra value heads (the goal critic and the twin critics, ppo.goal_critic_weight and ppo.vdag_beta). With this on, that pass looks only at the bot's own cars, not at the past versions' cars in self-play games, which the update leaves out anyway; and with run.fp16_sums at 'forward' or 'all' it adds up its half-precision products in half precision, as learning's own passes already do. The pass always skips the main head, whose answer it threw away; that part needs no setting, as it gives the same numbers to the bit.\n\n"
        "When to change it: for a bot with those heads, once it has been timed on the card it runs on. It does nothing for a bot without them.\n\n"
        "How to use it: true or false; it can be switched between a stop and a resume.\n\n"
        "Watch out for: the heads' values move by rounding (the products sum in other groupings, and in half precision with run.fp16_sums).\n\n"
        "Measured: on the processor, without FP16 sums, the bot's own cars get the same values to within a hundred-thousandth, on small networks and on a big GCO bot's (there about a millionth; tests/test_lean_heads_pass.py). The rounding of the FP16 sums is not measured: it needs a card. The speed is an estimate (2-3% of an iteration for a big GCO bot), not measured on a card."
    ),
    "ppo.adam_eps": (
        "What it does: a tiny number in the optimizer (Adam) that stops it dividing by almost zero. It sets how large a step a weight can take when its gradients have been very small and steady.\n"
        "\n"
        "When to change it: practically never. It is a numerical safety value, not a tuning knob for play.\n"
        "\n"
        "How to use it: default 1e-5; the presets leave it at that. PyTorch's own default is 1e-8. Must be above 0; the settings check refuses 0.\n"
        "\n"
        "Watch out for: very small values can let rarely-used weights take big jumps; very large values slow every weight's learning, a bit like a lower learning rate.\n"
        "\n"
        "Measured: not compared here; left out of the comparisons on purpose."
    ),
    "ppo.max_grad_norm": (
        "What it does: caps how big a single learning step can be. If the combined push on a network's weights is bigger than this, it is scaled down to this size. It guards against one odd batch (a huge reward, a rare situation) wrecking the network. Each network is capped separately.\n"
        "\n"
        "When to change it: rarely. If warnings keep reporting that updates moved the policy far more than usual, lowering the learning rate is the first fix; this cap is a second line.\n"
        "\n"
        "How to use it: default 0.5, a common value; the presets leave it at that. Lower is safer but can slow learning; higher lets big steps through. Must be above 0; the settings check refuses 0, since it would zero every step and quietly stop learning.\n"
        "\n"
        "Watch out for: the cap applies to each whole network, including an attention encoder that has its own learning rate (encoder_lr_scale).\n"
        "\n"
        "Measured: not compared here; a widely used value, left out of the comparisons on purpose."
    ),
    "ppo.reward_clip": (
        "What it does: before learning, rewards are scaled so the bot's running returns have a steady spread. After that scaling, any single step's reward bigger than this, plus or minus, is cut down to it. It stops a rare huge payout from dominating an update.\n"
        "\n"
        "When to change it: if you use very large event rewards (a big goal reward, say) and want them to count in full, raise it or set 0. Otherwise leave it.\n"
        "\n"
        "How to use it: default 10, a common value; the presets leave it at that. 0 means no clipping. Because it applies after scaling, it is not in the same units as your reward weights; 10 means ten times the unit the rewards are scaled to.\n"
        "\n"
        "Watch out for: setting it low silently flattens big rewards, so a goal could end up paying no more than a good touch.\n"
        "\n"
        "Measured: not compared here; a widely used value, left out of the comparisons on purpose."
    ),
    "ppo.normalize_advantages": (
        "What it does: rescales the advantages (how much better each action turned out than expected) in every learning step to a standard size, so the size of the policy's learning step does not swing with how big the rewards happen to be. It rescales them to zero mean and unit spread.\n"
        "\n"
        "When to change it: do not turn it off; it was measured clearly worse without it.\n"
        "\n"
        "How to use it: on by default; the presets leave it on. Some frameworks do not normalise advantages; this default is on purpose.\n"
        "\n"
        "Watch out for: the entropy bonus is sized for advantages of standard size; turning normalisation off or changing its method changes how strong the entropy bonus effectively is.\n"
        "\n"
        "Measured: on ball touch, off took 2.96 s to touch against 2.16 s with it on, worse on every seed, and the second evaluation agreed (2.89 s). It stays on."
    ),
    "ppo.popart": (
        "What it does: PopArt keeps the value network's targets at a steady scale and adjusts the network when that scale shifts, so its predictions do not have to chase returns that grow or shrink over a run. Without it, the value network learns raw returns.\n"
        "\n"
        "When to change it: it is meant for long runs whose rewards grow over time. There is no measured reason to turn it off.\n"
        "\n"
        "How to use it: on by default; the presets leave it on.\n"
        "\n"
        "Watch out for: fixed once the bot has saves: change it by making a new bot from this one.\n"
        "\n"
        "Measured: on ball touch, off made no measurable difference: 2.17 s against 2.16 s to touch (2 seeds better, 6 the same, 2 worse). It stays on because no task measured a cost, and the long-run case it is built for is untested."
    ),
    "selfplay.enabled": (
        "What it does: some games put a frozen past version of the bot in the Orange cars instead of the current bot. Copies of the bot are saved into a pool as it trains, each past version gets a rating from its games against the bot, and only the current bot's own cars are learned from.\n"
        "\n"
        "When to change it: on a long run where you want the bot to keep beating older play styles, not just its newest self. It is standard practice in self-play training, but nothing here has shown it makes a better bot.\n"
        "\n"
        "How to use it: switch it on and leave the other self-play settings at the presets' values (one game in five against a past version, 4 at once, a copy every 20M steps). It needs an opposing team (env.opponents = true).\n"
        "\n"
        "Watch out for: env.opponents is fixed once the bot has saves, so a bot made without opponents cannot get self-play later; make a new bot instead. The pool lives in the run folder and takes disk space (selfplay.pool_disk_mb).\n"
        "\n"
        "Measured: the speed cost of the pipeline itself is nothing measurable: 393,803 steps/s off against 401,011 on with 4 past versions, on a fresh run with an empty pool. Whether it helps learning is not measured yet: only full games could show it."
    ),
    "selfplay.past_share": (
        "What it does: sets how many of the games are against past versions. The rest are the current bot playing itself. The share is of the arenas, so with 1,024 games and 0.2, about 205 games have a past version as Orange.\n"
        "\n"
        "When to change it: raise it if you want more practice against older styles; lower it to keep more of the data as pure self-play. There is no measured best value; 0.2 is the presets' choice.\n"
        "\n"
        "How to use it: it must leave enough games for every past version in every processing group. The settings check needs past_share x env.num_arenas (rounded) to be at least env.groups x selfplay.active_opponents. With the usual 2 groups and 4 active past versions, that is at least 8 games. The 3v3 preset has only 342 games, so small shares there run out sooner.\n"
        "\n"
        "Watch out for: games against past versions still count toward agent steps, but only the current bot's cars are learned from. A larger share therefore means fewer learning samples per agent step (about 20% fewer at 0.2).\n"
        "\n"
        "Measured: not measured as a learning setting."
    ),
    "selfplay.active_opponents": (
        "What it does: picks how many different past versions are playing at once. The games given to past versions (selfplay.past_share) are split evenly between them. At every save a new set is drawn from the pool, using the matchmaking rule and the ratings, so a past version never changes in the middle of a game.\n"
        "\n"
        "When to change it: more variety per save means more different opponents at once; fewer means each one gets more games, so its rating settles faster. The presets use 4.\n"
        "\n"
        "How to use it: values 1 to 16. Each one needs at least one game in every processing group, so the settings check asks for env.groups x active_opponents games against past versions. If the check complains, raise past_share or lower this.\n"
        "\n"
        "Watch out for: early in a run the pool is still small. An empty slot plays the current bot itself, and those games count as normal learning data. Each active past version is an extra network pass over its games.\n"
        "\n"
        "Measured: with 4 active past versions on a fresh run, no measurable speed cost. Not measured with a full pool or for learning."
    ),
    "selfplay.snapshot_steps": (
        "What it does: decides how often a copy of the bot joins the pool of past versions. The check runs every iteration, so a copy is added as soon as the step count is reached, not only at saves. A new copy starts with the bot's current rating.\n"
        "\n"
        "When to change it: lower it for more past versions close together in time, raise it to spread the pool over a longer history. The presets use 20 million.\n"
        "\n"
        "How to use it: agent steps count every car, including the past versions' cars and both teams. In 2v2 one tick of one game is 4 agent steps, so the same number means less game time than in 1v1. Together with pool_size it sets how far back the pool reaches: 64 copies at 20M each cover about 1.3 billion steps before the oldest have to go.\n"
        "\n"
        "Watch out for: more frequent copies fill the pool and its disk cap sooner, and then older versions are removed sooner.\n"
        "\n"
        "Measured: not measured."
    ),
    "selfplay.pool_size": (
        "What it does: caps how many past versions are kept. When the pool is full, the past version whose rating is closest to its neighbours is removed, never the newest. That keeps a spread of different strengths over the bot's history instead of just the latest ones.\n"
        "\n"
        "When to change it: raise it to remember more of the history, lower it to save disk space or to keep the pool closer to the current bot. The default and the presets use 64.\n"
        "\n"
        "How to use it: works together with selfplay.snapshot_steps (how often a copy is added) and selfplay.pool_disk_mb (whichever cap is hit first wins). Each copy stores only the policy (and a shared trunk if the bot has one), not the value network.\n"
        "\n"
        "Watch out for: the pool lives in the run folder (its pool folder) with a games log, pool/games.csv, that keeps every result for later rating reports. A very small pool (minimum 2) leaves matchmaking little to choose from.\n"
        "\n"
        "Measured: not measured."
    ),
    "selfplay.pool_disk_mb": (
        "What it does: a second cap on the pool, by disk space instead of count. When the pool's files would go over this many megabytes, past versions are removed by the same rule as pool_size: the one rated closest to its neighbours goes, never the newest.\n"
        "\n"
        "When to change it: raise it for big networks, where 64 copies may not fit in the default 2,048 MB. Lower it if the drive holding the runs folder is short of space.\n"
        "\n"
        "How to use it: each copy holds only the policy's weights (and a shared trunk), so its size grows with the network's size. Whichever cap is reached first, this or pool_size, decides how many past versions stay. The minimum is 16 MB.\n"
        "\n"
        "Watch out for: with a big network and a small cap, the pool can end up much smaller than pool_size says, which leaves matchmaking few past versions to pick from. Check the pool folder's size on a long run.\n"
        "\n"
        "Measured: not measured."
    ),
    "selfplay.matchmaking": (
        "What it does: chooses the rule for which past versions play at each save. Every rule uses the predicted chance that the bot beats each past version, from the ratings.\n"
        "- variance (the default): favours even matches, near 50%.\n"
        "- uniform: any past version alike.\n"
        "- hardest: favours the ones the bot loses to.\n"
        "- squared: favours those much more strongly (AlphaStar's hard rule).\n"
        "- linear_capped: favours the hard ones, but treats every version the bot beats at most half the time the same, so one unbeatable version cannot take over.\n"
        "\n"
        'When to change it: if the bot\'s rating against its past versions stops rising, the training warning itself suggests trying "hardest". Otherwise keep the default.\n'
        "\n"
        "How to use it: the draw is repeatable: it depends on the run's seed and the save number, so a resumed run meets the same past versions a straight run would. Compare rules between saves with nitro eval.\n"
        "\n"
        "Watch out for: the rules that lean hard on the strongest versions can give the bot mostly lost games.\n"
        "\n"
        "Measured: which rule makes a stronger bot is not measured yet; only full self-play runs could show it."
    ),
    "selfplay.flat_rating_steps": (
        "What it does: a warning, not a change to training. It watches the bot's rating against its past versions. If the rating has not gone up by a real amount (about 3 percentage points of win chance) for this many agent steps, the run prints a warning that learning may have stalled.\n"
        "\n"
        "When to change it: lower it on a short run if you want to hear about a stall sooner. Raise it if the warning appears during slow but normal stretches. Set 0 to never warn. It only works while self-play is on.\n"
        "\n"
        'How to use it: the default is 500 million agent steps. The warning suggests what to look at next: watch the bot play (nitro watch), try selfplay.matchmaking = "hardest", or lower the learning rates.\n'
        "\n"
        "Watch out for: ratings move only on games that end in a goal; a game that ends without one says nothing about who is better. So a run with few goals against past versions moves its rating slowly. Treat the warning as a reason to look, not as proof.\n"
        "\n"
        "Measured: not measured."
    ),
    "pretrain.enabled": (
        "What it does: before training, a new bot learns to copy the players in your own replays (behaviour cloning). The replays of the run's team size are converted once and kept in the project's replays folder. The policy is then taught to choose what the players chose, and the copy is saved as the bot's first save, at 0 agent steps. Then it stops.\n"
        "\n"
        "When to change it: when you want a bot to start from human-like play instead of random flailing.\n"
        "\n"
        "How to use it: after the copy is saved you can watch it, export it, or try rewards on it with nitro reward-check (the Rewards tab's Try these rewards). The next Start trains from it: the copied policy plus a fresh value network. For thousands of replays, also set pretrain.batch_replays.\n"
        "\n"
        "Watch out for: only a new bot reads this; a resumed bot keeps its own networks. It cannot be used together with start_from.run, since both choose what the bot starts from.\n"
        "\n"
        "Measured: copying works on a test set of replays: the first choice matched the player's 36.1%, 51.5% and 55.8% of the time after 1 to 3 passes, against 30.6% for always picking the most common action. Whether this makes a better bot than training from scratch is not measured yet."
    ),
    "pretrain.replays": (
        "What it does: tells the copy where your .replay files are. Left empty, it uses the game's own replay folder (DemosEpic for Epic, Demos for Steam, inside Documents\\My Games\\Rocket League\\TAGame).\n"
        "\n"
        "When to change it: when your replays are somewhere else, such as an archive drive, or when you keep a folder for each game size.\n"
        "\n"
        "How to use it: give one folder, or a list of folders. With a list and batch_replays on, the folders are taken in turn, each newest first, so every batch mixes them. For example three folders, one each for 1v1, 2v2 and 3v3, with team_sizes = [1, 2, 3]. Relative paths are read from where the settings file is.\n"
        "\n"
        "Watch out for: the copy picks the newest replays first, not a random spread. A replay of another team size is noted in the replays folder's sizes.txt so it is not converted again for that size. Replays hold other players' names; the reports keep only counts.\n"
        "\n"
        "Measured: nothing to measure; it only names where the replays are."
    ),
    "pretrain.team_sizes": (
        "What it does: in the large replay mode, picks which game sizes are copied. For example [1, 2, 3] copies 1v1, 2v2 and 3v3 replays into one bot. Empty means this bot's own team size only.\n"
        "\n"
        "When to change it: when you want one bot that can play every game size.\n"
        "\n"
        "How to use it: it needs pretrain.batch_replays above 0 and a padded observation (env.obs_team_size) big enough for the largest size, for example 3 for [1, 2, 3]. pretrain.limit counts all sizes together, so a limit of 7,500 with three sizes gives about 2,500 of each. Give one folder per size in pretrain.replays so the batches mix them.\n"
        "\n"
        "Watch out for: env.obs_team_size is fixed once the bot has saves, so plan the room before the first save; change it by making a new bot from this one. Without the room the settings check refuses the list.\n"
        "\n"
        "Measured: a copy of 2,516 1v1, 2,518 2v2 and 2,516 3v3 replays took 1.87 hours on our test machine, held-back loss falling from 1.247 after round 1 to 1.122 after round 75."
    ),
    "pretrain.limit": (
        "What it does: caps how many replays are copied, newest first.\n"
        "\n"
        "When to change it: more replays copy the players better but take longer. The default of 30 is a quick first try. For a serious copy use the large replay mode (pretrain.batch_replays) and a big limit.\n"
        "\n"
        "How to use it:\n"
        "- Without batch_replays every replay is held in memory at once: about 300 to 400 at most on 32 GB. At most five times the limit are converted per start, so a rare team size does not take hours.\n"
        "- With batch_replays the limit can be 10,000 or more. Copies of 7,500 and of 20,000 more 2v2 replays have been run.\n"
        "\n"
        "Watch out for: changing the limit in the large mode makes the copy again from the start. Raising it on a finished copy also starts again; to add replays to a finished copy, use pretrain.continue_from with pretrain.new_only instead.\n"
        "\n"
        "Measured: on our test machine, 10,000 2v2 replays take about 2.2 to 2.4 hours to convert with 10 at once. In a copy of 7,500 replays the last 2,500 improved the held-back loss by only 0.008."
    ),
    "pretrain.passes": (
        "What it does: how many times the copy goes over its replays. More passes teach more from the same replays, up to the point where it starts to learn them by heart.\n"
        "\n"
        "When to change it: the default of 10 suits a small copy of a few dozen replays. In the large replay mode, 1 or 2 is enough, since each batch is new data. The presets suggest 2 there.\n"
        "\n"
        "How to use it: in the large mode the passes apply to each round (the newest batch plus the kept ones, pretrain.keep_batches). So passes = 2 with keep_batches = 2 means each replay is seen about 4 times. Each pass prints how often the copy's first choice matches the player's on held-back replays.\n"
        "\n"
        "Watch out for: in the large mode, changing passes makes the copy again from the start.\n"
        "\n"
        "Measured: on 30 1v1 replays, agreement rose from 36.1% to 51.5% to 55.8% over passes 1 to 3, with the held-back loss still falling. Whether two passes copy better than one in the large mode is untested."
    ),
    "pretrain.batch_replays": (
        "What it does: 0 keeps the simple mode: every replay is converted, then all are learned from at once. Above 0 switches on the large replay mode. Replays are converted this many at a time, kept only as compact samples in memory, learned from, then dropped. Memory and disk stay small however many replays you copy.\n"
        "\n"
        "When to change it: when you want more than a few hundred replays.\n"
        "\n"
        "How to use it: the presets suggest 200, which suits 32 GB for 2v2; 100 uses less memory. Then set pretrain.limit to the total and pretrain.passes to 1 or 2. Progress is saved after every batch, so a stop or a power cut loses at most the batch in progress, and the next start carries on. Several settings only work in this mode: team_sizes, keep_batches, workers, converter, continue_from and new_only.\n"
        "\n"
        "Watch out for: at most about 4 batches are in memory at once. Worked out from the code (not measured), that is about 4.5 GB for 2v2 at 200 a batch, plus the converter programs. Changing this number makes the copy again from the start.\n"
        "\n"
        "Measured: a copy of 7,500 replays of every size, 100 a batch, took 1.87 hours on our test machine, with converting the slow part."
    ),
    "pretrain.keep_batches": (
        "What it does: in the large replay mode, how many batches are learned from together in each round: the newest one plus the ones before it. With the default of 2, every replay is learned from in two rounds, once as the newest batch and once as the older one.\n"
        "\n"
        "When to change it: 1 means each batch is learned from once and dropped, as a big copy usually does (keep_batches = 1, passes = 1). Higher values let each replay be seen in more rounds, at the cost of memory.\n"
        "\n"
        "How to use it: memory holds this many batches at once, so memory use grows with keep_batches x batch_replays. It multiplies with pretrain.passes: each replay is seen about passes x keep_batches times in all.\n"
        "\n"
        "Watch out for: the kept batches are not stored on disk. After a stop and resume, the first round learns from the next batch alone. Changing this setting makes the copy again from the start.\n"
        "\n"
        "Measured: not measured as a comparison."
    ),
    "pretrain.workers": (
        "What it does: in the large replay mode, how many replays are converted at the same time. 0 picks half of this PC's threads, at most 16: 10 on a 10-core, 20-thread PC.\n"
        "\n"
        "When to change it: rarely. Leave it at 0 unless a measurement on your own PC says otherwise.\n"
        "\n"
        "How to use it: unlike limit, passes and the batch settings, changing it does not make the copy again. Each conversion runs its own converter program; replicar's took about 0.36 GB each for 3v3 (measured).\n"
        "\n"
        "Watch out for: more is not faster. Converting alone gains from more workers, but in a whole copy the learning needs the processor too, so extra workers slow it down.\n"
        "\n"
        "Measured on our test machine:\n"
        "- With replicar's converter, 18 at once was 22.8% slower than 10 (56.7 against about 73 replays a minute).\n"
        "- With the nitro converter, converting only, 14 at once made 22.9% more ready than 10. But in whole copies, 14 was 16.5% slower than 10.\n"
        "So the default stays at half the threads."
    ),
    "pretrain.converter": (
        "What it does: in the large replay mode, picks the program that turns replays into game states and controls.\n"
        "- nitro (the default): NitroLearn's own small build on top of replicar. It does the same conversion, but it writes only the parts the copy reads.\n"
        "- replicar: replicar's own tool.\n"
        "\n"
        "When to change it: there is normally no reason to. Choose replicar only if the nitro build fails on your PC.\n"
        "\n"
        "How to use it: nitro is built the first time it is used, which took 92 seconds on our test machine, and again when its source changes. A stopped copy can carry on with either converter, because the copy's progress does not depend on which one made the replays ready.\n"
        "\n"
        "Watch out for: a speed-tuned build for one processor was tried and rejected, because it changed results in the last bits.\n"
        "\n"
        "Measured on our test machine: the same arrays and samples bit for bit on 20 of 20 replays. nitro made 112.5 and 111.5 replays ready a minute against replicar's 103.0 and 93.2, 14.2% faster, which is why nitro became the default."
    ),
    "pretrain.continue_from": (
        "What it does: in the large replay mode, starts this copy from another run's finished copy (its pretrained folder) instead of from a fresh network. The replays here are learned on top of what that copy already knows.\n"
        "\n"
        "When to change it: to teach an existing copy more replays without starting over. For example, a copy of 7,500 replays can be carried on with 20,000 more 2v2 replays.\n"
        "\n"
        "How to use it: write the other run's name as it appears in the runs folder, not a path. Usually pair it with pretrain.new_only = true so the replays the first copy already used are skipped. Before any replay is converted, the network settings, the observation, tick_skip and action_delay are checked against the other run, because the copy's labels depend on them. The held-back replays are measured before the first round too, so you can see the starting point.\n"
        "\n"
        "Watch out for: it reads the other run's copy from before any training, not its later trained saves. The observation's width is only checked once the first batch is ready, a few minutes in.\n"
        "\n"
        "Measured: not measured as a learning comparison; checked working with the real converter."
    ),
    "pretrain.new_only": (
        "What it does: in the large replay mode, skips every replay that was already converted on this PC by an earlier copy or the converter. These are listed in the project's replays folder (sizes.txt). So a copy that carries on another copy learns only from new replays.\n"
        "\n"
        "When to change it: switch it on together with pretrain.continue_from, so the carried-on copy does not learn the same replays twice.\n"
        "\n"
        "How to use it: it needs pretrain.batch_replays above 0. Leave it off for a fresh copy that should use your newest replays whatever happened before.\n"
        "\n"
        "Watch out for: replays that were converted ahead but never used by a copy are not noted, so they are not skipped. Because the copy picks the newest replays first, with this on it digs further back into older replays once the newest are used up.\n"
        "\n"
        "Measured: not measured; tested with the real converter on real replays, where it left out the 3 replays the first copy had made ready."
    ),
    "pretrain.value_warmup_steps": (
        "What it does: after the copy, the bot gets a brand new value network, which knows nothing about your rewards. For this many agent steps at the start of training, only the value network learns, and the copied policy is held exactly as it was. That way the value network learns what situations are worth before it starts steering the policy.\n"
        "\n"
        "When to change it: when starting from a copy of replays. Without it, the early training can wash out some of the copied play while the value network is still guessing.\n"
        "\n"
        "How to use it: the presets' example is 10 million steps, about 200 updates at their settings. The save records that the bot started from replays, so the warm-up keeps counting correctly after a stop and resume. It can change while the bot trains: the bot's Overview shows how far the warm-up is, with the number to change (Apply carries on training with it). The steps already trained count: a lower number frees the policy sooner, and a number above them holds the policy still again until then.\n"
        "\n"
        "Watch out for: with a shared trunk (model.shared), the trunk is held too, so only the value network's own layers learn during the warm-up: the trunk belongs to the policy as well, and moving it would change the copied play.\n"
        "\n"
        "Measured: tested that the policy and a shared trunk keep every weight while the value network learns. How much a warm-up helps is not measured yet."
    ),
    "pretrain.keep_converted": (
        "What it does: keeps every replay the copy converts (any game size) in the project's replays\\store folder, and uses the kept ones before converting, so no replay is ever converted twice, by this bot or another.\n"
        "\n"
        "When to change it: leave it on. Turn it off only to save disk space.\n"
        "\n"
        "How to use it: on or off. The Resources page's Converted replays card shows what is kept, and converts replays on their own (learning nothing) while the PC is free.\n"
        "\n"
        "Watch out for: about 8-10 MB a 5-minute replay (20,000 replays: about 160-200 GB). Nothing more is kept while the disk has under 20 GB free.\n"
        "\n"
        "Measured (12 real replays): making a replay ready from the store took 0.16-0.29 s against 5.5-8.6 s converting it, the samples the same bit for bit."
    ),
    "pretrain.skill_every": (
        "What it does: every this many rounds of the copy (and after the last), the copy so far plays two earlier versions of itself in the simulator: the copy where it started, and the version from the last check. Each point starts at a kickoff and ends at a goal, or without one (a draw). From the goals it gets a skill rating, shown on the bot's Overview tab with how it did against each.\n"
        "\n"
        "When to change it: switch it on (Skill checks on the Learning from replays card) to see whether the copy is getting better at the game, not only closer to how the players choose. The agreement number says how closely it copies; the rating says whether that copying wins goals.\n"
        "\n"
        "How to use it: 0 = never. 10 checks every 10 rounds; with 100 replays a round on our test machine that is about every 12 minutes. Where the copy started counts as 25; above it is better.\n"
        "\n"
        "Watch out for: each check takes about a minute on a PC (the copy waits; converting goes on). A copy of few replays may not score at all yet: then the points are draws and the rating barely moves. A rating is a rough guide (about 200 points a match), not proof of a setting.\n"
        "\n"
        "Measured: a small copy here (4 replays, 24 points a check) played every check and kept its ratings. Not measured on a long copy yet."
    ),
    "pretrain.save_every": (
        "What it does: while the replays are being copied, every this many replays learned the copy as it is so far is saved as the bot's save (at 0 agent steps). Watch it play, Make RLBot bot and Try these rewards then work on it part-way through, not only once all the replays are copied.\n"
        "\n"
        "When to change it: lower it to see the copy more often, raise it (or 0) to spend less time saving.\n"
        "\n"
        "How to use it: 2,000 (the default) saves about every 25 minutes at 100 replays a round on our test machine. 0 saves only the finished copy. Each save replaces the one before; the copy's progress itself is kept after every round anyway, so this is only about using the copy early.\n"
        "\n"
        "Watch out for: a save of the copy so far is not training. Pressing Copy replays again carries on the copy, and training (Start training from the copy) begins only once it is finished. Each save takes about as long as a training save, and the copy waits for it (converting goes on meanwhile). A save that fails says why in the console and the copy goes on.\n"
        "\n"
        "Measured: tested for working on a small copy here; how long a save of a big copy takes is not measured yet."
    ),
    "pretrain.skill_points": (
        "What it does: how many points a skill check plays against each earlier version. A point starts at a kickoff and ends at a goal, after the bot's own no-touch time, or after 60 seconds.\n"
        "\n"
        "When to change it: raise it for a surer rating, lower it for quicker checks.\n"
        "\n"
        "How to use it: 200 is the start (all played at once in up to 200 games). Each point takes up to a minute of game time, played much faster than real time.\n"
        "\n"
        "Watch out for: few points give a rating that jumps around; the page shows each match's sure range.\n"
        "\n"
        "Measured: nothing yet beyond the small copy here."
    ),
    "pretrain.lr": (
        "What it does: how big a step the copy takes each time it learns from a batch of the players' decisions (Adam's learning rate).\n"
        "\n"
        "When to change it: lower it for big networks. Networks of 10 million parameters and more usually copy best at 1e-4 to 2e-4; 3e-4 can make their loss jump about. With a much bigger pretrain.minibatch, a somewhat higher rate is usual.\n"
        "\n"
        "How to use it: a number such as 1.5e-4. It can be changed between a stop and a resume of a copy: it does not make the copy again.\n"
        "\n"
        "Watch out for: too high, and the held-back loss stops falling or wobbles from round to round; too low, and the copy learns slowly from the same replays.\n"
        "\n"
        "Measured: only 3e-4 has been tried."
    ),
    "pretrain.lr_end_share": (
        "What it does: lets the copy's learning rate fall gradually, along a smooth curve, to this share of pretrain.lr by the time the last replay (pretrain.limit) is learned. Big early steps learn fast; small late steps fine-tune what was learned, which usually lowers the final loss. 1 keeps the rate the same throughout.\n"
        "\n"
        "When to change it: for a new, long copy, 0.1 is a common choice (a tenth of the rate at the end).\n"
        "\n"
        "How to use it: a number from 0 to 1. The curve follows the replays learned out of limit, so a stopped copy carries on along the same curve. With a copy of fewer replays than limit (the folder runs out), the rate never gets all the way down.\n"
        "\n"
        "Watch out for: it changes nothing about which replays are used, only how the learning steps are sized.\n"
        "\n"
        "Measured (a small check): on 6 real 2v2 replays, 4 learned from in 6 passes and 2 held back, a small transformer copied worse with the rate falling to a tenth than with it fixed (held-back loss 1.233 against 1.204, and 1.230 against 1.196, two seeds). A long copy of thousands of replays seen once may differ: not measured yet."
    ),
    "pretrain.prev_action_dropout": (
        "What it does: while copying, hides the bot's previous move (part of what it sees) on this share of the decisions it learns from, so it learns to choose from the game itself and not only from what it did last. In play it always sees its real previous move.\n"
        "\n"
        "When to change it: when a copy, playing its likeliest move, gets stuck repeating one move. One copy of a big 2v2 bot kept one move 117 decisions in a row (players: about 3), because repeating the previous move already matches the players about 71% of the time.\n"
        "\n"
        "How to use it: 0 to 0.9; 0 = never. Research on driving copies used about 0.5; 0.3-0.5 is a reasonable first try. It can change while a copy is under way; it does not make the copy again.\n"
        "\n"
        "Watch out for: the copy's agreement with the players drops a little (expected: it leans less on repeating); judge it by the copy check in the simulator (touches, how long it keeps one move), not by agreement.\n"
        "\n"
        "Measured: nothing yet in NitroLearn. In an offline check on 3v3 replays, hiding it on half the decisions cut how much the copy's choices depended on its previous move from 84% to 69%."
    ),
    "pretrain.minibatch": (
        "What it does: how many recorded decisions the copy learns from in each step.\n"
        "\n"
        "When to change it: raise it on a big graphics card (24 GB or more) with a big network: fewer, larger steps keep the card busy and finish the copy sooner. 8192 to 16384 suits a 30-million-parameter bot on a 24 GB card or bigger; on an 8 GB card stay near 4096 to 8192.\n"
        "\n"
        "How to use it: a whole number, 16 or more. It can be changed between a stop and a resume of a copy.\n"
        "\n"
        "Watch out for: twice the batch means half as many learning steps over the same replays, so raise pretrain.lr a little with it. Too big a batch can run out of graphics card memory.\n"
        "\n"
        "Measured: not yet; its speed is not measured yet."
    ),
    "pretrain.measure_every": (
        "What it does: in the large replay mode, how often the copy measures itself on the replays it holds back (the loss and the share of the players' first choices it agrees with). 1 measures after every round; 10 measures every tenth round and after the last.\n"
        "\n"
        "When to change it: for long copies. Measuring 50 held-back replays after each round of 100 costs about a sixth of the copy's learning time, and the numbers move little from one round to the next.\n"
        "\n"
        "How to use it: a whole number, 1 or more. Rounds in between say that they were not measured, and the charts skip them.\n"
        "\n"
        "Watch out for: nothing about the copy itself changes, only how often it is checked.\n"
        "\n"
        "Measured: the cost of measuring is estimated from the size of the held-back set; not timed."
    ),
    "replay_mix.enabled": (
        "What it does: while the bot trains, a share of its learning steps copy the players in your converted replays, the same learning the replay copy does before training, so a bot that is already good keeps picking up how real players play while it trains against itself. After each update the learner takes a few copy steps on decisions drawn from the replays. They move the policy and any shared trunk, never the value network, with an optimizer and learning rate of their own (replay_mix.lr), kept in the bot's saves.\n"
        "\n"
        "When to change it: late in a bot's training, when it has stopped improving or plays oddly against people, to mix some real play back in. It can also be on from the start.\n"
        "\n"
        "How to use it: convert replays of the bot's game size first (nitro convert-replays, or the Converted replays card on the Resources page; they go into the project's replays\\store). Then switch this on in the bot's Replays in training tab, keep share at 0.2 and lr at 3e-5 to begin with, and Apply (or Start). The bot carries on from its newest save: its steps, its own optimizer and everything else stay as they were; only the copy steps are new. The same tab, the console report's Replays line and the replay_mix columns in metrics.csv show the copy loss and how often the bot's first choice is the player's.\n"
        "\n"
        "Watch out for: without any converted replays of the bot's size it says so at the start, at each save and in its report, and trains as usual. While a value warm-up holds the policy still, no copy steps are taken. Each copy step takes about as long as a learning step of minibatch decisions, so an update takes a little longer.\n"
        "\n"
        "Measured: nothing measured yet. Whether it makes a better bot is for your own training to show."
    ),
    "replay_mix.share": (
        "What it does: the copy steps' share of all the learning steps that move the policy. With 0.2, one step in five copies replays: after an update of 8 policy steps (the presets' 2 passes of 4 minibatches) come 2 copy steps. A fraction left over is carried to the next update, so 0.1 gives one copy step every other update with 4 policy steps.\n"
        "\n"
        "When to change it: lower it if the bot's own play gets worse after switching the mixing on; raise it to lean harder on the players.\n"
        "\n"
        "How to use it: 0 to 0.9. 0 is the same as switching it off. It can change between a stop and a start.\n"
        "\n"
        "Watch out for: the copy steps pull toward how people play, the policy's own steps toward what the rewards pay; a high share can drown out the rewards. Steps that train only the value network (a value warm-up) do not count.\n"
        "\n"
        "Measured: nothing measured yet."
    ),
    "replay_mix.lr": (
        "What it does: the learning rate (Adam) of the copy steps, apart from the training's own ppo.policy_lr. It sets how hard each copy step pulls the policy toward the players' choices.\n"
        "\n"
        "When to change it: raise it a little if the copy loss in the console report never moves; lower it if the bot's own play suffers.\n"
        "\n"
        "How to use it: keep it below ppo.policy_lr; the settings check says so when it is not. The default 3e-5 is a tenth of the replay copy's own rate (pretrain.lr 3e-4) and under a third of a low training rate for a big bot (1e-4; the presets use 4.5e-4), so the copying nudges a trained bot instead of turning it back into a plain copy. It can change between a stop and a start; the copy steps' optimizer keeps its memory.\n"
        "\n"
        "Watch out for: the copy steps' optimizer is their own, so this changes nothing about the training's learning.\n"
        "\n"
        "Measured: nothing measured yet; 3e-5 is a starting value."
    ),
    "replay_mix.minibatch": (
        "What it does: how many of the players' decisions each copy step learns from, drawn at random from 16 replays held in memory at a time, so every step mixes many games.\n"
        "\n"
        "When to change it: rarely. Bigger steps are steadier; smaller ones cost less time per update.\n"
        "\n"
        "How to use it: a whole number, 16 or more; 4096 is what the replay copy uses (pretrain.minibatch). It can change between a stop and a start.\n"
        "\n"
        "Watch out for: with share fixed, a bigger minibatch copies more decisions per update, so each replay is used up sooner and the copying takes more of an update's time.\n"
        "\n"
        "Measured: nothing measured yet."
    ),
    "replay_mix.kickoff_only_seconds": (
        "What it does: above 0, the copy steps learn only from the replays' kickoffs. Each kickoff is kept from its start to this many seconds of play after the ball's first touch; a kickoff that nothing touches within 10 s ends there. Everything else in the replays is left out. 0 (the default) copies every decision.\n"
        "\n"
        "When to change it: to teach a bot better kickoffs from good players, for example a few thousand high-level 1v1 replays with 3 here: the kickoff itself and the 3 s after the first touch.\n"
        "\n"
        "How to use it: seconds of play, 0 to 30. Switch replay_mix on as usual, set this, and Apply (or Start). It can change between a stop and a start. The console report's Replays line says what share of the replays' decisions it keeps, and metrics.csv has replay_mix_rows_kept and replay_mix_rows_left_out.\n"
        "\n"
        "Watch out for: a replay gives only a small part of its decisions this way (in a 5-minute game about a tenth with 3), so the replays are used up about ten times faster and each cycle through them comes round sooner; more replays help. The copy steps still pull the whole policy toward the players at kickoffs, so keep share modest.\n"
        "\n"
        "Measured: tests only (the kept decisions are the same samples as without it, bit for bit). Whether it makes better kickoffs is for your own training to show."
    ),
    "replay_mix.replays": (
        "What it does: the folders the copy steps' replays come from: converted replays (.nrpl.zst files) as the project's replays\\store keeps them, subfolders included. Left empty, it is the store's folder of the bot's game size (replays\\store\\replicar-2.2.1\\1v1 for a 1v1 bot), or of each of pretrain.team_sizes when that is set.\n"
        "\n"
        "When to change it: to copy only some replays, such as a folder of high-ranked games you converted apart, or a store copied from another PC.\n"
        "\n"
        "How to use it: a folder, or a list of them. Replays are taken in a random order fixed by run.seed, each once per cycle; when all are used the cycle starts again in a new order. Replays added later join when the bot is started again.\n"
        "\n"
        "Watch out for: only converted replays are read, never .replay files: convert them first. Replays that do not fit the bot (another game size without room in its observation) are left out, and when none fit it says so (at the first iteration, at every save and in the report) and trains as usual.\n"
        "\n"
        "Measured: nothing measured yet."
    ),
    "wandb.enabled": (
        "What it does: also sends the run's numbers to Weights & Biases (wandb.ai): every metrics.csv column, all settings and the PC's CPU and graphics-card use. Training, copies of replays (round by round) and nitro pretrain all send. Practice tasks, nitro ab and nitro eval never do. The local metrics.csv and report stay the same.\n"
        "\n"
        "When to change it: when you want charts in the browser, from any device, or several runs side by side.\n"
        "\n"
        "How to use it: make a free W&B account and log in once on the PC with wandb login. The key stays with W&B, never in the settings. A resumed run continues the same W&B run, because its id is kept in the run folder. nitro wandb <run> uploads what was kept offline and sends anything older runs never sent.\n"
        "\n"
        "Watch out for: while a run sends, W&B's helper program uses a port on this PC that only this PC can reach. If online sending cannot start (no internet, not logged in), the run falls back to offline with a warning and keeps training.\n"
        "\n"
        "Measured: no speed measurement. Checked for working: nitro wandb sent the charts of two large copies (75 and 63 rounds)."
    ),
    "wandb.project": (
        "What it does: the W&B project the run's charts go into. A project is like a folder on wandb.ai that holds runs you want to compare.\n"
        "\n"
        "When to change it: when you want to keep groups of bots apart, for example one project for 1v1 experiments and one for your main bots.\n"
        "\n"
        'How to use it: the default is nitrolearn. The presets\' commented example shows "my-bots".\n'
        "\n"
        "Watch out for: it cannot be empty while W&B is on; the settings check refuses that. Runs in different projects cannot be shown on the same chart easily, so keep bots you want to compare in one project and use wandb.group or wandb.tags to sort them.\n"
        "\n"
        "Measured: not measured."
    ),
    "wandb.entity": (
        "What it does: chooses whose W&B account the project lives under: your own user name or a team name. Empty means your default, which for most people is your own account.\n"
        "\n"
        "When to change it: only when you are in a W&B team and want the runs to show up in the team's space instead of your own.\n"
        "\n"
        "How to use it: write the user or team name exactly as W&B shows it. Leave it empty if you are not sure.\n"
        "\n"
        "Watch out for: the settings check cannot know whether a user or team name is right. If W&B cannot start online, the run falls back to offline and keeps training, so watch the console for a W&B warning on the first start.\n"
        "\n"
        "Measured: not measured."
    ),
    "wandb.name": (
        "What it does: the name the run gets on wandb.ai. Empty means the run folder's name, which is usually what you want, since it matches the bot's name in the dashboard.\n"
        "\n"
        "When to change it: when the folder name is not descriptive enough on W&B, for example to add the reward set or the game size.\n"
        "\n"
        "How to use it: any text. For sorting runs, wandb.group and wandb.tags are usually better tools than a long name.\n"
        "\n"
        "Watch out for: a resumed run continues the same W&B run, found by an id kept in the run folder, not by name. Two different runs can share a name on W&B and still be separate runs.\n"
        "\n"
        "Measured: not measured."
    ),
    "wandb.group": (
        "What it does: puts the run in a named group on W&B, so related runs are shown together. W&B can fold a group into one line on a chart, or show its runs side by side.\n"
        "\n"
        "When to change it: when you train several bots that belong together, such as the same idea with different reward weights, or a bot and the runs started from it.\n"
        "\n"
        "How to use it: write the same group name in each related run's settings. Empty means no group. It only changes how W&B arranges the runs; nothing in training changes.\n"
        "\n"
        "Watch out for: a group is only a label on W&B. It does not link the runs in NitroLearn in any way, and runs in different projects are not grouped together.\n"
        "\n"
        "Measured: not measured."
    ),
    "wandb.tags": (
        'What it does: labels attached to the run on W&B, which you can filter and search by on wandb.ai. For example ["1v1", "nitro_v2"].\n'
        "\n"
        "When to change it: when you have many runs and want to find, say, every 2v2 bot or every bot with a certain observation or reward set.\n"
        "\n"
        "How to use it: a list of short words in quotes. Use the same spelling every time, since W&B treats different spellings as different tags. Empty means no tags. Tags only change how runs are sorted on W&B, not training.\n"
        "\n"
        "Watch out for: unlike a group, tags do not fold runs together on a chart; they are for filtering.\n"
        "\n"
        "Measured: not measured."
    ),
    "wandb.mode": (
        'What it does: "online" sends the numbers to wandb.ai while the run trains. "offline" writes W&B\'s files into the run folder only, so they can be uploaded later.\n'
        "\n"
        "When to change it: use offline when the PC has no internet, or when you do not want anything sent during training. Online is the normal choice.\n"
        "\n"
        "How to use it: to upload an offline run later, use nitro wandb <run>. It uploads what W&B kept offline, in order, and sends whatever the run has not sent yet, nothing twice. That command is not meant for a run that is training at the same time.\n"
        "\n"
        "Watch out for: even in online mode, a run that cannot reach W&B falls back to offline with a warning instead of stopping. When logged in but with the network off, that took about 45 seconds on the test before training went on.\n"
        "\n"
        "Measured: not measured for speed."
    ),
    "reward.weight": (
        "What it does: how much this reward counts. The reward's own value (often between -1 and 1) is multiplied by the weight. A negative weight turns the reward into a penalty.\n"
        "\n"
        "When to change it: when the bot farms one reward and ignores the rest, or never bothers with a behaviour you want.\n"
        "\n"
        "How to use it: what matters is how the rewards compare with each other. The presets start from a common mix, for example goal 150, strong_touch 60, velocity_player_to_ball 4, air 0.25. Rewards paid every step are scaled automatically with tick_skip, so the same weight means the same at any tick skip. The Rewards tab ranks each reward's share of all reward paid. Try these rewards (nitro reward-check) shows what each would pay a frozen bot before you train.\n"
        "\n"
        "Watch out for: training warns when one reward is over 90% of all reward paid. A reward that pays little to the bot now can still grow into the one it farms. Edits wait for Apply, which saves the bot and restarts it with them.\n"
        "\n"
        "Measured: weights are your own choice. The presets' mix was not measured here."
    ),
    "reward.zero_sum": (
        "What it does: makes a reward shared between teams. Each car is paid its own share and its team's share, minus the opponents' average. Formula: own x (1 - team_spirit) + team average x team_spirit - opponents' average x opponent_scale. So the bot cannot farm a reward that helps nobody: if both sides earn it equally, it cancels out.\n"
        "\n"
        "When to change it: on rewards both teams can earn, like ball speed toward goal, bumps and demos, so the bot learns to beat the opponents rather than just collect points.\n"
        "\n"
        "How to use it: written next to the weight, with team_spirit and opponent_scale. The presets use team spirit 1 on velocity_ball_to_goal and 0.5 on bump and demo. On an all_of group, the group carries the zero-sum setting, not its parts.\n"
        "\n"
        "Watch out for: a zero-sum reward adds up to about 0 across all cars, but the Rewards tab still ranks it by its size. In 1v1, team spirit makes no difference because there are no teammates.\n"
        "\n"
        "Measured: not measured."
    ),
    "reward.zero_sum.team_spirit": (
        "What it does: how much of the reward is shared with teammates. 0 means each car keeps its own reward. 1 means every car on the team gets the team's average. In between mixes the two.\n"
        "\n"
        "When to change it: low team spirit gives clearer credit, because each car sees what it did itself. High team spirit rewards teamwork, such as passing or rotating out for a teammate. Only matters in 2v2 and 3v3.\n"
        "\n"
        "How to use it: values 0 to 1. The presets use 1 on velocity_ball_to_goal and 0.5 on bump and demo. OpenAI Five raised it from 0.3 to 1 as its bots improved.\n"
        "\n"
        "Watch out for: in 1v1 there are no teammates, so it changes nothing.\n"
        "\n"
        "Measured: not yet. No practice task has teammates, so nothing narrow can test it."
    ),
    "reward.zero_sum.opponent_scale": (
        "What it does: how much of the opponents' average reward is taken away from each car. 1 takes all of it, so a reward both teams earn equally cancels out. 0 takes none, so the reward is only shared with teammates.\n"
        "\n"
        "When to change it: keep 1 for a true contest, where only doing better than the opponents pays. A value below 1 still pays a little when both teams earn the reward.\n"
        "\n"
        "How to use it: values 0 or more, default 1. The presets leave it at 1 everywhere. It combines with team_spirit: team spirit decides how a team splits its share, opponent_scale decides how much of the other team's share is subtracted.\n"
        "\n"
        "Watch out for: values above 1 punish the opponents' success more than your own success pays, which can make the bot overly defensive. Nothing here has measured that.\n"
        "\n"
        "Measured: not measured."
    ),
    "reward.shares": (
        "What it does: the list of what each reward pays, beside the rewards, can be sorted and filtered. Share of reward (the usual order) ranks by how much of all reward each pays. How often it triggers ranks by how many times a minute of one car's play it pays anything, before its weight, so a flick at weight 10 and a reset at weight 100 compare fairly. Size before its weight ranks by the reward's own value. A to Z sorts by name.\n"
        "\n"
        "When to change it: to see which moves the bot actually does often, whatever the weights. An event reward (a goal, flick or reset) fires only when it happens, so its count is how often the bot does it; a continuous reward such as speed toward the ball fires on almost every step, so its count tells you little.\n"
        "\n"
        "How to use it: pick the order in Sort by. Hide continuous rewards hides rewards marked every step, and any other that fires on more than half of the steps. The eye button on a row hides that reward; Show N hidden brings them back. With rewards hidden, the percentages are the share of the shown rewards, so they add up to 100%. The same order and hiding apply to Try these rewards.\n"
        "\n"
        "Watch out for: this only changes what the list shows. Training, the weights and settings.toml stay as they are, and hidden rewards still pay. Each bot remembers its own hidden rewards.\n"
        "\n"
        "Measured: on a busy shared processor, counting made no measurable difference to game or training speed; training results are unchanged. The speed on your machine is not measured."
    ),
    "start.weight": (
        "What it does: how often this way of starting is picked, compared with the others. Each time a game starts, one start is picked at random in proportion to the weights. The Starts tab shows each start's share of games (its weight over the sum).\n"
        "\n"
        "When to change it: to change the mix of practice, for example more kickoffs or more random situations. A common mix is about 70% kickoff and 30% random.\n"
        "\n"
        "How to use it: weights are relative: kickoff 7 and random 3 gives 70/30, the same as 0.7 and 0.3. Weight 0 keeps a start written but unused; at least one needs a weight above 0. Changes wait for Apply. Starts are not fixed by a bot's saves, so you can change them on any bot. With several starts, metrics.csv shows per-start figures: games started, the share of goals, the mean length and the touch share.\n"
        "\n"
        "Watch out for: each game's start is drawn at random, so a start with a small weight still turns up now and then; the share is an average over many games.\n"
        "\n"
        "Measured: the mix itself is not measured."
    ),
    "start.kickoff": (
        "What it does: a normal kickoff from one of the real kickoff spots. With no starts listed at all, every game starts this way, and the presets use a kickoff only.\n"
        "\n"
        "When to change it: kickoffs teach the most common moment of a real game. Add other starts beside it for variety, and keep kickoffs for a good share of games so the bot learns real kickoffs.\n"
        "\n"
        "How to use it: fuzz_uu moves each car up to this many units in a random direction, so kickoffs vary a little (a car is about 120 units long). 0 means exact spots; at most 300. The weight sets its share against the other starts. metrics.csv and the report show how fast the ball is first touched from each kind of spot: kickoff_touch_s_diagonal (the two diagonal spots), kickoff_touch_s_offcenter (the two back and to the side) and kickoff_touch_s_back (straight back), in seconds, with how many kickoffs each is over (kickoff_count_...).\n"
        "\n"
        "Watch out for: every save restarts all games from a kickoff anyway, so a resumed run continues exactly; that shows as a small blip in some charts at each save. Too much fuzz makes kickoffs unlike the real game's.\n"
        "\n"
        "Measured: not measured."
    ),
    "start.random": (
        "What it does: puts the ball and the cars anywhere on the field, with random speeds, directions and boost. It gives the bot practice in all kinds of situations a kickoff never leads to early in training.\n"
        "\n"
        "When to change it: add it next to kickoffs for variety. A common mix is about 70% kickoff and 30% random.\n"
        "\n"
        "How to use it:\n"
        "- max_ball_speed: the fastest the ball may start, in units per second (default 2,000; its top speed is 6,000).\n"
        "- ground_chance: the chance each car starts on its wheels instead of in the air (default 0.5).\n"
        "Lower ball speeds and a higher ground chance make easier situations.\n"
        "\n"
        "Watch out for: random situations are not like real play. Cars in the air at odd angles and balls flying fast are much more common than in a real game, so a bot trained mostly on random starts may get skills it rarely needs.\n"
        "\n"
        "Measured: not measured."
    ),
    "start.replay": (
        "What it does: starts games from real moments of recorded matches. It can use the five recordings RocketSim ships (25 minutes of 1v1, 2v2 and 3v3 by top players and bots) and recordings made from replays.\n"
        "\n"
        "When to change it: to give the bot practice in the situations real games actually produce. Top community bots such as Necto and Seer started about 70% of their games from real moments.\n"
        "\n"
        "How to use it: bundled = true uses RocketSim's recordings (119 MB, downloaded once from RocketSim's GitHub the first time a run needs them; nitro fetch-recordings gets them ahead, and without internet the run stops and says so); files lists your own (.rlpr files, wildcards allowed, read from where the settings file is). air_bias from 0 to 10 makes high moments likelier: 10 draws a ceiling-high moment 11 times as often. The presets' example is weight 2 beside kickoff 1, so two games in three start from a recording.\n"
        "\n"
        "Watch out for: the teams are worked out from the kickoffs, so a recording without a kickoff is refused, and one with too few cars per team for your game size is left out. Jump and flip state is not carried over. Half the moments are turned around so both ends of the field are used.\n"
        "\n"
        "Measured: not yet; whether real-game starts make a better bot is for full training."
    ),
}

# The "Nerd explanation" of each setting (keys as MORE): how it works inside, the maths where it helps, and a
# small settings-file example between lines of three backticks. Plain text; describe.py passes it on as 'nerd'.
NERD: dict[str, str] = {
    "paths.collision_meshes": (
        "The simulator (RocketSim) builds each arena's floor, walls, ramps and goals from triangle meshes dumped from the game. NitroLearn looks for them in this order: the path written here (relative to the settings file's folder), a collision_meshes folder next to the settings file, then NitroLearn's own collision_meshes folder. The folder must hold a soccar subfolder with the soccar mesh files.\n"
        "\n"
        "```toml\n"
        "[paths]\n"
        'collision_meshes = "D:/rocket league meshes"   # the folder that holds soccar/, not soccar itself\n'
        "```\n"
        "\n"
        "`nitro doctor` checks each file against the known meshes by its hash (a fingerprint of the file's bytes), so a wrong or damaged file is named instead of giving odd physics."
    ),
    "run.seed": (
        "Every random choice in a run comes from generators started from this one number. Each arena gets a generator of its own, made from the seed and the arena's number, so how the games run does not depend on how many processor threads step them. The network's starting weights, the order of the samples in each learning step and the sampled actions come from PyTorch's generator, also seeded from it.\n"
        "\n"
        "```toml\n"
        "[run]\n"
        "seed = 3          # another seed: other games, other starting weights\n"
        "```\n"
        "\n"
        "Two runs with the same seed and settings play the same games; for the learning to match bit for bit on a graphics card too, add run.deterministic = true."
    ),
    "run.device": (
        "\"auto\" asks PyTorch whether a CUDA graphics card is usable (torch.cuda.is_available()) and uses it, else the processor. The games always run on the processor's threads; only the networks' passes and the learning move.\n"
        "\n"
        "```toml\n"
        "[run]\n"
        'device = "cpu"    # force the processor, e.g. for a quick check while the card is busy\n'
        "```\n"
        "\n"
        "On the processor, FP16, the fused LayerNorm routine, recorded learning steps and overlapped passes have nothing to act on and quietly fall back to the plain way."
    ),
    "run.cards": (
        "With N cards, N processes start (one per card). Card k plays arenas k, k+N, ... (num_arenas / N each), collects its share of each iteration's steps (steps_per_iteration / N) and computes gradients on its share of every minibatch. After each backward pass the cards add their gradients with NCCL (NVIDIA's library for cards talking to each other) and divide by the total sample count, so every card applies the same update:\n"
        "\n"
        "g = (sum over cards of the gradient sums) / (total samples)\n"
        "\n"
        "which is exactly the gradient one card would get from the whole minibatch. The networks therefore stay identical; card 0 writes the saves.\n"
        "\n"
        "```toml\n"
        "[run]\n"
        "cards = 2\n"
        "[env]\n"
        "num_arenas = 1024         # must divide by cards\n"
        "```"
    ),
    "run.cards_sum_precision": (
        'Each learning step the cards add up their gradients (an all-reduce: every card sends its numbers and gets the total back). With "bf16" the gradients are rounded to bfloat16 before they travel: 16 bits each instead of 32, with FP32\'s range but only 8 bits of precision (about 3 significant digits). The sum is turned back into FP32 before the optimizer step.\n'
        "\n"
        "```toml\n"
        "[run]\n"
        "cards = 2\n"
        'cards_sum_precision = "bf16"\n'
        "```\n"
        "\n"
        "Traffic per step halves: for a network of P parameters, about 2P bytes instead of 4P go each way. The loss values and sample counts still travel in FP32, so the averages stay exact."
    ),
    "run.cards_pin_threads": (
        "On Linux each card's process is bound (with sched_setaffinity) to a set of whole cache groups: cores that share a level-3 cache. The game threads of that process then keep their data in one cache instead of moving between cores whose caches do not share it.\n"
        "\n"
        "```toml\n"
        "[run]\n"
        "cards = 2\n"
        "cards_pin_threads = true\n"
        "```\n"
        "\n"
        "On a processor with one cache group, or with one card, nothing changes."
    ),
    "run.cards_check_every": (
        "The check: every card lays its weights out in one long row, adds up each run of 4,096 of them and the squares of all of them (in 64-bit numbers), and the cards compare these sums exactly (the smallest and the largest over the cards must be equal). The same weights on the same kind of card give the same bits. The check waits for every card, so it costs the time the fastest card idles.\n"
        "\n"
        "```toml\n"
        "[run]\n"
        "cards = 2\n"
        "cards_check_every = 10    # every 10th iteration and before every save\n"
        "```\n"
        "\n"
        "With k, a drift would be found at most k - 1 iterations late, and never after a save has written it."
    ),
    "run.cards_graphed_steps": (
        "A CUDA graph is a recording of a fixed sequence of graphics-card jobs that the card can replay with one launch. On one card, ppo.graphed_steps records each learning step. On several cards each step needs the cards' gradient sum in the middle, so it is recorded as two graphs (the forward and backward pass, then the optimizer step) with the NCCL sum between them.\n"
        "\n"
        "```toml\n"
        "[run]\n"
        "cards = 2\n"
        "cards_graphed_steps = true\n"
        "[ppo]\n"
        "graphed_steps = true\n"
        "```\n"
        "\n"
        "A step then costs two launches plus the sum, instead of hundreds of small launches from Python."
    ),
    "run.precision": (
        "With \"fp16\" the hidden layers run under PyTorch's autocast: matrix products take FP16 inputs (16-bit numbers, about 3 significant digits, largest about 65,504) and the graphics card's fast FP16 units. The output layers, the softmax of the action probabilities, the probability ratios and the losses stay FP32. The gradients go through a loss scaler: the loss is multiplied by a large factor S before the backward pass so small gradients do not round to 0, then divided by S; if any gradient overflows, the step is skipped and S halves.\n"
        "\n"
        "```toml\n"
        "[run]\n"
        'precision = "fp16"\n'
        "[model]\n"
        "fused_norm = true          # pairs well with FP16 for big networks\n"
        "```\n"
        "\n"
        "The speed-up comes from the matrix products: for an n-by-n layer a pass costs about 2 x batch x n^2 operations, so big layers gain the most."
    ),
    "run.fp16_sums": (
        "A matrix product adds up many products of pairs: y = sum over i of a_i x b_i. In FP16 mode PyTorch normally keeps that running sum in FP32. This setting lets the card keep it in FP16 (torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction), which many GeForce cards do twice as fast. Each addition then rounds to about 3 significant digits, so a sum of n terms can drift by roughly sqrt(n) x 0.001 of its size.\n"
        "\n"
        "```toml\n"
        "[run]\n"
        'precision = "fp16"\n'
        'fp16_sums = "playing"      # only the passes that pick actions\n'
        "```\n"
        "\n"
        '"forward" also covers learning\'s forward passes; "all" also the backward pass, where sums run over thousands of samples and the drift is largest.'
    ),
    "run.deterministic": (
        "It switches PyTorch to its repeatable routines (torch.use_deterministic_algorithms), stops the card from picking the fastest routine afresh each run, and sets CUBLAS_WORKSPACE_CONFIG, which the card's matrix library needs to add up in a fixed order. Floating-point addition is not associative ((a + b) + c can differ from a + (b + c) in the last bit), so a fixed order is what makes two runs bit-identical.\n"
        "\n"
        "```toml\n"
        "[run]\n"
        "seed = 7\n"
        "deterministic = true\n"
        "```\n"
        "\n"
        "Resuming is exact too: a save holds every random generator's state."
    ),
    "run.on_training_problem": (
        "After each update the learner checks the losses and the networks' weights for NaN (not a number) and infinity, and counts updates the FP16 loss scaler skipped. A NaN spreads: any sum with a NaN in it is NaN, so one bad step can fill every weight within a few updates.\n"
        "\n"
        "```toml\n"
        "[run]\n"
        'on_training_problem = "warn"   # keep going and only warn\n'
        "```\n"
        "\n"
        'With "stop" the run ends before saving the broken state, so the newest save is the last good one.'
    ),
    "run.missing_data": (
        "At the start the trainer compares what the save refers to with what is on disk: the ratings table names past versions by their step counts, and each needs its file in the pool folder; [replay_mix] needs converted replay files (.nrpl.zst) in its folders.\n"
        "\n"
        "```toml\n"
        "[run]\n"
        'missing_data = "stop"   # refuse to train until everything is copied over\n'
        "```\n"
        "\n"
        'With "warn", a past version without its file is never drawn as an opponent; its rating stays in the table.'
    ),
    "run.checkpoint_minutes": (
        "A save writes checkpoints/<agent steps>/state.pt: the networks, the optimizer's state, the reward and value scaling, the self-play ratings and every random generator. Then every arena restarts from a start state, so a resume rebuilds exactly the same games. The cut games are marked as truncated (cut off, not ended): the learner still adds the value of the last state, so the cut teaches nothing wrong.\n"
        "\n"
        "```toml\n"
        "[run]\n"
        "checkpoint_minutes = 30\n"
        "```\n"
        "\n"
        "Lost work on a crash is at most this many minutes."
    ),
    "run.checkpoint_steps": (
        "With a step count, a save happens after the first iteration whose total agent steps reach the next multiple of it. Saves then sit at round numbers (each save's folder is named by its agent steps), which makes saves of different bots comparable.\n"
        "\n"
        "```toml\n"
        "[run]\n"
        "checkpoint_steps = 200_000_000\n"
        "keep_checkpoints = 5          # the newest 1,000,000,000 steps stay\n"
        "milestone_steps = 1_000_000_000\n"
        "```\n"
        "\n"
        "Time between saves = checkpoint_steps / (agent steps per second)."
    ),
    "run.keep_checkpoints": (
        "After each save the oldest saves beyond this count are deleted, except milestone saves (run.milestone_steps), which stay for good.\n"
        "\n"
        "```toml\n"
        "[run]\n"
        "keep_checkpoints = 10\n"
        "```\n"
        "\n"
        "Disk used is about (keep_checkpoints + milestones) x one save's size; a save's size grows with the networks (a few MB for the starters, a few hundred MB for very big bots, whose optimizer state is twice the weights)."
    ),
    "run.milestone_steps": (
        "A save is a milestone when it is the first save at or past a multiple of this many agent steps: with 1,000,000,000, the first save after 1B, after 2B and so on. Milestones are never deleted by keep_checkpoints.\n"
        "\n"
        "```toml\n"
        "[run]\n"
        "milestone_steps = 500_000_000    # a lasting save every half billion steps\n"
        "```\n"
        "\n"
        "0 keeps only the newest keep_checkpoints saves."
    ),
    "run.report_minutes": (
        "The report (report.html in the bot's folder) is written from metrics.csv: one row per iteration, every column the learner logs. Rewriting it reads the whole file, so on a long run a short interval costs a little processor time.\n"
        "\n"
        "```toml\n"
        "[run]\n"
        "report_minutes = 5\n"
        "```"
    ),
    "run.console": (
        '"line" prints one line per console_seconds: agent steps, steps per second, the mean reward, the policy\'s entropy and the value prediction. "report" prints blocks: speed and where the time went (collecting games, learning, saving), each reward\'s share of the total, the learning numbers (losses, KL divergence, clip fraction, explained variance), play statistics, self-play ratings and new warnings.\n'
        "\n"
        "```toml\n"
        "[run]\n"
        'console = "report"\n'
        "console_seconds = 30\n"
        "```\n"
        "\n"
        "It only changes what is printed; metrics.csv and the learning are the same."
    ),
    "run.console_seconds": (
        "The console is printed at most this often; a line or report covers the iterations since the last one.\n"
        "\n"
        "```toml\n"
        "[run]\n"
        "console_seconds = 60\n"
        "```"
    ),
    "env.team_size": (
        "Each arena holds 2 x team_size cars. Every car is an agent: in self-play both teams are the same bot, so one 1v1 arena gives 2 samples a decision, a 3v3 arena 6.\n"
        "\n"
        "```toml\n"
        "[env]\n"
        "team_size = 2\n"
        "num_arenas = 512     # 2,048 cars, as 1,024 1v1 arenas\n"
        "```\n"
        "\n"
        "Samples per second = arenas x 2 x team_size x decisions per second. The observation's size grows with the number of cars (each other car is a block of numbers), unless an attention network reads them as a set."
    ),
    "env.team_sizes": (
        "Arena i plays one size for the whole run; the sizes are dealt over the arenas in proportion to team_size_shares. Each batch then mixes every size, and the network must take every size's observation, so it needs room for the largest: a padded observation (env.obs_team_size) or an attention network.\n"
        "\n"
        "```toml\n"
        "[env]\n"
        "team_size = 3\n"
        "team_sizes = [1, 2, 3]\n"
        "obs_team_size = 3\n"
        "```\n"
        "\n"
        "Samples per size scale with cars: equal arena shares give 1v1, 2v2 and 3v3 samples in the ratio 2 : 4 : 6."
    ),
    "env.team_size_shares": (
        "Arena counts per size are num_arenas x share_i / sum(shares), rounded so they add up to num_arenas. A size's share of the samples is its arenas x its cars per arena.\n"
        "\n"
        "```toml\n"
        "[env]\n"
        "team_size = 3\n"
        "team_sizes = [1, 2, 3]\n"
        "obs_team_size = 3\n"
        "team_size_shares = [3, 1.5, 1]   # samples about 6 : 6 : 6, one third each\n"
        "```\n"
        "\n"
        "To give each size the same share of samples, make the arena shares inversely proportional to the cars: 1/2, 1/4, 1/6, i.e. [3, 1.5, 1]."
    ),
    "env.shuffle_slots": (
        'A padded observation has fixed places: teammate places, then opponent places, each a block of numbers ending in a "car present" flag. In a 1v1 game with room for 3, the opponent always sits in the first opponent place and the other places stay empty, so the network never learns what the later places mean. With shuffling, each game deals the cars a random permutation of their team\'s places.\n'
        "\n"
        "```toml\n"
        "[env]\n"
        "obs_team_size = 3\n"
        "shuffle_slots = true\n"
        "```\n"
        "\n"
        "The games are the same; only where each car is written in the observation changes."
    ),
    "env.opponents": (
        "Off, each arena holds only the Blue team's team_size cars. Rewards that compare teams (zero_sum, goals against) then see no opponents.\n"
        "\n"
        "```toml\n"
        "[env]\n"
        "opponents = false     # a lone car, e.g. to practise touching the ball\n"
        "```"
    ),
    "env.num_arenas": (
        "Each iteration collects ppo.steps_per_iteration agent steps over all arenas, so each arena advances about\n"
        "\n"
        "steps_per_iteration / (num_arenas x cars per arena)\n"
        "\n"
        "decisions an iteration. More arenas fill a batch faster (the graphics card handles all arenas' cars in one pass), but each game gets fewer steps per update. With the default 1,024 1v1 arenas and 50,000 steps, every game moves about 24 decisions (0.8 s of play) per update.\n"
        "\n"
        "```toml\n"
        "[env]\n"
        "num_arenas = 2048\n"
        "```\n"
        "\n"
        "The arenas are split into env.groups groups, so the processor steps one group while the card thinks for another."
    ),
    "env.tick_skip": (
        "The physics runs at 120 ticks a second. The bot picks an action every tick_skip ticks and the action is held in between, so the bot decides 120 / tick_skip times a second: 30 at 4, 15 at 8.\n"
        "\n"
        "```toml\n"
        "[env]\n"
        "tick_skip = 8\n"
        "action_delay = 7\n"
        "```\n"
        "\n"
        "Settings given in seconds convert on their own: the discount per decision is gamma = 0.5^(tick_skip / (120 x discount_half_life)), and rewards paid every step are scaled by tick_skip / 8 so a weight means the same per second. ppo.gae_lambda counts decisions: the same reach in seconds at another tick skip is lambda^(new / old)."
    ),
    "env.action_delay": (
        "The action chosen at tick t starts acting at tick t + action_delay and holds for tick_skip ticks. In a real match the bot reads the game, thinks and sends its controls, which arrive a tick or two later; the delay trains the bot with that lag built in.\n"
        "\n"
        "```toml\n"
        "[env]\n"
        "tick_skip = 4\n"
        "action_delay = 1      # must stay below tick_skip\n"
        "```\n"
        "\n"
        "At 0 the bot learns to react on the very tick it sees, which the exported bot cannot do in a real game."
    ),
    "env.observation": (
        "The observation is the list of numbers the network reads each decision. \"native\" (112 numbers in 1v1): the ball (9), the 34 boost pads (whether each is up), the previous action (8), the bot's own car (32) and the other car (29). \"nitro_v2\" (148 in 1v1): the same groups with more features: positions and speeds relative to the own car and to both goals, the ball's speed toward the goal, distances, the car's speed along its own axes, flip and demolition timers. Everything is written from the bot's team's side (Blue attacks +y; Orange sees the field turned 180 degrees), so one network plays both colours.\n"
        "\n"
        "```toml\n"
        "[env]\n"
        'observation = "native"    # the smaller observation\n'
        "```\n"
        "\n"
        "Sizes per other car: native 29, nitro_v2 38 numbers (plus a present flag with padding)."
    ),
    "env.obs_team_size": (
        "With room R (R >= team_size), the observation always has R - 1 teammate places and R opponent places. Each other-car block ends in a present flag (1 = a car is there, 0 = empty, with every number in the block 0). The size is fixed by R, so the same network reads any game up to R a side.\n"
        "\n"
        "```toml\n"
        "[env]\n"
        "team_size = 1\n"
        "obs_team_size = 3       # 1v1 now, 2v2 or 3v3 later with the same network\n"
        "```\n"
        "\n"
        "Size with room 3, native: 51 + 32 + 5 x 30 = 233 numbers (each block 29 + the flag)."
    ),
    "env.groups": (
        "The arenas are split into G groups. While the graphics card works out the actions for group k, the processor's threads step the physics of the other groups, so neither waits for the other. With G = 2 the work alternates; more groups mean smaller passes on the card.\n"
        "\n"
        "```toml\n"
        "[env]\n"
        "groups = 2\n"
        "```\n"
        "\n"
        "num_arenas must be at least groups."
    ),
    "env.pipelined_inference": (
        "Python launches graphics-card work asynchronously: a launch returns at once and the card runs it later. With this on, the next group's network pass is launched before the previous group's results are read back, so the time Python spends launching overlaps the card's work.\n"
        "\n"
        "```toml\n"
        "[env]\n"
        "pipelined_inference = false   # strictly one pass after another\n"
        "```\n"
        "\n"
        "The games and the collected data are bit-for-bit the same either way."
    ),
    "env.card_starts_games": (
        "Without it: Python waits for the card's actions, copies them to the processor and tells the game threads to step. With it: the copy and the start signal are queued on the card itself (a host callback runs as soon as the actions are on the processor), and Python moves on to prepare the next group.\n"
        "\n"
        "```toml\n"
        "[env]\n"
        "card_starts_games = false\n"
        "```\n"
        "\n"
        "The same actions reach the same games; only who starts them changes."
    ),
    "env.concurrent_passes": (
        "Each decision needs the policy's pass, the value network's pass and, in self-play, each past version's pass. With this on they go on separate CUDA streams (queues of card jobs that may run side by side), so several small passes share the card's units at once.\n"
        "\n"
        "```toml\n"
        "[env]\n"
        "concurrent_passes = false\n"
        "```\n"
        "\n"
        "Each pass computes exactly what it would alone, so the results are bit-identical."
    ),
    "env.threads": (
        "The physics of each group's arenas is split over this many processor threads. 0 picks all logical processors but two, leaving room for Python and the system. Each arena has its own random generator, so the thread count never changes the games.\n"
        "\n"
        "```toml\n"
        "[env]\n"
        "threads = 12\n"
        "```"
    ),
    "terminals.no_touch_seconds": (
        "A game is cut when nobody has touched the ball for this long. The cut is a truncation (the game is stopped, not finished): the learner adds the value of the last state, so the bot is not taught that the stall itself led nowhere.\n"
        "\n"
        "```toml\n"
        "[terminals]\n"
        "no_touch_seconds = 20\n"
        "```\n"
        "\n"
        "Decisions until the cut = no_touch_seconds x 120 / tick_skip (300 at 10 s and tick skip 4)."
    ),
    "terminals.max_episode_seconds": (
        "Every game is cut at this length whatever happens, as a truncation (the value of the last state still counts).\n"
        "\n"
        "```toml\n"
        "[terminals]\n"
        "no_touch_seconds = 0\n"
        "max_episode_seconds = 10    # practice tasks: a fixed-length try\n"
        "```"
    ),
    "terminals.end_on_touch": (
        'A touch ends the game as done (terminal): nothing after it counts, so the return of the touching step is its reward alone. With touch rewards that makes "touch as fast as possible" the whole task.\n'
        "\n"
        "```toml\n"
        "[terminals]\n"
        "end_on_touch = true\n"
        "max_episode_seconds = 10\n"
        "```"
    ),
    "terminals.end_on_landing": (
        'The game ends as done once every car has all four wheels on the floor. With a reward for landing upright, the task becomes "land well, quickly".\n'
        "\n"
        "```toml\n"
        "[terminals]\n"
        "end_on_landing = true\n"
        "max_episode_seconds = 5\n"
        "```"
    ),
    "terminals.truncate_on_ball_landing": (
        "When the ball comes down onto the floor after being in the air, a random draw cuts the game with this chance, as a truncation (the value of the last state still counts). With chance p, the expected number of landings before a cut is 1 / p.\n"
        "\n"
        "```toml\n"
        "[terminals]\n"
        "truncate_on_ball_landing = 0.1   # about one game in ten landings ends there\n"
        "```\n"
        "\n"
        "The draw happens only on a landing and only for 0 < p < 1, so runs without it keep their random sequence."
    ),
    "model.shared": (
        "Without a shared trunk there are two networks: policy(obs) -> 90 action scores and value(obs) -> one number. With one, both read the trunk's output: h = trunk(obs), then policy(h) and value(h). The trunk learns from both losses at once, so what helps judge a moment can also help choose an action.\n"
        "\n"
        "```toml\n"
        "[model]\n"
        "shared = [512, 512]\n"
        "policy = [256, 256]\n"
        "value = [256, 256]\n"
        "```\n"
        "\n"
        "Each hidden layer is linear, then LayerNorm, then the activation: h = act(LN(W x + b)). A layer from n to m numbers has n x m + m weights."
    ),
    "model.policy": (
        "The policy turns the observation (or the shared trunk's output) into one score per action, the logits z_1..z_90. Actions that make no sense at that moment are masked out (boosting with an empty tank, a jump when the car can neither jump nor flip, ground-only actions in the air); the rest get probabilities by softmax: p_i = exp(z_i) / sum over allowed j of exp(z_j). Training samples an action from p; the exported bot takes the likeliest.\n"
        "\n"
        "```toml\n"
        "[model]\n"
        "policy = [512, 512, 512]\n"
        "```\n"
        "\n"
        "Parameters for layers [a, b, c] on an observation of n numbers: n x a + a x b + b x c + c x 90, plus the biases and LayerNorms."
    ),
    "model.value": (
        "The value network maps the observation to one number V(s): the expected discounted sum of future rewards from this moment, sum over k of gamma^k x r_(t+k). Learning compares it with what really followed, and the policy is pushed toward actions that did better than V predicted (the advantage).\n"
        "\n"
        "```toml\n"
        "[model]\n"
        "value = [512, 512, 512]\n"
        "```\n"
        "\n"
        "With value_heads, these layers become the critic trunk that every head reads. Only training uses the value network; the exported bot leaves it out."
    ),
    "model.value_heads": (
        'Each head is a small network on top of the critic trunk: V_main(s), V_goal(s), V_vdag1(s), V_vdag2(s). They share the trunk\'s layers but have their own targets: "main" the usual discounted return, "goal" a much longer-horizon return (ppo.goal_critic_half_life), "vdag1"/"vdag2" an optimistic one-step target (ppo.vdag_beta).\n'
        "\n"
        "```toml\n"
        "[model]\n"
        "value = [1280, 1280, 1280]\n"
        "value_head_layers = [1280, 1280]\n"
        'value_heads = ["main", "goal", "vdag1", "vdag2"]\n'
        "```\n"
        "\n"
        "The trunk's gradient is the sum of the heads' gradients, so each head also shapes what the trunk learns."
    ),
    "model.value_head_layers": (
        "The layers a head has to itself between the critic trunk and its single output. Every head gets its own copy of them, so their cost multiplies by the number of heads: with k heads of layers [w, w] on a trunk ending in w, about k x (2 x w^2 + w) weights.\n"
        "\n"
        "```toml\n"
        "[model]\n"
        "value_head_layers = [600, 600]\n"
        "```\n"
        "\n"
        "With model.residual, a head layer as wide as the one before it is a residual block too."
    ),
    "model.residual": (
        "A plain hidden layer replaces its input: h_(i+1) = act(LN(W_i h_i + b_i)). A residual block adds to it: h_(i+1) = h_i + act(LN(W_i h_i + b_i)). The gradient then has a direct path through the sum (the derivative of h_i with respect to itself is 1), so deep stacks keep learning in their early layers. Only a layer with the same width in and out can be a block; the first layer of each network, which changes the width, stays plain.\n"
        "\n"
        "```toml\n"
        "[model]\n"
        "shared = [1024, 1024, 1024]   # first layer plain, the next two residual\n"
        "residual = true\n"
        "layer_norm = true\n"
        "```\n"
        "\n"
        "The sum is kept in FP32 even under FP16, so the stream of block outputs stays exact."
    ),
    "model.layer_norm": (
        "LayerNorm rescales each sample's layer output to mean 0 and spread 1, then applies a learned scale and shift per number: y = gamma x (x - mean) / sqrt(variance + 1e-5) + beta. It keeps the numbers in a steady range however the weights grow, which steadies learning.\n"
        "\n"
        "```toml\n"
        "[model]\n"
        "layer_norm = false\n"
        "```\n"
        "\n"
        "It sits after each hidden layer's linear part and before its activation. Residual layers, fused_norm and weight_decay need it."
    ),
    "model.fused_norm": (
        "Normally LayerNorm and the activation are separate graphics-card jobs: each reads the layer's numbers from memory and writes them back. The fused routine does both in one pass, compiled on first use with Triton (the compiler PyTorch ships). For big FP16 layers memory traffic, not arithmetic, is the limit there, so one pass instead of several saves time.\n"
        "\n"
        "```toml\n"
        "[model]\n"
        "fused_norm = true\n"
        "layer_norm = true\n"
        'activation = "leaky_relu"\n'
        "```\n"
        "\n"
        "It computes the same maths (to rounding), so it can be switched on or off between a stop and a resume."
    ),
    "model.init": (
        "\"orthogonal\" starts each hidden layer's weights as a random orthogonal matrix (its rows at right angles to each other) times sqrt(2), which keeps the size of the numbers steady through ReLU-like layers. The policy's output layer starts at 0.01 of that size, so all actions start almost equally likely; the value's output layer at 1. \"pytorch\" uses PyTorch's own default (uniform random numbers scaled by 1 / sqrt(inputs)).\n"
        "\n"
        "```toml\n"
        "[model]\n"
        'init = "pytorch"\n'
        "```\n"
        "\n"
        "Only the starting point changes; a bot started from another bot (start_from) takes that bot's weights instead."
    ),
    "model.activation": (
        "The activation bends each layer's output so stacked layers can learn more than a straight line:\n"
        "- relu(x) = max(0, x)\n"
        "- leaky_relu(x) = x for x > 0, 0.01 x otherwise\n"
        "- gelu(x) = x x Phi(x), Phi being the normal distribution's cumulative chance\n"
        "- tanh(x) squeezes into -1..1\n"
        'With relu, a neuron whose input stays negative gets gradient 0 and may never recover (a "dead" neuron); leaky_relu keeps a gradient of 0.01 there.\n'
        "\n"
        "```toml\n"
        "[model]\n"
        'activation = "relu"\n'
        "```\n"
        "\n"
        "The exported bot runs the same function, so it is fixed once the bot has saves."
    ),
    "model.encoder": (
        '"attention": the other cars become a set of tokens (one vector per car), and the own car asks each one a question: score_j = (q . k_j) / sqrt(d), weights w = softmax(scores), result = sum of w_j x v_j, where q comes from the own car and k_j, v_j from car j (all through shared weights). Empty places are masked out. "attention_communication" first lets every token look at every other one (self-attention) in each block, then the own car looks. "transformer": the own car, the ball, every other car (and optionally each pad) are tokens in a standard transformer, and the policy and value layers read one token at the end.\n'
        "\n"
        "```toml\n"
        "[env]\n"
        "team_size = 2\n"
        "[model]\n"
        'encoder = "attention"\n'
        "attention_fast = true\n"
        "```\n"
        "\n"
        "Because the weights are shared over cars, the network's size does not depend on the number of cars."
    ),
    "model.attention_width": (
        "The width d of each token: every car and the ball are turned into d numbers before the looks. Each head works on d / heads of them. Cost per look grows with d^2 (the query, key and value layers are d x d each).\n"
        "\n"
        "```toml\n"
        "[model]\n"
        'encoder = "attention"\n'
        "attention_width = 256\n"
        "attention_heads = 8      # 256 / 8 = 32 numbers per head\n"
        "```"
    ),
    "model.attention_blocks": (
        "Each block is one look: the own car's vector is updated from what it saw, h <- h + attention(h, cars), then a small feed-forward layer. The next block looks again with the updated vector, so it can ask a question that depends on the first answer.\n"
        "\n"
        "```toml\n"
        "[model]\n"
        'encoder = "attention"\n'
        "attention_blocks = 3\n"
        "```\n"
        "\n"
        "Cost and parameters grow about linearly with the number of blocks."
    ),
    "model.attention_heads": (
        "With H heads, the width d is split into H parts of d / H numbers; each part makes its own scores and weights, so the car can attend to H different things at once. The results are joined back into d numbers.\n"
        "\n"
        "```toml\n"
        "[model]\n"
        'encoder = "attention"\n'
        "attention_width = 128\n"
        "attention_heads = 4      # 32 numbers per head\n"
        "```\n"
        "\n"
        "The width must divide by the heads."
    ),
    "model.attention_skip": (
        "Without it the policy and value layers see only the encoder's output. With it they also get the observation's own part directly: [encoder output, ball, own car, pads, previous action], joined into one vector.\n"
        "\n"
        "```toml\n"
        "[model]\n"
        'encoder = "attention"\n'
        "attention_skip = true\n"
        "```\n"
        "\n"
        "The first layer after the encoder then has more inputs; nothing else changes."
    ),
    "model.attention_fast": (
        "The same attention maths, arranged for the graphics card: the other cars are normalised once instead of once per block, the keys and values of every block come out of one big matrix product, and the empty place's token is computed once per batch. Fewer, bigger jobs keep the card busier.\n"
        "\n"
        "```toml\n"
        "[model]\n"
        'encoder = "attention"\n'
        "attention_fast = true\n"
        "```\n"
        "\n"
        "Results match the plain path to rounding, so saves and the exported bot are the same."
    ),
    "model.small_attention": (
        "For a handful of tokens (the cars and the ball, at most 8), a general attention routine built for long text sequences wastes most of its work. NitroLearn's own routine (compiled with Triton on first use) computes softmax(Q K^T / sqrt(d)) V for these tiny sequences in one pass per head.\n"
        "\n"
        "```toml\n"
        "[model]\n"
        'encoder = "transformer"\n'
        "small_attention = false    # use PyTorch's routine\n"
        "```\n"
        "\n"
        "More than 8 tokens or more than 32 numbers a head falls back to PyTorch's routine."
    ),
    "model.lean_attention": (
        "When one query (the own car) looks at the tokens, the keys and values need not be made: q . (W_k x) = (W_k^T q) . x, so the key layer can be folded into the query, and the value layer applied once after the weighted sum: sum of w_j (W_v x_j) = W_v (sum of w_j x_j). That skips a matrix product per token. A key bias adds the same number to every score, which softmax ignores, so it has no effect.\n"
        "\n"
        "```toml\n"
        "[model]\n"
        'encoder = "transformer"\n'
        "lean_attention = true\n"
        "```\n"
        "\n"
        "The answers are the same to rounding."
    ),
    "model.transformer_layers": (
        "Each transformer block: every token t gets t <- t + attention(LN(t), all tokens), then t <- t + FF(LN(t)), where FF is two layers (width -> transformer_ff -> width). More blocks let information travel further: after two blocks a token knows what others learned from a third.\n"
        "\n"
        "```toml\n"
        "[model]\n"
        'encoder = "transformer"\n'
        "transformer_layers = 3\n"
        "```"
    ),
    "model.transformer_width": (
        "The width d of every token. Per block the attention costs about 4 d^2 weights (query, key, value and output) and the feed-forward 2 x d x ff.\n"
        "\n"
        "```toml\n"
        "[model]\n"
        'encoder = "transformer"\n'
        "transformer_width = 256\n"
        "transformer_heads = 8\n"
        "transformer_ff = 1024\n"
        "```"
    ),
    "model.transformer_heads": (
        "The width is split into heads of d / heads numbers; each makes its own attention scores, so a token can follow several others at once.\n"
        "\n"
        "```toml\n"
        "[model]\n"
        'encoder = "transformer"\n'
        "transformer_width = 128\n"
        "transformer_heads = 4     # 32 numbers per head\n"
        "```"
    ),
    "model.transformer_ff": (
        "The feed-forward part of a block: FF(t) = W2 act(W1 t + b1) + b2, with W1 from width to ff numbers and W2 back. It is where each token works on what it gathered.\n"
        "\n"
        "```toml\n"
        "[model]\n"
        'encoder = "transformer"\n'
        "transformer_ff = 512     # 4 x a width of 128\n"
        "```"
    ),
    "model.transformer_pads": (
        "Off: the 34 pads' up-or-down flags are part of the own car's token. On: each pad is a token of its own (its flag plus a learned vector for its place on the field), so the attention can pick out the pads that matter. Attention cost grows with the square of the number of tokens: with 2v2, 8 tokens become 42, about 27 times the attention work.\n"
        "\n"
        "```toml\n"
        "[model]\n"
        'encoder = "transformer"\n'
        "transformer_pads = true\n"
        "```"
    ),
    "model.transformer_readout": (
        'After the last block one token\'s vector goes to the policy and value layers. "own_car" reads the own car\'s token. "summary" adds one more learned token (it starts the same in every game and belongs to no object) that attends to everything, like the class token of vision transformers, and reads that.\n'
        "\n"
        "```toml\n"
        "[model]\n"
        'encoder = "transformer"\n'
        'transformer_readout = "summary"\n'
        "```"
    ),
    "start_from.enabled": (
        "At the bot's first start, with a run named, the trainer loads that bot's save and copies the chosen parts' weights into the new networks before the first iteration. Off: the run name is kept but nothing is copied. Later starts resume from the bot's own saves and never read [start_from] again.\n"
        "\n"
        "```toml\n"
        "[start_from]\n"
        "enabled = true\n"
        'run = "my-first-bot"\n'
        'parts = "policy"\n'
        "```"
    ),
    "start_from.run": (
        "A bot's name (its folder in the runs folder), a run folder's path, or a save's path. The networks must match in shape: the same layer sizes, layer_norm, activation and observation, so each weight has the same place in both.\n"
        "\n"
        "```toml\n"
        "[start_from]\n"
        'run = "my-first-bot"\n'
        "```\n"
        "\n"
        "The same as `nitro train <settings> --from my-first-bot`."
    ),
    "start_from.at": (
        "Saves are named by their agent steps (checkpoints/<steps>). 0 picks the newest save at the moment this bot starts; a number picks that save.\n"
        "\n"
        "```toml\n"
        "[start_from]\n"
        'run = "my-first-bot"\n'
        "at = 2_000_000_000\n"
        "```"
    ),
    "start_from.parts": (
        "What is copied:\n"
        '- "all": policy, value network, shared trunk, and the reward and value scaling (the running statistics that turn raw rewards into the learner\'s units), so learning continues as it was.\n'
        '- "policy": only the policy and the shared trunk; the value network starts fresh. Right after new rewards, whose values the old value network never learned.\n'
        '- "matching": the policy and shared trunk, plus every value part whose layer sizes are the same.\n'
        "\n"
        "```toml\n"
        "[start_from]\n"
        'run = "my-first-bot"\n'
        'parts = "policy"\n'
        "[pretrain]\n"
        "value_warmup_steps = 10_000_000   # let the fresh value network catch up first\n"
        "```"
    ),
    "ppo.steps_per_iteration": (
        "One iteration = collect this many agent steps (one step = one car's one decision), then learn from them. The count is rounded up to a whole number of decisions of every arena. Learning then splits the batch into minibatches and passes over it epochs times:\n"
        "\n"
        "learning steps per iteration = epochs x minibatches; samples per learning step = steps_per_iteration / minibatches\n"
        "\n"
        "```toml\n"
        "[ppo]\n"
        "steps_per_iteration = 100_000\n"
        "minibatches = 8       # 12,500 samples a step\n"
        "epochs = 2\n"
        "```\n"
        "\n"
        "Bigger batches give less noisy gradients but fewer updates per hour."
    ),
    "ppo.epochs": (
        "After collecting a batch, the learner goes over it this many times, each time in a new random order. Later passes reuse data the policy has already moved away from; PPO's clip (ppo.clip_range) limits how far that can push it.\n"
        "\n"
        "```toml\n"
        "[ppo]\n"
        "epochs = 3\n"
        "```\n"
        "\n"
        "Time spent learning grows in proportion to epochs."
    ),
    "ppo.minibatches": (
        "Each pass splits the batch into this many parts and takes one learning step (one optimizer update) per part. More parts mean more, smaller steps from the same data.\n"
        "\n"
        "```toml\n"
        "[ppo]\n"
        "minibatches = 8\n"
        "```\n"
        "\n"
        "With 50,000 steps and 4 minibatches, each step learns from 12,500 samples."
    ),
    "ppo.microbatch": (
        "A learning step on B samples can be worked out in parts of at most m samples: the gradient of a sum is the sum of the parts' gradients, so\n"
        "\n"
        "grad(loss over B) = sum over parts of grad(loss over the part) x (part size / B)\n"
        "\n"
        "gives the same step (to rounding) in the memory of m samples.\n"
        "\n"
        "```toml\n"
        "[ppo]\n"
        "microbatch = 4096\n"
        "```\n"
        "\n"
        "Slower, because each part is a separate pass; use it only when a step does not fit in the card's memory."
    ),
    "ppo.graphed_steps": (
        "A CUDA graph records a sequence of graphics-card jobs once and replays it with one launch. A learning step (forward pass, backward pass, optimizer update) is hundreds of small jobs; launched one by one from Python, a fast card spends much of the step waiting. Recorded, a step costs one launch. A minibatch smaller than the recorded size is padded with samples that count 0.\n"
        "\n"
        "```toml\n"
        "[ppo]\n"
        "graphed_steps = false\n"
        "```\n"
        "\n"
        "The recording is made on the first update and reused while the batch shapes stay the same."
    ),
    "ppo.policy_lr": (
        "Each learning step moves the policy's weights by about the learning rate times a normalised gradient. With Adam (the default optimizer): m <- 0.9 m + 0.1 g, v <- 0.999 v + 0.001 g^2, w <- w - lr x m / (sqrt(v) + eps), with the usual start-up corrections, so each weight moves by at most about lr per step.\n"
        "\n"
        "```toml\n"
        "[ppo]\n"
        "policy_lr = 4.5e-4\n"
        "```\n"
        "\n"
        "The policy rate also moves the shared trunk (model.shared). The built-in 4.5e-4 measured best on practice tasks; if the console warns that updates are too big (a large KL divergence or clip fraction), lower it."
    ),
    "ppo.value_lr": (
        "The value network's Adam rate (see ppo.policy_lr for the update rule). The value network learns to predict the return by lowering the squared error (V(s) - target)^2.\n"
        "\n"
        "```toml\n"
        "[ppo]\n"
        "value_lr = 2e-4\n"
        "```\n"
        "\n"
        "With value heads, every head and the critic trunk use this rate."
    ),
    "ppo.encoder_lr_scale": (
        "With an attention or transformer encoder, the encoder's weights get their own optimizer group with rate = network rate x this factor: policy_lr x scale for the policy's encoder, value_lr x scale for the value's.\n"
        "\n"
        "```toml\n"
        "[model]\n"
        'encoder = "attention"\n'
        "[ppo]\n"
        "encoder_lr_scale = 3.0\n"
        "```"
    ),
    "ppo.discount_half_life": (
        "The discount gamma is how much a reward one decision later counts. NitroLearn sets it from seconds:\n"
        "\n"
        "gamma = 0.5 ^ (tick_skip / (120 x half_life))\n"
        "\n"
        "so a reward half_life seconds away counts half, whatever the tick skip. 4.6 s gives 0.995 at tick skip 4. The value then predicts sum over k of gamma^k x r_(t+k); its effective reach is about 1 / (1 - gamma) decisions (200 at 0.995, about 6.7 s).\n"
        "\n"
        "```toml\n"
        "[ppo]\n"
        "discount_half_life = 10.0   # gamma 0.99769 at tick skip 4\n"
        "```"
    ),
    "ppo.gae_lambda": (
        "Generalised advantage estimation: with one-step errors delta_t = r_t + gamma x V(s_(t+1)) - V(s_t), the advantage is\n"
        "\n"
        "A_t = sum over k of (gamma x lambda)^k x delta_(t+k)\n"
        "\n"
        "lambda = 0 trusts the value network after one step (low noise, more bias); lambda = 1 adds up the real rewards to the end (no bias, more noise). It counts decisions, so at another tick skip the same game time is lambda^(new / old).\n"
        "\n"
        "```toml\n"
        "[env]\n"
        "tick_skip = 4\n"
        "[ppo]\n"
        "gae_lambda = 0.975     # about the reach of 0.95 at tick skip 8\n"
        "```"
    ),
    "ppo.goal_ends": (
        'With "terminal" the return stops at a goal: target_t = r_t (no gamma x V(next) after it). With "continue" the goal step\'s value target becomes r_t + gamma x V(s_next), where s_next is the next game\'s first state (the kickoff), as in a real match. The advantages still stop at the goal.\n'
        "\n"
        "```toml\n"
        "[ppo]\n"
        'goal_ends = "continue"\n'
        "```\n"
        "\n"
        'With rewards that pay throughout a game, "terminal" makes a goal end that stream, so a bot may learn to delay scoring; "continue" removes that pull.'
    ),
    "ppo.clip_range": (
        "PPO's clipped objective. For each sample, r = pi_new(a|s) / pi_old(a|s) is how much more (or less) likely the action became, and A its advantage. The policy maximises\n"
        "\n"
        "min(r x A, clip(r, 1 - epsilon, 1 + epsilon) x A)\n"
        "\n"
        "so once an action's probability has moved by more than epsilon (20% at 0.2) in the helpful direction, that sample stops pushing.\n"
        "\n"
        "```toml\n"
        "[ppo]\n"
        "clip_range = 0.1     # smaller, steadier updates\n"
        "```\n"
        "\n"
        "The console's clip fraction is the share of samples that hit the clip."
    ),
    "ppo.entropy": (
        "The policy's loss subtracts c x H, where H is the entropy of its action probabilities, normalised: H = -(sum of p_i ln p_i) / ln(n), n the number of allowed actions, so 0 = always the same action, 1 = all allowed actions equally likely. A larger c rewards keeping options open.\n"
        "\n"
        "```toml\n"
        "[ppo]\n"
        "entropy = 0.02\n"
        "```\n"
        "\n"
        "With ppo.entropy_target, c moves during training and this is only its start."
    ),
    "ppo.entropy_normalization": (
        'The entropy -sum of p_i ln p_i is divided by ln(n). "allowed": n is the number of actions allowed at that moment, so a car with few choices can still reach 1. "all": n = 90 always, so the same bonus is weaker whenever actions are masked.\n'
        "\n"
        "```toml\n"
        "[ppo]\n"
        'entropy_normalization = "all"\n'
        "```"
    ),
    "ppo.optimizer": (
        "Muon, for a hidden layer's weight matrix W with gradient G: momentum M <- 0.95 M + G, then the Nesterov step N = G + 0.95 M, then 5 Newton-Schulz steps turn N into an approximately orthogonal matrix O with the same directions but every size evened out (all singular values near 1). The step is W <- W - lr x sqrt(max(rows, cols)) x muon_lr_scale x O. Everything Muon does not take (biases, LayerNorms, the first and last layers, attention encoders) stays with Adam.\n"
        "\n"
        "```toml\n"
        "[ppo]\n"
        'optimizer = "muon"\n'
        "muon_batched = true\n"
        "weight_decay = 0.1\n"
        "```\n"
        "\n"
        "The Newton-Schulz steps cost about 15 matrix products per layer a step, run in BF16 where the card supports it."
    ),
    "ppo.muon_batched": (
        "Muon's step per matrix is many small jobs. Matrices of the same shape are stacked into one tensor of shape (count, rows, cols) and the momentum and Newton-Schulz steps run as batched products (torch.bmm) over all of them at once.\n"
        "\n"
        "```toml\n"
        "[ppo]\n"
        'optimizer = "muon"\n'
        "muon_batched = true\n"
        "```\n"
        "\n"
        "The maths is the same; a batched product may add in another order, so results agree to rounding."
    ),
    "ppo.muon_lr_scale": (
        "Multiplies the Muon matrices' step: W <- W - lr x sqrt(max(rows, cols)) x scale x O. Adam's parameters keep lr. Because every Muon matrix feeds a LayerNorm, the layer's output does not depend on W's overall size, only its direction; a step of size s on a matrix of size |W| turns it by about s / |W|. A scale of 1/r therefore makes a fresh matrix turn as one r times bigger would.\n"
        "\n"
        "```toml\n"
        "[ppo]\n"
        'optimizer = "muon"\n'
        "muon_lr_scale = 0.2\n"
        "```"
    ),
    "ppo.weight_decay": (
        "Decoupled weight decay: after each learning step that moves a layer, W <- W x (1 - lr x weight_decay). LayerNorm undoes the shrink, so the network computes the same, but a smaller |W| makes the next step turn it further. It balances when growth from the steps equals the shrink, which gives about\n"
        "\n"
        "weight_decay = 30 x t^2 / (2 x lr)\n"
        "\n"
        "for a turn t per step (30 measured).\n"
        "\n"
        "```toml\n"
        "[ppo]\n"
        'optimizer = "muon"\n'
        "weight_decay = 0.1     # at lr 1.5e-4 holds about 1e-3 turn a step\n"
        "```"
    ),
    "ppo.entropy_target": (
        "After each update the entropy bonus c is moved toward holding the measured entropy H at the target T:\n"
        "\n"
        "c <- c x exp(rate x (T - H)), then kept between entropy_target_min x c0 and 10 x c0\n"
        "\n"
        "where c0 is ppo.entropy. Below the target the bonus grows, above it shrinks, by a factor that depends on how far off it is.\n"
        "\n"
        "```toml\n"
        "[ppo]\n"
        "entropy = 0.035\n"
        "entropy_target = 0.5\n"
        "entropy_target_rate = 0.05\n"
        "entropy_target_min = 1.0\n"
        "```\n"
        "\n"
        "The entropy_coef column in metrics.csv records c each update."
    ),
    "ppo.entropy_target_rate": (
        "The rate in c <- c x exp(rate x (T - H)). At rate 1 and H 0.05 below the target, c grows by exp(0.05), about 5% an update; at rate 0.05 by about 0.25%.\n"
        "\n"
        "```toml\n"
        "[ppo]\n"
        "entropy_target = 0.5\n"
        "entropy_target_rate = 0.2\n"
        "```\n"
        "\n"
        "To change c by a factor F at a steady gap g takes ln(F) / (rate x g) updates."
    ),
    "ppo.entropy_target_min": (
        "The floor of the bonus: c >= entropy_target_min x ppo.entropy. The ceiling is always 10 x ppo.entropy.\n"
        "\n"
        "```toml\n"
        "[ppo]\n"
        "entropy = 0.035\n"
        "entropy_target = 0.5\n"
        "entropy_target_min = 1.0    # the bonus can only rise above 0.035\n"
        "```"
    ),
    "ppo.advantage_keep": (
        "In each learning step of n samples, the policy's loss uses only the k = round(keep x n) samples with the largest |A| (the others' policy loss is 0); the value heads and the entropy still use all n. The kept samples carry most of the signal: with goals as the only reward, most advantages are near 0.\n"
        "\n"
        "```toml\n"
        "[ppo]\n"
        "advantage_keep = 0.5\n"
        "```\n"
        "\n"
        "The column advantage_kept_size is (sum of |A| over the kept) / (sum of |A| over all)."
    ),
    "ppo.advantage_keep_rest": (
        "With keep k and rest s, each learning step runs on the kept samples plus a random share s of the others. The others' policy loss is 0 anyway; their value and entropy terms are counted with weight 1/s, so the expected sums equal those over every sample (an unbiased estimate, noisier for small s).\n"
        "\n"
        "```toml\n"
        "[ppo]\n"
        "graphed_steps = false       # needed below 1, for now\n"
        "advantage_keep = 0.5\n"
        "advantage_keep_rest = 0.2   # about 0.5 + 0.5 x 0.2 = 60% of the samples run\n"
        "```"
    ),
    "ppo.value_share": (
        "The value loss of each learning step is averaged over only the first share x n samples of the step (whose order is random), while the policy uses all n. The value side's work per step falls by about that share.\n"
        "\n"
        "```toml\n"
        "[ppo]\n"
        "value_share = 0.5\n"
        "```"
    ),
    "ppo.goal_critic_weight": (
        "The goal head learns V_goal with its own discount gamma_g = 0.5^(tick_skip / (120 x goal_critic_half_life)) and its own GAE advantages A_g. These are centred and scaled, then added:\n"
        "\n"
        "A <- A + w x std(A) x (A_g - mean(A_g)) / std(A_g)\n"
        "\n"
        "so with w = 0.25 the goal critic's opinion is a quarter of the size of the usual advantages.\n"
        "\n"
        "```toml\n"
        "[model]\n"
        'value_heads = ["main", "goal"]\n'
        "[ppo]\n"
        "goal_critic_weight = 0.25\n"
        "goal_critic_half_life = 77.0\n"
        "```"
    ),
    "ppo.goal_critic_half_life": (
        "The goal head's discount: gamma_g = 0.5^(tick_skip / (120 x half_life)). 77 s gives 0.99970 at tick skip 4; its reach 1 / (1 - gamma_g) is about 3,300 decisions (110 s).\n"
        "\n"
        "```toml\n"
        "[model]\n"
        'value_heads = ["main", "goal"]\n'
        "[ppo]\n"
        "goal_critic_weight = 0.25\n"
        "goal_critic_half_life = 60.0\n"
        "```"
    ),
    "ppo.vdag_beta": (
        "The twin critics V1, V2 learn the target y = r + gamma x min(V1(s'), V2(s')) with an expectile loss (ppo.vdag_expectile). Their value Phi(s) = min(V1(s), V2(s)) shapes the advantages: F_t = gamma x Phi(s_(t+1)) - Phi(s_t), the change of that value over the step. F is centred, scaled to beta x std(A), cut at 3 x beta x std(A), and added to A. Potential-based shaping like this cannot change which play is best in the end; it only points the bot sooner toward states with a high Phi.\n"
        "\n"
        "```toml\n"
        "[model]\n"
        'value_heads = ["main", "vdag1", "vdag2"]\n'
        "[ppo]\n"
        "vdag_beta = 0.3\n"
        "vdag_reinit_steps = 500_000_000\n"
        "```"
    ),
    "ppo.vdag_expectile": (
        "The expectile loss with level tau for an error u = target - prediction: |tau - 1(u < 0)| x u^2. Errors where the prediction is too low weigh tau, too high 1 - tau. tau = 0.5 learns the mean; tau = 0.75 learns a value above the mean, an optimistic one.\n"
        "\n"
        "```toml\n"
        "[model]\n"
        'value_heads = ["main", "vdag1", "vdag2"]\n'
        "[ppo]\n"
        "vdag_beta = 0.3\n"
        "vdag_expectile = 0.8\n"
        "```"
    ),
    "ppo.vdag_reinit_steps": (
        "Every this many agent steps, both twin heads get fresh random weights and their optimizer state is cleared. The critic trunk they share with the main value is kept.\n"
        "\n"
        "```toml\n"
        "[model]\n"
        'value_heads = ["main", "vdag1", "vdag2"]\n'
        "[ppo]\n"
        "vdag_beta = 0.3\n"
        "vdag_reinit_steps = 250_000_000\n"
        "```\n"
        "\n"
        "A save records the step of the last restart, so a resume keeps the same rhythm."
    ),
    "ppo.head_loss_scale": (
        "The extra heads' targets are in scaled-reward units, while the main value's loss is in PopArt's normalised units (divided by its variance sigma^2). \"raw\" adds the extra heads' losses as they are, so for the same relative error their gradients in the shared layers are about sigma^2 times the main value's. \"popart\" divides each extra head's loss by PopArt's current sigma^2, putting them in the main value's units.\n"
        "\n"
        "```toml\n"
        "[model]\n"
        'value_heads = ["main", "goal"]\n'
        "[ppo]\n"
        "goal_critic_weight = 0.25\n"
        "popart = true\n"
        'head_loss_scale = "popart"\n'
        "```"
    ),
    "ppo.lean_heads_pass": (
        'Before each update the learner computes the extra heads\' values for every collected state. In self-play, a share of the cars are past versions whose samples the update leaves out; this pass then skips them, and with run.fp16_sums "forward" or "all" it uses FP16 sums as learning\'s forward passes do.\n'
        "\n"
        "```toml\n"
        "[model]\n"
        'value_heads = ["main", "goal"]\n'
        "[ppo]\n"
        "goal_critic_weight = 0.25\n"
        "lean_heads_pass = true\n"
        "```"
    ),
    "ppo.adam_eps": (
        "Adam's step is m / (sqrt(v) + eps). eps keeps the step finite where the gradients have been near 0; a larger eps damps the steps of rarely-moving weights. 1e-5 is the common choice for PPO.\n"
        "\n"
        "```toml\n"
        "[ppo]\n"
        "adam_eps = 1e-8\n"
        "```"
    ),
    "ppo.max_grad_norm": (
        "Before each step, each network's gradients are viewed as one long vector g. If its length |g| = sqrt(sum of g_i^2) is above the limit, every gradient is multiplied by limit / |g|. The direction is kept; only too-large steps are shortened.\n"
        "\n"
        "```toml\n"
        "[ppo]\n"
        "max_grad_norm = 1.0\n"
        "```"
    ),
    "ppo.reward_clip": (
        "Rewards are first scaled: divided by a running estimate of the spread of the discounted returns, so the learner sees numbers of a steady size whatever the weights. The scaled reward is then cut to -clip..+clip, so a rare huge reward (a goal worth 150 next to small shaping rewards) cannot swamp a learning step.\n"
        "\n"
        "```toml\n"
        "[ppo]\n"
        "reward_clip = 0      # no clipping\n"
        "```"
    ),
    "ppo.normalize_advantages": (
        "In each learning step the advantages are rescaled to mean 0 and spread 1: A <- (A - mean(A)) / (std(A) + 1e-8). The policy's step size then does not depend on how big the rewards are.\n"
        "\n"
        "```toml\n"
        "[ppo]\n"
        "normalize_advantages = false\n"
        "```"
    ),
    "ppo.popart": (
        "PopArt keeps running estimates of the value targets' mean mu and spread sigma. The value network learns the normalised target (y - mu) / sigma; its output is turned back as V = sigma x out + mu. When mu and sigma change, the output layer's weights and bias are adjusted (w <- w x sigma_old / sigma_new, b <- (sigma_old x b + mu_old - mu_new) / sigma_new), so V itself stays the same.\n"
        "\n"
        "```toml\n"
        "[ppo]\n"
        "popart = false\n"
        "```"
    ),
    "selfplay.enabled": (
        "In a share of the arenas (selfplay.past_share) the Orange team is driven by a saved past version instead of the current bot. Only the current bot's cars become training samples. Every version, the bot included, has a rating (a Bradley-Terry model, like Elo, with an uncertainty) updated from the goals of their games, and the matchmaking picks opponents by those ratings.\n"
        "\n"
        "```toml\n"
        "[selfplay]\n"
        "enabled = true\n"
        "past_share = 0.2\n"
        "active_opponents = 4\n"
        "```\n"
        "\n"
        "The bot's own rating is shown in the report; a rating that stops rising for selfplay.flat_rating_steps gives a warning."
    ),
    "selfplay.past_share": (
        "Arenas against past versions = round(num_arenas x past_share); the rest are the current bot on both sides. Only the current bot's cars learn, so with past_share p in 1v1 the samples per iteration drop by p / 2 (one learning car in those arenas instead of two).\n"
        "\n"
        "```toml\n"
        "[selfplay]\n"
        "past_share = 0.3\n"
        "```"
    ),
    "selfplay.active_opponents": (
        "The arenas against past versions are dealt evenly among this many versions, drawn by the matchmaking at every save. Each version needs its own network pass per decision, so more versions cost more graphics-card time.\n"
        "\n"
        "```toml\n"
        "[selfplay]\n"
        "active_opponents = 2\n"
        "```\n"
        "\n"
        "The settings check warns when there are fewer past-version arenas than active versions."
    ),
    "selfplay.snapshot_steps": (
        "Every this many agent steps a copy of the policy (only the policy: past versions never learn) is written into the pool folder and given a starting rating.\n"
        "\n"
        "```toml\n"
        "[selfplay]\n"
        "snapshot_steps = 50_000_000\n"
        "```\n"
        "\n"
        "Agent steps count every learning car's decisions: with C learning cars playing at once, a copy comes every snapshot_steps / (C x 120 / tick_skip) seconds of game time."
    ),
    "selfplay.pool_size": (
        "When the pool holds more versions than this, the version whose rating is closest to its neighbours' goes first, so the pool keeps a spread of strengths rather than only the newest.\n"
        "\n"
        "```toml\n"
        "[selfplay]\n"
        "pool_size = 32\n"
        "```"
    ),
    "selfplay.pool_disk_mb": (
        "The pool's files may use at most this many megabytes; beyond it versions are dropped as with pool_size. A version's file is about 4 bytes per policy parameter.\n"
        "\n"
        "```toml\n"
        "[selfplay]\n"
        "pool_disk_mb = 4096\n"
        "```"
    ),
    "selfplay.matchmaking": (
        "With p the chance the ratings give the bot to beat a version, each version is drawn with weight:\n"
        '- "variance": p x (1 - p), largest for even matches\n'
        '- "uniform": 1\n'
        '- "hardest": 1 - p\n'
        '- "squared": (1 - p)^2\n'
        '- "linear_capped": min(0.5, 1 - p), the same for every version the bot beats at most half the time\n'
        "\n"
        "```toml\n"
        "[selfplay]\n"
        'matchmaking = "variance"\n'
        "```"
    ),
    "selfplay.flat_rating_steps": (
        "The trainer remembers the bot's best rating and the step it was reached. If no new best comes within this many agent steps, it warns (training goes on).\n"
        "\n"
        "```toml\n"
        "[selfplay]\n"
        "flat_rating_steps = 1_000_000_000\n"
        "```"
    ),
    "pretrain.enabled": (
        "Behaviour cloning: each replay is turned into the observations the bot would have seen and the action (of the 90) closest to what the player pressed. The policy then learns to give those actions high probability by lowering the cross-entropy loss -ln p(a_player | s). The result is saved as the bot's first save; the next start trains from it with a fresh value network.\n"
        "\n"
        "```toml\n"
        "[pretrain]\n"
        "enabled = true\n"
        "limit = 50\n"
        "passes = 10\n"
        "```"
    ),
    "pretrain.replays": (
        "Folders are searched for .replay files (subfolders included), newest first. With several folders and batch_replays, each batch takes replays from each folder in turn.\n"
        "\n"
        "```toml\n"
        "[pretrain]\n"
        "enabled = true\n"
        'replays = ["D:/replays/1v1", "D:/replays/ranked"]\n'
        "```"
    ),
    "pretrain.team_sizes": (
        "Each listed size's replays are converted into observations with room for env.obs_team_size players a side (empty places marked), so one network learns from all sizes at once.\n"
        "\n"
        "```toml\n"
        "[env]\n"
        "obs_team_size = 3\n"
        "[pretrain]\n"
        "enabled = true\n"
        "batch_replays = 200\n"
        "team_sizes = [1, 2, 3]\n"
        "```"
    ),
    "pretrain.limit": (
        "The newest replays of the right size are taken until this many are found. Without batch_replays every converted replay is held in memory: a 5-minute 1v1 replay is about 2 MB of observations.\n"
        "\n"
        "```toml\n"
        "[pretrain]\n"
        "limit = 10_000\n"
        "batch_replays = 200\n"
        "passes = 2\n"
        "```"
    ),
    "pretrain.passes": (
        "Epochs of behaviour cloning: each pass goes over every decision once, in a random order, in steps of pretrain.minibatch decisions. Learning steps = passes x decisions / minibatch.\n"
        "\n"
        "```toml\n"
        "[pretrain]\n"
        "passes = 5\n"
        "```"
    ),
    "pretrain.batch_replays": (
        "The large replay mode: replays are converted in batches of this size by several workers; each batch is learned from together with the batches kept with it (keep_batches), then dropped. Memory is bounded by about keep_batches + 2 batches whatever the limit, and progress is saved after each batch.\n"
        "\n"
        "```toml\n"
        "[pretrain]\n"
        "enabled = true\n"
        "batch_replays = 200\n"
        "limit = 5_000\n"
        "passes = 2\n"
        "```"
    ),
    "pretrain.keep_batches": (
        "Each round learns from the newest batch and the keep_batches - 1 before it, so each replay is seen in keep_batches rounds (times passes per round).\n"
        "\n"
        "```toml\n"
        "[pretrain]\n"
        "batch_replays = 200\n"
        "keep_batches = 3\n"
        "```"
    ),
    "pretrain.workers": (
        "Replays are converted in parallel processes. 0 = half the logical processors, at most 16.\n"
        "\n"
        "```toml\n"
        "[pretrain]\n"
        "batch_replays = 200\n"
        "workers = 8\n"
        "```"
    ),
    "pretrain.converter": (
        '"nitro" is a build of the replay converter (based on the open-source replicar) that writes only the columns the copy reads; "replicar" runs replicar\'s own tool and reads its output. Both produce the same states and controls.\n'
        "\n"
        "```toml\n"
        "[pretrain]\n"
        "batch_replays = 200\n"
        'converter = "replicar"\n'
        "```"
    ),
    "pretrain.continue_from": (
        "The copy starts from the named run's pretrained networks instead of fresh ones and learns this run's replays on top.\n"
        "\n"
        "```toml\n"
        "[pretrain]\n"
        "batch_replays = 200\n"
        'continue_from = "my-copy-bot"\n'
        "new_only = true\n"
        "```"
    ),
    "pretrain.new_only": (
        "The project's replays\\sizes.txt lists every replay converted here (by its file's fingerprint). With this on, those are skipped.\n"
        "\n"
        "```toml\n"
        "[pretrain]\n"
        "batch_replays = 200\n"
        "new_only = true\n"
        "```"
    ),
    "pretrain.value_warmup_steps": (
        "For the first N agent steps of training, the policy's loss is switched off and only the value network learns (with a shared trunk, the trunk is held too). PPO's advantages are only as good as the value network's predictions; a fresh value network gives noisy advantages that would pull a copied policy away from what it copied.\n"
        "\n"
        "```toml\n"
        "[pretrain]\n"
        "value_warmup_steps = 10_000_000\n"
        "```\n"
        "\n"
        "Updates spent warming up = N / steps_per_iteration (200 at 10M and 50,000)."
    ),
    "pretrain.keep_converted": (
        "Each converted replay is stored as a compressed file (.nrpl.zst) in the project's replays\\store, under its game size. A later copy, convert-replays or replay_mix reads these instead of converting again.\n"
        "\n"
        "```toml\n"
        "[pretrain]\n"
        "batch_replays = 200\n"
        "keep_converted = false\n"
        "```"
    ),
    "pretrain.skill_every": (
        "Every this many rounds the copy plays short games from kickoffs against earlier versions of itself (the copy at its start and at the last check), in the simulator. Goals give points; the points give each version an Elo-like rating, which shows whether the copy is getting better at the game, not only closer to the players.\n"
        "\n"
        "```toml\n"
        "[pretrain]\n"
        "batch_replays = 100\n"
        "skill_every = 10\n"
        "skill_points = 200\n"
        "```"
    ),
    "pretrain.skill_points": (
        "Points (each from a kickoff to a goal, or a draw) played against each earlier version per check. The rating's uncertainty shrinks like 1 / sqrt(points): 4 times the points halves it.\n"
        "\n"
        "```toml\n"
        "[pretrain]\n"
        "batch_replays = 100\n"
        "skill_every = 10\n"
        "skill_points = 400\n"
        "```"
    ),
    "pretrain.save_every": (
        "Every this many replays learned, the copy is written as the bot's save at 0 agent steps, so it can be watched or exported while the copy goes on.\n"
        "\n"
        "```toml\n"
        "[pretrain]\n"
        "batch_replays = 200\n"
        "save_every = 1000\n"
        "```"
    ),
    "pretrain.lr": (
        "The copy's Adam learning rate (see ppo.policy_lr for the update rule). Bigger networks usually take smaller rates.\n"
        "\n"
        "```toml\n"
        "[pretrain]\n"
        "lr = 1.5e-4\n"
        "```"
    ),
    "pretrain.lr_end_share": (
        "The rate follows a cosine curve from lr down to lr x end_share over the replays learned: at a fraction f of the way, rate = lr x (s + (1 - s) x (1 + cos(pi x f)) / 2), with s = end_share.\n"
        "\n"
        "```toml\n"
        "[pretrain]\n"
        "lr = 3e-4\n"
        "lr_end_share = 0.1\n"
        "```"
    ),
    "pretrain.prev_action_dropout": (
        "The observation includes the previous action (8 numbers). With dropout d, each decision's previous action is replaced by zeros with chance d while copying, so the copy must also learn from the rest of the observation.\n"
        "\n"
        "```toml\n"
        "[pretrain]\n"
        "prev_action_dropout = 0.3\n"
        "```"
    ),
    "pretrain.minibatch": (
        "Decisions per learning step of the copy. Steps per pass = decisions / minibatch.\n"
        "\n"
        "```toml\n"
        "[pretrain]\n"
        "minibatch = 8192\n"
        "lr = 4e-4\n"
        "```"
    ),
    "pretrain.measure_every": (
        "Some replays are held back and never learned from. Every this many rounds the copy is measured on them: how often its likeliest action is the player's (agreement) and the cross-entropy. Measuring takes time, so measuring less often speeds up the copy.\n"
        "\n"
        "```toml\n"
        "[pretrain]\n"
        "batch_replays = 100\n"
        "measure_every = 10\n"
        "```"
    ),
    "replay_mix.enabled": (
        "After each PPO update, extra learning steps copy the players in converted replays, with the same behaviour-cloning loss as [pretrain]: -ln p(a_player | s). They move only the policy (and a shared trunk), with their own Adam optimizer and rate, so the PPO optimizer's statistics are not disturbed.\n"
        "\n"
        "```toml\n"
        "[replay_mix]\n"
        "enabled = true\n"
        "share = 0.2\n"
        "lr = 3e-5\n"
        "```"
    ),
    "replay_mix.share": (
        "Of all steps that move the policy, this share are copy steps: with k PPO policy steps per update (epochs x minibatches), each update owes k x share / (1 - share) copy steps, the fraction carried to the next update. With 8 PPO steps and 0.2: 2 copy steps.\n"
        "\n"
        "```toml\n"
        "[replay_mix]\n"
        "enabled = true\n"
        "share = 0.1\n"
        "```"
    ),
    "replay_mix.lr": (
        "The copy steps' own Adam rate. Kept well below ppo.policy_lr so the copy nudges the trained policy rather than pulling it back to the players' play.\n"
        "\n"
        "```toml\n"
        "[replay_mix]\n"
        "enabled = true\n"
        "lr = 2e-5\n"
        "```"
    ),
    "replay_mix.minibatch": (
        "Each copy step learns from this many decisions drawn at random from 16 replays held in memory at a time.\n"
        "\n"
        "```toml\n"
        "[replay_mix]\n"
        "enabled = true\n"
        "minibatch = 8192\n"
        "```"
    ),
    "replay_mix.kickoff_only_seconds": (
        "Above 0, only decisions from a kickoff until this many seconds after the ball's first touch are kept for copying; the rest of each replay is left out.\n"
        "\n"
        "```toml\n"
        "[replay_mix]\n"
        "enabled = true\n"
        "kickoff_only_seconds = 3\n"
        "```\n"
        "\n"
        "metrics.csv's replay_mix_rows_kept and replay_mix_rows_left_out count what was kept."
    ),
    "replay_mix.replays": (
        "Folders of converted replays (.nrpl.zst). The files are shuffled once with run.seed and taken in that order, each once a cycle.\n"
        "\n"
        "```toml\n"
        "[replay_mix]\n"
        "enabled = true\n"
        'replays = ["D:/converted/1v1"]\n'
        "```"
    ),
    "wandb.enabled": (
        "Each iteration's metrics.csv row is also sent with wandb.log, and the settings with wandb.config. The run keeps training if the internet drops; `nitro wandb <bot>` uploads what is missing later.\n"
        "\n"
        "```toml\n"
        "[wandb]\n"
        "enabled = true\n"
        'project = "my-bots"\n'
        "```"
    ),
    "wandb.project": (
        "Runs in one project can be shown side by side on W&B.\n"
        "\n"
        "```toml\n"
        "[wandb]\n"
        "enabled = true\n"
        'project = "1v1-experiments"\n'
        "```"
    ),
    "wandb.entity": (
        "Your W&B user name or a team you belong to; empty uses the default of the logged-in account.\n"
        "\n"
        "```toml\n"
        "[wandb]\n"
        "enabled = true\n"
        'entity = "my-team"\n'
        "```"
    ),
    "wandb.name": (
        "The run's display name on W&B; empty uses the bot's folder name.\n"
        "\n"
        "```toml\n"
        "[wandb]\n"
        "enabled = true\n"
        'name = "1v1 leaky relu"\n'
        "```"
    ),
    "wandb.group": (
        "Runs with the same group are shown together and can be averaged, for example several seeds of one setting.\n"
        "\n"
        "```toml\n"
        "[wandb]\n"
        "enabled = true\n"
        'group = "seed-test"\n'
        "```"
    ),
    "wandb.tags": (
        "Labels to filter runs by on W&B.\n"
        "\n"
        "```toml\n"
        "[wandb]\n"
        "enabled = true\n"
        'tags = ["1v1", "nitro_v2"]\n'
        "```"
    ),
    "wandb.mode": (
        '"online" sends while training. "offline" writes W&B\'s files into the bot\'s folder only; upload them later with `wandb sync <folder>`.\n'
        "\n"
        "```toml\n"
        "[wandb]\n"
        "enabled = true\n"
        'mode = "offline"\n'
        "```"
    ),
    "reward.weight": (
        "Each step every car's reward is the weighted sum of its rewards:\n"
        "\n"
        "r = sum over rewards of weight_i x value_i (x tick_skip / 8 for rewards paid every step)\n"
        "\n"
        "The learner then divides r by a running estimate of the spread of the discounted returns, so only the weights' ratios matter, not their scale.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "touch_ball"\n'
        "weight = 10\n"
        "\n"
        "[[rewards]]\n"
        'name = "goal"\n'
        "weight = 150\n"
        "```"
    ),
    "reward.zero_sum": (
        "For car i on team T with opponents O, before weighting:\n"
        "\n"
        "r_i' = (1 - s) x r_i + s x mean(r over T) - o x mean(r over O)\n"
        "\n"
        "with s = team_spirit and o = opponent_scale. When both teams earn the same, the terms cancel.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "velocity_ball_to_goal"\n'
        "weight = 2\n"
        "zero_sum = { team_spirit = 1, opponent_scale = 1 }\n"
        "```"
    ),
    "reward.zero_sum.team_spirit": (
        "s in r_i' = (1 - s) x r_i + s x mean(r over the team) - o x mean(r over the opponents). At s = 1 every teammate gets the team's average; at 0 each keeps its own.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "bump"\n'
        "weight = 20\n"
        "zero_sum = { team_spirit = 0.5 }\n"
        "```"
    ),
    "reward.zero_sum.opponent_scale": (
        "o in r_i' = (1 - s) x r_i + s x mean(r over the team) - o x mean(r over the opponents). At o = 1 the sum over both teams is 0 when the teams are the same size.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "demo"\n'
        "weight = 80\n"
        "zero_sum = { team_spirit = 0.5, opponent_scale = 0.5 }\n"
        "```"
    ),
    "reward.shares": (
        "Each reward's share is sum over steps of |weight x value| for that reward divided by the same sum over all rewards, counted over the newest iterations. Trigger rate = steps where its value is not 0, per minute of one car's play.\n"
        "\n"
        "```text\n"
        "share_i = sum |w_i x v_i| / sum over j of sum |w_j x v_j|\n"
        "```\n"
        "\n"
        "Hiding a reward only removes it from the sums shown; training is unchanged."
    ),
    "start.weight": (
        "At each game start one start is drawn with chance weight_i / sum of the weights.\n"
        "\n"
        "```toml\n"
        "[[state_setters]]\n"
        'name = "kickoff"\n'
        "weight = 7\n"
        "\n"
        "[[state_setters]]\n"
        'name = "random"\n'
        "weight = 3\n"
        "```"
    ),
    "start.kickoff": (
        "The cars are placed on the real kickoff spots (two diagonal, two off-centre, one straight back per team), at rest, the ball still at the centre. fuzz_uu moves each car by up to that many units in a random direction.\n"
        "\n"
        "```toml\n"
        "[[state_setters]]\n"
        'name = "kickoff"\n'
        "fuzz_uu = 50\n"
        "```"
    ),
    "start.random": (
        "The ball and every car get a random position on the field, a random velocity, rotation and boost; ground_chance is each car's chance to start on its wheels.\n"
        "\n"
        "```toml\n"
        "[[state_setters]]\n"
        'name = "random"\n'
        "weight = 1\n"
        "max_ball_speed = 1500\n"
        "ground_chance = 0.7\n"
        "```"
    ),
    "start.replay": (
        "A start is a recorded frame: every car's and the ball's position, velocity and rotation at one moment of a real match. Half the moments are turned around so both ends are used. air_bias b draws a moment with weight 1 + b x height / ceiling height (height of the highest thing in it), so high moments come up more often.\n"
        "\n"
        "```toml\n"
        "[[state_setters]]\n"
        'name = "replay"\n'
        "weight = 2\n"
        "bundled = true\n"
        'files = ["recordings/*.rlpr"]\n'
        "air_bias = 1.0\n"
        "```"
    ),
}


def more(path: str) -> str:
    """The long explanation of `path`, or "" when there is none."""
    return MORE.get(path, "")


def under(prefix: str) -> dict[str, str]:
    """Every explanation whose key starts with `prefix` (the Rewards tab's "reward.", the Starts tab's)."""
    return {key: text for key, text in MORE.items() if key.startswith(prefix)}
