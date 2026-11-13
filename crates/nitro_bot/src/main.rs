//! The bot program RLBot starts:
//! `nitro_bot <policy.nwt> [--sample] [--seed N] [--latency T] [--trained-size-only] [--threads N]
//! [--no-decider]`.
//!
//! RLBot sets the server address and the agent id in the environment (`rlbot::util::AgentEnvironment`).
//! One process drives every car RLBot hands it (`hivemind = true` in bot.toml), each on its own thread.
//! The cars share one copy of the policy and one decider thread: each packet is answered at once,
//! the network runs while the game plays the action delay, and cars deciding at the same tick share one
//! pass. `--no-decider` decides on each car's packet thread instead, answering after the decision.

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::{Arc, OnceLock};

use nitro_bot::decider::Decider;
use nitro_bot::{Choice, NitroBot, controller_state};
use nitro_core::policy::Policy;
use rlbot::RLBotConnection;
use rlbot::agents::{BotAgent, run_bot_agents};
use rlbot::flat::{ControllableInfo, FieldInfo, GamePacket, MatchConfiguration, PlayerInput};
use rlbot::util::{AgentEnvironment, PacketQueue};

struct Settings {
    policy: PathBuf,
    choice: Choice,
    seed: u64,
    latency: u32,
    /// Coast in matches bigger than the policy trained for, even with an attention network.
    trained_size_only: bool,
    /// Threads each decision pass is split across.
    threads: usize,
    /// Decide on each car's packet thread instead of the shared decider.
    no_decider: bool,
}

// `BotAgent::new` takes no arguments of ours, so the command line is kept here for it.
static SETTINGS: OnceLock<Settings> = OnceLock::new();
/// The policy, loaded once for every car.
static POLICY: OnceLock<Arc<Policy>> = OnceLock::new();
/// The cars' decider (absent with `--no-decider`).
static DECIDER: OnceLock<Arc<Decider>> = OnceLock::new();

struct Agent {
    index: u32,
    bot: Option<NitroBot>,
}

impl BotAgent for Agent {
    fn new(
        _team: u32,
        controllable: ControllableInfo,
        _match_configuration: Arc<MatchConfiguration>,
        field_info: Arc<FieldInfo>,
        _queue: &mut PacketQueue,
    ) -> Self {
        let settings = SETTINGS
            .get()
            .expect("settings are set before the agents start");
        let policy = POLICY
            .get()
            .expect("the policy is loaded before the agents start");
        let bot = NitroBot::new(
            policy.clone(),
            &field_info,
            controllable.index as usize,
            settings.choice,
            settings.seed ^ u64::from(controllable.index),
            settings.latency,
        )
        .map(|mut bot| {
            if settings.trained_size_only {
                bot.keep_to_trained_size();
            }
            bot.set_threads(settings.threads);
            if let Some(decider) = DECIDER.get() {
                bot.use_decider(decider.seat());
            }
            bot
        });
        match &bot {
            Ok(_) => println!("nitro_bot: car {} ready.", controllable.index),
            Err(problem) => println!(
                "nitro_bot: car {} cannot play: {problem}",
                controllable.index
            ),
        }
        Self {
            index: controllable.index,
            bot: bot.ok(),
        }
    }

    fn tick(&mut self, packet: &GamePacket, queue: &mut PacketQueue) {
        let Some(bot) = self.bot.as_mut() else {
            return;
        };
        let controls = bot.controls(packet);
        queue.push(PlayerInput {
            player_index: self.index,
            controller_state: controller_state(controls),
        });
    }
}

impl Drop for Agent {
    fn drop(&mut self) {
        if let Some(bot) = &self.bot {
            let stats = bot.rhythm_stats();
            println!(
                "nitro_bot: car {}: {} decisions, {} late (dropped packets), {} missed.",
                self.index, stats.decisions, stats.late, stats.missed
            );
            let times = bot.decision_times();
            if times.decisions > 0 {
                println!(
                    "nitro_bot: car {}: deciding took {:.1} ms on average, {:.1} ms at most; {} of {} slower \
                     than the {:.1} ms the action delay leaves (those acted late); {} packets waited for a \
                     decision; {} decisions shared a pass with another car.",
                    self.index,
                    times.total_ms / times.decisions as f64,
                    times.slowest_ms,
                    times.over_budget,
                    times.decisions,
                    times.budget_ms,
                    times.waited,
                    times.batched
                );
            }
        }
    }
}

fn parse_args() -> Result<Settings, String> {
    let mut policy = None;
    let mut choice = Choice::MostLikely;
    let mut seed = 0;
    // Ticks the game takes to apply a sent control beyond the training rhythm (`--latency`).
    let mut latency = 0;
    let mut trained_size_only = false;
    // 2 halves a large network's decision time with the same decisions (a big 2v2 transformer:
    // 11.4 ms -> 4.9 ms a car on 4 cores); more threads than cores free makes it slower. With the
    // decider the two cars of a 2v2 share one pass on these threads.
    let mut threads = 2;
    let mut no_decider = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--sample" => choice = Choice::Sample,
            "--seed" => {
                seed = args
                    .next()
                    .and_then(|s| s.parse().ok())
                    .ok_or("--seed needs a whole number")?;
            }
            "--latency" => {
                latency = args
                    .next()
                    .and_then(|s| s.parse().ok())
                    .ok_or("--latency needs a whole number of ticks")?;
            }
            "--trained-size-only" => trained_size_only = true,
            "--threads" => {
                threads = args
                    .next()
                    .and_then(|s| s.parse().ok())
                    .filter(|&n| n >= 1)
                    .ok_or("--threads needs a whole number, 1 or more")?;
            }
            "--no-decider" => no_decider = true,
            "--help" | "-h" => return Err(String::new()),
            other if policy.is_none() && !other.starts_with('-') => {
                policy = Some(PathBuf::from(other))
            }
            other => return Err(format!("unknown argument '{other}'")),
        }
    }
    Ok(Settings {
        policy: policy.ok_or("the policy file is missing")?,
        choice,
        seed,
        latency,
        trained_size_only,
        threads,
        no_decider,
    })
}

fn main() -> ExitCode {
    let settings = match parse_args() {
        Ok(settings) => settings,
        Err(problem) => {
            if !problem.is_empty() {
                eprintln!("nitro_bot: {problem}");
            }
            eprintln!(
                "usage: nitro_bot <policy.nwt> [--sample] [--seed N] [--latency T] [--trained-size-only] \
                 [--threads N] [--no-decider]"
            );
            return ExitCode::FAILURE;
        }
    };
    // Load once up front so a bad file fails here, with a reason, instead of in every agent thread.
    let policy = match Policy::load(&settings.policy) {
        Ok(policy) => policy,
        Err(err) => {
            eprintln!("nitro_bot: {err}");
            return ExitCode::FAILURE;
        }
    };
    let header = policy.header();
    println!(
        "nitro_bot: policy from run '{}' ({} agent steps, {}v{}, tick skip {}, action delay {}; controls \
         sent {} tick(s) early for the game's input latency, see --latency).",
        header.run,
        header.agent_steps,
        header.game.team_size,
        header.game.team_size,
        header.game.tick_skip,
        header.game.action_delay,
        settings.latency
    );
    if policy.reads_any_cars() {
        println!(
            "nitro_bot: an attention network: it plays any number of cars a side (its skill is measured \
             only at the sizes it trained at)."
        );
    }
    let policy = Arc::new(policy);
    if !settings.no_decider {
        DECIDER
            .set(Decider::start(policy.clone(), settings.threads))
            .ok()
            .expect("set once");
    }
    POLICY.set(policy).expect("set once");
    SETTINGS.set(settings).ok().expect("set once");

    let AgentEnvironment {
        server_addr,
        agent_id,
    } = AgentEnvironment::from_env();
    let agent_id = agent_id.unwrap_or_else(|| "nitrolearn/nitro-bot".into());
    let connection = match RLBotConnection::new(&server_addr) {
        Ok(connection) => connection,
        Err(err) => {
            eprintln!("nitro_bot: cannot connect to RLBot at {server_addr}: {err}");
            return ExitCode::FAILURE;
        }
    };
    println!("nitro_bot: connected to RLBot at {server_addr} as '{agent_id}'.");
    let outcome = run_bot_agents::<Agent>(agent_id, false, false, connection);
    match outcome {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("nitro_bot: stopped with an error: {err}");
            ExitCode::FAILURE
        }
    }
}
