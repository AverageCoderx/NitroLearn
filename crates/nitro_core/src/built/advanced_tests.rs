//! Advanced built rewards: rewards of the library rebuilt from formulas, variables and remembering
//! functions alone, paid as the hand-written ones on recorded games; sequences; caps.

use super::*;
use crate::custom::kit::testing::{game, quiet, touched};
use crate::custom::prelude::{Mat3A, Team};
use crate::env::{RecordedStep, record_episodes};
use crate::rewards::{RewardSet, RewardStats, RewardTerm};
use crate::testing::{meshes_loaded, pool_config};
use serde_json::json;

fn term(kind: Value) -> RewardTerm {
    serde_json::from_value(json!({ "kind": kind, "weight": 1.0, "zero_sum": null })).expect("term")
}

fn built(vars: Value, terms: Value) -> Value {
    json!({ "name": "built", "title": "advanced", "vars": vars, "terms": terms })
}

/// A built reward as `make` reads it (the tag already taken off).
fn own(vars: Value, terms: Value) -> Value {
    json!({ "title": "advanced", "vars": vars, "terms": terms })
}

/// Each car's pay for every recorded step from a set of `terms` at `tick_skip`, a new episode starting
/// where the recording's did.
fn paid(
    terms: Vec<RewardTerm>,
    tick_skip: u32,
    steps: &[RecordedStep],
    starts: &[bool],
) -> Vec<Vec<f32>> {
    let mut set = RewardSet::new(terms.clone(), tick_skip);
    let mut stats = RewardStats::new(terms.len());
    steps
        .iter()
        .zip(starts)
        .map(|((prev, cur, events), &start)| {
            if start {
                set.reset();
            }
            let cars = cur.cars.len();
            let mut out = vec![0.0; cars];
            set.compute(prev, cur, events, &vec![true; cars], &mut out, &mut stats);
            out
        })
        .collect()
}

/// A test-only hand-written reward: on the car's team's goal, 1 when the car touched the ball within the
/// last `seconds` (the touch on the goal's step included).
struct TouchThenGoal {
    seconds: f32,
}

impl Reward for TouchThenGoal {
    fn memory(&self) -> usize {
        2
    }

    fn value_remembering(
        &self,
        _: &GameSnapshot,
        cur: &GameSnapshot,
        events: &StepEvents,
        car: usize,
        m: &mut [f32],
    ) -> f32 {
        if events.cars[car].touched_ball {
            m[0] = 1.0;
            m[1] = 0.0;
        } else if m[0] != 0.0 {
            m[1] += events.step_seconds;
        }
        let scored = cur.goal_scored == Some(cur.cars[car].team);
        f32::from(u8::from(scored && m[0] != 0.0 && m[1] <= self.seconds))
    }
}

/// A test-only hand-written reward: a shot (the car's touch sends the ball heading into the other goal)
/// within `seconds` of the car regaining its flip in the air.
struct ResetThenShot {
    seconds: f32,
}

impl Reward for ResetThenShot {
    fn memory(&self) -> usize {
        2
    }

    fn value_remembering(
        &self,
        prev: &GameSnapshot,
        cur: &GameSnapshot,
        events: &StepEvents,
        car: usize,
        m: &mut [f32],
    ) -> f32 {
        let me = &cur.cars[car];
        if me.can_flip && !prev.cars[car].can_flip && !me.on_ground {
            m[0] = 1.0;
            m[1] = 0.0;
        } else if m[0] != 0.0 {
            m[1] += events.step_seconds;
        }
        let goal_y = crate::custom::kit::attack_sign(me.team) * 5120.0;
        let shot =
            events.cars[car].touched_ball && crate::mechanics::heading_into_goal(&cur.ball, goal_y);
        f32::from(u8::from(shot && m[0] != 0.0 && m[1] < self.seconds))
    }
}

/// The dashboard's templates (one file for the page and these tests): every one is checked here, and the
/// ones that copy a reward are compared with it.
const TEMPLATES: &str = include_str!("../../../../python/nitro/config/built_templates.json");

fn templates() -> Vec<Value> {
    serde_json::from_str::<Vec<Value>>(TEMPLATES).expect("the templates file reads")
}

/// The templates that copy a reward: (title, the hand-written reward, vars, terms).
fn pairs() -> Vec<(String, RewardTerm, Value, Value)> {
    templates()
        .into_iter()
        .filter_map(|t| {
            let copies = t.get("copies")?.clone();
            let hand = match (copies["name"].as_str(), copies["kind"].as_str()) {
                (Some("test"), Some(kind)) => {
                    let mut hand = term(json!({ "name": "touch_ball" }));
                    hand.kind = crate::rewards::RewardKind::Custom(match kind {
                        "touch_then_goal" => {
                            crate::custom::CustomReward::of("t", TouchThenGoal { seconds: 3.0 })
                        }
                        _ => crate::custom::CustomReward::of("r", ResetThenShot { seconds: 2.0 }),
                    });
                    hand
                }
                _ => term(copies),
            };
            let title = t["title"].as_str().expect("a title").to_string();
            Some((
                title,
                hand,
                t.get("vars").cloned().unwrap_or(json!([])),
                t["terms"].clone(),
            ))
        })
        .collect()
}

#[test]
fn every_template_is_a_reward_the_core_takes_and_names_no_other_reward() {
    let list = templates();
    assert!(list.len() >= 10);
    for t in &list {
        let title = t["title"].as_str().unwrap();
        let settings = json!({ "title": title, "vars": t.get("vars").cloned().unwrap_or(json!([])), "terms": t["terms"] });
        make(settings.as_object().unwrap()).unwrap_or_else(|e| panic!("{title}: {e}"));
        assert!(
            !crate::custom::REWARDS.iter().any(|r| r.name == title),
            "{title} is a library reward's name"
        );
        assert!(
            t["about"].as_str().is_some_and(|a| a.len() > 20),
            "{title} says what it pays"
        );
    }
}

/// `config` with starts near the ball, shots on goal and aerials as well, and room for touches.
fn busy(mut config: crate::config::PoolConfig, team_size: usize) -> crate::config::PoolConfig {
    config.team_size = team_size;
    config.terminals.no_touch_seconds = 8.0;
    for kind in [
        json!({"name": "ball_touch", "min_distance_uu": 150.0, "max_distance_uu": 500.0}),
        json!({"name": "save", "min_speed": 1500.0, "max_speed": 2500.0}),
        json!({"name": "shot", "min_distance_uu": 300.0, "max_distance_uu": 900.0, "car_distance_uu": 150.0}),
        json!({"name": "aerial_ball", "min_height_uu": 300.0, "max_height_uu": 900.0, "max_ball_speed": 300.0}),
        json!({"name": "airborne", "min_height_uu": 200.0, "max_height_uu": 1200.0, "min_ball_distance_uu": 100.0, "max_ball_distance_uu": 400.0}),
    ] {
        config
            .state_setters
            .push(serde_json::from_value(json!({ "kind": kind, "weight": 2.0 })).unwrap());
    }
    config
}

#[test]
fn advanced_rewards_built_from_formulas_pay_what_the_hand_written_ones_do_on_recorded_games() {
    if !meshes_loaded() {
        return;
    }
    let mut totals = vec![0usize; pairs().len()];
    for (team_size, tick_skip) in [(1, 8), (2, 8), (1, 4), (2, 4)] {
        let mut config = busy(pool_config(1, 1, 1, 11), team_size);
        config.tick_skip = tick_skip;
        let (steps, starts) = record_episodes(&config, 8000, 5);
        for (n, (what, hand, vars, terms)) in pairs().into_iter().enumerate() {
            let expected = paid(vec![hand], tick_skip, &steps, &starts);
            let got = paid(vec![term(built(vars, terms))], tick_skip, &steps, &starts);
            let mut fired = 0;
            for (step, (a, b)) in expected.iter().zip(&got).enumerate() {
                for (x, y) in a.iter().zip(b) {
                    assert!(
                        (x - y).abs() <= 1e-6 * x.abs().max(1.0),
                        "{what}, {team_size}v{team_size}, tick skip {tick_skip}, step {step}: {x} vs {y}"
                    );
                    fired += usize::from(*x != 0.0);
                }
            }
            eprintln!(
                "{what} ({team_size}v{team_size}, tick skip {tick_skip}): paid on {fired} car-steps"
            );
            totals[n] += fired;
        }
    }
    for ((what, ..), fired) in pairs().iter().zip(&totals) {
        eprintln!("{what}: paid on {fired} car-steps over the four recordings");
    }
    let unpaid = totals.iter().filter(|&&f| f == 0).count();
    assert!(
        unpaid <= 2,
        "most are paid somewhere in the recordings: {totals:?}"
    );
}

/// The sequences and the rare ones on states made for them.
#[test]
fn sequences_wait_for_their_first_event() {
    let made = |terms: Value| make(own(json!([]), terms).as_object().unwrap()).unwrap();
    let reward = made(json!([{ "when": "goal", "value": "since(me.touch) <= 3" }]));
    let mut memory = vec![0.0; reward.memory()];
    let g = game(Vec3A::ZERO, &[(Team::Blue, Vec3A::ZERO)]);
    let mut scored = g.clone();
    scored.goal_scored = Some(Team::Blue);
    let q = quiet(1);
    assert_eq!(
        reward.value_remembering(&g, &scored, &q, 0, &mut memory),
        0.0,
        "never touched"
    );
    assert_eq!(
        reward.value_remembering(&g, &g, &touched(quiet(1), 0), 0, &mut memory),
        0.0
    );
    for _ in 0..44 {
        reward.value_remembering(&g, &g, &q, 0, &mut memory); // 44 steps of 1/15 s: 2.93 s
    }
    assert_eq!(
        reward.value_remembering(&g, &scored, &q, 0, &mut memory),
        1.0,
        "within 3 s"
    );
    assert_eq!(
        reward.value_remembering(&g, &scored, &q, 0, &mut memory),
        0.0,
        "3.07 s after"
    );
}

#[test]
fn the_rare_ones_pay_as_the_hand_written_ones_on_states_made_for_them() {
    let (hand, built_terms) = {
        let p = pairs();
        let find = |start: &str| {
            p.iter()
                .find(|(w, ..)| w == start)
                .cloned()
                .expect("listed")
        };
        (find("wall_ready"), find("reset_then_shot"))
    };
    // On the side wall at 600 uu, the ball just above it, 60 boost.
    let mut wall = game(
        Vec3A::new(4000.0, 0.0, 800.0),
        &[(Team::Blue, Vec3A::new(4079.0, 0.0, 600.0))],
    );
    wall.cars[0].boost = 60.0;
    wall.cars[0].body.rot = Mat3A::from_rotation_y(-std::f32::consts::FRAC_PI_2);
    // A reset in the air, three quiet steps, then a touch with the ball heading into Orange's goal.
    let mut before = game(
        Vec3A::new(0.0, 4000.0, 300.0),
        &[(Team::Blue, Vec3A::new(0.0, 3800.0, 250.0))],
    );
    before.cars[0].on_ground = false;
    before.cars[0].can_flip = false;
    let mut reset = before.clone();
    reset.cars[0].can_flip = true;
    let mut shot = reset.clone();
    shot.ball.vel = Vec3A::new(0.0, 2000.0, 0.0);
    let steps = vec![
        (wall.clone(), wall.clone(), quiet(1)),
        (wall.clone(), wall.clone(), touched(quiet(1), 0)),
        (before.clone(), reset.clone(), quiet(1)),
        (reset.clone(), reset.clone(), quiet(1)),
        (reset.clone(), reset.clone(), quiet(1)),
        (reset.clone(), shot.clone(), touched(quiet(1), 0)),
    ];
    let starts = [true, false, true, false, false, false];
    for (what, hand, vars, terms) in [hand, built_terms] {
        let expected = paid(vec![hand], 8, &steps, &starts);
        let got = paid(vec![term(built(vars, terms))], 8, &steps, &starts);
        assert_eq!(expected, got, "{what}");
        assert!(
            expected.iter().flatten().any(|&v| v != 0.0),
            "{what} is paid here"
        );
    }
}

#[test]
fn caps_and_once_per_episode_bound_the_pay() {
    let made = |terms: Value| make(own(json!([]), terms).as_object().unwrap()).unwrap();
    let g = game(Vec3A::ZERO, &[(Team::Blue, Vec3A::ZERO)]);
    let q = quiet(1);
    let pay = |reward: &crate::custom::CustomReward, steps: usize| -> Vec<f32> {
        let mut memory = vec![0.0; reward.memory()];
        (0..steps)
            .map(|_| reward.value_remembering(&g, &g, &q, 0, &mut memory))
            .collect()
    };
    let capped = made(json!([{ "value": "3", "weight": 2, "cap_step": 5, "cap_episode": 12 }]));
    assert_eq!(pay(&capped, 4), [5.0, 5.0, 2.0, 0.0]);
    let once = made(json!([{ "value": "1", "once_per_episode": true }]));
    assert_eq!(pay(&once, 3), [1.0, 0.0, 0.0]);
    let charge = made(json!([{ "value": "-3", "cap_episode": 1 }]));
    assert_eq!(
        pay(&charge, 2),
        [-3.0, -3.0],
        "charges are not capped per episode"
    );
}

#[test]
fn formulas_are_checked_before_training() {
    let refused = |vars: Value, terms: Value| -> String {
        make(own(vars, terms).as_object().unwrap()).expect_err("refused")
    };
    let one = |value: &str| refused(json!([]), json!([{ "value": value }]));
    assert!(
        one("me.pos").contains("not a direction"),
        "a term pays a number"
    );
    assert!(one("me.speed +").contains("ends too soon"));
    assert!(one("me.sped").contains("no field 'sped'"));
    assert!(one("it.z").contains("'it'"));
    assert!(one("prev(me.touch)").contains("prev(...) cannot read"));
    assert!(one("sum_mates(sum_opps(1))").contains("another sum"));
    assert!(one("since(since(me.touch) > 1)").contains("cannot be inside"));
    assert!(one("me.pos + 1").contains("mixes numbers and directions"));
    assert!(one("clamp(1, 2)").contains("takes 3"));
    assert!(one("foo(1)").contains("no function called 'foo'"));
    let long = format!("{}1", "1 + ".repeat(400));
    assert!(one(&long).contains("too long"));
    assert!(
        refused(json!([]), json!([{ "value": "ball.y", "potential": true }]))
            .contains("not in this version")
    );
    assert!(refused(json!([]), json!([{ "measure": "one", "value": "1" }])).contains("not both"));
    assert!(
        refused(
            json!([{ "name": "me", "next": "1" }]),
            json!([{ "value": "1" }])
        )
        .contains("taken")
    );
    assert!(refused(json!([{ "name": "x" }]), json!([{ "value": "1" }])).contains("next"));
    assert!(refused(json!([]), json!([{ "value": "1", "cap_step": 0 }])).contains("above 0"));
    // The work per step is bounded: one term cannot hold 10 000 formulas of 400 parts each.
    let many = vec![json!("me.z > 0"); 17];
    assert!(refused(json!([]), json!([{ "when": many, "value": "1" }])).contains("at most 16"));
    let many = vec![json!("one"); 17];
    assert!(refused(json!([]), json!([{ "measure": many }])).contains("at most 16"));
}

/// A shared settings file is not trusted: a formula of signs nested past the part limit is refused like a
/// long one, before it can run the checker out of stack (100 000 signs crashed the program).
#[test]
fn formulas_nested_past_the_limit_are_refused_not_a_crash() {
    for text in [
        format!("{}1", "-".repeat(100_000)),
        format!("{}1", "not ".repeat(100_000)),
    ] {
        let found = make(
            own(json!([]), json!([{ "value": text }]))
                .as_object()
                .unwrap(),
        )
        .expect_err("refused");
        assert!(found.contains("too long"), "{found}");
    }
}

/// Numbers finite as written but past what a 32-bit number holds (1e300) are refused, wherever they are
/// written; and a big weight times a big value is never paid as infinity.
#[test]
fn numbers_too_big_for_the_core_are_refused_and_never_paid_as_infinity() {
    let refused = |vars: Value, terms: Value| -> String {
        make(own(vars, terms).as_object().unwrap()).expect_err("refused")
    };
    let term = |extra: Value| {
        let mut t = json!({ "value": "1" });
        t.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        refused(json!([]), json!([t]))
    };
    assert!(term(json!({ "weight": 1e300 })).contains("weight"));
    assert!(term(json!({ "weight": -1e39 })).contains("weight"));
    assert!(term(json!({ "cap_step": 1e300 })).contains("cap_step"));
    assert!(term(json!({ "cap_episode": 1e300 })).contains("cap_episode"));
    assert!(
        refused(
            json!([{ "name": "a", "start": 1e300, "next": "a" }]),
            json!([{ "value": "a" }])
        )
        .contains("start")
    );
    assert!(
        refused(json!([]), json!([{ "value": "1e39" }])).contains("not a number"),
        "a number in a formula too"
    );
    let reward = make(
        own(
            json!([]),
            json!([{ "value": "1e10", "weight": 1e30 }, { "value": "2" }]),
        )
        .as_object()
        .unwrap(),
    )
    .expect("finite as written and as held");
    let g = game(Vec3A::ZERO, &[(Team::Blue, Vec3A::ZERO)]);
    let paid = reward.value(&g, &g, &quiet(1), 0);
    assert!(paid.is_finite(), "paid {paid}");
}

#[test]
fn every_entity_field_and_function_reads_what_it_says() {
    let made = |terms: Value| make(own(json!([]), terms).as_object().unwrap()).unwrap();
    let mut g = game(
        Vec3A::new(0.0, 1000.0, 93.0),
        &[
            (Team::Blue, Vec3A::new(0.0, 0.0, 17.0)),
            (Team::Blue, Vec3A::new(300.0, 0.0, 17.0)),
            (Team::Orange, Vec3A::new(0.0, 2000.0, 17.0)),
        ],
    );
    g.cars[0].body.vel = Vec3A::new(0.0, 1000.0, 0.0);
    let q = quiet(3);
    let value = |text: &str| {
        let reward = made(json!([{ "value": text }]));
        let mut memory = vec![0.0; reward.memory()];
        reward.value_remembering(&g, &g, &q, 0, &mut memory)
    };
    let near = |a: f32, b: f32| (a - b).abs() < 1e-3;
    assert!(near(
        value("dist(me.pos, ball.pos)"),
        (1000.0f32.powi(2) + 76.0f32.powi(2)).sqrt()
    ));
    assert!(near(value("dist(mate.pos, me.pos)"), 300.0) && near(value("opp.y"), 2000.0));
    assert!(
        near(value("local(ball.pos - me.pos).y"), 1000.0),
        "the ball is to the car's right (+y)"
    );
    assert!(near(value("angle(me.forward, vec(0, 1, 0))"), 90.0));
    assert!(near(value("me.wall_dist"), 4096.0) && near(value("me.ceiling_dist"), 2027.0));
    assert!(near(value("count_mates(1) + 10 * count_opps(1)"), 11.0));
    assert!(
        near(value("me.team"), 1.0)
            && near(value("opp_goal.y"), 5120.0)
            && near(value("own_net.y"), -6000.0)
    );
    assert!(
        near(value("decay(200, 100)"), 0.25) && near(value("sign(-3) + abs(-2) + sqrt(16)"), 5.0)
    );
    assert!(near(
        value("me.speed / 1000 + clamp(5, 0, 2) + max(1, 3, 2) - min(4, 2)"),
        4.0
    ));
    assert!(near(
        value("nearest_big_pad(me.pos)"),
        (3584.0f32.powi(2)).sqrt()
    ));
    assert!(near(value("nearest_pad(vec(0, -1024, 0))"), 0.0));
    assert!(near(
        value("if(me.on_floor and not me.on_wall and not me.in_air, 1, 0)"),
        1.0
    ));
    assert!(near(
        value("1 < 2 and 2 <= 2 and 3 > 2 and 2 >= 2 and 1 == 1 and 1 != 2 or 0"),
        1.0
    ));
    assert!(near(
        value("exp(log(5)) + pow(2, 3) + -(-1) + length(cross(vec(1,0,0), vec(0,1,0)))"),
        15.0
    ));
    assert!(near(
        value("dot(norm(me.vel), me.forward) + flat(vec(1, 2, 3)).z"),
        0.0
    ));
    assert!(near(
        value("max_opps(it.y) + min_mates(it.x) + sum_mates(1)"),
        2301.0
    ));
    assert_eq!(value("min_mates(it.x) < 1e30"), 1.0);
}

#[test]
fn variables_count_and_remember_from_step_to_step() {
    let reward = make(own(
            json!([{ "name": "streak", "start": 2, "next": "if(me.touch, streak + 1, streak)" }]),
            json!([{ "value": "streak * 10 + prev(streak)" }, { "value": "count(me.touch) * 100 + held(me.on_ground) * 0" }, { "value": "once(me.touch) * 1000" }]),
        )
        .as_object()
        .unwrap(),
    )
    .unwrap();
    let g = game(Vec3A::ZERO, &[(Team::Blue, Vec3A::ZERO)]);
    let mut memory = vec![0.0; reward.memory()];
    let step = |touch: bool, memory: &mut [f32]| {
        let events = if touch {
            touched(quiet(1), 0)
        } else {
            quiet(1)
        };
        reward.value_remembering(&g, &g, &events, 0, memory)
    };
    assert_eq!(step(false, &mut memory), 22.0, "starts at 2");
    assert_eq!(
        step(true, &mut memory),
        32.0 + 100.0 + 1000.0,
        "3, from 2; one touch; the first"
    );
    assert_eq!(
        step(true, &mut memory),
        43.0 + 200.0,
        "4, from 3; two touches; not the first"
    );
    memory.fill(0.0); // a new episode
    assert_eq!(step(false, &mut memory), 22.0);
}

/// Processor time per car-step of the templates against the rewards they copy, on recorded 2v2 games. Run
/// on a quiet machine: `cargo test --release -p nitro_core reward_builder_speed -- --ignored --nocapture`.
#[test]
#[ignore]
fn reward_builder_speed_advanced() {
    if !meshes_loaded() {
        return;
    }
    let (steps, starts) = record_episodes(&busy(pool_config(1, 1, 1, 11), 2), 4000, 5);
    let rounds = 10;
    for (title, hand, vars, terms) in pairs().into_iter().filter(|(t, ..)| t != "chase_the_ball") {
        let compiled = vec![hand];
        let interpreted = vec![term(built(vars, terms))];
        let time = |terms: &[RewardTerm]| {
            let start = std::time::Instant::now();
            for _ in 0..rounds {
                std::hint::black_box(paid(terms.to_vec(), 8, &steps, &starts));
            }
            start.elapsed().as_secs_f64() * 1e9 / (rounds * steps.len() * 4) as f64
        };
        let (mut a, mut b): (Vec<f64>, Vec<f64>) = (0..5)
            .map(|_| (time(&compiled), time(&interpreted)))
            .unzip();
        a.sort_by(f64::total_cmp);
        b.sort_by(f64::total_cmp);
        eprintln!(
            "REWARD_BUILDER_SPEED {title} (2v2): compiled {a:.1?} ns, built {b:.1?} ns per car-step (median {:.1} vs \
             {:.1}; each includes the reward set's own work)",
            a[2], b[2]
        );
    }
}
