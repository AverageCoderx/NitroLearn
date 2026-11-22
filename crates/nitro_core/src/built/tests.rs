use super::*;
use crate::custom::kit::testing::{game, quiet, touched};
use crate::env::{RecordedStep, record_random_play};
use crate::rewards::{RewardSet, RewardStats, RewardTerm};
use crate::snapshot::Team;
use crate::testing::{meshes_loaded, pool_config};
use rocketsim::Mat3A;
use serde_json::json;

fn term(kind: Value) -> RewardTerm {
    serde_json::from_value(json!({ "kind": kind, "weight": 1.0, "zero_sum": null })).expect("term")
}

fn built(terms: Value) -> Value {
    json!({ "name": "built", "title": "test", "terms": terms })
}

/// Each car's pay for every recorded step from a set of `terms` at `tick_skip`.
fn paid(terms: Vec<RewardTerm>, tick_skip: u32, steps: &[RecordedStep]) -> Vec<Vec<f32>> {
    let mut set = RewardSet::new(terms.clone(), tick_skip);
    let mut stats = RewardStats::new(terms.len());
    steps
        .iter()
        .map(|(prev, cur, events)| {
            let cars = cur.cars.len();
            let mut out = vec![0.0; cars];
            set.compute(prev, cur, events, &vec![true; cars], &mut out, &mut stats);
            out
        })
        .collect()
}

/// Built rewards and the hand-written ones they copy: `(what, hand-written terms, built terms)`.
fn pairs() -> Vec<(&'static str, Vec<Value>, Value)> {
    vec![
        (
            "air",
            vec![json!({"name": "air"})],
            json!([{"when": ["in_air"], "measure": "one"}]),
        ),
        (
            "face_ball",
            vec![json!({"name": "face_ball"})],
            json!([{"measure": "facing_ball"}]),
        ),
        (
            "velocity_player_to_ball",
            vec![json!({"name": "velocity_player_to_ball"})],
            json!([{"measure": "speed_toward_ball"}]),
        ),
        (
            "velocity_ball_to_goal",
            vec![json!({"name": "velocity_ball_to_goal"})],
            json!([{"measure": "ball_speed_toward_goal"}]),
        ),
        (
            "alignment",
            vec![json!({"name": "alignment"})],
            json!([{"measure": "alignment"}]),
        ),
        (
            "save_boost",
            vec![json!({"name": "save_boost", "exponent": 1.0})],
            json!([{"measure": "boost"}]),
        ),
        (
            "touch_ball",
            vec![json!({"name": "touch_ball"})],
            json!([{"measure": "touch"}]),
        ),
        (
            "bump",
            vec![json!({"name": "bump"})],
            json!([{"measure": "bump"}]),
        ),
        (
            "demo",
            vec![json!({"name": "demo"})],
            json!([{"measure": "demo"}]),
        ),
        (
            "goal",
            vec![json!({"name": "goal", "concede_scale": -1.0})],
            json!([{"measure": "goal"}, {"measure": "concede", "weight": -1.0}]),
        ),
        (
            "example_ball_height",
            vec![json!({"name": "custom", "kind": "example_ball_height", "settings": {}})],
            json!([{"measure": "ball_height"}]),
        ),
        (
            "a mix: 4 x velocity_player_to_ball + 60 x touch_ball",
            vec![
                json!({"name": "velocity_player_to_ball"}),
                json!({"name": "touch_ball"}),
            ],
            json!([
                {"measure": "speed_toward_ball", "weight": 4.0},
                {"measure": "touch", "weight": 60.0}
            ]),
        ),
    ]
}

fn mix_weights(what: &str, terms: &mut [RewardTerm]) {
    if what.starts_with("a mix") {
        terms[0].weight = 4.0;
        terms[1].weight = 60.0;
    }
}

/// `config` with starts near the ball and shots on goal as well, so random play touches it, flies and
/// concedes.
fn busy(mut config: crate::config::PoolConfig) -> crate::config::PoolConfig {
    config.terminals.no_touch_seconds = 8.0;
    for kind in [
        json!({"name": "ball_touch", "min_distance_uu": 150.0, "max_distance_uu": 500.0}),
        json!({"name": "shot", "min_distance_uu": 500.0, "max_distance_uu": 1500.0, "car_distance_uu": 200.0}),
        json!({"name": "save", "min_speed": 1500.0, "max_speed": 2500.0}),
        json!({"name": "aerial_ball", "min_height_uu": 300.0, "max_height_uu": 900.0, "max_ball_speed": 300.0}),
    ] {
        config
            .state_setters
            .push(serde_json::from_value(json!({ "kind": kind, "weight": 2.0 })).unwrap());
    }
    config
}

#[test]
fn built_rewards_pay_what_the_hand_written_ones_do_on_recorded_games() {
    if !meshes_loaded() {
        return;
    }
    for tick_skip in [8, 4] {
        let mut config = pool_config(1, 1, 1, 11);
        config.tick_skip = tick_skip;
        let steps = record_random_play(&busy(config), 3000, 5);
        let touches = steps
            .iter()
            .filter(|(_, _, e)| e.cars.iter().any(|c| c.touched_ball))
            .count();
        let cars: usize = steps.iter().map(|(_, _, e)| e.cars.len()).sum();
        let air: usize = steps
            .iter()
            .filter(|(_, c, _)| !c.cars[0].on_ground)
            .count();
        assert!(
            touches > 0,
            "the recording holds touches ({cars} car events, {air} in the air)"
        );
        for (what, hand, terms) in pairs() {
            let mut written: Vec<RewardTerm> = hand.into_iter().map(term).collect();
            mix_weights(what, &mut written);
            let expected = paid(written, tick_skip, &steps);
            let got = paid(vec![term(built(terms))], tick_skip, &steps);
            let mut fired = 0;
            for (step, (a, b)) in expected.iter().zip(&got).enumerate() {
                for (x, y) in a.iter().zip(b) {
                    assert!(
                        (x - y).abs() <= 1e-6 * x.abs().max(1.0),
                        "{what}, tick skip {tick_skip}, step {step}: {x} vs {y}"
                    );
                    fired += usize::from(*x != 0.0);
                }
            }
            eprintln!(
                "{what} (tick skip {tick_skip}): {fired} non-zero car-steps of {}",
                2 * steps.len()
            );
            if !matches!(what, "bump" | "demo") {
                assert!(fired > 0, "{what} is paid somewhere in the recording");
            }
        }
    }
}

#[test]
fn demos_and_bumps_pay_as_the_hand_written_ones() {
    let g = game(
        Vec3A::ZERO,
        &[(Team::Blue, Vec3A::ZERO), (Team::Orange, Vec3A::X)],
    );
    let mut events = quiet(2);
    events.cars[0].demoed_opponent = true;
    events.cars[1].bumped_opponent = true;
    let steps = vec![(g.clone(), g, events)];
    for name in ["demo", "bump"] {
        let hand = paid(vec![term(json!({ "name": name }))], 8, &steps);
        let got = paid(vec![term(built(json!([{ "measure": name }])))], 8, &steps);
        assert_eq!(hand, got, "{name}");
        assert!(hand[0].contains(&1.0), "{name} is paid");
    }
}

#[test]
fn the_built_reward_is_named_by_its_title_and_paid_every_step_only_when_every_term_is() {
    let kind: crate::rewards::RewardKind = serde_json::from_value(json!({
        "name": "built", "title": "chase", "terms": [{"measure": "speed_toward_ball"}]
    }))
    .unwrap();
    assert_eq!(kind.label(), "chase");
    let every_step = |terms: Value| made(terms).every_step();
    assert!(every_step(json!([{"measure": "speed_toward_ball"}])));
    assert!(!every_step(json!([{"measure": "touch"}])));
    assert!(!every_step(
        json!([{"when": ["touching_ball"], "measure": "ball_height"}])
    ));
    assert!(!every_step(
        json!([{"measure": "one", "once_per_touch": true}])
    ));
}

#[test]
fn mistakes_are_refused_with_their_term_and_word() {
    let refused = |settings: Value| -> String {
        let map = settings.as_object().unwrap().clone();
        make(&map).expect_err("refused")
    };
    assert!(refused(json!({"title": "x", "terms": []})).contains("at least one term"));
    assert!(refused(json!({"title": "X y", "terms": [{"measure": "one"}]})).contains("title"));
    assert!(refused(json!({"terms": [{"measure": "one"}]})).contains("title"));
    assert!(
        refused(json!({"title": "x", "terms": [{"measure": "one"}], "weight": 2}))
            .contains("'weight'")
    );
    let msg = refused(json!({"title": "x", "terms": [{"measure": "one"}, {"measure": "speed"}]}));
    assert!(
        msg.contains("term 2") && msg.contains("no measure called 'speed'"),
        "{msg}"
    );
    assert!(
        refused(json!({"title": "x", "terms": [{"when": ["flying"], "measure": "one"}]}))
            .contains("no condition called 'flying'")
    );
    assert!(
        refused(
            json!({"title": "x", "terms": [{"when": [{"ball_above": "high"}], "measure": "one"}]})
        )
        .contains("needs a number")
    );
    assert!(refused(json!({"title": "x", "terms": [{"when": [{"facing_ball_within": 200}], "measure": "one"}]}))
        .contains("0 to 180"));
    assert!(
        refused(json!({"title": "x", "terms": [{"when": ["in_air"]}]})).contains("needs a measure")
    );
    assert!(
        refused(json!({"title": "x", "terms": [{"measure": "one", "combine": "max"}]}))
            .contains("combine")
    );
    assert!(
        refused(json!({"title": "x", "terms": [{"measure": "one", "wieght": 2}]}))
            .contains("'wieght'")
    );
}

#[test]
fn numbers_too_big_for_the_core_are_refused_not_paid_as_infinity() {
    let refused = |term: Value| -> String {
        make(json!({"title": "x", "terms": [term]}).as_object().unwrap()).expect_err("refused")
    };
    // Finite as written, but past what a 32-bit number holds: an infinite weight would pay inf (or NaN
    // where the measure is 0) into training.
    assert!(refused(json!({"measure": "one", "weight": 1e300})).contains("weight"));
    assert!(refused(json!({"measure": "one", "weight": -1e39})).contains("weight"));
    assert!(
        refused(json!({"when": [{"ball_above": 1e300}], "measure": "one"}))
            .contains("needs a number")
    );
    assert!(
        refused(json!({"when": [{"speed_above": 1e300}], "measure": "one"}))
            .contains("needs a number")
    );
    assert!(
        make(
            json!({"title": "x", "terms": [{"measure": "one", "weight": 1e30}]})
                .as_object()
                .unwrap()
        )
        .is_ok()
    );
}

/// A built reward made straight from its terms.
fn made(terms: Value) -> CustomReward {
    make(
        json!({ "title": "test", "terms": terms })
            .as_object()
            .unwrap(),
    )
    .unwrap()
}

fn value(terms: Value, prev: &GameSnapshot, cur: &GameSnapshot, events: &StepEvents) -> f32 {
    let made = made(terms);
    let mut memory = vec![0.0; made.memory()];
    made.value_remembering(prev, cur, events, 0, &mut memory)
}

#[test]
fn conditions_hold_where_they_say() {
    let blue = Team::Blue;
    let mut g = game(
        Vec3A::new(0.0, 1000.0, 500.0),
        &[(blue, Vec3A::new(0.0, 0.0, 17.0))],
    );
    let q = quiet(1);
    let pays = |when: Value, g: &GameSnapshot, e: &StepEvents| {
        value(json!([{ "when": when, "measure": "one" }]), g, g, e) != 0.0
    };
    // The car rests at the centre facing +x; the ball is 1000 uu toward Orange's goal, 500 uu up.
    assert!(pays(json!(["on_ground", "on_floor", "has_flip"]), &g, &q));
    assert!(!pays(json!(["in_air"]), &g, &q) && !pays(json!(["on_wall"]), &g, &q));
    assert!(pays(
        json!(["ball_opponent_half", {"not": "ball_own_half"}]),
        &g,
        &q
    ));
    assert!(pays(
        json!([{"ball_above": 400}, {"ball_below": 600}, {"car_below": 100}]),
        &g,
        &q
    ));
    assert!(pays(
        json!([{"ball_closer_than": 1200}, {"ball_farther_than": 1000}]),
        &g,
        &q
    ));
    assert!(
        !pays(json!([{"facing_ball_within": 45}]), &g, &q),
        "the ball is to its left"
    );
    assert!(pays(json!([{"facing_ball_within": 90}]), &g, &q));
    assert!(pays(
        json!([{"boost_above": 30}, {"boost_below": 40}, {"speed_below": 1}]),
        &g,
        &q
    ));
    assert!(pays(json!(["closest_to_ball"]), &g, &q) && !pays(json!(["touching_ball"]), &g, &q));
    assert!(pays(
        json!(["touching_ball", "last_to_touch"]),
        &g,
        &touched(quiet(1), 0)
    ));
    g.cars[0].body.rot = Mat3A::from_rotation_z(std::f32::consts::FRAC_PI_2);
    assert!(pays(
        json!([{"facing_goal_within": 5}, {"facing_ball_within": 30}]),
        &g,
        &q
    ));
    g.cars[0].team = Team::Orange;
    assert!(
        pays(json!(["ball_own_half"]), &g, &q)
            && !pays(json!([{"facing_goal_within": 90}]), &g, &q)
    );
    g.cars[0].on_ground = false;
    g.cars[0].body.pos.y = 2000.0;
    assert!(pays(json!(["in_air", "own_half"]), &g, &q));
    g.cars[0].on_ground = true;
    g.cars[0].body.rot = Mat3A::from_rotation_y(std::f32::consts::FRAC_PI_2);
    assert!(pays(json!(["on_wall"]), &g, &q));
}

#[test]
fn once_per_touch_pays_once_until_the_next_touch() {
    let made = made(json!([{ "when": ["in_air"], "measure": "one", "once_per_touch": true }]));
    let mut g = game(Vec3A::ZERO, &[(Team::Blue, Vec3A::new(0.0, 0.0, 300.0))]);
    g.cars[0].on_ground = false;
    let mut memory = vec![0.0; made.memory()];
    let mut pay = |e: &StepEvents| made.value_remembering(&g, &g, e, 0, &mut memory);
    assert_eq!([pay(&quiet(1)), pay(&quiet(1))], [1.0, 0.0]);
    assert_eq!([pay(&touched(quiet(1), 0)), pay(&quiet(1))], [1.0, 0.0]);
    assert!(made.memory() == 1 && !made.every_step());
}

#[test]
fn measures_of_the_step_and_its_events() {
    let mut prev = game(Vec3A::new(0.0, 0.0, 93.0), &[(Team::Blue, Vec3A::ZERO)]);
    let mut cur = prev.clone();
    cur.cars[0].boost = 13.0;
    cur.ball.vel = Vec3A::new(0.0, 3000.0, 0.0);
    cur.ball.pos.y = 4000.0;
    let q = quiet(1);
    let near = |a: f32, b: f32| (a - b).abs() < 1e-6;
    assert!(near(
        value(json!([{"measure": "boost_used"}]), &prev, &cur, &q),
        0.2
    ));
    assert!(near(
        value(json!([{"measure": "boost_gained"}]), &cur, &prev, &q),
        0.2
    ));
    assert!(near(
        value(json!([{"measure": "ball_speed_change"}]), &prev, &cur, &q),
        0.5
    ));
    assert_eq!(
        value(json!([{"measure": "shot"}]), &prev, &cur, &q),
        0.0,
        "no touch"
    );
    assert_eq!(
        value(
            json!([{"measure": "shot"}]),
            &prev,
            &cur,
            &touched(quiet(1), 0)
        ),
        1.0
    );
    cur.goal_scored = Some(Team::Orange);
    assert_eq!(
        value(
            json!([{"measure": "concede"}, {"measure": "goal", "weight": 9}]),
            &prev,
            &cur,
            &q
        ),
        1.0
    );
    cur.cars[0].has_jumped = true;
    prev.cars[0].has_jumped = false;
    assert_eq!(
        value(
            json!([{"measure": ["jump", "boost"], "weight": 2}]),
            &prev,
            &cur,
            &q
        ),
        0.26
    );
    // Mixed: the continuous term is scaled to the step (4 ticks = half the reference), the event not.
    let mut short = quiet(1);
    short.step_seconds = 4.0 / 120.0;
    let mixed = json!([{"measure": "one"}, {"measure": "jump", "weight": 10}]);
    assert!(near(value(mixed, &prev, &cur, &short), 10.5));
}

/// Processor time per car-step of a built reward vs its compiled twins on recorded games. Run on a quiet
/// machine: `cargo test --release -p nitro_core reward_builder_speed -- --ignored --nocapture`.
#[test]
#[ignore]
fn reward_builder_speed() {
    if !meshes_loaded() {
        return;
    }
    let steps = record_random_play(&busy(pool_config(1, 1, 1, 11)), 4000, 5);
    let cases = [
        (
            "4 terms (chase, touch, goal)",
            vec![
                json!({"name": "velocity_player_to_ball"}),
                json!({"name": "touch_ball"}),
                json!({"name": "goal", "concede_scale": -1.0}),
            ],
            json!([
                {"measure": "speed_toward_ball"},
                {"measure": "touch"},
                {"measure": "goal"}, {"measure": "concede", "weight": -1}
            ]),
        ),
        (
            "1 term, 4 conditions",
            vec![json!({"name": "velocity_player_to_ball"})],
            json!([{"when": ["in_air", {"ball_above": 300}, {"facing_ball_within": 30}, {"boost_above": 10}],
                    "measure": "speed_toward_ball"}]),
        ),
    ];
    let rounds = 20;
    for (what, hand, terms) in cases {
        let time = |terms: &[RewardTerm]| {
            let start = std::time::Instant::now();
            for _ in 0..rounds {
                std::hint::black_box(paid(terms.to_vec(), 8, &steps));
            }
            start.elapsed().as_secs_f64() * 1e9 / (rounds * steps.len() * 2) as f64
        };
        let compiled: Vec<RewardTerm> = hand.into_iter().map(term).collect();
        let interpreted = vec![term(built(terms))];
        // Five repeats, taken in turn, so a slower moment of the machine hits both alike.
        let (mut a, mut b): (Vec<f64>, Vec<f64>) = (0..5)
            .map(|_| (time(&compiled), time(&interpreted)))
            .unzip();
        a.sort_by(f64::total_cmp);
        b.sort_by(f64::total_cmp);
        eprintln!(
            "REWARD_BUILDER_SPEED {what}: compiled {a:.1?} ns, built {b:.1?} ns per car-step (median {:.1} vs {:.1}; \
             each includes the reward set's own work)",
            a[2], b[2]
        );
    }
}
