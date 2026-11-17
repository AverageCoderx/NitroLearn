//! The rewards a GigaLearn bot's import makes from its project, held to GigaLearn's own formulas on simulated
//! 2v2 games: each reward re-typed here from GigaLearnCPP's `CommonRewards.h` and `ZeroSumReward.cpp` (per
//! GigaLearn step, never scaled to the tick skip), weighted as the project weighs it, against NitroLearn's
//! reward set built from the terms the import writes (`tests/fixtures/gigalearn/rewards_tick*.json`).

use rocketsim::Vec3A;
use serde::Deserialize;
use serde_json::Value;

use crate::rewards::{RewardSet, RewardStats, RewardTerm};
use crate::sim::Sim;
use crate::snapshot::{Body, Controls, GameSnapshot, StepEvents, Team};
use crate::testing::meshes_loaded;

#[derive(Deserialize)]
struct Origin {
    class: String,
    args: Vec<Value>,
    weight: f32,
    zero_sum: Option<[f32; 2]>,
}

#[derive(Deserialize)]
struct Fixture {
    tick_skip: u32,
    gigalearn: Vec<Origin>,
    terms: Vec<RewardTerm>,
}

fn kph(v: f32) -> f32 {
    v * (250.0 / 9.0)
}

fn arg(origin: &Origin, i: usize, default: f32) -> f32 {
    origin.args.get(i).map_or(default, |v| {
        v.as_f64()
            .map(|x| x as f32)
            .or_else(|| v.as_bool().map(f32::from))
            .expect("a number")
    })
}

/// GigaLearn's per-player value of one reward class (CommonRewards.h), before ZeroSum and weight.
fn gigalearn(
    origin: &Origin,
    prev: &GameSnapshot,
    cur: &GameSnapshot,
    ev: &StepEvents,
    car: usize,
) -> f32 {
    let me = &cur.cars[car];
    let happened = &ev.cars[car];
    let dir_to_ball = (cur.ball.pos - me.body.pos).normalize_or_zero();
    let net = |orange: bool| Vec3A::new(0.0, if orange { 6000.0 } else { -6000.0 }, 642.775 / 2.0);
    match origin.class.as_str() {
        "AirReward" => f32::from(u8::from(!me.on_ground)),
        "FaceBallReward" => me.body.forward().dot(dir_to_ball),
        "VelocityPlayerToBallReward" => dir_to_ball.dot(me.body.vel / 2300.0),
        "VelocityBallToGoalReward" => {
            let mut orange = me.team == Team::Blue;
            if arg(origin, 0, 0.0) != 0.0 {
                orange = !orange;
            }
            (net(orange) - cur.ball.pos)
                .normalize_or_zero()
                .dot(cur.ball.vel / 6000.0)
        }
        "TouchBallReward" => f32::from(u8::from(happened.touched_ball)),
        "StrongTouchReward" => {
            if !happened.touched_ball {
                return 0.0;
            }
            let force = (cur.ball.vel - prev.ball.vel).length();
            if force < kph(arg(origin, 0, 20.0)) {
                0.0
            } else {
                (force / kph(arg(origin, 1, 130.0))).min(1.0)
            }
        }
        "PickupBoostReward" => {
            let before = prev.cars[car].boost;
            if me.boost > before {
                (me.boost / 100.0).sqrt() - (before / 100.0).sqrt()
            } else {
                0.0
            }
        }
        "SaveBoostReward" => (me.boost / 100.0).powf(arg(origin, 0, 0.5)).clamp(0.0, 1.0),
        "BumpReward" => f32::from(u8::from(happened.bumped_opponent)),
        "DemoReward" => f32::from(u8::from(happened.demoed_opponent)),
        "BumpedPenalty" => -f32::from(u8::from(happened.bumped_by_opponent)),
        "DemoedPenalty" => -f32::from(u8::from(happened.demoed_by_opponent)),
        "GoalReward" => match cur.goal_scored {
            None => 0.0,
            Some(_) => {
                let scored = me.team != Team::from_field_y(cur.ball.pos.y);
                if scored { 1.0 } else { arg(origin, 0, -1.0) }
            }
        },
        "VelocityReward" => {
            let sign = 1.0 - 2.0 * arg(origin, 0, 0.0);
            me.body.vel.length() / 2300.0 * sign
        }
        "SpeedReward" => me.body.vel.length() / 2300.0,
        "WavedashReward" => {
            let flipping = ev.prev_detail(car).is_flipping && !prev.cars[car].on_ground;
            f32::from(u8::from(me.on_ground && flipping))
        }
        "TouchAccelReward" => {
            if !happened.touched_ball {
                return 0.0;
            }
            let top = kph(110.0);
            let before = (prev.ball.vel.length() / top).min(1.0);
            let after = (cur.ball.vel.length() / top).min(1.0);
            if after > before { after - before } else { 0.0 }
        }
        // Two rewards of community GigaLearn bots, as their CommonRewards.h writes them.
        "KickoffProximityReward" => {
            if cur.ball.vel.length() >= 1.0 {
                return 0.0;
            }
            let mine = (me.body.pos - cur.ball.pos).length();
            let nearest = cur
                .cars
                .iter()
                .filter(|c| c.team != me.team)
                .map(|c| (c.body.pos - cur.ball.pos).length())
                .fold(f32::INFINITY, f32::min);
            if nearest == f32::INFINITY {
                0.0
            } else if mine < nearest {
                1.0
            } else {
                -1.0
            }
        }
        "SpeedTowardsOpponentReward" => {
            let Some(opponent) = cur.cars.iter().find(|c| c.team != me.team) else {
                return 0.0;
            };
            let direction = (opponent.body.pos - me.body.pos).normalize_or_zero();
            direction.dot(me.body.vel / 2300.0).clamp(-1.0, 1.0) * arg(origin, 0, 1.0)
        }
        other => panic!("no GigaLearn formula for {other} in this test"),
    }
}

/// GigaLearn's total for every player: each reward, ZeroSum-wrapped where the project wraps it, times its
/// weight (ZeroSumReward::GetAllRewards).
fn gigalearn_totals(
    f: &Fixture,
    prev: &GameSnapshot,
    cur: &GameSnapshot,
    ev: &StepEvents,
) -> Vec<f32> {
    let cars = cur.cars.len();
    let mut total = vec![0.0; cars];
    for origin in &f.gigalearn {
        let own: Vec<f32> = (0..cars)
            .map(|c| gigalearn(origin, prev, cur, ev, c))
            .collect();
        let shaped: Vec<f32> = match origin.zero_sum {
            None => own,
            Some([spirit, scale]) => {
                let mut sums = [0.0f32; 2];
                let mut counts = [0f32; 2];
                for (c, &value) in own.iter().enumerate() {
                    let team = cur.cars[c].team.index();
                    sums[team] += value;
                    counts[team] += 1.0;
                }
                let avg = |t: usize| sums[t] / counts[t].max(1.0);
                (0..cars)
                    .map(|c| {
                        let t = cur.cars[c].team.index();
                        own[c] * (1.0 - spirit) + avg(t) * spirit - avg(1 - t) * scale
                    })
                    .collect()
            }
        };
        for c in 0..cars {
            total[c] += shaped[c] * origin.weight;
        }
    }
    total
}

fn axis(rng: &mut fastrand::Rng) -> f32 {
    [-1.0, 0.0, 1.0][rng.usize(0..3)]
}

fn check(file: &str) {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/gigalearn/"
    )
    .to_owned()
        + file;
    let fixture: Fixture =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("fixture")).expect("parse");
    assert_eq!(fixture.terms.len(), fixture.gigalearn.len());
    let mut set = RewardSet::new(fixture.terms.clone(), fixture.tick_skip);
    let mut stats = RewardStats::new(fixture.terms.len());
    let mut sim = Sim::new(2, true, 7);
    let cars = sim.num_cars();
    let mut rng = fastrand::Rng::with_seed(11);
    let (mut prev, mut cur) = (GameSnapshot::default(), GameSnapshot::default());
    let mut events = StepEvents::default();
    let mut out = vec![0.0; cars];
    let (mut fresh, mut checked, mut largest) = (true, 0usize, 0.0f32);
    let mut fired = std::collections::BTreeSet::new();
    sim.reset_to_kickoff(1);
    events.clear(cars);
    sim.details(&mut events);
    sim.read(&mut cur);
    for step in 0..6000 {
        if step % 150 == 0 {
            // Cars thrown at the ball and at each other now and then, for touches, bumps and demolitions.
            let mut placed = Vec::with_capacity(cars);
            for _ in 0..cars {
                let at = Vec3A::new(
                    rng.f32() * 1200.0 - 600.0,
                    rng.f32() * 1200.0 - 600.0,
                    17.0 + 300.0 * rng.f32(),
                );
                let toward = (Vec3A::new(0.0, 0.0, 200.0) - at).normalize_or_zero() * 2200.0;
                let body = Body {
                    pos: at,
                    vel: toward,
                    ..Body::RESTING
                };
                placed.push((body, 100.0 * rng.f32(), at.z < 20.0));
            }
            sim.place(sim.ball(), &placed);
            fresh = true;
        }
        for car in 0..cars {
            sim.set_controls(
                car,
                Controls {
                    throttle: axis(&mut rng),
                    steer: axis(&mut rng),
                    pitch: axis(&mut rng),
                    yaw: axis(&mut rng),
                    roll: axis(&mut rng),
                    jump: rng.bool(),
                    boost: rng.u8(0..4) == 0,
                    handbrake: rng.bool(),
                },
            );
        }
        std::mem::swap(&mut prev, &mut cur);
        events.clear(cars);
        sim.step(fixture.tick_skip, &mut events);
        std::mem::swap(&mut events.details, &mut events.prev_details);
        sim.details(&mut events);
        sim.read(&mut cur);
        set.compute(
            &prev,
            &cur,
            &events,
            &vec![true; cars],
            &mut out,
            &mut stats,
        );
        if !fresh {
            let expected = gigalearn_totals(&fixture, &prev, &cur, &events);
            for car in 0..cars {
                let gap = (out[car] - expected[car]).abs();
                largest = largest.max(gap / expected[car].abs().max(1.0));
                assert!(
                    gap <= 1e-4 * expected[car].abs().max(1.0),
                    "{file}, step {step}, car {car}: NitroLearn {} against GigaLearn {}",
                    out[car],
                    expected[car]
                );
            }
            for origin in &fixture.gigalearn {
                if (0..cars).any(|c| gigalearn(origin, &prev, &cur, &events, c) != 0.0) {
                    fired.insert(origin.class.clone());
                }
            }
            checked += 1;
        }
        fresh = false;
        if cur.goal_scored.is_some() {
            sim.reset_to_kickoff(step as u64);
            set.reset();
            fresh = true;
        }
    }
    assert!(checked > 5000, "{checked} steps checked");
    // Every reward paid something at least once, so none was checked only at 0.
    for needed in [
        "StrongTouchReward",
        "BumpReward",
        "BumpedPenalty",
        "TouchAccelReward",
        "PickupBoostReward",
        "KickoffProximityReward",
        "SpeedTowardsOpponentReward",
    ] {
        assert!(fired.contains(needed), "{needed} never paid in {file}");
    }
    eprintln!(
        "{file}: {checked} steps, largest relative difference {largest:.2e}, rewards seen: {fired:?}"
    );
}

#[test]
fn imported_rewards_pay_what_gigalearn_pays_at_tick_skip_8() {
    if meshes_loaded() {
        check("rewards_tick8.json");
    }
}

#[test]
fn imported_rewards_pay_what_gigalearn_pays_at_tick_skip_4() {
    if meshes_loaded() {
        check("rewards_tick4.json");
    }
}
