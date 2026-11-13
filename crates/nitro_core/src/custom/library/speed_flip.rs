//! Speed flips: a jump and a forward-diagonal dodge, cancelled so the nose stays level, paid by how much
//! farther the car got than boosting straight would have.
//! group: moves

use crate::custom::prelude::*;

use crate::custom::library::aerial_kit::*;

/// Speed flips: once per jump, up to max_credit (1) by how much farther along its line the car got than boosting
/// would have (full at 800 uu ahead); nothing for any other dodge.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpeedFlip {
    /// On the floor: 3+ wheels down with the car's centre below this...
    #[serde(default = "floor_max_z")]
    pub floor_max_z: f32,
    /// ...and the roof's up.z above this (not a wall or ramp).
    #[serde(default = "floor_min_up_z")]
    pub floor_min_up_z: f32,
    /// Seconds from the jump to the dodge (a later one is a late dodge).
    #[serde(default = "max_jump_to_dodge")]
    pub max_jump_to_dodge: f32,
    /// A forward diagonal: the dodge's sideways torque at least this...
    #[serde(default = "min_torque_side")]
    pub min_torque_side: f32,
    /// ...and its forward torque at least this.
    #[serde(default = "min_torque_forward")]
    pub min_torque_forward: f32,
    /// The cancel: the net nose rotation from the jump to wheels-down at most this (deg)...
    #[serde(default = "max_pitch_rotation")]
    pub max_pitch_rotation: f32,
    /// ...and the largest nose up/down angle in the air at most this (deg).
    #[serde(default = "max_nose_angle")]
    pub max_nose_angle: f32,
    /// Seconds from the jump to wheels-down on the floor.
    #[serde(default = "max_air_time")]
    pub max_air_time: f32,
    /// The roof's tilt from vertical at wheels-down at most this (deg).
    #[serde(default = "max_land_tilt")]
    pub max_land_tilt: f32,
    /// Wheels down off the floor with the ball centre this close: on the ball.
    #[serde(default = "ball_land_dist")]
    pub ball_land_dist: f32,
    /// The roof's tilt while the body is at the floor before wheels-down at most this (180 = off).
    #[serde(default = "max_body_tilt")]
    pub max_body_tilt: f32,
    /// A hitbox corner this close to the floor: the body is at the floor (uu).
    #[serde(default = "body_floor_margin")]
    pub body_floor_margin: f32,
    /// Seconds on the floor after wheels-down before the speed is measured.
    #[serde(default = "settle_time")]
    pub settle_time: f32,
    /// The lowest speed after wheels-down at least this share of the landing speed.
    #[serde(default = "min_kept_frac")]
    pub min_kept_frac: f32,
    /// The travel direction at the pay point within this of the one before the jump (deg).
    #[serde(default = "max_veer_deg")]
    pub max_veer_deg: f32,
    /// Seconds from the step before the jump to the pay point.
    #[serde(default = "pay_window")]
    pub pay_window: f32,
    /// The comparison starts from the highest speed in this many seconds before the jump.
    #[serde(default = "ref_window")]
    pub ref_window: f32,
    /// What the gain is: "lead_along" (uu ahead of boosting, along the pre-jump line), "lead" (in any direction),
    /// "speed_vs_boost" (speed over boosting's) or "speed_vs_ref" (speed over the speed before the jump).
    #[serde(default = "pay_measure", deserialize_with = "measure_name")]
    pub pay_measure: String,
    /// Also give the boost-only comparison the boost pads picked up during the attempt.
    #[serde(default = "pads_in_baseline")]
    pub pads_in_baseline: bool,
    /// No credit at or below this gain.
    #[serde(default = "min_gain")]
    pub min_gain: f32,
    /// Full credit at this gain.
    #[serde(default = "gain_full")]
    pub gain_full: f32,
    /// The credit for a full speed flip.
    #[serde(default = "max_credit")]
    pub max_credit: f32,
    /// Paid (negative) when a forward-diagonal attempt loses its speed or lands badly.
    #[serde(default = "lost_penalty")]
    pub lost_penalty: f32,
    /// The most credit per episode, 0 = no cap (the speed flip bot: 3).
    #[serde(default = "max_per_episode")]
    pub max_per_episode: f32,
    /// Training aid: credit per second while a forward diagonal off a floor jump stays cancelled during its
    /// dodge, 0 = off (the speed flip bot's pilot B: 0.15, then off).
    #[serde(default = "cancel_shaping")]
    pub cancel_shaping: f32,
}

defaults! { SpeedFlip {
    floor_max_z: f32 = 60.0,
    floor_min_up_z: f32 = 0.8,
    max_jump_to_dodge: f32 = 0.25,
    min_torque_side: f32 = 0.5,
    min_torque_forward: f32 = 0.5,
    max_pitch_rotation: f32 = 30.0,
    max_nose_angle: f32 = 25.0,
    max_air_time: f32 = 1.4,
    max_land_tilt: f32 = 30.0,
    ball_land_dist: f32 = 250.0,
    max_body_tilt: f32 = 180.0,
    body_floor_margin: f32 = 8.0,
    settle_time: f32 = 0.3,
    min_kept_frac: f32 = 0.75,
    max_veer_deg: f32 = 50.0,
    pay_window: f32 = 1.5,
    ref_window: f32 = 2.0,
    pay_measure: String = "lead_along".into(),
    pads_in_baseline: bool = false,
    min_gain: f32 = 0.0,
    gain_full: f32 = 800.0,
    max_credit: f32 = 1.0,
    lost_penalty: f32 = 0.0,
    max_per_episode: f32 = 0.0,
    cancel_shaping: f32 = 0.0,
} }

const MEASURES: [&str; 4] = ["lead", "speed_vs_boost", "speed_vs_ref", "lead_along"];

fn measure_name<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<String, D::Error> {
    let name = String::deserialize(deserializer)?;
    if MEASURES.contains(&name.as_str()) {
        Ok(name)
    } else {
        Err(serde::de::Error::custom(format!(
            "pay_measure '{name}' is not one of {}",
            MEASURES.join(", ")
        )))
    }
}

// The attempt's phase.
const IDLE: i32 = 0;
const JUMPED: i32 = 1;
const FLIPPING: i32 = 2;
const SETTLING: i32 = 3;

// Dodge types.
const DODGE_FRONT: i32 = 1;
const DODGE_BACK: i32 = 2;
const DODGE_SIDE: i32 = 3;
const DODGE_DIAG_FORWARD: i32 = 4;
const DODGE_DIAG_BACK: i32 = 5;
const DODGE_STALL: i32 = 6;

// Why an attempt fails (the ones with a penalty).
const SPEED_LOST: i32 = 1;
const BAD_LANDING: i32 = 2;
const OTHER: i32 = 0;

/// The boost pads picked up during an attempt kept for the comparison (more than an attempt can reach).
const MAX_PICKUPS: usize = 8;

kept! {
    /// Each car's attempt in progress (ticks on the episode clock).
    struct Track {
        phase: i32,
        jump_tick: f32,
        land_tick: f32,
        /// The horizontal travel direction before the jump.
        dir0: Vec3A,
        /// Where the car was at the step before the jump...
        jump_pos: Vec3A,
        /// ...and when: the distance and the boost-only comparison start there.
        base_tick: f32,
        /// The pay point's measurements are taken.
        captured: bool,
        min_speed_settle: f32,
        /// Credit paid this episode.
        paid: f32,
        dodge: i32,
        /// The highest horizontal speed in ref_window before the jump.
        ref_speed: f32,
        boost_jump: f32,
        /// Net rotation about the car's right axis (nose up/down) since the jump, deg.
        pitch_rot: f32,
        max_nose: f32,
        /// The worst roof tilt while the body was at the floor before wheels-down.
        body_tilt: f32,
        speed_land: f32,
        veer: f32,
        gain: f32,
        pickups: i32,
        /// The horizontal speed history's ring: how many entries since it was last emptied, and where the next goes.
        hist_count: i32,
        hist_head: i32,
    }
}

/// Rotation from orientation `a` to `b` as a rotation vector in `a`'s axes (deg): x about the forward axis (roll),
/// y about the right axis (pitch), z about the up axis (yaw).
fn local_rotation(a: &Mat3A, b: &Mat3A) -> Vec3A {
    let (aa, bb) = (
        [a.x_axis, a.y_axis, a.z_axis],
        [b.x_axis, b.y_axis, b.z_axis],
    );
    let r = |i: usize, j: usize| aa[i].dot(bb[j]);
    let trace = r(0, 0) + r(1, 1) + r(2, 2);
    let th = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0).acos();
    let skew = Vec3A::new(r(2, 1) - r(1, 2), r(0, 2) - r(2, 0), r(1, 0) - r(0, 1)); // 2 sin(th) x axis
    let s = th.sin();
    let k = if s > 1e-4 { th / (2.0 * s) } else { 0.5 };
    skew * (k * 180.0 / std::f32::consts::PI)
}

fn classify_dodge(torque: Vec3A, min_side: f32, min_forward: f32) -> i32 {
    let (x, y) = (torque.x, torque.y);
    let side = x.abs() >= min_side;
    let forward = y >= min_forward;
    let back = y <= -min_forward;
    if x.abs() < 0.1 && y.abs() < 0.1 {
        DODGE_STALL
    } else if side && forward {
        DODGE_DIAG_FORWARD
    } else if side && back {
        DODGE_DIAG_BACK
    } else if forward {
        DODGE_FRONT
    } else if back {
        DODGE_BACK
    } else {
        DODGE_SIDE
    }
}

/// RocketSim's ground throttle acceleration at speed `v`.
fn throttle_accel(v: f32) -> f32 {
    if v < 1400.0 {
        1600.0 * (1.0 - 0.9 * v / 1400.0)
    } else if v < 1410.0 {
        1600.0 * 0.1 * (1410.0 - v) / 10.0
    } else {
        0.0
    }
}

/// What driving straight with full throttle and boost would have done in `t` seconds from speed `v0` with `boost`
/// (RocketSim's ground model at 120 Hz): (speed, distance). `pickups`: (seconds in, amount) of pads picked up.
fn baseline(v0: f32, boost: f32, t: f32, pickups: &[(f32, f32)]) -> (f32, f32) {
    let (mut v, mut d, mut b) = (v0, 0.0, boost);
    let n = (t * 120.0 + 0.5) as i32;
    let mut next = 0;
    for i in 0..n {
        while next < pickups.len() && pickups[next].0 * 120.0 <= i as f32 + 0.5 {
            b = (b + pickups[next].1).min(100.0);
            next += 1;
        }
        let a = throttle_accel(v) + if b > 0.0 { 2975.0 / 3.0 } else { 0.0 };
        if b > 0.0 {
            b -= 100.0 / 3.0 / 120.0;
        }
        let v1 = (v + a / 120.0).min(2300.0);
        d += (v + v1) * 0.5 / 120.0;
        v = v1;
    }
    (v, d)
}

impl SpeedFlip {
    /// Ticks of history the reference speed looks back.
    fn ref_ticks(&self) -> f32 {
        (self.ref_window * 120.0 + 0.5).floor()
    }

    /// The speed history's ring (one entry per step since it was emptied: at any tick skip from 1 tick on, it
    /// holds every step of ref_window).
    fn hist_len(&self) -> usize {
        self.ref_ticks().max(0.0) as usize + 1
    }

    fn layout(&self) -> (usize, usize, usize) {
        let pickups = Clock::SIZE + Track::SIZE;
        let hist = pickups + 2 * MAX_PICKUPS;
        (pickups, hist, hist + self.hist_len())
    }

    /// Wheels down on the floor: 3+ wheels touching, low, roof roughly up (on_ground is also true on the ball).
    fn on_floor(&self, c: &CarSnapshot) -> bool {
        c.on_ground && c.body.pos.z < self.floor_max_z && c.body.up().z > self.floor_min_up_z
    }

    /// The height of the hitbox's lowest corner above the floor.
    fn lowest_corner_z(c: &Body) -> f32 {
        let centre = c.pos + c.rot * HITBOX_OFFSET;
        let half = HITBOX_SIZE * 0.5;
        let dz = c.rot.x_axis.z.abs() * half.x
            + c.rot.y_axis.z.abs() * half.y
            + c.rot.z_axis.z.abs() * half.z;
        centre.z - dz
    }

    /// The highest speed in the history within ref_window of `now` (the latest entry is the step before).
    fn ref_speed(&self, tr: &Track, hist: &[f32], step: f32) -> f32 {
        let len = hist.len() as i32;
        let mut best = 0.0f32;
        for j in 1..=tr.hist_count {
            if j as f32 * step > self.ref_ticks() {
                break;
            }
            best = best.max(hist[(tr.hist_head - j).rem_euclid(len) as usize]);
        }
        best
    }

    fn accumulate(&self, tr: &mut Track, a: &CarSnapshot, b: &CarSnapshot) {
        let r = local_rotation(&a.body.rot, &b.body.rot);
        tr.pitch_rot += r.y;
        tr.max_nose = tr
            .max_nose
            .max(deg(b.body.forward().z.abs().min(1.0).asin()));
        if !self.on_floor(b) && Self::lowest_corner_z(&b.body) < self.body_floor_margin {
            tr.body_tilt = tr.body_tilt.max(deg(b.body.up().z.clamp(-1.0, 1.0).acos()));
        }
    }

    /// The pay point: the speed, direction and distance measured from car state `c` at `tick`.
    fn capture(&self, tr: &mut Track, c: &CarSnapshot, tick: f32, pickups: &[(f32, f32)]) {
        let v = Vec3A::new(c.body.vel.x, c.body.vel.y, 0.0);
        let hs = v.length();
        let along_end = v.dot(tr.dir0);
        tr.veer = if hs > 1.0 {
            deg((v.dot(tr.dir0) / hs).clamp(-1.0, 1.0).acos())
        } else {
            180.0
        };
        let pay_time = (tick - tr.base_tick) / 120.0;
        let pads = if self.pads_in_baseline { pickups } else { &[] };
        let (base_along, base_dist) = baseline(tr.ref_speed, tr.boost_jump, pay_time, pads);
        let mut dp = c.body.pos - tr.jump_pos;
        dp.z = 0.0;
        tr.gain = match self.pay_measure.as_str() {
            "lead" => dp.length() - base_dist,
            "lead_along" => dp.dot(tr.dir0) - base_dist,
            "speed_vs_boost" => along_end - base_along,
            _ => along_end - tr.ref_speed,
        };
        tr.captured = true;
    }

    /// The attempt ends unpaid (with the optional penalty for a failed forward diagonal).
    fn fail(&self, tr: &mut Track, why: i32) -> f32 {
        tr.phase = IDLE;
        if (why == SPEED_LOST || why == BAD_LANDING) && tr.dodge == DODGE_DIAG_FORWARD {
            -self.lost_penalty
        } else {
            0.0
        }
    }

    /// The pay point reached (or the car jumped again): check the speed kept and the direction, and pay.
    fn resolve(&self, tr: &mut Track, c: &CarSnapshot, now: f32, pickups: &[(f32, f32)]) -> f32 {
        if !tr.captured {
            self.capture(tr, c, now, pickups);
        }
        let kept = if tr.speed_land > 1.0 {
            tr.min_speed_settle / tr.speed_land
        } else {
            1.0
        };
        if kept < self.min_kept_frac {
            return self.fail(tr, SPEED_LOST);
        }
        if tr.veer > self.max_veer_deg {
            return self.fail(tr, OTHER);
        }
        let mut credit = self.max_credit
            * clamp01((tr.gain - self.min_gain) / (self.gain_full - self.min_gain).max(1.0));
        if self.max_per_episode > 0.0 {
            credit = credit.min(self.max_per_episode - tr.paid).max(0.0);
        }
        tr.paid += credit;
        tr.phase = IDLE;
        credit
    }

    #[allow(clippy::too_many_arguments)]
    fn update(
        &self,
        tr: &mut Track,
        pickups: &mut Vec<(f32, f32)>,
        hist: &[f32],
        prev: &GameSnapshot,
        cur: &GameSnapshot,
        events: &StepEvents,
        car: usize,
        now: f32,
    ) -> f32 {
        let (p, pp) = (&cur.cars[car], &prev.cars[car]);
        let touched = events.cars[car].touched_ball;
        let step = step_ticks(events);
        let last_tick = now - step; // The step before
        let mut reward = 0.0;
        let mut since_jump = (now - tr.jump_tick) / 120.0;

        // ---- 1. A jump from the floor (a new one ends a settle window early) ----
        if tr.phase == IDLE || (tr.phase == SETTLING && p.has_jumped && !pp.has_jumped) {
            if tr.phase == SETTLING {
                reward += self.resolve(tr, p, now, pickups); // Jumped again before the pay point
            }
            if self.on_floor(pp) && p.has_jumped && !pp.has_jumped && !p.is_demoed {
                let paid = tr.paid;
                *tr = Track {
                    phase: JUMPED,
                    jump_tick: now,
                    ref_speed: self.ref_speed(tr, hist, step).max(hspeed(pp.body.vel)),
                    boost_jump: pp.boost, // Pads picked up from now on: see pads_in_baseline
                    jump_pos: pp.body.pos,
                    base_tick: last_tick,
                    paid,
                    hist_count: tr.hist_count,
                    hist_head: tr.hist_head,
                    ..Track::default()
                };
                pickups.clear();
                let v = Vec3A::new(pp.body.vel.x, pp.body.vel.y, 0.0);
                tr.dir0 = if v.length() > 100.0 {
                    v.normalize()
                } else {
                    let f = Vec3A::new(pp.body.forward().x, pp.body.forward().y, 0.0);
                    if f.length() > 1e-3 {
                        f.normalize()
                    } else {
                        Vec3A::X
                    }
                };
            }
            since_jump = 0.0;
        }

        // Boost pads picked up during the attempt (for the boost-only comparison)
        if tr.phase >= JUMPED
            && !tr.captured
            && p.boost > pp.boost + 5.0
            && pickups.len() < MAX_PICKUPS
        {
            pickups.push(((now - tr.base_tick) / 120.0, p.boost - pp.boost));
        }

        // ---- 2. The dodge ----
        if tr.phase == JUMPED {
            let detail = events.detail(car);
            if touched {
                tr.phase = IDLE; // Hit the ball before dodging: no speed flip in progress
            } else if p.has_flipped && !pp.has_flipped {
                tr.dodge = classify_dodge(
                    detail.flip_rel_torque,
                    self.min_torque_side,
                    self.min_torque_forward,
                );
                if tr.dodge != DODGE_DIAG_FORWARD || since_jump > self.max_jump_to_dodge {
                    return reward + self.fail(tr, OTHER); // Not a forward diagonal, or a late dodge
                }
                tr.phase = FLIPPING;
                self.accumulate(tr, pp, p); // The rotation since the jump step
            } else if self.on_floor(p) && (since_jump > 0.1 || !p.has_jumped) {
                tr.phase = IDLE; // A plain jump, back on the floor
            } else if p.has_double_jumped || since_jump > 1.3 {
                tr.phase = IDLE; // A double jump, or no dodge at all
            } else {
                self.accumulate(tr, pp, p);
            }
            return reward;
        }

        // ---- 3./4. In the air until wheels-down ----
        if tr.phase == FLIPPING {
            self.accumulate(tr, pp, p);
            tr.captured = false;
            if self.cancel_shaping > 0.0
                && events.detail(car).is_flipping
                && tr.pitch_rot.abs() <= self.max_pitch_rotation
                && tr.max_nose <= self.max_nose_angle
                && (self.max_per_episode <= 0.0 || tr.paid < self.max_per_episode)
                && last_tick < now
            {
                reward += self.cancel_shaping * ((now - last_tick) / 120.0).min(0.1);
            }
            if p.on_ground
                && !self.on_floor(p)
                && p.body.pos.distance(cur.ball.pos) < self.ball_land_dist
            {
                return reward + self.fail(tr, OTHER); // Wheels on the ball
            }
            if touched {
                return reward + self.fail(tr, OTHER); // Hit the ball before the wheels were down again
            }
            if self.on_floor(p) {
                let land_tilt = deg(p.body.up().z.clamp(-1.0, 1.0).acos());
                tr.speed_land = hspeed(p.body.vel);
                if tr.pitch_rot.abs() > self.max_pitch_rotation || tr.max_nose > self.max_nose_angle
                {
                    return reward + self.fail(tr, OTHER); // The nose rotated over: no cancel
                }
                if land_tilt > self.max_land_tilt || tr.body_tilt > self.max_body_tilt {
                    return reward + self.fail(tr, BAD_LANDING);
                }
                tr.phase = SETTLING;
                tr.land_tick = now;
                tr.captured = false;
                tr.min_speed_settle = tr.speed_land;
                return reward;
            }
            if since_jump > self.max_air_time {
                return reward + self.fail(tr, BAD_LANDING); // Not back on the wheels in time
            }
            return reward;
        }

        // ---- 5./6. Settle on the floor, then pay ----
        if tr.phase == SETTLING {
            // A ball touch after wheels-down: pay from the step before the touch (the impact is not the flip's)
            if touched {
                if !tr.captured {
                    self.capture(tr, pp, last_tick, pickups);
                }
                return reward + self.resolve(tr, p, now, pickups);
            }
            tr.min_speed_settle = tr.min_speed_settle.min(hspeed(p.body.vel));
            if !tr.captured && (now - tr.base_tick) / 120.0 >= self.pay_window - 1e-4 {
                self.capture(tr, p, now, pickups);
            }
            if tr.captured && (now - tr.land_tick) / 120.0 >= self.settle_time {
                reward += self.resolve(tr, p, now, pickups);
            }
        }
        reward
    }
}

impl Reward for SpeedFlip {
    fn memory(&self) -> usize {
        self.layout().2
    }

    fn value_remembering(
        &self,
        prev: &GameSnapshot,
        cur: &GameSnapshot,
        events: &StepEvents,
        car: usize,
        memory: &mut [f32],
    ) -> f32 {
        let (pickups_at, hist_at, _) = self.layout();
        let (head, hist) = memory.split_at_mut(hist_at);
        let (clock_mem, rest) = head.split_at_mut(Clock::SIZE);
        let (track_mem, pickup_mem) = rest.split_at_mut(pickups_at - Clock::SIZE);
        let mut clock = Clock::load(clock_mem);
        clock.advance(events); // Nothing to set up: the history starts empty, the step before is `prev`
        let mut tr = Track::load(track_mem);
        let mut pickups: Vec<(f32, f32)> = (0..(tr.pickups.max(0) as usize).min(MAX_PICKUPS))
            .map(|i| (pickup_mem[2 * i], pickup_mem[2 * i + 1]))
            .collect();
        let p = &cur.cars[car];
        let reward = if p.is_demoed {
            // Demolished: whatever was in progress is over, and the speed history starts again
            let r = if tr.phase >= FLIPPING {
                self.fail(&mut tr, BAD_LANDING)
            } else {
                0.0
            };
            tr.phase = IDLE;
            tr.hist_count = 0;
            r
        } else {
            let r = self.update(
                &mut tr,
                &mut pickups,
                hist,
                prev,
                cur,
                events,
                car,
                clock.now,
            );
            // This step's speed joins the history (a jump seen now compares with the steps before it)
            let len = hist.len() as i32;
            hist[tr.hist_head.rem_euclid(len) as usize] = hspeed(p.body.vel);
            tr.hist_head = (tr.hist_head + 1).rem_euclid(len);
            tr.hist_count = (tr.hist_count + 1).min(len);
            r
        };
        tr.pickups = pickups.len() as i32;
        for (i, (t, amount)) in pickups.iter().enumerate() {
            pickup_mem[2 * i] = *t;
            pickup_mem[2 * i + 1] = *amount;
        }
        clock.store(clock_mem);
        tr.store(track_mem);
        finite_or_zero(reward)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::kit::testing::{game, quiet};

    #[test]
    fn listed_with_every_setting_optional_and_described() {
        assert_listed("speed_flip", 25);
    }

    fn reward() -> SpeedFlip {
        serde_json::from_value(serde_json::json!({})).unwrap()
    }

    /// A car driving at `speed` along +y (Blue's view, mirrored for Orange), ball far away.
    fn driving(team: Team, y: f32, speed: f32) -> GameSnapshot {
        let m = |v: Vec3A| as_blue(team, v);
        let mut g = game(
            m(Vec3A::new(3000.0, 3000.0, 93.0)),
            &[(team, m(Vec3A::new(0.0, y, 17.0)))],
        );
        let yaw = if team == Team::Blue { 0.5 } else { -0.5 };
        g.cars[0].body.rot = Mat3A::from_rotation_z(yaw * std::f32::consts::PI);
        g.cars[0].body.vel = m(Vec3A::new(0.0, speed, 0.0));
        g
    }

    /// Steps the reward through `games`; the dodge's torque is `torque` on every step.
    fn run(r: &SpeedFlip, games: &[GameSnapshot], torque: Vec3A) -> (Vec<f32>, Vec<f32>) {
        let mut memory = vec![0.0; r.memory()];
        let mut ev = quiet(1);
        ev.step_seconds = 4.0 / 120.0;
        ev.details[0].flip_rel_torque = torque;
        let out = games
            .windows(2)
            .map(|pair| r.value_remembering(&pair[0], &pair[1], &ev, 0, &mut memory))
            .collect();
        (out, memory)
    }

    /// Drive, jump, dodge (cancelled: the car stays level), land at 2200 uu/s, keep going: a speed flip that
    /// gets farther than boosting from 1000 uu/s would.
    fn speed_flip(team: Team) -> Vec<GameSnapshot> {
        let mut games = Vec::new();
        let (mut y, dt) = (-3000.0, 4.0 / 120.0);
        let mut push = |y: f32, speed: f32, air: bool, flipped: bool| {
            let mut g = driving(team, y, speed);
            let c = &mut g.cars[0];
            c.on_ground = !air;
            c.has_jumped = air;
            c.has_flipped = flipped;
            if air {
                c.body.pos.z = 80.0;
            }
            games.push(g);
        };
        for _ in 0..10 {
            push(y, 1000.0, false, false);
            y += 1000.0 * dt;
        }
        push(y, 1000.0, true, false); // The jump
        y += 1000.0 * dt;
        for i in 0..36 {
            push(y, 2200.0, true, i >= 2); // The dodge two steps later; 1.2 s in the air, level
            y += 2200.0 * dt;
        }
        for _ in 0..15 {
            push(y, 2200.0, false, false); // Wheels down, keeping the speed
            y += 2200.0 * dt;
        }
        games
    }

    #[test]
    fn a_cancelled_forward_diagonal_that_keeps_its_speed_is_paid_and_orange_the_same() {
        let r = reward();
        let (blue, _) = run(&r, &speed_flip(Team::Blue), Vec3A::new(0.7, 0.7, 0.0));
        let paid: f32 = blue.iter().sum();
        assert!(paid > 0.5 && paid <= 1.0, "{blue:?}");
        assert_eq!(blue.iter().filter(|&&v| v > 0.0).count(), 1, "paid once");
        let (orange, _) = run(&r, &speed_flip(Team::Orange), Vec3A::new(0.7, 0.7, 0.0));
        assert!((paid - orange.iter().sum::<f32>()).abs() < 1e-4);
    }

    #[test]
    fn a_front_flip_or_driving_alone_earns_nothing_and_memory_starts_again() {
        let r = reward();
        let (front, mut memory) = run(&r, &speed_flip(Team::Blue), Vec3A::new(0.0, 1.0, 0.0));
        assert!(front.iter().all(|&v| v == 0.0), "{front:?}");
        let drive: Vec<_> = (0..20)
            .map(|i| driving(Team::Blue, -3000.0 + 33.0 * i as f32, 1000.0))
            .collect();
        assert!(run(&r, &drive, Vec3A::ZERO).0.iter().all(|&v| v == 0.0));
        memory.fill(0.0);
        let mut ev = quiet(1);
        ev.step_seconds = 4.0 / 120.0;
        r.value_remembering(&drive[0], &drive[1], &ev, 0, &mut memory);
        assert_eq!(Track::load(&memory[Clock::SIZE..]).hist_count, 1);
        let capped = SpeedFlip {
            max_per_episode: 0.25,
            ..reward()
        };
        let (out, _) = run(&capped, &speed_flip(Team::Blue), Vec3A::new(0.7, 0.7, 0.0));
        assert!(
            (out.iter().sum::<f32>() - 0.25).abs() < 1e-6,
            "capped: {out:?}"
        );
        assert!(
            serde_json::from_value::<SpeedFlip>(serde_json::json!({ "pay_measure": "fast" }))
                .is_err()
        );
    }
}
