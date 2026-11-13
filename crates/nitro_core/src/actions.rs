//! The 90-action lookup table and action masks.
//! The 90-action table is adapted from LookupTableAction in RLGym (rlgym-rocket-league 2.0.1,
//! rlgym/rocket_league/action_parsers/lookup_table_action.py, Copyright 2021 Lucas Emery).
//! Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0; the
//! full text is in LICENSES/Apache-2.0.txt).
//! Modified: written in Rust; each action is a `Controls` value; the action masks added.

use serde::Deserialize;

use crate::snapshot::{CarSnapshot, Controls};

/// Size of the action table.
pub const NUM_ACTIONS: usize = 90;
const NUM_GROUND_ACTIONS: usize = 24;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaskRule {
    /// Ground actions on the ground, air actions in the air, jumps only while a jump/flip is available
    /// (or the car is turtled), no boosting actions with an empty tank.
    #[default]
    Standard,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Ground,
    Air,
}

pub struct ActionTable {
    controls: Vec<Controls>,
    /// Every mask there can be, by situation: a mask depends on nothing else, so writing one is a copy.
    masks: Vec<u8>,
}

/// Which actions a car may take depends only on these three facts about it.
#[derive(Clone, Copy)]
struct Situation {
    on_ground: bool,
    can_jump: bool,
    has_boost: bool,
}

const SITUATIONS: usize = 8;

impl Situation {
    fn of(car: &CarSnapshot) -> Self {
        Self {
            on_ground: car.on_ground,
            can_jump: car.can_flip || car.is_turtled(),
            has_boost: car.boost > 0.0,
        }
    }

    fn all() -> impl Iterator<Item = Self> {
        (0..SITUATIONS).map(|i| Self {
            on_ground: i & 1 != 0,
            can_jump: i & 2 != 0,
            has_boost: i & 4 != 0,
        })
    }

    fn index(self) -> usize {
        usize::from(self.on_ground)
            | usize::from(self.can_jump) << 1
            | usize::from(self.has_boost) << 2
    }
}

impl Default for ActionTable {
    fn default() -> Self {
        Self::new()
    }
}

impl ActionTable {
    /// The 90-action table.
    pub fn new() -> Self {
        let mut controls = Vec::with_capacity(NUM_ACTIONS);
        const AXIS: [f32; 3] = [-1.0, 0.0, 1.0];
        for throttle in AXIS {
            for steer in AXIS {
                for boost in [false, true] {
                    for handbrake in [false, true] {
                        // Boosting already means full throttle.
                        if boost && throttle != 1.0 {
                            continue;
                        }
                        controls.push(Controls {
                            throttle,
                            steer,
                            yaw: steer,
                            boost,
                            handbrake,
                            ..Controls::default()
                        });
                    }
                }
            }
        }
        for pitch in AXIS {
            for yaw in AXIS {
                for roll in AXIS {
                    for jump in [false, true] {
                        for boost in [false, true] {
                            // Flips only need roll for sideways; pure-yaw air actions duplicate ground ones.
                            if (jump && yaw != 0.0) || (pitch == 0.0 && roll == 0.0 && !jump) {
                                continue;
                            }
                            // Handbrake on directional jumps allows wavedashes.
                            let handbrake = jump && (pitch != 0.0 || yaw != 0.0 || roll != 0.0);
                            controls.push(Controls {
                                throttle: f32::from(u8::from(boost)),
                                steer: yaw,
                                pitch,
                                yaw,
                                roll,
                                jump,
                                boost,
                                handbrake,
                            });
                        }
                    }
                }
            }
        }
        assert_eq!(controls.len(), NUM_ACTIONS);
        let kinds: Vec<Kind> = (0..NUM_ACTIONS)
            .map(|i| {
                if i < NUM_GROUND_ACTIONS {
                    Kind::Ground
                } else {
                    Kind::Air
                }
            })
            .collect();

        let throttle_is_boost = |c: &Controls| c.throttle == f32::from(u8::from(c.boost));
        let len = controls.len();
        // The air part of the table skips pure-yaw actions; these ground actions are exactly them.
        let pure_yaw = |c: &Controls| throttle_is_boost(c) && !c.handbrake;
        let ground: Vec<bool> = kinds.iter().map(|&k| k == Kind::Ground).collect();
        let air: Vec<bool> = controls
            .iter()
            .zip(&kinds)
            .map(|(c, &kind)| match kind {
                Kind::Air => !c.jump,
                Kind::Ground => pure_yaw(c),
            })
            .collect();
        let mut masks = vec![0u8; SITUATIONS * len];
        for situation in Situation::all() {
            let start = situation.index() * len;
            let out = &mut masks[start..start + len];
            let base = if situation.on_ground { &ground } else { &air };
            for i in 0..len {
                let (jump, boost) = (controls[i].jump, controls[i].boost);
                let allowed =
                    (base[i] || (jump && situation.can_jump)) && (situation.has_boost || !boost);
                out[i] = u8::from(allowed);
            }
        }
        Self { controls, masks }
    }

    pub fn len(&self) -> usize {
        self.controls.len()
    }

    pub fn is_empty(&self) -> bool {
        self.controls.is_empty()
    }

    pub fn controls(&self, action: usize) -> Controls {
        self.controls[action]
    }

    /// Tests only: for every action, the action that does the same thing mirrored left-right (steer, yaw
    /// and roll reversed). The table is closed under mirroring, so this is an involution (tested).
    #[cfg(test)]
    pub(crate) fn mirror_permutation(&self) -> Vec<usize> {
        self.controls
            .iter()
            .map(|c| {
                let mirrored = Controls {
                    steer: -c.steer,
                    yaw: -c.yaw,
                    roll: -c.roll,
                    ..*c
                };
                self.controls
                    .iter()
                    .position(|other| *other == mirrored)
                    .expect("the action table is closed under mirroring")
            })
            .collect()
    }

    /// Writes 1 for every action `car` may take now, 0 otherwise (`out.len() == self.len()`).
    pub fn write_mask(&self, rule: MaskRule, car: &CarSnapshot, out: &mut [u8]) {
        let len = self.len();
        let MaskRule::Standard = rule;
        let start = Situation::of(car).index() * len;
        out.copy_from_slice(&self.masks[start..start + len]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::snapshot::Body;
    use rocketsim::{Mat3A, Vec3A};

    fn car(on_ground: bool, can_flip: bool, boost: f32) -> CarSnapshot {
        CarSnapshot {
            on_ground,
            can_flip,
            boost,
            body: Body {
                pos: Vec3A::new(0.0, 0.0, 17.0),
                ..Body::RESTING
            },
            ..CarSnapshot::default()
        }
    }

    fn allowed(table: &ActionTable, rule: MaskRule, car: &CarSnapshot) -> Vec<usize> {
        let mut mask = [0u8; NUM_ACTIONS];
        table.write_mask(rule, car, &mut mask);
        (0..NUM_ACTIONS).filter(|&i| mask[i] == 1).collect()
    }

    #[test]
    fn table_has_90_distinct_actions_with_24_ground_ones() {
        let table = ActionTable::new();
        for i in 0..NUM_ACTIONS {
            for j in (i + 1)..NUM_ACTIONS {
                assert_ne!(
                    table.controls(i),
                    table.controls(j),
                    "actions {i} and {j} are equal"
                );
            }
        }
        assert!((0..24).all(|i| !table.controls(i).jump && table.controls(i).pitch == 0.0));
        assert_eq!(
            (0..NUM_ACTIONS).filter(|&i| table.controls(i).jump).count(),
            18
        );
    }

    #[test]
    fn ground_with_flip_allows_ground_and_jump_actions() {
        let table = ActionTable::new();
        let ok = allowed(&table, MaskRule::Standard, &car(true, true, 50.0));
        assert_eq!(ok.len(), 24 + 18);
        assert!(ok.iter().all(|&i| i < 24 || table.controls(i).jump));
    }

    #[test]
    fn air_without_flip_allows_air_control_only() {
        let table = ActionTable::new();
        let ok = allowed(&table, MaskRule::Standard, &car(false, false, 50.0));
        assert_eq!(ok.len(), 48 + 6);
        assert!(ok.iter().all(|&i| !table.controls(i).jump));
        assert!(ok.contains(&24), "action 24 must be usable in the air");
        // The ground actions allowed in the air are exactly the pure-yaw air actions the table skips.
        for &i in ok.iter().filter(|&&i| i < 24) {
            let c = table.controls(i);
            assert!(!c.handbrake && c.throttle == f32::from(u8::from(c.boost)) && c.steer == c.yaw);
        }
    }

    #[test]
    fn empty_tank_removes_boosting_actions() {
        let table = ActionTable::new();
        for on_ground in [true, false] {
            let ok = allowed(&table, MaskRule::Standard, &car(on_ground, true, 0.0));
            assert!(ok.iter().all(|&i| !table.controls(i).boost));
        }
    }

    #[test]
    fn mirroring_pairs_every_action_and_undoes_itself() {
        let table = ActionTable::new();
        let mirror = table.mirror_permutation();
        for (i, &j) in mirror.iter().enumerate() {
            assert_eq!(mirror[j], i, "action {i} -> {j} -> {}", mirror[j]);
            let (a, b) = (table.controls(i), table.controls(j));
            assert_eq!((a.steer, a.yaw, a.roll), (-b.steer, -b.yaw, -b.roll));
            assert_eq!(
                (a.throttle, a.pitch, a.jump, a.boost, a.handbrake),
                (b.throttle, b.pitch, b.jump, b.boost, b.handbrake)
            );
        }
        // Straight-ahead actions are their own mirror image; turning ones are not.
        assert!(mirror.iter().enumerate().filter(|&(i, &j)| i == j).count() > 0);
        assert!(mirror.iter().enumerate().filter(|&(i, &j)| i != j).count() > 0);
    }

    #[test]
    fn turtled_car_may_jump() {
        let table = ActionTable::new();
        let mut upside_down = car(false, false, 50.0);
        upside_down.body.rot = Mat3A::from_rotation_x(std::f32::consts::PI);
        upside_down.body.pos.z = 32.0;
        assert!(upside_down.is_turtled());
        let ok = allowed(&table, MaskRule::Standard, &upside_down);
        assert!(ok.iter().any(|&i| table.controls(i).jump));
    }

    /// The car states of `testdata/action_table_original.txt`, by its names.
    fn reference_states() -> Vec<(String, CarSnapshot)> {
        let mut out = Vec::new();
        for on_ground in [true, false] {
            for can_flip in [true, false] {
                for boost in [0.0f32, 50.0] {
                    for turtled in [false, true] {
                        if turtled && on_ground {
                            continue;
                        }
                        let mut state = car(on_ground, can_flip, boost);
                        if turtled {
                            state.body.rot = Mat3A::from_rotation_x(std::f32::consts::PI);
                            state.body.pos.z = 32.0;
                        }
                        out.push((
                            format!(
                                "ground={on_ground} flip={can_flip} boost={boost} turtled={turtled}"
                            ),
                            state,
                        ));
                    }
                }
            }
        }
        out
    }

    /// The same listing as the reference file.
    fn listing(table: &ActionTable) -> String {
        let mut text = String::new();
        for i in 0..NUM_ACTIONS {
            text += &format!("action {i} {:?}\n", table.controls(i).to_floats());
        }
        for rule in [MaskRule::Standard] {
            for (name, state) in reference_states() {
                let mut mask = vec![0u8; table.len()];
                table.write_mask(rule, &state, &mut mask);
                let bits: String = mask[..NUM_ACTIONS]
                    .iter()
                    .map(|&b| if b == 1 { '1' } else { '0' })
                    .collect();
                text += &format!("mask {rule:?} {name} {bits}\n");
            }
        }
        text
    }

    #[test]
    fn table_and_masks_match_the_pinned_listing() {
        // A Windows checkout may turn the file's line ends into CRLF; the listing is the same.
        let reference = include_str!("../testdata/action_table_original.txt").replace("\r\n", "\n");
        assert_eq!(listing(&ActionTable::new()), reference);
    }
}
