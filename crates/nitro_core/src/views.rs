//! Other bots' views of a game: the observation and action mask a bot trained in other games
//! (another observation, room or game size) sees from a car of this one. A multi-model bot's models each
//! keep their own view; the pool writes them on request (`EnvPool::views_into`) and the RLBot bot builds
//! them from the packet the same way.

use serde::Deserialize;

use crate::actions::{ActionTable, MaskRule};
use crate::obs::ObsKind;
use crate::snapshot::{Body, Controls, GameSnapshot};

/// The game a bot learned to see: its observation, its game's size and its action mask rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ViewConfig {
    pub observation: ObsKind,
    pub team_size: usize,
    pub opponents: bool,
    /// Players per team its observation has places for; 0 = its game's own.
    #[serde(default)]
    pub obs_team_size: usize,
    pub action_masks: MaskRule,
    /// Its ball prediction's times in seconds, at the game's tick skip; none by default.
    #[serde(default)]
    pub ball_prediction: crate::ball_prediction::Times,
}

impl ViewConfig {
    /// Numbers in its observation; an error for settings no observation has.
    pub fn obs_size(&self) -> Result<usize, String> {
        if !self.ball_prediction.is_empty() && !self.observation.pads() {
            return Err("ball prediction needs the native or nitro_v2 observation".to_owned());
        }
        Ok(self
            .observation
            .checked_layout(self.team_size, self.opponents, self.obs_team_size)?
            .with_ball_prediction(self.ball_prediction.len())
            .size)
    }

    /// The decisions of `tick_skip` ticks its ball prediction reads (empty = none).
    pub fn prediction(&self, tick_skip: u32) -> Result<Vec<u32>, String> {
        crate::ball_prediction::decisions(&self.ball_prediction, tick_skip)
    }

    /// Why a game of `team_size` a side (with or without `opponents`) cannot give this view, if it cannot.
    pub fn problem(&self, team_size: usize, opponents: bool) -> Option<String> {
        if let Err(err) = self.obs_size() {
            return Some(err);
        }
        if self.team_size > team_size || (self.opponents && !opponents) {
            return Some(format!(
                "a bot that played {} a side{} cannot see a game of {team_size} a side{}",
                self.team_size,
                if self.opponents {
                    " with opponents"
                } else {
                    " alone"
                },
                if opponents {
                    " with opponents"
                } else {
                    " alone"
                },
            ));
        }
        None
    }

    /// Why a match with `same` players on the car's team and `other` on the other cannot give this view, if
    /// it cannot (the RLBot bot coasts then). A view keeps the nearest `team_size` a side; one with room
    /// (`obs_team_size`) pads fewer, one without needs them all.
    pub fn match_problem(&self, same: usize, other: usize) -> Option<String> {
        let short = same < self.team_size || (self.opponents && other < self.team_size);
        (self.obs_team_size == 0 && short).then(|| {
            format!(
                "one of its models learned {0}v{1} with no room for fewer players, and this match has {same} \
                 on its team and {other} on the other",
                self.team_size,
                if self.opponents { self.team_size } else { 0 },
            )
        })
    }

    /// Car `me`'s view of `snap` into `obs` (`obs_size` numbers) and its mask of `actions` into `mask`
    /// (`actions.len()` entries): `me`, its nearest-to-the-ball teammates and opponents, as `view_of` keeps them.
    /// With ball prediction, the bodies of `path` (a decision apart) at `steps` (`prediction`).
    #[allow(clippy::too_many_arguments)]
    pub fn write(
        &self,
        snap: &GameSnapshot,
        me: usize,
        prev_action: Controls,
        steps: &[u32],
        path: &[Body],
        actions: &ActionTable,
        obs: &mut [f32],
        mask: &mut [u8],
    ) {
        let (view, at) = view_of(snap, me, self.team_size, self.opponents);
        let mut predicted = Vec::new();
        if !steps.is_empty() {
            crate::ball_prediction::pick(path, steps, &mut predicted);
        }
        self.observation.build_predicted(
            &view,
            at,
            prev_action,
            self.obs_team_size,
            &predicted,
            obs,
        );
        actions.write_mask(self.action_masks, &snap.cars[me], mask);
    }
}

/// The game as a specialist that played `team_size` a side (with or without opponents) sees it from car
/// `me`: `me`, its nearest-to-the-ball teammates and opponents, in the game's car order; and where `me` is
/// in it.
pub fn view_of(
    snap: &GameSnapshot,
    me: usize,
    team_size: usize,
    opponents: bool,
) -> (GameSnapshot, usize) {
    let team = snap.cars[me].team;
    let by_nearness = |same: bool| {
        let mut cars: Vec<usize> = (0..snap.cars.len())
            .filter(|&c| c != me && (snap.cars[c].team == team) == same)
            .collect();
        cars.sort_by(|&a, &b| {
            let d = |c: usize| snap.cars[c].body.pos.distance(snap.ball.pos);
            d(a).total_cmp(&d(b))
        });
        cars
    };
    let mut keep = vec![me];
    keep.extend(by_nearness(true).into_iter().take(team_size - 1));
    if opponents {
        keep.extend(by_nearness(false).into_iter().take(team_size));
    }
    keep.sort_unstable();
    let mut view = snap.clone();
    view.cars = keep.iter().map(|&c| snap.cars[c]).collect();
    let at = keep.iter().position(|&c| c == me).expect("me is kept");
    (view, at)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::snapshot::{CarSnapshot, Team};

    fn game(per_side: usize) -> GameSnapshot {
        let mut snap = GameSnapshot::default();
        for i in 0..per_side * 2 {
            let mut car = CarSnapshot {
                team: if i % 2 == 0 { Team::Blue } else { Team::Orange },
                ..CarSnapshot::default()
            };
            car.body.pos.x = 100.0 * i as f32;
            car.body.pos.y = -300.0 + 50.0 * i as f32;
            car.body.pos.z = 17.0;
            car.on_ground = true;
            car.boost = 0.1 * i as f32;
            snap.cars.push(car);
        }
        snap
    }

    fn view(observation: ObsKind, team_size: usize, opponents: bool, room: usize) -> ViewConfig {
        ViewConfig {
            observation,
            team_size,
            opponents,
            obs_team_size: room,
            action_masks: MaskRule::Standard,
            ball_prediction: Default::default(),
        }
    }

    #[test]
    fn the_games_own_view_is_its_observation() {
        let snap = game(2);
        let table = ActionTable::new();
        for kind in [ObsKind::Native, ObsKind::NitroV2] {
            let own = view(kind, 2, true, 0);
            let size = own.obs_size().unwrap();
            for me in 0..4 {
                let (mut obs, mut mask) = (vec![0.0; size], vec![0; table.len()]);
                own.write(
                    &snap,
                    me,
                    Controls::default(),
                    &[],
                    &[],
                    &table,
                    &mut obs,
                    &mut mask,
                );
                let mut direct = vec![0.0; size];
                kind.build(&snap, me, Controls::default(), 0, &mut direct);
                assert_eq!(obs, direct, "{kind:?} car {me}");
                let mut direct_mask = vec![0; table.len()];
                table.write_mask(MaskRule::Standard, &snap.cars[me], &mut direct_mask);
                assert_eq!(mask, direct_mask);
            }
        }
    }

    #[test]
    fn a_lone_bots_view_leaves_the_other_cars_out() {
        let snap = game(1);
        let table = ActionTable::new();
        let alone = view(ObsKind::NitroV2, 1, false, 0);
        let size = alone.obs_size().unwrap();
        assert!(size < view(ObsKind::NitroV2, 1, true, 0).obs_size().unwrap());
        let (mut obs, mut mask) = (vec![0.0; size], vec![0; table.len()]);
        alone.write(
            &snap,
            1,
            Controls::default(),
            &[],
            &[],
            &table,
            &mut obs,
            &mut mask,
        );
        let mut solo = snap.clone();
        solo.cars = vec![snap.cars[1]];
        let mut direct = vec![0.0; size];
        ObsKind::NitroV2.build(&solo, 0, Controls::default(), 0, &mut direct);
        assert_eq!(obs, direct, "the Orange car alone, as it trained");
    }

    /// A model trained 2v2 without room, in a 1v1 match of a bot with room, must not make the RLBot bot
    /// panic: every match a view accepts is written without a panic; the rest are named.
    #[test]
    fn a_match_too_small_for_a_view_is_named_and_every_other_is_written() {
        let table = ActionTable::new();
        assert!(
            view(ObsKind::Native, 2, true, 0)
                .match_problem(1, 1)
                .is_some()
        );
        assert_eq!(view(ObsKind::Native, 2, true, 2).match_problem(1, 1), None);
        for kind in [ObsKind::Native, ObsKind::NitroV2] {
            for team_size in 1..=3 {
                for opponents in [true, false] {
                    for room in [0, 3] {
                        let seen = view(kind, team_size, opponents, room);
                        let size = seen.obs_size().unwrap();
                        for same in 1..=3 {
                            for other in 0..=3 {
                                if seen.match_problem(same, other).is_some() {
                                    continue;
                                }
                                let mut snap = GameSnapshot::default();
                                for i in 0..same + other {
                                    let mut car = CarSnapshot {
                                        team: if i < same { Team::Blue } else { Team::Orange },
                                        ..CarSnapshot::default()
                                    };
                                    car.body.pos.x = 100.0 * i as f32;
                                    snap.cars.push(car);
                                }
                                let (mut obs, mut mask) = (vec![0.0; size], vec![0; table.len()]);
                                seen.write(
                                    &snap,
                                    0,
                                    Controls::default(),
                                    &[],
                                    &[],
                                    &table,
                                    &mut obs,
                                    &mut mask,
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn views_a_game_cannot_give_are_named() {
        assert!(view(ObsKind::Native, 2, true, 0).problem(1, true).is_some());
        assert!(
            view(ObsKind::Native, 1, true, 0)
                .problem(1, false)
                .is_some()
        );
        assert!(
            view(ObsKind::GigalearnAdvanced, 1, true, 2)
                .problem(2, true)
                .is_some()
        );
        assert_eq!(view(ObsKind::Native, 1, false, 0).problem(2, true), None);
        assert_eq!(view(ObsKind::Native, 1, true, 3).problem(1, true), None);
    }
}
