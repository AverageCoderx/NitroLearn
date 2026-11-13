//! Out of position: a penalty for being upfield of both the ball and the opponent challenging for it.

use crate::custom::prelude::*;

const BALL_RADIUS: f32 = 92.75;

/// Every step an opponent is within `near_radii` ball radii (8, about 740 uu) of the ball and this car is
/// further toward the opponents' net than the ball and every such opponent (by more than `margin`): −1.
/// 0 otherwise, with no opponent near the ball, or while demolished. Teammates do not count.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutOfPosition {
    /// An opponent this many ball radii (centre to centre) from the ball or closer is challenging for it.
    #[serde(default = "eight")]
    pub near_radii: f32,
    /// The car must be this far (uu) past the ball and the opponents to count as in front of them.
    #[serde(default = "no_margin")]
    pub margin: f32,
}

fn eight() -> f32 {
    8.0
}

fn no_margin() -> f32 {
    0.0
}

impl Reward for OutOfPosition {
    fn every_step(&self) -> bool {
        true
    }

    fn value(&self, _prev: &GameSnapshot, cur: &GameSnapshot, _: &StepEvents, car: usize) -> f32 {
        let me = &cur.cars[car];
        if me.is_demoed {
            return 0.0;
        }
        // How far toward the opponents' net each thing is, for this car's team
        let upfield = |pos: Vec3A| pos.y * attack_sign(me.team);
        let mine = upfield(me.body.pos) - self.margin;
        let reach = self.near_radii * BALL_RADIUS;
        let mut near = opponents(cur, car)
            .map(|o| &cur.cars[o])
            .filter(|o| !o.is_demoed && o.body.pos.distance(cur.ball.pos) <= reach)
            .peekable();
        if near.peek().is_none() || mine <= upfield(cur.ball.pos) {
            return 0.0;
        }
        if near.all(|o| mine > upfield(o.body.pos)) {
            -1.0
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::kit::testing::{game, quiet};

    fn reward() -> OutOfPosition {
        serde_json::from_value(serde_json::json!({})).unwrap()
    }

    /// The ball at y = 0; Blue's car at `blue_y`, Orange's at (`orange_x`, `orange_y`), both on the floor.
    fn at(blue_y: f32, orange_x: f32, orange_y: f32) -> GameSnapshot {
        game(
            Vec3A::new(0.0, 0.0, 93.0),
            &[
                (Team::Blue, Vec3A::new(0.0, blue_y, 17.0)),
                (Team::Orange, Vec3A::new(orange_x, orange_y, 17.0)),
            ],
        )
    }

    #[test]
    fn listed_with_every_setting_optional_and_described() {
        let info = crate::custom::REWARDS
            .iter()
            .find(|i| i.name == "out_of_position")
            .expect("built in");
        let names: Vec<_> = info.params.iter().map(|p| p.name).collect();
        assert_eq!(names, ["near_radii", "margin"]);
        assert!(info.every_step && info.params.iter().all(|p| p.optional && !p.doc.is_empty()));
    }

    #[test]
    fn upfield_of_the_ball_and_a_challenging_opponent_costs_one_for_either_team() {
        let r = reward();
        let pay = |g: &GameSnapshot, car: usize| r.value(g, g, &quiet(2), car);
        // Blue attacks +y: Blue past the ball, Orange 300 uu from the ball on its own side of it
        let g = at(1000.0, 0.0, 300.0);
        assert_eq!(pay(&g, 0), -1.0);
        // Orange is the one upfield (toward -y) when mirrored
        let mirrored = game(
            Vec3A::new(0.0, 0.0, 93.0),
            &[
                (Team::Blue, Vec3A::new(0.0, -300.0, 17.0)),
                (Team::Orange, Vec3A::new(0.0, -1000.0, 17.0)),
            ],
        );
        assert_eq!(pay(&mirrored, 1), -1.0);
        assert_eq!(pay(&mirrored, 0), 0.0, "Blue is behind the ball");
    }

    #[test]
    fn nothing_behind_the_ball_behind_the_opponent_or_with_nobody_near_the_ball() {
        let r = reward();
        let pay = |g: &GameSnapshot| r.value(g, g, &quiet(2), 0);
        assert_eq!(pay(&at(-1000.0, 0.0, 300.0)), 0.0, "behind the ball");
        assert_eq!(
            pay(&at(500.0, 0.0, 700.0)),
            0.0,
            "past the ball but not past the opponent"
        );
        assert_eq!(
            pay(&at(1000.0, 0.0, 800.0)),
            0.0,
            "the opponent is 8.6 ball radii away"
        );
        assert_eq!(
            pay(&at(1000.0, 0.0, 700.0)),
            -1.0,
            "7.5 ball radii: challenging"
        );
        let wider = OutOfPosition {
            near_radii: 12.0,
            ..reward()
        };
        let g = at(1500.0, 0.0, 1000.0);
        assert_eq!(wider.value(&g, &g, &quiet(2), 0), -1.0);
        let margin = OutOfPosition {
            margin: 800.0,
            ..reward()
        };
        let g = at(1000.0, 0.0, 300.0);
        assert_eq!(
            margin.value(&g, &g, &quiet(2), 0),
            0.0,
            "only 700 uu past the opponent: within the margin"
        );
    }
}
