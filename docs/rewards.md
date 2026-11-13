# The rewards that come with NitroLearn

Every reward here is in each bot's **+ Add reward** list in the dashboard, and its **See more** button says
what it pays, when to use it, how to set it and what to watch out for. In a settings file a reward is a
`[[rewards]]` block with its `name`, a `weight` and its own settings (`python/nitro/config/rewards.py`
explains the format).

- **Built in** (13): the everyday rewards, part of NitroLearn itself.
- **Moves** (17): rewards for trick moves (air dribbles, flip resets, flicks, speed flips, dashes). Their
  files' headers have a `//! group: moves` line, which lists them as their own group.
- **More rewards** (67): everything else in `crates/nitro_core/src/custom/library/`.
- `example_ball_height` (`crates/nitro_core/src/custom/example.rs`): the example to copy for a reward of
  your own (`docs/your_own_rewards.md`).

Eight **reward sets** (`configs/reward-sets/`) load several of these at once with weights that suit one
move: `air_dribble_bot`, `breezi_bot`, `flip_reset_simple`, `mawkzy_flick_bot`, `musty_bot`, `raze_bot`,
`reset_musty_bot` and `speed_flip_bot`.

## Built in

| Reward | Pays |
|---|---|
| `air` | every step the car is off the ground |
| `face_ball` | every step: how straight the car's nose points at the ball |
| `velocity_player_to_ball` | every step: the car's speed towards the ball |
| `velocity_ball_to_goal` | every step: the ball's speed towards the opponents' net |
| `touch_ball` | a touch of the ball |
| `strong_touch` | a touch, by how much it changed the ball's velocity |
| `pickup_boost` | boost picked up, more on an emptier tank |
| `save_boost` | every step: boost held |
| `bump` | bumping an opponent |
| `demo` | demolishing an opponent |
| `goal` | a goal for the team, a cost for conceding |
| `alignment` | every step: the ball lined up between the car and the opponents' net |
| `all_of` | several rewards tied together as an "and" |

## Moves

| File | Rewards |
|---|---|
| `air_dribble.rs` | `air_dribble` |
| `dashes.rs` | `wavedash` |
| `flip_reset.rs` | `flip_reset` |
| `mawkzy_flick.rs` | `mawkzy_flick` |
| `musty_flick.rs` | `musty_flick`, `breezi_flick` |
| `musty_rewards.rs` | `musty_pose_shaping`, `tornado_shaping` |
| `raze_flick.rs` | `raze_flick` |
| `raze_flick_extras.rs` | `goal_in_mouth`, `ball_to_roof_shaping` |
| `raze_flick_setup.rs` | `raze_flick_setup` |
| `reset_musty.rs` | `reset_musty`, `reset_musty_pose_shaping` |
| `speed_flip.rs` | `speed_flip` |
| `wall_dash.rs` | `wall_dash` |
| `wizard_flick.rs` | `wizard_flick` |

## More rewards

| File | Rewards |
|---|---|
| `aerial.rs` | `jump_touch`, `aerial_fly_at_ball`, `aerial_chain_distance` |
| `aerial_approach.rs` | `aerial_approach` |
| `aerial_distance.rs` | `aerial_chain_distance_v2` |
| `aerial_navigation.rs` | `aerial_navigation`, `aerial_navigation_strict` |
| `air_dribble_setup_touch.rs` | `air_dribble_setup_touch` |
| `air_dribble_under_ball.rs` | `air_dribble_under_ball` |
| `air_dribble_upfield.rs` | `air_dribble_upfield` |
| `air_roll.rs` | `air_roll`, `air_roll_on_kickoff` |
| `air_touch_long_flight.rs` | `air_touch_long_flight`, `aerial_touch_airtime` |
| `align_ball_goal.rs` | `align_ball_goal` |
| `ball_touch_scaled_by_height.rs` | `ball_touch_scaled_by_height`, `aerial_touch_scaled_by_height`, `touch_ball_on_wall_scaled_by_height` |
| `ball_y_coordinate.rs` | `ball_y_coordinate` |
| `behind_ball.rs` | `behind_ball` |
| `behind_ball_alignment.rs` | `behind_ball_alignment` |
| `boost_changes.rs` | `boost_gain`, `small_boost_pickup`, `boost_use` |
| `boost_lose.rs` | `boost_lose` |
| `boost_pad_pickup.rs` | `boost_pad_pickup` |
| `boost_pickup_sqrt.rs` | `boost_pickup_sqrt` |
| `distance_to_ball.rs` | `distance_to_ball`, `aerial_distance_to_ball` |
| `dont_boost_while_supersonic.rs` | `dont_boost_while_supersonic` |
| `dribble_carry.rs` | `dribble_carry` |
| `dribble_cradle.rs` | `dribble_cradle`, `dribble_upfield`, `dribble_flick` |
| `dribble_match_speed.rs` | `dribble_match_speed` |
| `dribble_roof.rs` | `dribble_roof` |
| `flip_reset_height.rs` | `flip_reset_height` |
| `flip_reset_height_v2.rs` | `flip_reset_height_v2` |
| `flip_reset_simple.rs` | `flip_reset_simple`, `flip_reset_approach` |
| `goal_speed.rs` | `goal_speed`, `goal_speed_and_placement` |
| `goal_speed_bonus.rs` | `goal_speed_bonus`, `goal_speed_top_corner` |
| `high_touch.rs` | `high_touch` |
| `kickoff.rs` | `kickoff` |
| `kickoff_nearest.rs` | `kickoff_nearest`, `kickoff_nearest_v2` |
| `kickoff_speed_nearest.rs` | `kickoff_speed_no_boost`, `kickoff_nearest_team` |
| `kickoff_speed_no_boost.rs` | `kickoff_speed_no_boost_v2` |
| `landing_recovery.rs` | `landing_recovery` |
| `on_ground.rs` | `on_ground` |
| `out_of_position.rs` | `out_of_position` |
| `player_on_wall.rs` | `player_on_wall` |
| `possession.rs` | `possession` |
| `speed_forward.rs` | `speed_forward` |
| `speed_linear.rs` | `speed_linear` |
| `speed_sqrt.rs` | `speed_sqrt` |
| `speed_sqrt_v2.rs` | `speed_sqrt_v2` |
| `speed_toward_ball.rs` | `speed_toward_ball` |
| `speedy_touch.rs` | `speedy_touch` |
| `supersonic.rs` | `supersonic` |
| `touch_ball_scaled_by_hit_force.rs` | `touch_ball_scaled_by_hit_force` |
| `touch_ball_speed_gain.rs` | `touch_ball_speed_gain` |
| `velocity_ball_to_goal_aligned.rs` | `velocity_ball_to_goal_aligned` |
| `wall_positioning.rs` | `wall_positioning` |

## Credits

Each reward file's header says where its reward came from; in short:

- **RLGym** (RLGym 1.2.2, Copyright 2021 Lucas Emery) and **rlgym-sim**
  (github.com/AechPro/rocket-league-gym-sim), both under the Apache License, Version 2.0 (full text in
  `LICENSES/Apache-2.0.txt`): seven built-in rewards follow RLGym's common rewards
  (`crates/nitro_core/src/rewards.rs` names them and what was changed), and `align_ball_goal.rs` and
  `ball_y_coordinate.rs` are adapted from rlgym-sim.
- **rlgym-tools** (github.com/RLGym/rlgym-tools, Rolv Arild Braaten, Apache License, Version 2.0):
  `aerial_chain_distance` (in `aerial.rs`) also follows its AerialDistanceReward.
- **Necto** (github.com/Rolv-Arild/Necto): `behind_ball_alignment.rs`, `boost_lose.rs` and
  `goal_speed_bonus.rs` are adapted from it. Those three files are licensed CC BY-NC-SA 4.0
  (https://creativecommons.org/licenses/by-nc-sa/4.0/), as Necto is, not under NitroLearn's licence.
- **apollo-bot** by redd-rl: the rewards in `aerial.rs`, `boost_pad_pickup.rs`, `dribble_match_speed.rs`,
  `goal_speed.rs`, `kickoff_speed_nearest.rs`, `player_on_wall.rs`, `possession.rs`, `speed_sqrt.rs` and
  `touch_ball_scaled_by_hit_force.rs` are based on it; `goal_speed` also builds on one by fredrik and
  `kickoff_nearest_team` on one by aether6837.
- **rishieissocool's reward collection**: `dribble_cradle.rs`, `landing_recovery.rs` and `speed_forward.rs`.
- **Opti**: `aerial_distance.rs` and `flip_reset_simple.rs`.
- **The RLGym-PPO guide** (github.com/ZealanL/RLGym-PPO-Guide): `on_ground.rs`, `speed_toward_ball.rs`,
  `high_touch.rs` and `air_touch_long_flight.rs` (its air-touch reward).
- **walldashcoach** by samuelkassi (github.com/samuelkassi/walldashcoach, Copyright (c) 2026 samuelkassi,
  MIT licence, full text in `LICENSES/walldashcoach-MIT.txt`): `wall_dash.rs` follows its rules for
  a wall dash (what counts as a make, the timing and flip angle bands, the speed window, wall types, chains
  and the spam guard).
- `all_of` follows the Lucy-SKG paper's way of combining reward parts by multiplying them (it pays only
  when every part does), and the zero-sum sharing is the "team spirit" of OpenAI Five.
