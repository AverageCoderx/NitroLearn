# Your own rewards, in Rust

NitroLearn's rewards are written in Rust and compiled into it, which is why they cost almost nothing
while training. A reward of your own is written the same way, in one `.rs` file, and compiled in by the
dashboard. Once built, it is in every bot's **+ Add reward** list, with its
weight, zero-sum, the share it is paid and everything else, like the built-in rewards.

## Before the first one: the build tools

Building needs two things that most people do not have yet. Both are needed only once.

- **Rust, in NitroLearn's folder:** run the installer again (NitroLearn-Setup into the same folder, or
  `Install NitroLearn.cmd` in NitroLearn's folder) and tick **Tools to build your own Rust rewards** (part 5
  of its list; on Linux: `bash install.sh --rust`). It puts Rust inside NitroLearn's folder (about 1.5 GB)
  and changes nothing else on the PC.
- **On Windows, Microsoft's C++ build tools:** NitroLearn's installer never installs them, because they
  change Windows itself. Get **Build Tools for Visual Studio 2022** from
  <https://visualstudio.microsoft.com/downloads/> (under "Tools for Visual Studio"), and in its installer
  tick **Desktop development with C++**. On Linux, install your system's build tools instead (on Ubuntu or
  Debian: `sudo apt install build-essential`).

**Check this PC** on the dashboard says whether Rust is there, and the **Build** button says exactly what is
missing if a build cannot start.

## The three steps

1. **Write the file.** Copy the example, `crates/nitro_core/src/custom/example.rs`, somewhere of your own
   (for example `my-settings\rewards\flip_reset.rs`), rename the struct and change the maths.
2. **Add it** on the dashboard's Resources page (**Add a file…**). It is recognised as a Rust reward.
3. **Build.** The Resources page's **Build** button compiles NitroLearn again with your rewards
   (a few minutes; `.\nitro rebuild` does the same in a console). A mistake in your file is shown with
   its file and line. When it says Built, close the dashboard and open it again.

## What a reward file looks like

```rust
use crate::custom::prelude::*;

/// 1 when the car gets its flip back by touching the ball in the air.
#[derive(Deserialize)]
pub struct FlipResetTouch {
    /// What one is worth.
    pub scale: f32,
}

impl Reward for FlipResetTouch {
    fn value(&self, prev: &GameSnapshot, cur: &GameSnapshot, events: &StepEvents, car: usize) -> f32 {
        let (before, now) = (&prev.cars[car], &cur.cars[car]);
        let regained = !before.can_flip && now.can_flip && !now.on_ground;
        if regained && events.cars[car].touched_ball { self.scale } else { 0.0 }
    }
}
```

- **Its name** in a settings file is the struct's name in lower case with underscores:
  `FlipResetTouch` is `flip_reset_touch`. It cannot take a built-in reward's name.
- **Its settings** are the struct's fields, one per line, each with `pub`. A settings file writes them in
  the reward's block (`scale = 1.5`); the dashboard shows them as boxes. A field marked
  `#[serde(default)]` (or `#[serde(default = "a_function")]`) may be left out. The `///` lines above the
  struct and its fields become the descriptions the dashboard shows.
- **`value`** is what car number `car` earns for the step from `prev` to `cur`, before its weight.
  `events` says what happened in the step (`touched_ball`, `bumped_opponent`, `demoed_opponent` per car).
- **Paid every step?** A reward of the state (like "how high is the ball") adds
  `fn every_step(&self) -> bool { true }`: it is then scaled to the tick skip like the built-in ones.
  Rewards of events (a touch, a flip reset) leave it out.
- **`#[derive(Deserialize)]`** is needed on every reward (it reads the settings).

## What `value` can look at

`cur.ball` and each car's `body` have `pos` (position, uu), `vel` (velocity, uu/s), `ang_vel` and `rot`
(`forward()` and `up()` give directions). Each car (`cur.cars[car]`) also has `team`, `boost` (0 to 100),
`on_ground`, `has_jumped`, `has_double_jumped`, `has_flipped`, `can_flip`, `is_demoed`, `is_supersonic`.
`cur.goal_scored` is the team that scored on this step, if any. Useful numbers in the prelude:
`CAR_MAX_SPEED` (2300), `BALL_MAX_SPEED` (6000), `CEILING_Z` (2044), `GOAL_LINE_Y` (5120; Blue defends −y).

## More to look at, and rewards that remember

- `events.detail(car)` holds what the snapshot does not, as the step ended (`events.prev_detail(car)`: the
  step before): the controls the car used, which wheels touch something, whether it is jumping, flipping or
  boosting, its time in the air, when it last hit the ball and where the ball was, and its last bump.
  `events.last_touch` is the car that touched the ball last; `events.tick` counts physics ticks (120 a
  second) and `events.step_seconds` is one step's length.
- A reward that must remember something from step to step (a count, a timer, "was dribbling") says how many
  numbers it keeps a car with `fn memory(&self) -> usize`, and is written as `value_remembering(..., memory)`
  instead of `value`: `memory` is that car's numbers, zeroed when a game starts.
- Contacts are checked on every physics tick of the step: `events.cars[car]` also says whether the
  car was on the floor (`on_floor`), touched any part of the arena (`on_arena`: floor, walls, ceiling,
  goals; wheels on the ball never count), touched the ball from the floor (`touched_ball_on_floor`) or with
  nothing under it (`touched_ball_in_air`), and the ticks of its first and last touch
  (`first_touch_tick`, `last_touch_tick`), of its first touch in the air (`first_air_touch_tick`) and of its
  first touch in the air with the ball off the field's floor (`air_start_tick`, the touch that can start an
  air dribble).
  `events.ball_on_field_floor` says the ball touched the field's floor (inside a goal does not count), and
  `events.ball_field_floor_last_tick` on which tick it last did.
- The prelude has helpers for field numbers and teams: `attack_goal(team)`, `own_goal(team)`,
  `teammates(cur, car)`, `opponents(cur, car)`, `as_blue(team, v)` (maths written for Blue), `is_kickoff`,
  `kph`, `cosine` and constants such as `BALL_RADIUS` and `BACK_WALL_Y` (`crates/nitro_core/src/custom/kit.rs`).
- The rewards that come with NitroLearn (`crates/nitro_core/src/custom/library/`, listed in `docs/rewards.md`)
  are complete examples of all of this.

## Good to know

- The list of your reward files is `my-settings\rust-files.txt` (the Build button writes it). An update,
  or running the installer again, puts in the release's ready-made NitroLearn, which has none of your own
  rewards: with Rust in NitroLearn's folder and the build tools on the PC, the installer then builds them in
  again by itself. Without them it says so, and your rewards are missing until you press **Build** again
  (a settings file that names one fails its check, saying so). A listed file that has moved is left out
  with a warning. Take a file off the Resources page and Build to remove it.
- If an update ever changes what a reward file can use and yours no longer compiles, the build names your
  file and line: fix it, or take it off the list, and build again.
- Observation builders and action parsers of your own come later: they must also reach the exported
  RLBot bot.
- C++ rewards from other frameworks cannot be loaded as they are; their maths can be written again in a file like
  the one above.
