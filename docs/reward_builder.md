# Building a reward without code

A *built reward* is put together from ready-made blocks in your settings file (or in the dashboard's
Rewards tab, **Add reward → Build your own (no code)**). Nothing is compiled and nothing needs installing:
NitroLearn reads the blocks and adds them up while the games run. It is paid, shown and checked like any
other reward: its share in the Rewards tab, its `paid_<title>` column in metrics.csv, *Try these rewards*.

## The idea

A built reward is a sum of **terms**. Each term says:

- **when** it pays: conditions that must all be true (in the air, touching the ball, ball above 300, ...);
- **what** it pays: a measure (speed toward the ball, ball height, a goal, ...), or several measures
  multiplied together;
- **how much**: the term's weight.

The reward then has its own weight and the usual options (zero-sum against the opponents, team spirit).

```toml
[[rewards]]
name = "built"
title = "touch_up_high"          # its name in the dashboard and metrics: small letters, digits and _
weight = 20

[[rewards.terms]]             # touching the ball high in the air: pays the ball's height
when = ["in_air", "touching_ball", { ball_above = 300 }]
measure = "ball_height"

[[rewards.terms]]             # a little for chasing the ball while on the ground
when = ["on_ground"]
measure = "speed_toward_ball"
weight = 0.05
```

A short term can also be written on one line: `terms = [{ measure = "speed_toward_ball" }]`.

## Conditions

Plain ones, written in quotes: `in_air`, `on_ground` (wheels on anything), `on_floor`, `on_wall` (wall or
ceiling), `touching_ball`, `has_flip` (can still jump or flip), `supersonic`, `boosting`, `own_half`,
`opponent_half`, `ball_own_half`, `ball_opponent_half`, `kickoff`, `closest_to_ball` (nearest on its team),
`last_to_touch`.

With a number, written `{ name = number }`:

| Condition | True when | Number in |
|---|---|---|
| `ball_above`, `ball_below` | the ball is higher / lower than this | game units (ball on the floor: 93) |
| `car_above`, `car_below` | the car is higher / lower than this | game units (car on the floor: 17) |
| `ball_closer_than`, `ball_farther_than` | the ball is nearer / farther than this from the car | game units |
| `facing_ball_within` | the car's nose points at the ball within this angle | degrees, 0 to 180 |
| `facing_goal_within` | the car's nose points at the other goal within this angle | degrees, 0 to 180 |
| `boost_above`, `boost_below` | the car has more / less boost | 0 to 100 |
| `speed_above`, `speed_below` | the car is faster / slower | km/h |
| `ball_speed_above`, `ball_speed_below` | the ball is faster / slower | km/h |

`{ not = "in_air" }` turns any condition around. For scale: the field is 10240 units long, 8192 wide,
and the ceiling is at 2044.

## Measures

Paid **every step** (these are scaled to the tick skip like the built-in rewards, so the same weights mean
the same at any tick skip):

| Measure | Pays |
|---|---|
| `one` | 1, so the term pays its weight while its conditions hold |
| `speed_toward_ball` | how fast the car moves toward the ball, as a share of its top speed |
| `ball_speed_toward_goal` | how fast the ball moves toward the other goal, as a share of its top speed |
| `car_speed`, `ball_speed` | speed, as a share of the top speed |
| `car_height`, `ball_height` | height, 0 on the floor to 1 at the ceiling |
| `boost` | the car's boost, 0 to 1 |
| `facing_ball` | 1 nose straight at the ball, -1 away |
| `alignment` | 1 lined up behind the ball toward the other goal, -1 between the ball and the goal |
| `distance_to_ball` | the distance, as a share of the field's length |
| `closeness_to_ball` | 1 at the ball, falling to 0 a field's length away |
| `air_time` | time in the air, as a share of 1.75 seconds (at most 1) |

Paid for **what happened in the step**: `boost_used`, `boost_gained` (0 to 1 of a tank),
`ball_speed_change` (how hard the ball was hit), and one-off events, 1 when they happen: `touch`, `goal`,
`concede`, `shot` (the car's touch sends the ball heading into the other goal), `demo`, `demoed`, `bump`,
`jump`, `flip`. Give `concede` and `demoed` a negative weight.

## Term options

- `measure = ["air_time", "ball_height"]` multiplies them; add `combine = "min"` to pay the smaller one.
- `weight = 2`: how much the term counts inside the reward (default 1).
- `once_per_touch = true`: the term pays at most once between two of the car's ball touches.

For more than the blocks can say, a reward can still be written in Rust (`docs/your_own_rewards.md`); that
needs Rust and Microsoft's C++ build tools, a built reward needs nothing.

## Advanced rewards: formulas

When the blocks are not enough, a term can pay a **formula** instead of a measure, a condition can be a
formula, and a reward can keep **variables** from step to step. Everything is still checked before training
starts (a mistake is named with its place, e.g. `rewards[0].terms[1].value: 'me' has no field 'sped'`), and
nothing is compiled.

```toml
[[rewards]]
name = "built"
title = "long_air_dribble"
weight = 3

[[rewards.vars]]                 # a counter of touches in a row without landing
name = "chain"
start = 0
next = "if(me.on_ground, 0, if(me.touch, chain + 1, chain))"

[[rewards.terms]]                # a touch after at least a second in the air pays the chain so far
when = "me.touch and me.in_air and held(me.in_air) >= 1"
value = "chain"
cap_episode = 5                  # at most 5 an episode
```

**Things a formula can read**

- `me` (this car), `mate` (its nearest teammate), `opp` (its nearest opponent), `ball`, and `it` (each car
  inside `sum_mates(...)`, `max_opps(...)` and the like). Fields: `pos`, `vel`, `angvel`, `forward`, `up`,
  `right` (directions and positions), `x`, `y`, `z`, `speed`, `boost`, `on_ground`, `on_floor`, `on_wall`,
  `on_ceiling`, `in_air`, `has_flip`, `jumped`, `double_jumped`, `flipped`, `flipping`, `boosting`,
  `supersonic`, `demoed`, `air_time`, `team` (+1 Blue, -1 Orange), `wall_dist`, `ceiling_dist`, `closest`;
  and what happened this step: `touch`, `touch_air`, `touch_strength`, `shot`, `bump`, `demo`, `got_demoed`,
  `jump`, `flip`, `regained_flip`, `flip_reset`, `last_touch`. The ball has `pos`, `vel`, `angvel`, `x`, `y`,
  `z`, `speed`, `wall_dist`, `ceiling_dist`, `touched`.
- Named values: `dt` (the step's seconds), `goal`, `concede`, `kickoff`, `has_mate`, `has_opp`, `own_goal`,
  `opp_goal` (goal mouth centres), `own_net`, `opp_net` (backs of the nets), `pi`.
- Your variables, by name.

**Maths**: `+ - * /`, comparisons `< <= > >= == !=`, `and`, `or`, `not`, `if(c, a, b)`, `min`, `max`,
`clamp(x, lo, hi)`, `abs`, `sign`, `sqrt`, `pow`, `exp`, `log`, `decay(x, half)` (1 at 0, halving every
`half`). Directions: `dot`, `cross`, `dist`, `angle` (degrees), `length`, `norm`, `flat` (seen from above),
`local` (in the car's own frame: `.x` forward, `.y` right, `.z` up), `vec(x, y, z)`, `.x .y .z`,
`nearest_pad(p)`, `nearest_big_pad(p)`. Over cars: `sum_mates`, `sum_opps`, `min_mates`, `min_opps`,
`max_mates`, `max_opps`, `count_mates`, `count_opps`.

**Time and memory**: `prev(x)` (x at the step before) and `delta(x)` (the change); `since(c)` (seconds since
c was last true; a billion before it ever was), `held(c)` (seconds c has been true without a break),
`count(c)` (steps c was true this episode), `once(c)` (1 the first time only). A sequence such as "a flip
reset, then a shot within 2 seconds" is `me.shot and since(me.flip_reset) < 2`.

**More term options**: `once_per_episode = true`; `cap_step = 5` (at most 5 a step) and `cap_episode = 20`
(at most 20 an episode; charges are not capped); `every_step = true` scales a term that reads an event to the tick skip
like an every-step reward.

**Templates**: the dashboard's Add reward → *Start from a template* has ready-made built rewards to start
from; the ones that say "a copy of ..." are proven to pay exactly what that reward pays.

**Limits**: at most 400 parts a formula, 16 conditions and 16 measures a term, 32 terms, 16 variables and 32
`since`-like functions a reward; no loops. A value that comes out as not-a-number or infinite pays 0.

## Good to know

- Built rewards can be part of an `all_of` group, except ones that remember (`once_per_touch`, variables,
  `since`-like functions, caps per episode, once per episode).
- They stay with your bot's settings: an exported bot does not need its rewards.
- How fast: checked on recorded games, a built reward of blocks costs about what the compiled rewards it
  copies cost, and one of formulas about 1.5 to 2 times as much (tens of nanoseconds per car and step either
  way: a tiny share of a game step).
- Proof it pays what it says: the tests compare built copies of `air`, `face_ball`,
  `velocity_player_to_ball`, `velocity_ball_to_goal`, `alignment`, `save_boost`, `touch_ball`, `goal`,
  `bump` and `demo` with the real ones on recorded games, to 1e-6; and the templates rebuild
  `aerial_touch_airtime`, `kickoff_nearest`, `wall_positioning`, `goal_speed_and_placement` and
  `landing_recovery` from formulas, equal to 1e-6 on recorded 1v1 and 2v2 games, plus two sequences (touch
  then goal, flip reset then shot) against hand-written versions.
