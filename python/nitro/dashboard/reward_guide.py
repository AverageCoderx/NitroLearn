# ruff: noqa: E501  (long explanation texts, kept one paragraph line each)
"""The rewards' "See more" explanations in the dashboard's Rewards tab: for every built-in
reward and every reward that comes with NitroLearn, what it pays and when, when to use it, how to set it and
what to watch out for. Keyed by reward name; paragraphs open with their label, as `guide.MORE` does."""

from __future__ import annotations

MORE: dict[str, str] = {
    "aerial_approach": (
        "What it does: every step the car is in the air and rising, below ball_height_min with the ball above that height and closer than three times the ball's height. It pays how straight the car flies at the ball times its speed over 2,300 units/s, plus half that straightness with beginner on, plus 0.5 when the car is nearer the ball than on its last paying step. A touch on that step multiplies it by 1.5. 0 to 3, never negative.\n"
        "\n"
        "When to use it: early aerial training, to teach taking off fast and straight toward a high ball.\n"
        "\n"
        "How to use it:\n"
        "- ball_height_min: default 400 units. The ball must be above it and the car below it, so only the take-off pays. Raise it to pay for more of the climb.\n"
        "- beginner: default on. Adds half the straightness, for bots that cannot yet fly fast; turn it off later so speed matters more.\n"
        "- Weight: up to 3 a step, about 15 times a second, but only during take-offs. Try 0.5 to 2 next to a goal at 150.\n"
        "\n"
        "Watch out for: since only the low, rising part of a flight pays, a bot can farm it by jumping again and again under a high ball without going for it. The nearer-than-last-time check compares with the car's last paying step, even from an earlier take-off, so a new jump can get the 0.5 at once.\n"
        "\n"
        "Measured: not measured."
    ),
    "aerial_chain_distance": (
        "What it does: pays touches in an aerial, with one car at a time holding the lead. A car's first touch while it is above 256 units up pays height_scale times (its height plus the ball's height minus 512), over 10,240. Each later touch by that leader, while it stays above 256 units, pays distance_scale times how far it and the ball moved since its last touch, over 10,240. A first touch pays up to about 0.35 (car and ball near the ceiling); a later one about 0.1 per 1,000 units the car and ball travelled together. Another car's air touch takes the lead; dropping below 256 ends it.\n"
        "\n"
        "When to use it: air dribbles and long multi-touch aerials, in a bot that already flies.\n"
        "\n"
        "How to use it:\n"
        "- height_scale: pay per unit of height for the first touch (default 1). Raise it to favour high first touches.\n"
        "- distance_scale: pay per unit travelled for later touches (default 1). Raise it to favour carrying the ball far.\n"
        "- Event reward, small values: a weight of about 10 to 50 next to a goal at 150.\n"
        "\n"
        "Watch out for: the car's own travel counts, so flying a long way between two touches pays even if the ball barely moved. The height test is the car's height, not whether it is airborne, so touches while driving high on a wall count too.\n"
        "\n"
        "Measured: not measured."
    ),
    "aerial_chain_distance_v2": (
        "What it does: it pays chains of air touches. The car's first touch while it is at least 256 units up pays height_scale x (car height + ball height - 512) / 10,240: about 0.08 with the car at 1,000 and the ball at 1,100. Each further touch before the car drops below 256 units pays distance_scale x (how far the car and the ball travelled since the last touch) / 10,240: about 0.1 if each moved 1,000 units. Another car touching the ball 256+ units up takes the chain over. Paid on touches only.\n"
        "\n"
        "When to use it: to teach going up for the ball and keeping several air touches going: multi-touch aerials and air dribbles. For a bot that can already fly, later in training.\n"
        "\n"
        "How to use it:\n"
        "- height_scale: what the first touch's height is worth (default 0.5).\n"
        "- distance_scale: what the travel between touches is worth (default 0.5).\n"
        "- Weight: the values are small, about 0.03 to 0.2 per touch, so it needs a large weight to matter, in the tens to low hundreds next to a goal at 150.\n"
        "\n"
        "Watch out for: it checks the car's height, not that it is flying, so touches while driving on a wall above 256 units count. The travel pay has no time limit, so a bot could touch, fly around, and touch again for more. An opponent's touch on the ground does not end the chain.\n"
        "\n"
        "Measured: not measured."
    ),
    "aerial_distance_to_ball": (
        "What it does: every step the car's wheels are off every surface, it pays 1 minus its distance to the ball divided by 600 units: about 0.75 when touching the ball (distance is measured centre to centre), 0.5 at 300 units, 0 from 600 units away. On the ground it pays 0. Range 0 to 1.\n"
        "\n"
        "When to use it: to teach staying close to the ball while flying: aerials and air dribbles. For a bot that can already jump and boost into the air.\n"
        "\n"
        "How to use it:\n"
        "- No settings of its own; the weight is all there is.\n"
        "- It pays up to about 0.8 per step, about 15 times a second. With a goal at 150, a weight of about 0.1 to 1.\n"
        "\n"
        "Watch out for: it does not care which side of the ball the car is on or where the ball goes, so the bot can learn to hover near the ball, or jump beside a ball on the floor, instead of hitting it somewhere useful. Pair it with touch and goal rewards.\n"
        "\n"
        "Measured: not measured."
    ),
    "aerial_fly_at_ball": (
        "What it does: pays every step the car is in the air and moving towards a ball that is at least 300 units up. The pay is the car's speed towards the ball as a share of top speed (2,300 units/s), shrunk with distance from the ball (to about a third at 4,600 units away), plus 0.1 once the car has double jumped. About 0 to 1.1 a step. It pays 0 on the ground, when moving away, or with the ball lower.\n"
        "\n"
        "When to use it: early aerial training, to get a bot leaving the ground and flying hard at high balls.\n"
        "\n"
        "How to use it:\n"
        "- No settings of its own; the weight is all there is.\n"
        "- It adds up about 15 times a second at tick skip 8, so keep it small: about 0.1 to 1 next to a goal at 150. Pair it with a reward for the touch itself.\n"
        "\n"
        "Watch out for: it pays the flight, not a touch, so a bot can fly near the ball and never hit it. The 0.1 for a double jump is paid even with almost no speed towards the ball, so a bot may learn to double jump on every aerial just to collect it. Flying past the ball stops the pay at once.\n"
        "\n"
        "Measured: not measured."
    ),
    "aerial_navigation": (
        "What it does: pays every step the car is in the air, rising, higher than 200 units, toward a ball above 400 units that is no farther away across the field than 3 times its height. Up to 0.5 for flying straight at the ball, up to 0.25 for its nose pointing at it, and up to 0.1 more while boosting. 0 to 0.85 a step with beginner on; never negative.\n"
        "\n"
        "When to use it: early aerial training, to get a bot to jump and fly at high balls.\n"
        "\n"
        "How to use it:\n"
        "- ball_height_min: the ball must be higher than this (default 400 units).\n"
        "- player_height_min: the car must be higher than this (default 200).\n"
        "- beginner: on (default), the flying part is half of how straight the car flies at the ball, never below 0. Off, it is how straight times its speed over 2,300 units/s, up to 1, which can go negative and eat into the facing pay; the total then reaches 1.35.\n"
        "- Weight: every step, so it adds up over a flight. With a goal reward around 10, start around 0.5 to 2.\n"
        "\n"
        "Watch out for: it needs no touch and does not get smaller with distance, so a car rising toward a ball it cannot reach still pays; a bot can farm it by jumping and boosting under high balls without hitting them. Pair it with a touch reward. The boost bonus can teach wasting boost in the air.\n"
        "\n"
        "Measured: not measured."
    ),
    "aerial_navigation_strict": (
        "What it does: pays every step the car is in the air, rising, below a ball higher than 600 units, itself higher than 450, with more than 20 boost. Up to 0.5 for flying straight at the ball and up to 0.15 for its nose pointing at it, both scaled by nearness (full at the ball, nothing from max_distance away), plus 0.85 on a step it touches the ball. Kept between 0 and 1 a step.\n"
        "\n"
        "When to use it: early aerial training, to teach flying to high balls and hitting them.\n"
        "\n"
        "How to use it:\n"
        "- ball_height_min: the ball must be higher than this (default 600 units).\n"
        "- player_height_min: the car must be higher than this (default 450).\n"
        "- beginner: on (default), the flying part is half of how straight the car flies at the ball, never negative. Off, it is how straight times its speed over 2,300 units/s (up to 1), and flying away takes pay off.\n"
        "- max_distance: from this far from the ball, centre to centre, flying and facing pay nothing (default 2,044, the field's height).\n"
        "- Weight: every step, so it adds up over a flight. With a goal reward around 10, start around 0.5 to 2.\n"
        "\n"
        "Watch out for: it pays only while rising and below the ball with boost, so it stops once the car passes the ball's height, and the touch bonus counts only if those still hold on that step. As an every-step reward, even the touch bonus is scaled by the tick skip.\n"
        "\n"
        "Measured: not measured."
    ),
    "aerial_touch_airtime": (
        "What it does: on a touch, pays the smaller of two shares: the car's time in the air before this step over 1.75 seconds (at most 1), and the ball's height over 2,044 units (the ceiling). 0 to 1, on every touch. The time counts the steps that ended with the car off its wheels and starts again at 0 each time it is on them.\n"
        "\n"
        "When to use it: aerial training, rewarding touches after a real flight with the ball high.\n"
        "\n"
        "How to use it:\n"
        "- No settings of its own; the weight is all there is.\n"
        "- Event reward, 0 to 1: a weight of about 10 to 50 next to a goal at 150.\n"
        "\n"
        "Watch out for: full pay needs 1.75 seconds in the air and the ball at the ceiling, so most touches pay well under 1. Every touch pays, with no wait, so an air dribble collects on each step it touches the ball. Wheels on a wall, or on the ball (the game counts that as being on the ground), restart the timer. Each car keeps its own timer.\n"
        "\n"
        "Measured: not measured."
    ),
    "aerial_touch_scaled_by_height": (
        "What it does: when the car touches the ball while its wheels are off every surface, it pays the ball's height divided by 2,000 units. A ball near the floor pays about 0.05, a ball 1,000 units up pays 0.5, one near the ceiling about 1. It pays on each step with a touch, 0 otherwise.\n"
        "\n"
        "When to use it: to make a bot go up and meet high balls: aerials and air touches. Best once the bot already touches the ball on the ground and needs a reason to take to the air.\n"
        "\n"
        "How to use it:\n"
        "- No settings of its own; the weight is all there is.\n"
        "- It pays at most about 1 per touch, like a touch reward. With a goal at 150, a weight of about 5 to 30 keeps it well below scoring (the starter's strong_touch is 60).\n"
        "\n"
        "Watch out for: it pays the ball's height, not how useful the touch was, so a bot can learn to jump into high balls aimlessly or keep popping the ball up. A long contact over several steps pays on each of them. A jump touch on a ball at the floor still pays a little. Pair it with goal or ball-toward-goal rewards.\n"
        "\n"
        "Measured: not measured."
    ),
    "air": (
        "What it does: pays 1 on every step the car is off the ground, and 0 while its wheels are on the floor, a wall or the ceiling. It is paid every step, about 15 times a second at tick skip 8.\n"
        "\n"
        "When to use it: early in training, so a new bot leaves the floor at all and can stumble onto jumps, aerials and recoveries. The default mix, used when you list no rewards, has it at weight 0.25. A negative weight does the opposite and teaches getting back on the wheels quickly (the recovery practice task uses -1).\n"
        "\n"
        "How to use it:\n"
        "- No settings of its own; the weight is all there is.\n"
        "- Weight: keep it small, about 0.1 to 0.5 next to a goal worth 150. At 0.25 a full second in the air earns about 3.75.\n"
        "\n"
        "Watch out for: it pays for being in the air, not for doing anything useful there. Too big a weight teaches jumping and flipping around aimlessly, away from the ball. Driving on walls earns nothing, because the wheels are on a surface. Consider lowering it once the bot jumps on its own.\n"
        "\n"
        "Measured: not measured."
    ),
    "air_dribble": (
        "What it does: pays air dribbles by one car (separate, useful air touches on a ball kept off the floor). A goal off a valid chain pays goal_reward times bonuses (extra touches, air roll, flip resets, distance carried): 2 to about 28. Capped shaping pays touches, control and placement; an own goal off a chain costs 0.5. Paid on events.\n"
        "\n"
        "When to use it: main reward of the air dribble bot set.\n"
        "\n"
        "How to use it:\n"
        "- Touches needed: min_touches_ground 2; min_touches_dribble, min_touches_wall, min_touches_aerial 3.\n"
        "- Touch spacing: max_touch_gap 1.5 s, min_separation_ticks 6, min_separation_gap 25, ball_radius 91.25.\n"
        "- Starts: setup_window 1.2, aerial_catch_height 450, aerial_min_air_time 0.35, wall_up_z 0.7, floor_contact_height 112.75, floor_bounce_height 300, dribble_min_up 105, dribble_max_up 185, dribble_max_planar 110, dribble_min_time 0.25.\n"
        "- Goal: goal_reward 2, per_extra_touch 0.15, max_extra_touches 6, goal_window 2.5, require_goal_mouth true, goal_half_width 892.755, goal_height 642.775, own_goal_penalty 0.5, allow_floor_before_goal false.\n"
        "- Chain rules: min_air_touch_height 300, max_car_ball_distance 550, above_margin 50, ahead_margin 50, min_touch_lift 100, min_touch_push 125, max_push_touch_drop 25, min_boost_at_start 50.\n"
        "- Style: air_roll_rate 0.15, air_roll_min_rate 3, max_air_roll_per_chain 0.2, air_roll_goal_bonus 0.25, reset_goal_bonus 0.5, max_resets_for_bonus 4, shot_dodge_window 0.5, min_reset_interval 0.5, cross_map_bonus 1, cross_map_start 1500, cross_map_full 6000.\n"
        "- Shaping: touch_reward 0.15, max_paid_touches_per_chain 10, control_rate 0.25, control_full_height 1000, max_control_per_chain 0.6, progress_rate 0.3, valid_chain_bonus 0.25, catch_bonus 0.15, catch_soft_speed 1200, max_shaping_per_episode 1.5.\n"
        "- Positioning: position_weight 0.1 (0 off), position_gamma 0.99, position_full_dist 200, position_below_full 120, position_behind_full 100, position_match_full 300, position_match_zero 1500; none after falling at position_fall_fraction 0.7 g for over position_fall_grace 0.5 s.\n"
        '- aim_at_goal (default false): "toward goal" in the pay aims at the goal\'s centre, not straight downfield, so carrying the ball toward the net beats the side wall; "in front", which ends the chain, stays downfield.\n'
        "- Weight: 10 in the set.\n"
        "\n"
        "Watch out for: under 50 boost at takeoff a chain earns nothing; another car's touch ends it; positioning pay stays only when a touch extends the chain.\n"
        "\n"
        "Measured: not measured."
    ),
    "air_dribble_setup_touch": (
        "What it does: on a touch with the car in the air, the ball above 250 units, above the car and farther upfield than it, pays (1 minus the ball's distance from the middle of the field over 2,200) times the number of such touches since the car left the ground. The first touch pays up to 1, the second up to 2, and so on; the count restarts when the car is on its wheels. A ball more than 2,200 units off centre pays a minus.\n"
        "\n"
        "When to use it: air dribble training, to reward pushing the ball upfield through the middle with touch after touch in one aerial.\n"
        "\n"
        "How to use it:\n"
        "- No settings of its own; the weight is all there is.\n"
        "- Start small, about 1 to 5 next to a goal at 150, because the pay grows with every touch.\n"
        "\n"
        "Watch out for: there is no cap. A car that keeps its nose pressed on the ball can touch it every step, so the count and the pay can climb very fast; watch for bots that carry the ball slowly instead of shooting. Wheels on the ball count as on the ground and restart the count. Each car counts its own touches, on either team.\n"
        "\n"
        "Measured: not measured."
    ),
    "air_dribble_under_ball": (
        "What it does: every step the car is off the ground. It pays 1 when the ball is within 300 units of the car, above it, and the car's nose points upward (when car and ball both fall, only while both fall slower than 650 units/s). It adds 0.5 whenever car and ball move in the same general direction, near or not. 0 to 1.5 a step; 0 on the ground or a wall.\n"
        "\n"
        "When to use it: air dribble training for bots that can already aerial, keeping the car under the ball and moving with it.\n"
        "\n"
        "How to use it:\n"
        "- No settings of its own; the weight is all there is.\n"
        "- Weight: up to 1.5 a step, about 15 times a second, so a second of air dribbling pays about 22 at weight 1. Try 0.5 to 2 next to a goal at 150.\n"
        "\n"
        "Watch out for: the 0.5 pays for any time in the air moving the ball's way, even far from it, so a bot can farm it by jumping often while chasing the ball. Nose up means any amount above level. It pays for staying under the ball, not for taking it toward the goal, so pair it with a goal or shot reward.\n"
        "\n"
        "Measured: not measured."
    ),
    "air_dribble_upfield": (
        "What it does: every step the car is in the air below a ball that is above 250 units and in the opponents' half, within 400 units of it and facing it. It pays up to 7.5 for closeness, up to 1 for facing, 5 for being behind the ball and 40 on a touch: up to 53.5. With 15 boost or less it pays half; when the car or the ball is not rising, half again. Then it is multiplied by 2 in the middle of the field, falling to 0 at the side walls. So 0 to 107 a step, about 100 times a usual 0-to-1 reward.\n"
        "\n"
        "When to use it: an air dribble bot carrying the ball up the middle.\n"
        "\n"
        "How to use it:\n"
        "- No settings of its own; the weight is all there is.\n"
        "- Weight: about 0.01 to 0.05 with a goal at 150, since it pays about 100 times a usual reward and it adds up about 15 times a second at tick skip 8.\n"
        "\n"
        "Watch out for: a weight of 1 dwarfs every other reward. A touch is worth 40 of the 53.5, so a bot may learn to tap the ball rather than carry it. It does not check that the ball heads toward the net, only that it is in the opponents' half. Near the side walls it pays almost nothing.\n"
        "\n"
        "Measured: not measured."
    ),
    "air_roll": (
        "What it does: pays every step the car is higher than 200 units: how fast it spins about its own nose (an air roll), in radians a second, 0 to 5.5. With in_air_with_flip on, it pays only in the air with a flip still left, and takes off a fifth of its pitch and yaw spin: -2.2 to 5.5.\n"
        "\n"
        "When to use it: to teach a bot to air roll, as a step toward aerial control or roll-heavy tricks.\n"
        "\n"
        "How to use it:\n"
        "- in_air_with_flip: off (default) pays the roll speed whenever the car is above 200 units; on is the version described above.\n"
        "- Weight: it pays up to 5.5 a step, more than most rewards, every step. Use a small weight, around 0.01 to 0.1 with a goal reward around 10.\n"
        "\n"
        "Watch out for: the bot can roll whenever it is up high with no link to the ball, so at a large weight rolling crowds out real play. The default version checks only the height, not that the car is off the ground.\n"
        "\n"
        "Measured: not measured."
    ),
    "air_roll_on_kickoff": (
        "What it does: pays every step of a kickoff, while the ball still sits exactly on the centre spot: the car's roll speed about its own nose over its top spin of 5.5 radians a second, -1 to 1. Rolling one way pays, rolling the other way costs the same. Once the ball has moved it pays 0.\n"
        "\n"
        "When to use it: to push a bot to air roll during its kickoff, such as rolling through a flip kickoff, in a bot that already gets to the ball.\n"
        "\n"
        "How to use it:\n"
        "- No settings of its own; the weight is all there is.\n"
        "- Weight: every step, but only until the kickoff's first touch. A small weight, around 0.05 to 0.5 with a goal reward around 10.\n"
        "\n"
        "Watch out for: it teaches one roll direction and punishes the other, and it does not care whether the kickoff is won. It only pays with start types that use real kickoffs (ball at rest on the centre spot).\n"
        "\n"
        "Measured: not measured."
    ),
    "air_touch_long_flight": (
        "What it does: on a touch, pays the smaller of two shares: the car's time in the air as the game counts it, over 1.75 seconds (at most 1), and the ball's height over the ceiling's (2,044 units). 0 to 1, on every touch. It pays 0 without a touch.\n"
        "\n"
        "When to use it: aerial training, rewarding touches after a real flight with the ball high.\n"
        "\n"
        "How to use it:\n"
        "- No settings of its own; the weight is all there is.\n"
        "- Event reward, 0 to 1: a weight of about 10 to 50 next to a goal at 150.\n"
        "\n"
        "Watch out for: full pay needs 1.75 seconds in the air and the ball at the ceiling, so most touches pay well under 1. Every touch pays, with no wait, so an air dribble collects on each step it touches the ball. Being on its wheels, on the floor or a wall, restarts the game's air time.\n"
        "\n"
        "Measured: not measured."
    ),
    "align_ball_goal": (
        "What it does: every step it adds two parts. Defence: how well the car sits on the line from its own net to the ball. Offence: how well the ball sits on the line from the car to the opponents' net. Each part runs from -1 (exactly the wrong side) to 1 (perfectly lined up), times its setting, so -1.1 to 1.1 with the defaults. It looks at positions only, not where the car faces.\n"
        "\n"
        "When to use it: to teach positioning: coming at the ball from behind so a hit sends it toward the opponents' net, and staying goal-side in defence. Any game size, early to mid training.\n"
        "\n"
        "How to use it:\n"
        "- defense: what being between the ball and the own net is worth (default 0.1; 1 in rlgym-sim).\n"
        "- offense: what having the ball between the car and the opponents' net is worth (default 1).\n"
        "- Weight: up to about 1.1 per step, about 15 a second. With a goal at 150, about 0.1 to 1.\n"
        "\n"
        "Watch out for: it pays a position every step, so the bot can learn to sit lined up behind the ball instead of hitting it. Pair it with a reward for moving to the ball. With both settings high the parts can pull in different directions.\n"
        "\n"
        "Measured: not measured."
    ),
    "alignment": (
        'What it does: pays, every step, how well the car, the ball and the opponents\' goal line up: 1 when the ball is exactly between the car and the back of their net (a touch would send it goalward), 0 when the car is beside the ball, and -1 when the car is between the ball and their goal. It is the angle between "car to ball" and "ball to their net", from 1 to -1, paid about 15 times a second at tick skip 8.\n'
        "\n"
        "When to use it: to teach getting on the right side of the ball before hitting it, for any bot that should send the ball toward goal. It works best tied to moving toward the ball with all_of, so it pays only while the car approaches from the right side.\n"
        "\n"
        "How to use it:\n"
        "- No settings of its own; the weight is all there is.\n"
        "- Weight: about 1 next to a goal of 150 when used alone. Inside all_of with velocity_player_to_ball, the presets' commented example uses weight 4, in place of velocity_player_to_ball's own 4.\n"
        "\n"
        "Watch out for: on its own it pays for being behind the ball whether the car moves or not, so a bot can sit behind the ball and collect. It ignores distance: lined up from far away pays as much as lined up close.\n"
        "\n"
        "Measured: On the kickoff practice task (10 seeds each), alignment added on its own at weight 1 barely learned the kickoff (5.8 s to the first touch). Tied to velocity_player_to_ball with all_of, the bot reached the ball in 4.2 s and sent it about 2,600 units toward goal, against 490. One task only."
    ),
    "all_of": (
        'What it does: ties two or more rewards together as an "AND". The group pays only when every part is positive, and then the geometric mean of the parts (for two parts, the square root of their product: 0.25 and 1 give 0.5). It pays 0 when any part is 0, and a negative amount (minus the geometric mean of their sizes) when any part is negative. If every part is paid every step, so is the group; if any part is an event, the group pays only on that event.\n'
        "\n"
        "When to use it: when a behaviour is only good in combination, such as moving toward the ball while lined up to send it goalward.\n"
        "\n"
        "How to use it:\n"
        "- parts: the rewards to combine, at least two, each written with its name and its own settings. A part has no weight, zero-sum or on/off switch of its own; the group carries those. A group cannot hold another group, nor a reward that remembers things between steps.\n"
        "- Weight: like the parts it replaces. The presets' example uses 4 for velocity_player_to_ball with alignment, in place of velocity_player_to_ball's 4.\n"
        "\n"
        "Watch out for: one negative part makes the whole group negative, however good the others are, and two negatives stay negative. A tiny part drags the result toward 0 (0.01 and 1 give 0.1). Parts are multiplied, not added, so a part with a big range scales the whole group.\n"
        "\n"
        "Measured: On the kickoff practice task (10 seeds each), velocity_player_to_ball and alignment tied with all_of reached the first touch in 4.2 s against 5.8 s for the two added up, better on every seed, and sent the ball about 2,600 units toward goal against 490. One task only."
    ),
    "ball_to_roof_shaping": (
        "What it does: helps the bot pick the ball up onto its roof. Each step, until the ball first counts as under control on the roof, it pays how much closer the ball came to a spot just above the roof (20 units forward of the car's centre, 133 up), as a share of d_max. Control means the car on its wheels with the ball resting on the roof and moving with it: plus when closer, minus when farther. Over an episode it adds up to at most 1, then stops. If the episode starts with the ball on the roof, it never pays.\n"
        "\n"
        "When to use it: early in a dribble or flick bot, to teach the pickup. The Raze set uses it at 30, with goal_in_mouth at 25.\n"
        "\n"
        "How to use it:\n"
        "- d_max: distances count up to this, in units (default 1,500).\n"
        "- dribble_min_up and dribble_max_up: how high above the car the ball's centre must be for control (defaults 105 and 185 units).\n"
        "- dribble_target_forward: the best spot, forward of the centre (default 20 units).\n"
        "- dribble_max_planar_dist: how far sideways from that spot control ends (default 90 units).\n"
        "- dribble_max_rel_speed: control ends at this speed against the car (default 600 units/s).\n"
        "- control_threshold: the control score, 0 to 1, that counts (default 0.25).\n"
        "- Weight: at most the weight once per episode; 20 to 40 next to a goal at 150.\n"
        "\n"
        "Watch out for: it switches off at the first control, so it does not teach keeping the ball; pair it with a dribble reward. It also pays driving to the ball, since that brings the spot nearer.\n"
        "\n"
        "Measured: not measured."
    ),
    "ball_touch_scaled_by_height": (
        "What it does: on every touch, on the ground or in the air, it pays the ball's height divided by 2,000 units: about 0.05 for a ball on the floor, 0.5 at 1,000 units, about 1 near the ceiling. With high_touch_penalty on, it does the opposite: touches above 200 units cost 1 minus the square root of (height / 20), from about -2.2 just above 200 units to about -8.9 at the ceiling, and lower touches pay 0. It pays on each step with a touch.\n"
        "\n"
        "When to use it: the normal formula rewards hitting the ball high, for a bot learning aerials. The penalty formula (high_touch_penalty) punishes high touches, for a bot that should keep the ball low and play on the ground.\n"
        "\n"
        "How to use it:\n"
        "- high_touch_penalty: false (default) pays more the higher the ball; true is a penalty that grows with the height.\n"
        "- Weight: for the normal formula about 5 to 30 with a goal at 150, like a touch reward. The penalty's values are several times bigger and always negative, so use a smaller positive weight, about 1 to 5. A negative weight would turn that penalty into a large reward for high touches.\n"
        "\n"
        "Watch out for: the normal formula pays even dull touches a little, and every step of a long contact. With high_touch_penalty the penalty is large: the bot may learn to stay away from the ball whenever it is in the air, even when it should save or clear it.\n"
        "\n"
        "Measured: not measured."
    ),
    "ball_y_coordinate": (
        "What it does: pays every step by how far up the field the ball is, as the car's team sees it: the ball's position along the field's length over 5,212.75 units (back wall plus the ball's radius), raised to the power exponent. About -1 at the car's own back wall, 0 at midfield, about +1 at the opponent's.\n"
        "\n"
        "When to use it: general bots, to reward keeping the ball in the opponent's half and pushing it upfield. Useful early, before goals come often. It comes from the rlgym-sim library.\n"
        "\n"
        "How to use it:\n"
        "- exponent: bends the curve (default 1, a straight line). Keep it odd: 3 or 5 flatten the middle so only deep positions count. An even number pays a plus in the car's own half too.\n"
        "- It adds up about 15 times a second at tick skip 8, so keep it small: about 0.05 to 0.5 next to a goal at 150.\n"
        "\n"
        "Watch out for: it pays where the ball is, whoever put it there, so a bot may learn to just clear the ball long and leave it in a corner rather than score. Opponents see the opposite sign, so it is already zero-sum between teams. Only whole-number exponents are allowed here.\n"
        "\n"
        "Measured: not measured."
    ),
    "behind_ball": (
        "What it does: every step, pays 1 while the car is between the ball and its own back wall (nearer its own end of the field than the ball is), else 0. Only the length of the field counts, not sideways distance or height.\n"
        "\n"
        "When to use it: early training or defence, to teach a bot to stay goal-side of the ball.\n"
        "\n"
        "How to use it:\n"
        "- No settings of its own; the weight is all there is.\n"
        "- Weight: about 0.1 to 0.5 with a goal at 150; it adds up about 15 times a second at tick skip 8 while it holds.\n"
        "\n"
        "Watch out for: it pays just as much 8,000 units away as right behind the ball, so a bot can sit in its own net and collect it. Pair it with rewards for going to and hitting the ball. Too big a weight can make a bot afraid to go past the ball to challenge.\n"
        "\n"
        "Measured: not measured."
    ),
    "behind_ball_alignment": (
        "What it does: every step, align_w times how well the ball lines up in front of the car on the way to the back of the opponents' net: +0.75 with car, ball and net in one line and the ball ahead, 0 with them at right angles, -0.75 with the car between the ball and that net. Distance does not matter.\n"
        "\n"
        "When to use it: to teach a bot to get behind the ball, facing the opponents' net, before it hits.\n"
        "\n"
        "How to use it:\n"
        "- align_w: default 0.75. The pay with everything in line, and the penalty when the car is exactly on the wrong side.\n"
        "- Weight: it pays every step, about 15 times a second, positive or negative. Try 0.25 to 2 next to a goal at 150.\n"
        "\n"
        "Watch out for: it ignores distance, so a car far away on the right line is paid as much as one about to shoot, and a bot may hang back behind the ball rather than challenge. It uses the net the car attacks.\n"
        "\n"
        "Measured: not measured."
    ),
    "boost_gain": (
        "What it does: pays on the step the car's boost goes up. With sqrt on (the default) it pays the square root of the boost gained as a share of a full tank: a small pad (12 boost) pays about 0.35, a big pad from empty pays 1. With sqrt off it pays the share itself, so a small pad pays 0.12. 0 when boost stays the same or goes down. Each car is compared with its own boost the step before.\n"
        "\n"
        "When to use it: early in training, to teach a bot to collect boost and drive through pads.\n"
        "\n"
        "How to use it:\n"
        "- sqrt: default on. On makes small pads worth more for the boost they give; off makes big pads count far more than small ones. With sqrt on it pays the same as boost_pickup_sqrt.\n"
        "- Weight: an event reward that pays at most 1 per pickup. The preset's pickup_boost uses 10 next to a goal at 150, a sensible start. Lower it once the bot collects boost on its own.\n"
        "\n"
        "Watch out for: a high weight teaches pad farming: driving from pad to pad instead of playing. A big pad taken when nearly full pays only for what fits. Each car is compared with its own boost on the step before, and the boost an episode starts with is not a pickup.\n"
        "\n"
        "Measured: not measured."
    ),
    "boost_lose": (
        "What it does: on every step the car's boost goes down, it pays boost_lose_w x the drop in the square root of (boost / 100) x (1 - car height / 643, the goal's height). Spending boost low is a penalty, at goal height it is free, and above it is a reward. A whole tank spent at floor level costs about 1 in all; near the ceiling it pays about +2.2. Because of the square root the last part of the tank costs most: 100 to 64 costs 0.2, 36 to 0 costs 0.6.\n"
        "\n"
        "When to use it: to teach saving boost on the ground and spending it for aerials. For a bot that wastes boost, mid training.\n"
        "\n"
        "How to use it:\n"
        "- boost_lose_w: multiplies the value (default 1); it does the same job as the weight.\n"
        "- Weight: about -1 per tank on the ground. With a goal at 150, about 2 to 20 (the starter's pickup_boost is 10).\n"
        "\n"
        "Watch out for: above goal height it is a reward, so a bot can learn to fly up and burn boost for nothing, refill on pads and do it again. Any fall in boost counts, whatever caused it. Too high a weight can make the bot afraid to boost on the ground even to save a goal.\n"
        "\n"
        "Measured: not measured."
    ),
    "boost_pad_pickup": (
        "What it does: pays when the car's boost goes up: the gain as a share of a full tank times big, plus the gain times small when the boost ends below 98 having started below 88 (as with a small pad). With the defaults a small pad (12 boost) pays 1.56 and a big pad from empty pays 10. It is paid on the event, once per step where boost rose; 0 otherwise.\n"
        "\n"
        "When to use it: to teach collecting boost, with small pads worth 30% more per unit of boost than a pickup that fills the tank.\n"
        "\n"
        "How to use it:\n"
        "- small: the extra pay per full tank for a gain that ends below 98 from below 88 (default 3). 0 removes the small-pad bonus.\n"
        "- big: the pay per full tank of any gain (default 10).\n"
        "- Weight: its values are already large (up to 10), so about 0.5 to 1.5 next to a goal of 150.\n"
        "\n"
        "Watch out for: any rise in boost counts, including the refill to a third of a tank when a demolished car respawns with less. A large weight teaches running pad routes instead of playing.\n"
        "\n"
        "Measured: not measured."
    ),
    "boost_pickup_sqrt": (
        "What it does: pays on the step the car gains boost: the square root of the gain as a share of a full tank. A big pad from empty pays 1, a small pad (12 boost) about 0.35. 0 when boost stays the same or goes down. It pays the same as boost_gain with sqrt on.\n"
        "\n"
        "When to use it: early in training, to teach a bot to collect boost.\n"
        "\n"
        "How to use it:\n"
        "- No settings of its own; the weight is all there is.\n"
        "- Weight: an event reward that pays at most 1 per pickup. The preset's pickup_boost uses 10 next to a goal at 150, a sensible start; lower it once the bot collects boost on its own.\n"
        "\n"
        "Watch out for: a high weight teaches pad farming: driving from pad to pad instead of playing. The square root makes small pads worth more for the boost they give, and a big pad taken when nearly full pays only for what fits. Use only one of this and boost_gain, or the bot is paid twice for every pad.\n"
        "\n"
        "Measured: not measured."
    ),
    "boost_use": (
        "What it does: pays on each step the car's boost goes down: the square root of the boost spent as a share of a full tank. A step of full boosting at tick skip 8 (about 2.2 boost) pays about 0.15. It is a reward for using boost, not a penalty. 0 when boost stays the same or goes up. Each car is compared with its own boost the step before.\n"
        "\n"
        "When to use it: to stop a bot from hoarding boost and teach it to use boost for speed.\n"
        "\n"
        "How to use it:\n"
        "- No settings of its own; the weight is all there is.\n"
        "- Weight: it pays on every boosting step, about 2.2 a second of full boost at weight 1 and tick skip 8, so keep it small: around 0.5 to 3 next to a goal at 150.\n"
        "\n"
        "Watch out for: it pays for spending boost on anything, so at a high weight the bot boosts all the time, even into walls or while already at top speed, and ends up empty when it matters. It works against save_boost. Because of the square root, and because event rewards are not scaled for tick skip, it pays more per second at a smaller tick skip (about 3 a second at tick skip 4). Each car is compared with its own boost on the step before.\n"
        "\n"
        "Measured: not measured."
    ),
    "breezi_flick": (
        "What it does: a bonus for a Breezi: the ball controlled on the roof, a pop, a tornado spin in the air, then a ground musty, judged as by musty_flick. When that musty is judged, it pays breezi_scale times the musty's credit times four spin parts (total turn, roll, yaw, not flipping over), plus musty_share times the musty. About 0 to 1 by default; never negative while musty_share is 0.\n"
        "\n"
        "When to use it: a Breezi bot (1v1, tick skip 4, the breezi_bot reward set), beside musty_flick and tornado_shaping. For a bot that already mustys.\n"
        "\n"
        "How to use it:\n"
        "- Pay (defaults after =): musty_share=0, breezi_scale=1.\n"
        "- Spin, degrees from lift-off to dodge: spin_zero_deg=200, spin_full_deg=290, roll_zero_deg=30, roll_full_deg=80, yaw_zero_deg=60, yaw_full_deg=150, pitch_free_deg=180, pitch_zero_deg=300, nose_up_free_deg=30, nose_up_zero_deg=90, tornado_sign=on (roll and yaw the same way).\n"
        "- Pop: pop_window=0.25 seconds, ground_only=on; roof control: dribble_min_up=105, dribble_max_up=185, dribble_target_forward=20, dribble_max_planar_dist=90, dribble_max_rel_speed=600, control_threshold=0.25.\n"
        "- Clean air: ball_rest_margin=8, bounce_margin=80, bounce_dv=150, car_clear_margin=4.\n"
        "- Air goal: air_goal_bonus=0: paid again if the Breezi scores before any bounce or touch.\n"
        "- The musty, as musty_flick's with the same defaults: arm_max_torque_y, arm_min_nose_down_deg, arm_max_nose_down_deg, arm_max_ball_dist, arm_min_ball_above, arm_min_up_dot_ball, arm_max_lateral, strike_window, top_min_up, hitbox_length, hitbox_width, hitbox_height, hitbox_offset_forward, hitbox_offset_right, hitbox_offset_up, min_impulse, full_impulse, min_lift, full_lift, full_gain, full_launch_speed, scoop_rear_x, scoop_front_x, scoop_level_deg, scoop_zero_deg, scoop_floor, shot_on_target, corner_from_x, corner_full_x, shot_max_time, w_power, w_impulse, w_lift, w_shot, aim_cos_zero, aim_cos_full, elev_zero_low, elev_full_low, elev_full_high, elev_zero_high, aerial_elev_zero_low, aerial_elev_full_low, base_credit, aerial_bonus, on_net_bonus, on_net_power_mix, stretch_shot_weight, stretch_shot_from, stretch_shot_full, stretch_gain_weight, stretch_gain_from, stretch_gain_full, cancel_on_car_contact, car_contact_margin, car_contact_lookback, flick_weight, arm_credit, backward_penalty, min_penalty_impulse, separation_dist, resolve_time, release_steps, ride_hover_gap, ride_leave_speed, wall_margin, surface_margin, start_jump_margin, reset_max_ball_dist, start_reset_max_ball_dist.\n"
        "- Weight: rare, so large; the Breezi set uses 400 beside musty_flick 200, musty extras on.\n"
        "\n"
        "Watch out for: every part must happen, so it pays rarely; tornado_shaping does the early teaching. A surface touch by ball or car, or another car's touch, between lift-off and the hit cancels it. Keep musty_share 0 beside musty_flick, or the musty pays twice. A start in the air has no pop, so its first flight cannot pay.\n"
        "\n"
        "Measured: not measured."
    ),
    "bump": (
        "What it does: pays 1 on a step where this car bumps an opponent. Only the bumping car is paid, not the car that was hit, and bumping a teammate pays nothing. A demolition also counts as a bump. It is paid on the event, at most 1 per step.\n"
        "\n"
        "When to use it: for bots that should play physically: bumping opponents off the ball, out of position or away from a save. Usually added once the bot can already reach and hit the ball.\n"
        "\n"
        "How to use it:\n"
        "- No settings of its own; the weight is all there is.\n"
        "- Weight: the default mix uses 20, shared with zero-sum (team spirit 0.5), next to a goal of 150. Between 5 and 20 keeps it below scoring.\n"
        "\n"
        "Watch out for: a large weight teaches hunting cars instead of playing the ball. With zero-sum, the bumped opponent is charged what the bumper earns, which also teaches avoiding being bumped. Together with demo, a demolition pays both rewards.\n"
        "\n"
        "Measured: not measured."
    ),
    "demo": (
        "What it does: pays 1 on a step where this car demolishes an opponent. It is paid on the event, once per demolition. A demolition also counts as a bump, so the bump reward pays for it too if you use both.\n"
        "\n"
        "When to use it: for bots that should demolish opponents when it helps, typically in later training once the bot plays the ball well. A demolished opponent is out of play for about 3 seconds, which already helps scoring.\n"
        "\n"
        "How to use it:\n"
        "- No settings of its own; the weight is all there is.\n"
        "- Weight: the default mix uses 80, shared with zero-sum (team spirit 0.5), next to a goal of 150. Keeping it well below the goal reward stops demos from becoming the main goal.\n"
        "\n"
        "Watch out for: a large weight teaches chasing demolitions instead of the ball, and leaving the net open to get one. With zero-sum, the demolished car is charged what the demolisher earns, which also teaches dodging demos. Demolishing teammates pays nothing.\n"
        "\n"
        "Measured: not measured."
    ),
    "distance_to_ball": (
        "What it does: every step it pays the car's distance to the ball, centre to centre, divided by 2,000 units: under 0.1 when touching, 1 at 2,000 units, about 6.6 from the far corner. It grows the farther away the car is, so a positive weight pays staying away. Give it a negative weight to pay being near the ball.\n"
        "\n"
        "When to use it: with a negative weight, to teach a brand-new bot to go to the ball, in the first stage of training.\n"
        "\n"
        "How to use it:\n"
        "- No settings of its own; the weight is all there is.\n"
        "- Weight: negative, about -0.05 to -0.5 with a goal at 150. At -0.5 a car 4,000 units away loses 1 per step, about 15 a second.\n"
        "\n"
        "Watch out for: with a negative weight it is a penalty on every step that never pays anything positive, so the bot can learn that anything ending the game sooner, even a goal against, is welcome. It ignores where the ball is going, so it teaches ball chasing, and in team games every car crowds the ball. A positive weight teaches the bot to keep away.\n"
        "\n"
        "Measured: not measured."
    ),
    "dont_boost_while_supersonic": (
        "What it does: pays -1 on each step where the car's boost went down while it is faster than 2,200 units/s (supersonic), and 0 otherwise. It is counted as an event reward, so it is paid on each such step without the usual tick-skip scaling.\n"
        "\n"
        "When to use it: to teach a bot not to waste boost once it is already near top speed (top speed is 2,300, so there is little left to gain). Best once the bot drives and boosts well.\n"
        "\n"
        "How to use it:\n"
        "- No settings of its own; the weight is all there is. Use a positive weight: the -1 already makes it a penalty.\n"
        "- Weight: about 0.1 to 1 next to a goal of 150. At weight 1, a second of boosting while supersonic costs about 15 at tick skip 8.\n"
        "\n"
        "Watch out for: it also charges boost used while supersonic in the air, for example to keep an aerial going, and when holding top speed for a demo. Because it is not scaled to the tick skip, the same weight costs twice as much per second at tick skip 4 as at 8. A negative weight would reward wasting boost.\n"
        "\n"
        "Measured: not measured."
    ),
    "dribble_carry": (
        "What it does: every step the ball is between about 113 and 293 units off the floor and within 150 units of the car both across and along the field, it pays 0.2; on a step the car also touches the ball, 1. Otherwise 0. Range 0 to 1.\n"
        "\n"
        "When to use it: to teach ground dribbling: keeping the ball on or just above the car. For a bot that already reaches the ball, early to mid training.\n"
        "\n"
        "How to use it:\n"
        "- No settings of its own; the weight is all there is.\n"
        "- Weight: up to 1 per step, about 15 a second. With a goal at 150, about 0.2 to 2.\n"
        "\n"
        "Watch out for: it does not check that the car is under the ball, so a car beside or above a bouncing ball also passes. The distances are compared from the centre lines, so a car at the mirrored spot across a centre line from the ball also counts. It pays for as long as the ball stays up, so the bot may dribble forever instead of shooting.\n"
        "\n"
        "Measured: not measured."
    ),
    "dribble_cradle": (
        "What it does: every step the ball sits just above the car: between about 113 and 293 units off the floor, its centre within 170 units of the car's, the car below it, and the ball at least about 150 units from the side and back walls, and no opponent is within minimum_barrier of the ball, it pays 1. Otherwise 0.\n"
        "\n"
        "When to use it: to teach ground dribbling: keeping the ball on the car's roof. For a bot that already reaches and touches the ball and should start controlling it.\n"
        "\n"
        "How to use it:\n"
        "- minimum_barrier: an opponent this close to the ball, in units centre to centre, stops the pay (default 200; 0 turns the check off). Higher values pay only dribbles in open space.\n"
        "- Weight: 1 per step adds up to about 15 a second. With a goal at 150, about 0.2 to 2 keeps a long dribble worth less than scoring.\n"
        "\n"
        "Watch out for: it pays for as long as the ball stays on the roof, so the bot can learn to dribble forever and never shoot; add goal and ball-toward-goal rewards. The height test is measured from the floor, so wall dribbles never count. Because an opponent near the ball stops the pay, it can teach running away from challenges.\n"
        "\n"
        "Measured: not measured."
    ),
    "dribble_flick": (
        "What it does: every step the ball sits on the car (dribble_cradle's test: about 113 to 293 units up, within 170 units, car below it, away from the walls), it pays 0.5. If the carry is steady (ball under 173 units up, moving within max_vel_diff of the car), it adds: +1 with no opponent within minimum_barrier of the ball; with one that close, -0.5 on the ground, +2 in the air with the flip left, +5 once the flip is gone (the flick). -0.5 to 5.5 per step; 0 without the ball on the car.\n"
        "\n"
        "When to use it: to teach dribbling and flicking when an opponent comes to challenge. For a ground dribbling bot that plays against opponents, mid to late training.\n"
        "\n"
        "How to use it:\n"
        "- minimum_barrier: how close an opponent must be to the ball, in units centre to centre, to count as a challenge (default 400).\n"
        "- max_vel_diff: how far the ball's velocity may differ from the car's for a steady carry, in units per second (default 400). Lower is stricter.\n"
        "- training: true (default) pays the 0.5 for any hold; false keeps only the carry and flick parts.\n"
        "- Weight: up to 5.5 per step, about 15 times a second, so keep it small: about 0.1 to 1 with a goal at 150.\n"
        "\n"
        "Watch out for: the +5 only checks that the car is in the air, challenged, with no flip left and the ball still on it, not that a flick happened, so a car whose flip simply ran out earns it too. Under 173 units, in the air means a small hop. With no opponent it is never challenged.\n"
        "\n"
        "Measured: not measured."
    ),
    "dribble_match_speed": (
        "What it does: every step the car is on the ground within 197 units (centre to centre) of a ball 109 to 180 units high, it pays the car's speed out of 2,300 units/s plus 2 times how closely the car's speed matches the ball's (2 for the same speed, less as they differ). 0 to 3 a step; 0 otherwise.\n"
        "\n"
        "When to use it: ground dribbling, carrying the ball on the roof at speed.\n"
        "\n"
        "How to use it:\n"
        "- No settings of its own; the weight is all there is.\n"
        "- Weight: about 0.1 to 1 with a goal at 150, since it pays up to 3 a step and it adds up about 15 times a second at tick skip 8.\n"
        "\n"
        "Watch out for: the speed match pays 2 even when car and ball are both still, so a bot can balance the ball on a stopped car and collect 2 a step. Only the speeds are compared, not their directions. The ball need not be on the roof: a slightly lifted ball right beside the car counts, while a ball rolling on the floor (93 units high) does not.\n"
        "\n"
        "Measured: not measured."
    ),
    "dribble_roof": (
        "What it does: every step the ball is between about 113 and 293 units high and within 150 units of the car both across and along the field, away from the walls (within 3,900 units of the centre across, 5,000 along): 0.2, or 1 on a step the car touches the ball. 0 otherwise.\n"
        "\n"
        "When to use it: to teach carrying the ball on the roof.\n"
        "\n"
        "How to use it:\n"
        "- No settings of its own; the weight is all there is.\n"
        "- Weight: 0.2 a step, about 3 a second at weight 1, and up to 15 a second with a touch every step. Try 1 to 5 next to a goal at 150.\n"
        "\n"
        "Watch out for: it does not check that the ball is on the car: a ball bouncing low beside the car, or a car in the air under it, also pays. It compares distances from the centre lines, so a car at the mirror spot across a centre line (say 500 units left with the ball 500 right) passes too. Touching every step pays five times more, which can teach pushing the ball along rather than balancing it. It pays nothing for moving toward the goal.\n"
        "\n"
        "Measured: not measured."
    ),
    "dribble_upfield": (
        "What it does: every step the ball sits just above the car: between about 113 and 293 units off the floor, its centre within 170 units of the car's, the car below it, and the ball at least about 150 units from the side and back walls, and no opponent is within minimum_barrier of the ball, it pays by how far up the field the car is, seen from its own team's side: about 0.1 near its own back wall, 0.71 at the centre line, 1 near the opponents' back wall. Otherwise 0.\n"
        "\n"
        "When to use it: to teach carrying the ball on the roof up the field toward the opponents' goal. For a ground dribbling bot, once it reaches the ball reliably.\n"
        "\n"
        "How to use it:\n"
        "- minimum_barrier: an opponent this close to the ball, in units centre to centre, stops the pay (default 200; 0 turns the check off).\n"
        "- Weight: up to 1 per step, about 15 a second. With a goal at 150, about 0.2 to 2.\n"
        "\n"
        "Watch out for: it pays for as long as the ball stays on the roof, so the bot can learn to dribble forever instead of shooting. The pay rises slowly (a square root), so a dribble in the own half already pays well. Wall dribbles never count.\n"
        "\n"
        "Measured: not measured."
    ),
    "example_ball_height": (
        "What it does: pays, every step, how high the ball is: 0 with the ball on the floor, 1 with it at the ceiling (2,044 units), in a straight line between. With power above 1 the curve bends, so low balls pay almost nothing and high ones much more: at power 2, a ball halfway up pays 0.25. Every car on the field is paid the same, because the ball's height says nothing about who lifted it.\n"
        "\n"
        "When to use it: it is the example that shows how a reward of your own is written in Rust, always built into NitroLearn. Copy its file (crates/nitro_core/src/custom/example.rs), rename it, change the maths, add the copy on the Resources page and press Build. As a reward for a real bot it is weak.\n"
        "\n"
        "How to use it: leave it out of real bots. To try a reward of your own, start from its file, which has every part a reward needs: the settings as fields, every_step, and the value for one car.\n"
        "\n"
        "Watch out for: because every car gets the same pay, zero-sum cancels it completely, and without zero-sum both teams are paid for a high ball whoever lifted it. Paid every step, it is scaled to the tick skip like the other every-step rewards."
    ),
    "face_ball": (
        "What it does: pays, every step, how directly the car's nose points at the ball: 1 straight at it, 0 sideways, -1 facing away. It is paid about 15 times a second at tick skip 8.\n"
        "\n"
        "When to use it: early in training, as a small nudge to keep the ball in front of the car, which makes driving to it and hitting it easier to learn. The default mix has it at 0.25.\n"
        "\n"
        "How to use it:\n"
        "- No settings of its own; the weight is all there is.\n"
        "- Weight: small, about 0.1 to 0.5 next to a goal of 150. At 0.25, facing the ball for a whole second earns about 3.75.\n"
        "\n"
        "Watch out for: it pays while standing still, so on its own it teaches watching the ball, not playing it. It charges the car for turning away, for example to rotate back, grab boost or wait in goal, so a large weight makes a bot reluctant to do those. The pointing is in 3D: a ball high above the car scores near 0.\n"
        "\n"
        "Measured: not measured."
    ),
    "flip_reset": (
        "What it does: pays flip resets off the ball in the air, in parts, on events: the approach (only new progress, at most 4.5 an episode), the reset (10 times its style), following it through (3 times style), using it within 2 seconds (8 times style for a full use, twice that for a delayed one) and a goal off a used reset (20). Style grows for flipping or spinning into the reset, a later reset, low boost or a long way from goal.\n"
        "\n"
        "When to use it: air dribble and flip reset bots (the air dribble set: weight 5).\n"
        "\n"
        "How to use it:\n"
        "- Approach: setup_scale 3, setup_max_dist 2,000, setup_full_dist 200, setup_close_exponent 3, setup_flip_ready_factor 0.4, max_above_ball 100, setup_above_looseness 600, max_setup_per_episode 4.5.\n"
        "- Reset: min_ball_height 300, min_air_time 0 (off), reset_reward 10, max_reset_ball_dist 350, max_resets_per_episode 5, min_reset_interval 0.5, multi_reset_full_progress 800.\n"
        "- Style: flip_into_reset_bonus 0.4, backflip_into_reset_bonus 0.75, flip_into_reset_window 1, spin_reset_bonus 0.3, spin_min_rate 3.5, spin_full_rate 5.25, multi_reset_step 0.25, max_multi_reset_mult 2, need_bonus 0.5, need_full_dist 8,000.\n"
        "- Follow: follow_reward 3, follow_time 0.75, max_follow_distance 550, above_margin 0, ahead_margin 50.\n"
        "- Use: use_reward 8, use_window 2, use_flip_base 0.3, use_full_gain 1,500, use_reset_quality 0.6, use_touch_quality 0.4, dodge_hit_window 0.65, min_use_ball_height 0, delayed_setup_time 0.25, delayed_hit_time 0.3, delayed_use_bonus 1, separation_dist 180.\n"
        "- Shooting (off by default): use_aim pays a dodge use only when aimed at the net; quick_use_factor (1) scales a dodge before delayed_setup_time; wasted_dodge_cost (0) charges a missed or off-target dodge; use_aim_reach (false): the ball must reach the net in the air. reset_height_bonus (0) pays a higher reset more, full at reset_height_full (1,500).\n"
        "- Goal: goal_reward 20, goal_window 3. start_reset_counts (false): an airborne start holding its flip near a high ball holds a reset.\n"
        "- One reset pays 10 or more: weights near 5 already matter.\n"
        "\n"
        "Watch out for: resets pay big: too high a weight teaches resets that go nowhere (a reset shot bot farmed them). A reset still open at a time-out is not judged.\n"
        "\n"
        "Measured: not measured."
    ),
    "flip_reset_approach": (
        "What it does: every step, it multiplies five parts that describe setting up a flip reset: the ball lies toward the car's underside (counted double), the ball is out toward a side wall (full from 1,300 units off the centre line), the ball is high (full at 1,700 units), the car's nose points at the opponents' net, and the car's wheels face the ceiling while it is close to the ball. The top is 2 on paper, but the closeness part stays small (about 0.2 even touching the ball), so in practice it pays about 0.4 at most. It goes negative when one part is negative.\n"
        "\n"
        "When to use it: flip-reset training for bots that already fly, to guide them into the upside-down pose under a high ball. It is the helper half of a pair, used with flip_reset_simple (both weight 1 in the flip_reset_simple reward set).\n"
        "\n"
        "How to use it:\n"
        "- flip_reset_help_w: default 1. Multiplies the whole thing; 0 turns it off. It does the same job as the weight, so leave it at 1.\n"
        "- Weight: the flip_reset_simple set uses 1. It pays every step, so it adds up over long flights.\n"
        "\n"
        "Watch out for: a ball near the middle of the field (left to right) pays little or nothing. Two negative parts multiply into a positive, so a car the right way up under a high ball, nose to the net, also earns a little. It pays for holding the pose, not for getting the reset.\n"
        "\n"
        "Measured: not measured."
    ),
    "flip_reset_height": (
        "What it does: pays when the car, in the air, touches the ball and gets its flip back that step (a flip reset). It pays the car's height out of 1,500 units, at most 1: a reset 750 units up pays 0.5, one at 1,500 or higher pays 1. Once per reset; 0 otherwise.\n"
        "\n"
        "When to use it: a flip reset bot. It rewards the reset itself, higher ones more.\n"
        "\n"
        "How to use it:\n"
        "- No settings of its own; the weight is all there is.\n"
        "- Weight: about 5 to 30 with a goal at 150. Resets are rare events, so it has to stand out.\n"
        "\n"
        "Watch out for: there is no limit per episode and no wait between resets, so a bot that can reset, dodge and reset again is paid every time. It does not pay for using the reset (a shot or a goal), so pair it with shooting rewards. Low resets near the floor pay little but still pay.\n"
        "\n"
        "Measured: not measured."
    ),
    "flip_reset_height_v2": (
        "What it does: when the car gets its flip back while it is higher than minimum_height, it pays the car's height x height_scaling: 250 at 1,000 units, about 500 near the ceiling with the defaults. Once per regained flip. A real flip reset off the ball counts, but so does landing on a wall or the ceiling above that height after using the jump or flip.\n"
        "\n"
        "When to use it: to teach flip resets, for a freestyle or air dribble bot late in training.\n"
        "\n"
        "How to use it:\n"
        "- height_scaling: what each unit of height is worth (default 0.25).\n"
        "- minimum_height: only higher than this, in units (default 100). A car on the floor is about 17 units up; raise it, say to 300 to 500, to skip low wall landings.\n"
        "- Weight: the values are huge, so the weight must be tiny: about 0.01 to 0.1 with a goal at 150 (a reset at 1,000 units at weight 0.05 pays 12.5).\n"
        "\n"
        "Watch out for: it does not check that the ball was involved, so a bot can farm it by jumping off a wall and landing back on it high up. flip_reset_event and flip_reset_height check for the ball.\n"
        "\n"
        "Measured: not measured."
    ),
    "flip_reset_simple": (
        "What it does: pays on the step the car gets a flip reset: its jump comes back while in the air, with the ball above 200 units, within 110 units of the car and under its wheels. With the defaults: 10 for the first reset of each flight, plus 0.5 for the car's first reset of the episode, so 10.5 at most. Further resets in the same flight pay nothing unless prevent_chain_reset is off. Landing starts a new flight.\n"
        "\n"
        "When to use it: to teach flip resets to bots that already fly, with flip_reset_approach guiding them into position.\n"
        "\n"
        "How to use it:\n"
        "- flip_reset_w: default 10. What a counted reset pays.\n"
        "- quick_flip_reset_w: default 5. The car's first reset of the episode adds a tenth of it (0.5).\n"
        "- inc_flip_reset_w: default 2. With every reset counted, the second, third and later reset in a row adds up to this (about 0.65 for the second, the full 2 from the sixth). It does nothing while prevent_chain_reset is on.\n"
        "- prevent_chain_reset: default on: only the first reset until the car is on the ground again. Off pays every reset plus the in-a-row bonus.\n"
        "- Weight: 1 makes a reset worth about 10, under a tenth of a goal at 150. Raise it if resets stay rare.\n"
        "\n"
        "Watch out for: with prevent_chain_reset off, a bot can farm it by bouncing its wheels off the ball again and again without shooting. Nothing here pays for what happens after the reset. Each car counts its own resets.\n"
        "\n"
        "Measured: not measured."
    ),
    "goal": (
        "What it does: when a goal is scored, pays 1 to every player of the scoring team and concede_scale (by default -1) to every player of the other team. It is paid on the event, once per goal. Everyone on the team is paid the same, whoever touched last. An own goal counts as a goal for the other team.\n"
        "\n"
        "When to use it: almost always: it is the real aim, and every other reward should help toward it. It carries the most weight in most mixes, from the first step of training to the last.\n"
        "\n"
        "How to use it:\n"
        "- concede_scale: what conceding is worth compared with scoring (default -1, as bad as scoring is good). -0.8 makes conceding a little less bad, 0 ignores it. A positive value would reward conceding, so do not use one.\n"
        "- Weight: the default mix uses 150; the shot and save practice tasks use 50. Set the other rewards against it.\n"
        "\n"
        "Watch out for: with the default -1 it is already zero-sum (what one side gains the other loses); adding zero_sum on top charges the opponents twice and doubles the pay. It gives no credit to the actual scorer in team modes. It is rare early on, so a new bot learns little from it alone.\n"
        "\n"
        "Measured: not measured."
    ),
    "goal_in_mouth": (
        "What it does: pays +1 on the step the ball first gets inside the goal mouth this car attacks (past the line by half a ball's radius, between the posts and under the crossbar), and -1 when it gets into the car's own. Both teams are paid. Once per entry, usually a moment before the game counts the goal.\n"
        "\n"
        "When to use it: as the goal reward of a trick bot. The Raze, Musty and Breezi sets use it at 25, with velocity_player_to_ball at 4 and velocity_ball_to_goal at 2.\n"
        "\n"
        "How to use it:\n"
        "- No settings of its own; the weight is all there is.\n"
        "- Event reward of 1: treat the weight like a goal's (the flick sets use 25).\n"
        "\n"
        "Watch out for: with the built-in goal reward on as well, every goal pays twice, so set both weights knowing that. It pays a little earlier than the goal itself, since here the game runs on until the goal counts, a little deeper in the net.\n"
        "\n"
        "Measured: not measured."
    ),
    "goal_speed": (
        "What it does: pays on the step your team scores: the ball's speed at that moment, in units/s, times reward_factor times 2. With the default that is 600 for a 3,000 units/s goal and up to 1,200 at the ball's top speed of 6,000. Every car of the scoring team is paid, once per goal.\n"
        "\n"
        "When to use it: once a bot scores, to make it prefer hard shots over slow rollers.\n"
        "\n"
        "How to use it:\n"
        "- reward_factor: the pay per unit/s of ball speed before the doubling (default 0.1). Lowering it does the same as lowering the weight.\n"
        "- Weight: it pays hundreds per goal, so give it a very low weight: about 0.005 to 0.02 makes a 3,000 units/s goal pay 3 to 12, next to your normal goal reward.\n"
        "\n"
        "Watch out for: even a slow goal pays something. It looks only at which team scored, so the other team's own goal pays too. At a normal weight it drowns out every other reward.\n"
        "\n"
        "Measured: not measured."
    ),
    "goal_speed_and_placement": (
        "What it does: pays on the step your team scores: the ball's speed on the step before the goal over 6,000 units/s (0 to 1), times 1.75 if the ball was then off the floor (its centre higher than about 103 units; a ball resting on the floor sits at about 93). So 0 to 1.75, once per goal, to every car of the scoring team.\n"
        "\n"
        "When to use it: as a bonus on top of a goal reward, for a bot that already scores, to favour fast and lifted shots that are harder to save.\n"
        "\n"
        "How to use it:\n"
        "- No settings of its own; the weight is all there is.\n"
        "- Weight: an event that pays at most 1.75. A quarter to the same as your goal reward's weight makes a fast lifted shot worth clearly more than a slow roller.\n"
        "\n"
        'Watch out for: "off the floor" is any height above about 103 units, so a ball that bounces slightly counts as lifted. It pays the scoring team whoever touched the ball last.\n'
        "\n"
        "Measured: not measured."
    ),
    "goal_speed_bonus": (
        "What it does: on a goal, pays the ball's speed on the step before the goal over 6,000 units/s (its top speed), times goal_speed_bonus_w. Plus for every car of the scoring team, minus for every car of the other: from 0 for a slow roller up to 1 at top speed with the default. Once per goal.\n"
        "\n"
        "When to use it: once a bot scores, to teach hard shots, and to make conceding fast shots cost more.\n"
        "\n"
        "How to use it:\n"
        "- goal_speed_bonus_w: what a goal at the ball's top speed is worth (default 1). It multiplies the weight, so leave it at 1 and use the weight.\n"
        "- Event reward, 0 to 1: a weight of about 20 to 75 next to a goal at 150, so it adds to the goal rather than replacing it.\n"
        "\n"
        "Watch out for: it pays only goals, so it adds nothing until a bot scores. Too big a weight next to the goal reward can make a bot pass up easy slow goals for hard shots. Conceding costs as much as scoring pays. Every car of both teams is paid or charged.\n"
        "\n"
        "Measured: not measured."
    ),
    "goal_speed_top_corner": (
        "What it does: on a goal, pays the ball's speed on the step before the goal over 6,000 units/s (its top speed), times goal_speed_bonus_w, times where the ball went in: its height over 640 times its distance from the middle over 640. Plus for every car of the scoring team, minus for every car of the other. About 0 for a goal along the floor or through the middle, roughly 1 or a little more for a fast top-corner goal. Once per goal.\n"
        "\n"
        "When to use it: once a bot scores, to teach fast, well-placed shots that are hard to save.\n"
        "\n"
        "How to use it:\n"
        "- goal_speed_bonus_w: what a goal at the ball's top speed is worth before the placement (default 1). It multiplies the weight, so leave it at 1 and use the weight.\n"
        "- Event reward: a weight of about 20 to 75 next to a goal at 150, so it adds to the goal rather than replacing it.\n"
        "\n"
        "Watch out for: height and side multiply, so a high shot through the middle or a low one into a corner pays almost nothing. A slow corner goal pays little too. Conceding costs as much as scoring pays. Every car of both teams is paid or charged.\n"
        "\n"
        "Measured: not measured."
    ),
    "high_touch": (
        "What it does: on a step the car touches the ball, the smaller of two parts: its time in the air (full at 1.75 seconds) and the ball's height as a share of the ceiling's (2,044 units). So a touch at 1,022 units after a full second in the air pays 0.5. 0 to 1; 0 for a touch from the ground or a wall, or without a touch.\n"
        "\n"
        "When to use it: aerial training: it rewards high touches after a long flight.\n"
        "\n"
        "How to use it:\n"
        "- No settings of its own; the weight is all there is.\n"
        "- Weight: an event reward of at most 1 per touch step. Try 5 to 30 next to a goal at 150 (the preset's strong_touch uses 60).\n"
        "\n"
        "Watch out for: it pays for every step with a touch, so an air dribble with a touch each step collects it many times. It does not care where the ball goes.\n"
        "\n"
        "Measured: not measured."
    ),
    "jump_touch": (
        "What it does: on a touch with the car in the air and the ball at least min_height up, pays by the ball's height: from 0 with the ball resting on the floor up to 1 with the ball at 1,022 units (half the ceiling) or higher. Pays only above 0.05. After a paid touch, that car's touches pay nothing for the next 149 steps (about 5 seconds at the default tick skip of 4, 10 at tick skip 8).\n"
        "\n"
        "When to use it: early aerial training, to reward reaching and hitting high balls at all.\n"
        "\n"
        "How to use it:\n"
        "- min_height: the lowest ball height, in units, for a touch to count (default 200).\n"
        "- exp: the curve (default 1, straight). Below 1, such as 0.5, pays low touches more; above 1 saves the pay for high ones.\n"
        "- car_on_ground: off by default. On, it pays touches with the car on its wheels (floor or wall) instead.\n"
        "- Event reward, 0 to 1: a weight of about 10 to 50 next to a goal at 150.\n"
        "\n"
        "Watch out for: the wait is counted in steps, so at tick skip 4 it is only about 5 seconds. Only one touch per wait pays, so it does not teach air dribbles. A small hop under a bouncing ball counts as in the air.\n"
        "\n"
        "Measured: not measured."
    ),
    "kickoff": (
        "What it does: pays, every step while the ball still sits exactly on the centre spot, 1 if this car is nearer the ball than every opponent and -0.8 otherwise (also in a tie, and always -0.8 with no opponents). Once the ball has moved it pays 0. Paid about 15 times a second at tick skip 8.\n"
        "\n"
        "When to use it: to teach winning the race to the ball at kickoffs, for bots trained with kickoff starts.\n"
        "\n"
        "How to use it:\n"
        "- No settings of its own; the weight is all there is.\n"
        "- Weight: about 0.5 to 2 next to a goal of 150. A kickoff lasts a couple of seconds, so the pay adds up over about 30 steps.\n"
        "\n"
        "Watch out for: in 2v2 and 3v3 every car is compared with the nearest opponent, so a teammate who stays back is charged -0.8 every step, which pushes every car to rush the ball. Alone (no opponents) it is a penalty until the ball is hit, which teaches hitting it sooner. Only the exact centre spot counts, so with other start types it usually pays nothing. It pays being nearer, not a good kickoff touch.\n"
        "\n"
        "Measured: not measured."
    ),
    "kickoff_nearest": (
        "What it does: during a kickoff (while the ball is still exactly on the centre spot), every step it pays 1 if the car is nearer the ball than every opponent, else -0.8. A tie, or no opponent at all, pays -0.8. Once the ball has moved it pays 0. Teammates do not count.\n"
        "\n"
        "When to use it: to teach getting to the ball first at kickoffs, mainly in 1v1. Early to mid training.\n"
        "\n"
        "How to use it:\n"
        "- No settings of its own; the weight is all there is.\n"
        "- Weight: about plus or minus 1 per step until the first touch, a few seconds. With a goal at 150, about 0.5 to 3.\n"
        "\n"
        "Watch out for: in 2v2 and 3v3 every car that is not the nearest is charged -0.8 each step, so a teammate who should stay back is punished; it teaches every car to rush. It pays being first, not winning the kickoff, and in practice with no opponent it is only a penalty. Any start that leaves a still ball on the centre spot counts as a kickoff.\n"
        "\n"
        "Measured: not measured."
    ),
    "kickoff_nearest_team": (
        "What it does: every step while the ball stands still (a kickoff, before the first touch), it pays +1 to every car of the team whose nearest car is closer to the ball than the other team's nearest car, and -1 to every car of the other team. A tie, or no opponent at all, pays -1. 0 once the ball moves.\n"
        "\n"
        "When to use it: teaching a team, in 1v1, 2v2 or 3v3, to win the race to the ball on kickoffs.\n"
        "\n"
        "How to use it:\n"
        "- No settings of its own; the weight is all there is.\n"
        "- Weight: about 0.1 to 1 with a goal at 150. If a kickoff takes 2 to 3 seconds, that is about 30 to 45 steps at tick skip 8, so at weight 1 it can add up to about 40 either way.\n"
        "\n"
        "Watch out for: on an even kickoff both teams start equally near, so both get -1 until one pulls ahead. Any start with the ball standing still counts, not only centre kickoffs. The whole team is paid for its nearest car, so a second car is paid or charged for its teammate. With no opponent it only ever costs, so leave it out of one-car training.\n"
        "\n"
        "Measured: not measured."
    ),
    "kickoff_nearest_v2": (
        "What it does: during a kickoff (while the ball is still exactly on the centre spot), every step the car is farther than min_distance from the ball it pays 1 if the car is nearer the ball than every opponent, else -1 (a tie, or no opponent at all, is -1). Within min_distance, and once the ball has moved, it pays 0. Teammates do not count.\n"
        "\n"
        "When to use it: to teach winning the race to the ball at kickoffs, mainly in 1v1. Early to mid training.\n"
        "\n"
        "How to use it:\n"
        "- min_distance: how close to the ball, in units, the pay stops (default 1,200). The kickoff spots are about 3,300 to 4,600 units from the ball; lower values keep paying for the race nearer the ball, 0 all the way.\n"
        "- Weight: plus or minus 1 per step for a second or two. With a goal at 150, about 0.5 to 3.\n"
        "\n"
        "Watch out for: in 2v2 and 3v3 every car that is not the nearest gets -1 each step, so a teammate who should stay back is punished; it teaches every car to rush. It only pays being first, not winning the kickoff. Any start that leaves a still ball on the centre spot counts as a kickoff.\n"
        "\n"
        "Measured: not measured."
    ),
    "kickoff_speed_no_boost": (
        "What it does: every step while the ball stands still (a kickoff) and the car has less than 2 boost out of 100, it pays the car's speed toward the ball out of 2,300 units/s: 0 to 1, and 0 when moving away. Nothing once the ball moves, or while the car still has boost.\n"
        "\n"
        "When to use it: kickoff practice. It teaches spending all the kickoff boost and arriving at the ball at top speed, as a speed flip kickoff does. Despite its name it does not check for a speed flip.\n"
        "\n"
        "How to use it:\n"
        "- No settings of its own; the weight is all there is.\n"
        "- Weight: about 0.5 to 2 with a goal at 150. It pays only for the moments between the boost running out and the first touch, about 15 times a second at tick skip 8.\n"
        "\n"
        "Watch out for: a car that keeps 2 or more boost gets nothing, so it pushes the bot to empty its tank on every kickoff, even when saving boost would be better. It stops at the first touch, so it does not pay for winning the kickoff or for a good hit. Any start with a still ball and an empty tank counts too.\n"
        "\n"
        "Measured: not measured."
    ),
    "kickoff_speed_no_boost_v2": (
        "What it does: every step of a kickoff (the ball exactly on the centre spot) while the car's boost is below max_boost: goal_speed times its speed toward the ball as a share of 2,300 units/s, but never less than 2.5% of goal_speed. So 0.0125 to 0.5 a step with the defaults, even when moving away. 0 at any other time.\n"
        "\n"
        "When to use it: kickoff training: it pays for keeping full speed toward the ball once the starting boost (33) is used up, as in a fast speed-flip kickoff.\n"
        "\n"
        "How to use it:\n"
        "- goal_speed: default 0.5. The pay at top speed straight at the ball.\n"
        "- max_boost: default 2 (boost runs 0 to 100). Pays only below it. 101 pays at any boost.\n"
        "- Weight: paid every step until the first touch, about 15 times a second, so a kickoff is worth up to about 10 times the weight. Try 1 to 5 next to a goal at 150.\n"
        "\n"
        "Watch out for: it does not check for a flip; only speed counts. With the default it pays nothing until the car has burned its boost, which teaches using it all. Every car under the boost limit is paid, even one that stays back.\n"
        "\n"
        "Measured: not measured."
    ),
    "landing_recovery": (
        "What it does: pays every step the car is falling in the air higher than 250 units: half of two scores added together, how well its nose points the way it travels seen from above (-1 to 1) and how much its roof points up (-1 to 1). So -1 to 1 a step: 1 for falling wheels down and nose first, -1 for upside down going backwards.\n"
        "\n"
        "When to use it: to teach a bot to come down ready to land on its wheels and keep its speed, for bots that crash-land after aerials and jumps.\n"
        "\n"
        "How to use it:\n"
        "- No settings of its own; the weight is all there is.\n"
        "- Weight: every step while falling. With a goal reward around 10, start around 0.1 to 0.5.\n"
        "\n"
        "Watch out for: it punishes falling upside down, which fights trick training (flip resets, mustys) at a large weight. It stops at 250 units, so the landing itself is not checked. Falling straight down with no sideways speed leaves only the roof part. It pays for time spent falling upright, so a large weight may favour high jumps just to fall.\n"
        "\n"
        "Measured: not measured."
    ),
    "mawkzy_flick": (
        "What it does: pays the Mawkzy flick in parts, as events: 0.1 once for jumping off a controlled hood dribble; up to 0.3 for turning into the Mawkzy pose after the pop (kept only by a Mawkzy dodge); up to 0.15 for the dodge's shape; then, once per possession, the launch, from -0.5 (backward) to 1.5 (full power at the net).\n"
        "\n"
        "When to use it: a one-car Mawkzy flick bot (the mawkzy_flick_bot set); other sets use only its older dribble pay.\n"
        "\n"
        "How to use it:\n"
        "- Dribble test (defaults in brackets; degrees, units, seconds): dribble_min_up (105), dribble_max_up (185), dribble_target_forward (20), dribble_max_planar_dist (90), dribble_max_rel_speed (600), control_threshold (0.25).\n"
        "- Dribble pay, off by default: dribble_rate (0), setup_rate (0), setup_full_speed (1400), possession_decay_time (2), max_dense_per_episode (1); jump_setup_bonus (0.1).\n"
        "- Attempt: min_possession_time (0.25), possession_memory (1), arm_max_ball_dist (280), window_time (0.5), separation_dist (300), wall_margin (200).\n"
        "- Launch: min_gain (250), full_gain (1800), full_launch_speed (3200), aim_cos_zero (0), aim_cos_full (0.85), flick_weight (1), power_bonus (0.5), strong_gain_start (800), strong_gain_full (1600), backward_penalty (0.5), min_penalty_impulse (500), use_elevation (on), elev_full_lo (5), elev_full_hi (10), elev_zero_lo (-5), elev_zero_hi (25).\n"
        "- Shape: require_shape (on), heading_min (25), heading_max (95), heading_edge (10), nose_down_min (-10), nose_down_max (30), nose_down_edge (10), tilt_zero (25), tilt_full (40), roof_to_goal_zero (0), roof_to_goal_full (0.3), roof_to_ball_zero (0.3), roof_to_ball_full (0.6), backflip_credit (0.6), wrong_side_credit (0), setup_shape_bonus (0.15), style_weight (0), reverse_weight (0), require_front_contact (on), front_contact_min (20), reject_floor_crash (on), floor_crash_height (15).\n"
        "- Pose shaping: pose_shaping_weight (0.3; 0 off), pose_gamma (0.99, the training's discount), pose_max_dist (450).\n"
        "- Mid-move starts: reset_grants_possession (off), reset_possession_dist (280).\n"
        "- Weight: 60 to 100 (the reward sets).\n"
        "\n"
        "Watch out for: by default the dribble pays nothing; other rewards must teach it. A miss, another car's touch or a launch near a wall pays nothing. A Raze or floor crash takes back the pose and shape pay.\n"
        "\n"
        "Measured: not measured."
    ),
    "musty_flick": (
        "What it does: pays once per musty, when the ball has left the car: a backflip or back-diagonal dodge with the nose pointing down at the ball, then a roof hit within 0.7 seconds. A quarter for any musty, the rest for goal-directed power, hit strength, lift and a shot on net, times aim and launch angle; 1.5 times as much off a wall, the ceiling or a flip reset; half for a roof push. 0 to 1 from the ground, 1.5 otherwise; down to -0.5 toward its own goal.\n"
        "\n"
        "When to use it: a Musty bot (1v1, tick skip 4, the musty_bot set), beside musty_pose_shaping and dribble pay. For bots that already dribble.\n"
        "\n"
        "How to use it:\n"
        "- The dodge (defaults after =): arm_max_torque_y=-0.4, arm_min_nose_down_deg=88, arm_max_nose_down_deg=140, arm_max_ball_dist=350, arm_min_ball_above=-100, arm_min_up_dot_ball=-0.35, arm_max_lateral=1, strike_window=0.7, top_min_up=28.\n"
        "- Octane hitbox: hitbox_length=120.5, hitbox_width=86.7, hitbox_height=38.7, hitbox_offset_forward=13.9, hitbox_offset_right=0, hitbox_offset_up=20.8.\n"
        "- Hit quality (speeds in units/s): min_impulse=300, full_impulse=2000, min_lift=0, full_lift=600, full_gain=1000, full_launch_speed=2400, w_power=0.35, w_impulse=0.1, w_lift=0.2, w_shot=0.35, shot_on_target=0.5, corner_from_x=250, corner_full_x=760, shot_max_time=5.\n"
        "- Scoop, aim, angle (degrees): scoop_rear_x=-30, scoop_front_x=40, scoop_level_deg=30, scoop_zero_deg=75, scoop_floor=0.5, aim_cos_zero=0, aim_cos_full=0.85, elev_zero_low=0, elev_full_low=8, elev_full_high=45, elev_zero_high=70, aerial_elev_zero_low=-45, aerial_elev_full_low=-20.\n"
        "- Pay: flick_weight=1, base_credit=0.25, aerial_bonus=0.5, backward_penalty=0.5, min_penalty_impulse=500, arm_credit=0 (paid at a good dodge).\n"
        "- Extras, off by default (the musty_bot set turns them on): on_net_bonus=0, on_net_power_mix=1, stretch_shot_weight=0, stretch_shot_from=1600, stretch_shot_full=3000, stretch_gain_weight=0, stretch_gain_from=500, stretch_gain_full=1500.\n"
        "- Judging: cancel_on_car_contact=on, car_contact_margin=40, car_contact_lookback=1, separation_dist=300, resolve_time=0.6, release_steps=2, ride_hover_gap=45, ride_leave_speed=500, wall_margin=200, surface_margin=150, start_jump_margin=450, reset_max_ball_dist=350, start_reset_max_ball_dist=600.\n"
        "- Weight: rare and about 1, so large; the Musty set uses 200 beside goal_in_mouth 25.\n"
        "\n"
        "Watch out for: it pays 0 if another car touches the ball or comes within 40 units, if the car lands or misses, or if the ball leaves near a wall. The Musty set's extras raise a ground musty to about 1.8. An attempt open at a timeout is never judged (episode ends are known only on goals).\n"
        "\n"
        "Measured: not measured."
    ),
    "musty_pose_shaping": (
        "What it does: guides the car into a musty flick's set-up. In the air with its dodge left and the ball close, its pose gets a 0-to-1 score: nose down toward the ball, roof facing it, ball near. Each step pays the score's change (times gamma), so the pay adds up to the score reached. A dodge musty_flick would count keeps it; any other dodge, or a goal, takes it back: at most about 1 per dodge.\n"
        "\n"
        "When to use it: Musty and Breezi bots, before they land mustys.\n"
        "\n"
        "How to use it:\n"
        "- max_dist: no score with the ball this far (default 450 units).\n"
        "- full_dist: full score this close (default 250).\n"
        "- min_ball_above: no score with the ball's centre over 100 below the car's (default -100).\n"
        "- nose_full_deg: nose angle below level for full score; 90 is straight down (default 100).\n"
        "- gamma: your training discount (default 0.995, which suits tick skip 4).\n"
        "- arm_max_torque_y: the arm settings pick the dodge that keeps the pay; a backflip or back-diagonal (default -0.4).\n"
        "- arm_min_nose_down_deg: nose down toward the ball, at least (default 88).\n"
        "- arm_max_nose_down_deg: at most (default 140).\n"
        "- arm_max_ball_dist: ball within (default 350).\n"
        "- arm_min_ball_above: ball centre no more than 100 below (default -100).\n"
        "- arm_min_up_dot_ball: roof facing the ball, -1 to 1 (default -0.35).\n"
        "- arm_max_lateral: sideways limit; 1 is none (default 1).\n"
        "- Weight: the Musty set uses 30, beside musty_flick at 200 and a goal at 25.\n"
        "\n"
        "Watch out for: it pays nothing for the flick itself; use it with musty_flick. Holding the pose costs a little each step, so hovering cannot farm it.\n"
        "\n"
        "Measured: not measured."
    ),
    "on_ground": (
        "What it does: every step it pays 1 while the car's wheels are on the floor, a wall or the ceiling, and 0 in the air. It is the opposite of the built-in air reward.\n"
        "\n"
        'When to use it: with a negative weight it is "the floor is lava", pushing a bot to spend time in the air, for aerial or freestyle bots. A small positive weight keeps a beginner from jumping around for nothing.\n'
        "\n"
        "How to use it:\n"
        "- No settings of its own; the weight is all there is.\n"
        "- Weight: 1 per step, about 15 a second, so keep it small: about 0.05 to 0.5 either way next to a goal at 150.\n"
        "\n"
        "Watch out for: a positive weight can stop the bot from ever learning to jump and fly. A negative weight can make it jump constantly, and walls and the ceiling still count as ground.\n"
        "\n"
        "Measured: not measured."
    ),
    "out_of_position": (
        "What it does: every step an opponent is within near_radii ball radii of the ball, and this car is further toward the opponents' net than the ball and every such opponent: -1. 0 otherwise, with no opponent near the ball, or while this car is demolished. Only the position along the length of the field counts. Teammates and demolished opponents do not count.\n"
        "\n"
        "When to use it: to teach a bot not to be caught upfield, ahead of the play, while an opponent challenges for the ball, so it rotates back.\n"
        "\n"
        "How to use it:\n"
        "- near_radii: default 8, about 740 units centre to centre. Larger counts opponents further from the ball as challenging.\n"
        "- margin: default 0 units. How far past the ball and the opponents the car must be before it is punished; a few hundred forgives being about level.\n"
        "- Weight: positive, since the reward is already -1. It costs about 15 a second at weight 1. Try 0.1 to 1 next to a goal at 150.\n"
        "\n"
        "Watch out for: a negative weight would reward being out of position. It is never positive, so a heavy weight can make a bot hang back in its own half; pair it with rewards for attacking. In team games it also punishes a teammate who is upfield on purpose, waiting for a pass.\n"
        "\n"
        "Measured: not measured."
    ),
    "pickup_boost": (
        "What it does: pays when the car's boost goes up: the rise in the square root of its boost, so the same pad pays more on an empty tank. Empty to full pays 1, a small pad (12 boost) from empty about 0.35, from half a tank about 0.08, and topping up from 88 about 0.06. It is paid on the event, once per step where boost rose; 0 otherwise.\n"
        "\n"
        "When to use it: to teach collecting boost, especially when low on it, from early training on. The default mix uses it at 10.\n"
        "\n"
        "How to use it:\n"
        "- No settings of its own; the weight is all there is.\n"
        "- Weight: the default mix and the boost practice task use 10 next to a goal of 150; 5 to 15 is a sensible range.\n"
        "\n"
        "Watch out for: it pays any rise in boost, including the refill to a third of a tank when a demolished car respawns with less than that. A large weight teaches running pad routes instead of playing. It pairs with save_boost, which pays holding boost; too much of both and the bot hoards.\n"
        "\n"
        "Measured: not measured."
    ),
    "player_on_wall": (
        "What it does: pays 1 every step the car is on its wheels higher than 300 units, which means on a wall or the ceiling; 0 otherwise.\n"
        "\n"
        "When to use it: to get a bot comfortable driving up walls, for wall play and wall touches, or before ceiling and air dribble training.\n"
        "\n"
        "How to use it:\n"
        "- No settings of its own; the weight is all there is.\n"
        "- Weight: every step, so a bot that lives on the wall collects about 15 a second at weight 1 (tick skip 8). Keep it small, around 0.05 to 0.3 with a goal reward around 10, and lower it once the bot can drive on walls.\n"
        "\n"
        "Watch out for: it does not care about the ball, so at a large weight the bot learns to stay on the wall all game. A car low on the wall, 300 units or less, is not paid.\n"
        "\n"
        "Measured: not measured."
    ),
    "possession": (
        "What it does: every step the car's team touched the ball last, the car is within min_dist of it and no opponent is nearer, it pays the possession's count divided by 10. The count goes up by 1 each such step for each such car (teammates share it) and starts again only when the other team touches the ball or a new game starts. So 0.1, 0.2, 0.3 and on, with no limit: after 10 seconds of control it pays about 15 per step. Otherwise 0.\n"
        "\n"
        "When to use it: to teach keeping the ball close after touching it: dribbling and control. Mid training.\n"
        "\n"
        "How to use it:\n"
        "- min_dist: how near the ball the car must be, in units centre to centre (default 200, close to touching).\n"
        "- Weight: it grows without end, so keep it very small, about 0.01 to 0.1 next to a goal at 150, and check what it pays with Try these rewards.\n"
        "\n"
        "Watch out for: a long possession pays more and more per step, so the bot can learn to keep the ball forever rather than score. The count is not cleared when the car moves away, only when the other team touches the ball. Its numbers depend on the tick skip: more steps a second make the count climb faster.\n"
        "\n"
        "Measured: not measured."
    ),
    "raze_flick": (
        "What it does: pays once per Raze flick, when the ball has left the car: a roof dribble, a dodge with the roof tilted, the upside-down car bouncing off the floor, then a hit with its rear. A quarter for any Raze, the rest for pop (lift given to the ball), pogo (the car's bounce), power and a shot on net, times aim and launch angle. 0 to 1 by default; down to -0.5 toward its own goal. One attempt per dribble.\n"
        "\n"
        "When to use it: a Raze bot (1v1, tick skip 4, the raze_bot set), beside raze_flick_setup and dribble pay, for bots that already dribble.\n"
        "\n"
        "How to use it:\n"
        "- Dribble (defaults after =): dribble_min_up=105, dribble_max_up=185, dribble_target_forward=20, dribble_max_planar_dist=90, dribble_max_rel_speed=600, control_threshold=0.25, min_possession_time=0.25, possession_memory=1.\n"
        "- Dodge, bounce, hit: arm_max_ball_dist=400, arm_min_ball_above=60, arm_max_car_height=250, arm_max_up_z=0.5, rebound_window=0.75, rebound_max_up_z=-0.2, rebound_max_car_z=140, rebound_min_vz=80, rebound_min_dvz=150, same_step_min_ball_above=50, strike_window=0.5, rear_max_forward=-20, require_turned_setup=off, turned_min_deg=25, turned_max_deg=110.\n"
        "- Octane hitbox: hitbox_length=120.5, hitbox_width=86.7, hitbox_height=38.7, hitbox_offset_forward=13.9, hitbox_offset_right=0, hitbox_offset_up=20.8.\n"
        "- Quality (units/s, degrees): min_impulse=300, full_impulse=2000, min_pop=200, full_pop=1500, min_pogo=100, full_pogo=600, min_gain=0, full_gain=1000, full_launch_speed=2400, shot_on_target=0.5, corner_from_x=250, corner_full_x=760, shot_max_time=5, w_power=0.1, w_impulse=0.1, w_pop=0.3, w_pogo=0.3, w_shot=0.2, aim_cos_zero=0, aim_cos_full=0.85, elev_zero_low=0, elev_full_low=8, elev_full_high=45, elev_zero_high=70.\n"
        "- Pay: flick_weight=1, base_credit=0.25, backward_penalty=0.5, min_penalty_impulse=500.\n"
        "- Extras, off by default (the raze_bot set turns them on): arm_credit=0, rebound_credit=0, rebound_credit_max_dist=450, on_net_bonus=0, on_net_power_mix=1, stretch_pogo_weight=0, stretch_pogo_from=600, stretch_pogo_full=1200, stretch_pop_weight=0, stretch_pop_from=1500, stretch_pop_full=2500, stretch_impulse_weight=0, stretch_impulse_from=2000, stretch_impulse_full=3500, stretch_gain_weight=0, stretch_gain_from=1000, stretch_gain_full=2500, stretch_shot_weight=0, stretch_shot_from=1600, stretch_shot_full=4000, possession_at_reset=off, possession_at_reset_max_dist=500, possession_at_reset_min_ball_z=0.\n"
        "- Judging: cancel_on_car_contact=on, car_contact_margin=40, car_contact_lookback=1, separation_dist=300, resolve_time=0.3, wall_margin=200.\n"
        "- Weight: rare and about 1, so large; the Raze set uses 400 beside goal_in_mouth 25.\n"
        "\n"
        "Watch out for: a flick without the floor bounce, a front hit, another car's touch or contact, or a launch near a wall or the ceiling pays 0. arm_credit and rebound_credit pay even when the flick fails, so keep them small. An attempt open at a timeout is never judged (episode ends are known only on goals).\n"
        "\n"
        "Measured: not measured."
    ),
    "raze_flick_setup": (
        "What it does: pays for setting up a Raze flick, from the pop jump off a roof dribble to the dodge: each step, setup_rate times how closely car and ball match reference Razes made in RocketSim (0 to 1), times the step's seconds. At best about 0.87 times setup_rate per attempt, one attempt per possession. It stops at the dodge, on landing, at a later hit or after about 0.87 s.\n"
        "\n"
        "When to use it: for a Raze flick bot, with raze_flick, which pays the finished flick.\n"
        "\n"
        "How to use it:\n"
        "- dribble_min_up (105), dribble_max_up (185), dribble_target_forward (20), dribble_max_planar_dist (90), dribble_max_rel_speed (600 units/s): where the ball must sit on the roof and how fast it may move against it.\n"
        "- control_threshold: the control quality (0 to 1) that counts (default 0.25).\n"
        "- min_possession_time: seconds of control before the pop (default 0.25; the Raze set: 0.05).\n"
        "- possession_memory: the pop must come within this many seconds of control (default 1).\n"
        "- setup_rate: pay per second at a perfect match (default 1).\n"
        "- possession_at_reset: a ball on the roof at the start counts as a dribble (default false; the Raze set: true).\n"
        "- pop_ticks (24), land_ticks (24): ticks after the jump (120 a second) when contacts are still the pop, and after which landing ends it.\n"
        "- max_z2 (16), q_scale (1): cap each mismatch; how fast the match falls.\n"
        "- min_spread_car_z (12), min_spread_car_vz (60), min_spread_heading (10 degrees), min_spread_roof_tilt (10 degrees), min_spread_nose_pitch (0.1), min_spread_ball_along_goal (35), min_spread_ball_lateral (40), min_spread_ball_z (35), min_spread_ball_vel_along_goal (120), min_spread_ball_vz (100), min_spread_car_speed_along_goal (300), min_spread_roof_lateral (0.15): the least tolerance per measurement; larger forgives more.\n"
        "- Weight: under 1 per attempt, so large, below raze_flick's (the Raze set: 150 against 400).\n"
        "\n"
        "Watch out for: it pays the pose, not the result. Losing control past possession_memory opens a new attempt, so repeated pops can pay again.\n"
        "\n"
        "Measured: not measured."
    ),
    "reset_musty": (
        "What it does: pays a musty flick off a high flip reset, once, when the shot is judged. When it dodges the car must hold an unused reset counted by the flip_reset rules; the dodge is a backflip or back diagonal, nose well down toward a close ball, and the ball must hit the car's top within 0.7 seconds. It pays the shot's quality (power, aim, angle, on target), times 0.5 for a smack up to 1 for a sling (ball rolling along the roof). At most about 2.7; a backward shot costs up to 0.5.\n"
        "\n"
        "When to use it: a reset musty bot that already gets high resets, to turn them into mustys (the reset_musty_bot set: flip_reset 5, reset_musty_pose_shaping 20).\n"
        "\n"
        "How to use it:\n"
        "- reset_only: only mustys off a counted reset (default true).\n"
        "- wall_margin: the ball's gap from walls and ceiling, reset to shot (default 0, off; runs 300).\n"
        "- sling_floor, sling_ride_from, sling_ride_full: a smack's share (0.5), full as the roof ride grows 15 to 70 units.\n"
        "- on_net_bonus, on_net_power_mix: extra for a shot on target (0.4), this share flat, the rest by speed (0.5).\n"
        "- stretch_shot_weight, stretch_shot_from, stretch_shot_full: extra for shot speed on target (0.25, 1,600 to 3,000).\n"
        "- stretch_gain_weight, stretch_gain_from, stretch_gain_full: extra for speed added toward the net (0.15, 500 to 1,500).\n"
        "- reset_min_ball_height, reset_min_air_time, reset_min_use_ball_height, start_reset_counts: reset height, seconds off every surface, ball height at the hit, a start holding its flip counts (300, 0, 0, false; runs 1,000, 1, 600).\n"
        "- Weight: 100 to 200 with a goal at 150 (the reset_musty_bot set: 150).\n"
        "\n"
        "Watch out for: match its reset settings to flip_reset's, or the two disagree on resets. Another car touching the ball or this car, or a release by a wall, cancels it.\n"
        "\n"
        "Measured: not measured."
    ),
    "reset_musty_pose_shaping": (
        "What it does: while the car holds an unused flip reset that counts by the flip_reset rules, and still has its dodge, it pays for turning toward the musty pose and takes it back for turning away. The pose score, 0 to 1: nose tipped down toward the ball, roof to the ball, ball close. Each step pays gamma times the new score minus the old. A dodge the musty would count keeps what was paid; any other dodge, or losing the reset, pays it all back. At most 1 per dodge, not changed by the tick skip.\n"
        "\n"
        "When to use it: with reset_musty, to guide the turn from the reset into the musty.\n"
        "\n"
        "How to use it:\n"
        "- max_dist, full_dist: the pose counts within 450 units of the ball, fully within 250 (defaults).\n"
        "- min_ball_above: the ball's centre at least this high above the car's (default -100).\n"
        "- nose_full_deg: nose angle below level, toward the ball, for full credit (default 115; 90 is straight down).\n"
        "- roof_zero, roof_full: roof-to-ball credit, none at -1 (facing away), full at 0.6 (defaults).\n"
        "- gamma: the training's discount per step (default 0.995, the runs' at tick skip 4).\n"
        "- reset_min_ball_height, reset_min_air_time, reset_min_use_ball_height, start_reset_counts: the reset rules (300, 0, 0, false; runs 1,000, 1, 600). Match flip_reset's and reset_musty's.\n"
        "- Weight: 10 to 30 with reset_musty at 150 (the reset_musty_bot set: 20); an attempt pays at most the weight.\n"
        "\n"
        "Watch out for: a gamma unlike the training's discount pushes or drags a little on its own. Holding the pose loses a little each step. An episode ending while the pose is held keeps its pay (NitroLearn has no last-step flag), so a bot could hover in the pose until time runs out. It checks only the dodge's start, not whether a musty followed.\n"
        "\n"
        "Measured: not measured."
    ),
    "save_boost": (
        "What it does: pays, every step, the car's boost as a share of a full tank raised to the power exponent: 0 when empty, 1 when full. With the default square root, 25 boost already pays 0.5, so the first bits of boost matter most. Paid about 15 times a second at tick skip 8.\n"
        "\n"
        "When to use it: to stop a bot burning all its boost for nothing, so it keeps some for when it matters. The default mix uses it at a small 0.2.\n"
        "\n"
        "How to use it:\n"
        "- exponent: the shape of the curve (default 0.5, the square root). 1 is a straight line, every unit of boost worth the same; values between 0.3 and 1 are sensible. It must be above 0.\n"
        "- Weight: small, about 0.1 to 0.5 next to a goal of 150. At 0.2 a full tank earns about 3 a second.\n"
        "\n"
        "Watch out for: it charges every use of boost in lost pay, so a large weight teaches hoarding and never boosting, even when a boost would win the ball. With an exponent below 1, keeping a little boost is what pays most; topping up is worth little. It pairs with pickup_boost.\n"
        "\n"
        "Measured: not measured."
    ),
    "small_boost_pickup": (
        "What it does: pays 1 on a step the car's boost goes up by less than 15 out of 100: a small pad (12 boost), or a big pad taken when the tank is nearly full. Otherwise 0, so a big pad taken with room in the tank pays nothing. Each car is compared with its own boost the step before.\n"
        "\n"
        "When to use it: to teach a bot to pick up small pads along its route and keep its boost topped up, instead of detouring to the big pads.\n"
        "\n"
        "How to use it:\n"
        "- No settings of its own; the weight is all there is.\n"
        "- Weight: an event reward of 1 per small pad. Start low, around 1 to 5 next to a goal at 150, since a bot drives over many pads in a game.\n"
        "\n"
        "Watch out for: it pays the same whether the pad filled the tank or added almost nothing, so a nearly full bot is still paid for driving over pads, and a bot can learn to circle between small pads (they come back after 4 seconds) at a high weight. Two small pads in one step (24 boost) pay nothing. Each car is compared with its own boost on the step before.\n"
        "\n"
        "Measured: not measured."
    ),
    "speed_flip": (
        "What it does: pays speed flips on the floor: a floor jump, a forward-diagonal dodge soon after, cancelled so the nose stays level, wheels down flat and the speed kept. Once per jump, up to 1, by how much farther along its starting line the car got than driving straight with full throttle and its boost would have: full at 800 units ahead. Measured 1.5 seconds after the jump, paid once the car has been 0.3 seconds back on the floor. Other dodges pay nothing.\n"
        "\n"
        "When to use it: a speed flip or kickoff bot. The speed_flip_bot set uses it at 10, at most 3 an episode, with velocity_player_to_ball at 0.1.\n"
        "\n"
        "How to use it:\n"
        "- floor_max_z 60, floor_min_up_z 0.8: what counts as the floor.\n"
        "- max_jump_to_dodge 0.25 s; min_torque_side 0.5, min_torque_forward 0.5: how soon and how diagonal the dodge is.\n"
        "- max_pitch_rotation 30, max_nose_angle 25 (degrees): the cancel.\n"
        "- max_air_time 1.4 s, max_land_tilt 30, max_body_tilt 180 (off), body_floor_margin 8: the landing.\n"
        "- ball_land_dist 250: landing on the ball fails.\n"
        "- settle_time 0.3 s, min_kept_frac 0.75 (speed kept), max_veer_deg 50 (direction kept).\n"
        "- pay_window 1.5 s; ref_window 2 s (the comparison starts from the best speed then).\n"
        '- pay_measure: "lead_along" (default), "lead", "speed_vs_boost" or "speed_vs_ref".\n'
        "- pads_in_baseline: off; on gives the comparison the boost pads picked up.\n"
        "- min_gain 0, gain_full 800, max_credit 1.\n"
        "- lost_penalty 0: a minus when a try loses speed or lands badly.\n"
        "- max_per_episode 0 (no cap).\n"
        "- cancel_shaping 0 (off): small pay while the cancel holds; a first trial used 0.15 early on, then off.\n"
        "\n"
        "Watch out for: the comparison uses the car's own boost, so with none any decent flip beats it. Near top speed little is left to gain. Touching the ball in the air ends the try.\n"
        "\n"
        "Measured: not measured."
    ),
    "speed_forward": (
        "What it does: pays every step: the car's speed over 2,300 units/s (its top speed), positive moving nose first and negative reversing, -1 to 1. It counts in the air too.\n"
        "\n"
        "When to use it: early training, to get a bot moving instead of sitting still and to discourage driving backwards.\n"
        "\n"
        "How to use it:\n"
        "- No settings of its own; the weight is all there is.\n"
        "- Weight: every step, so it adds up quickly. Keep it small next to ball and goal rewards, around 0.05 to 0.3 with a goal reward around 10, and lower or remove it later.\n"
        "\n"
        "Watch out for: it pays for speed in any direction, ball or no ball, so at a large weight a bot drives fast circles. The speed includes up and down, so falling or flying fast pays as well.\n"
        "\n"
        "Measured: not measured."
    ),
    "speed_linear": (
        "What it does: pays every step the car's speed over 1,000 units/s: 0 at rest, 1 at 1,000 units/s, 2.3 at top speed (2,300 units/s). It is not capped at 1. Speed in any direction counts, falling and flying too.\n"
        "\n"
        "When to use it: early training, to get a bot that drives fast.\n"
        "\n"
        "How to use it:\n"
        "- No settings of its own; the weight is all there is.\n"
        "- It adds up about 15 times a second at tick skip 8, and tops out at 2.3 rather than 1, so a given weight counts more than for most rewards: about 0.05 to 0.3 next to a goal at 150.\n"
        "\n"
        "Watch out for: it pays speed for its own sake, so driving in circles or away from the ball pays as much as going somewhere useful. Supersonic driving pays most, which can teach a bot to boost everywhere. Too big a weight teaches a bot that just drives around.\n"
        "\n"
        "Measured: not measured."
    ),
    "speed_sqrt": (
        "What it does: pays every step by the car's speed on a square-root curve: the square root of its speed over 1,800 units/s, at most 1. 0 at rest, 0.5 at 450 units/s, about 0.7 at 900, and 1 from 1,800 units/s up. Speed in any direction counts, falling and flying too.\n"
        "\n"
        "When to use it: early training, to get a bot that keeps moving instead of sitting still.\n"
        "\n"
        "How to use it:\n"
        "- No settings of its own; the weight is all there is.\n"
        "- It adds up about 15 times a second at tick skip 8, so keep it small: about 0.1 to 0.5 next to a goal at 150.\n"
        "\n"
        "Watch out for: it pays speed for its own sake, so driving in circles or away from the ball pays as much as going somewhere useful. The curve pays half at only a quarter of the speed, so it rewards always moving more than going fast, and nothing extra above 1,800. Too big a weight teaches a bot that just drives around.\n"
        "\n"
        "Measured: not measured."
    ),
    "speed_sqrt_v2": (
        "What it does: every step, pays the square root of the car's speed out of 2,300 units/s: 0 standing still, 1 at top speed. The curve pays most for the first bit of speed: a quarter of top speed already pays 0.5. Any direction counts, in the air too.\n"
        "\n"
        "When to use it: early training, to get a bot moving instead of sitting still.\n"
        "\n"
        "How to use it:\n"
        "- No settings of its own; the weight is all there is.\n"
        "- Weight: about 0.1 to 1 with a goal at 150; it adds up about 15 times a second at tick skip 8.\n"
        "\n"
        "Watch out for: it pays speed in any direction, so a bot can drive in circles or away from the ball and still collect it. Too big a weight makes it ignore the ball. Because the curve flattens near the top, it barely rewards going from fast to supersonic.\n"
        "\n"
        "Measured: not measured."
    ),
    "speed_toward_ball": (
        "What it does: every step, pays the car's speed toward the ball out of 2,300 units/s: 0 to 1. Moving away from the ball pays 0, not less.\n"
        "\n"
        "When to use it: early training, to get a bot to chase the ball. It is like the built-in velocity_player_to_ball, except that one also costs for moving away.\n"
        "\n"
        "How to use it:\n"
        "- No settings of its own; the weight is all there is.\n"
        "- Weight: about 0.5 to 4 with a goal at 150 (the starter files give velocity_player_to_ball 4); it adds up about 15 times a second at tick skip 8.\n"
        "\n"
        "Watch out for: since moving away never costs, a bot can drive back and forth and be paid on every approach. Too big a weight makes ball-chasing win over positioning and teamwork. Use it or velocity_player_to_ball, not both, or the chase counts twice.\n"
        "\n"
        "Measured: not measured."
    ),
    "speedy_touch": (
        "What it does: pays when the car touches the ball, by how much the ball's speed changed over that step, up or down, divided by 2,777 units/s (100 km/h), at most 1. Only speed counts, not direction. It pays 0 without a touch. It is paid on the event, once per step with a touch.\n"
        "\n"
        "When to use it: to teach touches that change the ball's speed a lot, such as powerful hits, once the bot reaches the ball. Compare strong_touch, which counts direction changes too and has a minimum.\n"
        "\n"
        "How to use it:\n"
        "- No settings of its own; the weight is all there is.\n"
        "- Weight: like strong_touch, about 20 to 60 next to a goal of 150.\n"
        "\n"
        "Watch out for: slowing the ball down pays as much as speeding it up, so killing a fast ball dead earns the full reward, while redirecting it at the same speed earns nothing. It does not care where the ball goes, so pair it with a reward for direction. Every car touching in that step gets the same pay.\n"
        "\n"
        "Measured: not measured."
    ),
    "strong_touch": (
        "What it does: pays when the car touches the ball, by how hard it hit: the change in the ball's velocity over that step (direction counts as well as speed), on a straight ramp from 0 at min_speed_kph to 1 at max_speed_kph. With the defaults a change of 20 km/h or less pays 0, 60 km/h pays 0.5 and 100 km/h or more pays 1. It is paid on the event, once per step with a touch.\n"
        "\n"
        "When to use it: to teach powerful touches instead of pushing or dribbling the ball, once the bot reaches it. The default mix uses it at 60.\n"
        "\n"
        "How to use it:\n"
        "- min_speed_kph: changes up to this, in km/h, earn nothing; the pay starts rising here (default 20, about 556 units/s). Raise it to ignore soft touches.\n"
        "- max_speed_kph: the change that earns the full 1 (default 100, about 2,778 units/s). It must be above min_speed_kph.\n"
        "- Weight: the default mix uses 60 next to a goal of 150; 20 to 60 is a sensible range.\n"
        "\n"
        "Watch out for: it pays hard hits in any direction, including toward your own goal or into the wall, so pair it with a reward for the ball's direction. Every car touching the ball in that step gets the same pay, so 50/50 challenges pay both cars. Stopping a fast ball dead also counts as a big change.\n"
        "\n"
        "Measured: not measured."
    ),
    "supersonic": (
        "What it does: pays 1 every step the car is supersonic (the game's own flag: on from 2,200 units/s and kept a moment while slowing), otherwise 0.\n"
        "\n"
        "When to use it: to teach a bot to reach and keep top speed, for faster rotations and challenges.\n"
        "\n"
        "How to use it:\n"
        "- No settings of its own; the weight is all there is.\n"
        "- Weight: every step, up to about 15 a second at weight 1 (tick skip 8). Keep it small, around 0.05 to 0.3 with a goal reward around 10.\n"
        "\n"
        "Watch out for: it does not care about the ball or the direction, so a bot can farm it with fast circles or boosting around the field, wasting boost.\n"
        "\n"
        "Measured: not measured."
    ),
    "tornado_shaping": (
        "What it does: shapes the Breezi's tornado spin. After a pop (lift-off just after roof control of the ball) it adds up roll and yaw turning the same way; that turn over full_deg is a 0-to-1 score while the dodge is left and the ball near and up. Each step pays its change (times gamma). A dodge musty_flick counts keeps the pay; others, or a goal, take it back: at most about 1 per dodge.\n"
        "\n"
        "When to use it: Breezi bots, with breezi_flick.\n"
        "\n"
        "How to use it:\n"
        "- full_deg: full-score turn (default 290 degrees).\n"
        "- max_dist: no score with the ball this far (default 700).\n"
        "- min_ball_z: nor lower (default 140).\n"
        "- min_ball_above: nor over 150 below the car (default -150).\n"
        "- gamma: your discount (default 0.995).\n"
        "- pop_window: roof control to lift-off, seconds (default 0.25).\n"
        "- surface_margin: touching something this near a surface counts as landed (default 150).\n"
        "- dribble_min_up: roof control: ball this high above the car (default 105).\n"
        "- dribble_max_up: at most (default 185).\n"
        "- dribble_target_forward: best this far forward (default 20).\n"
        "- dribble_max_planar_dist: none this far sideways (default 90).\n"
        "- dribble_max_rel_speed: nor this fast relative (default 600).\n"
        "- control_threshold: control quality that counts (default 0.25).\n"
        "- arm_max_torque_y: arm settings pick the dodge that keeps pay: back or back-diagonal (default -0.4).\n"
        "- arm_min_nose_down_deg: nose down, at least (default 88).\n"
        "- arm_max_nose_down_deg: at most (default 140).\n"
        "- arm_max_ball_dist: ball within (default 350).\n"
        "- arm_min_ball_above: ball under 100 below (default -100).\n"
        "- arm_min_up_dot_ball: roof facing ball (default -0.35).\n"
        "- arm_max_lateral: sideways limit, 1 = none (default 1).\n"
        "- Weight: the Breezi set uses 30, goal 25.\n"
        "\n"
        "Watch out for: opposite roll and yaw score 0; the flick pays nothing.\n"
        "\n"
        "Measured: not measured."
    ),
    "touch_ball": (
        "What it does: pays 1 on each step where the car touches the ball, 0 otherwise. It is paid on the event, at most 1 per step, however hard or soft the touch.\n"
        "\n"
        "When to use it: early in training, to teach a new bot to reach and hit the ball at all. It is the core of the ball-touch practice task. Lower it once the bot touches the ball reliably.\n"
        "\n"
        "How to use it:\n"
        "- No settings of its own; the weight is all there is.\n"
        "- Weight: the practice tasks use 2 to 10 next to a goal of 50; against a goal of 150, about 1 to 20.\n"
        "\n"
        "Watch out for: every step of contact pays again, so a bot can farm it by dribbling, pinning the ball against a wall or pushing it slowly, rather than shooting. It does not care where the ball goes, so pair it with a reward for the ball's speed or direction. A big weight late in training can keep a bot playing for touches instead of goals.\n"
        "\n"
        "Measured: not measured."
    ),
    "touch_ball_on_wall_scaled_by_height": (
        "What it does: when the car touches the ball while its wheels are on a surface and the car is more than 200 units up, it pays the car's height divided by 2,044 (the ceiling's height): about 0.1 just above 200 units, 0.5 halfway up a wall, 1 on the ceiling. It pays on each step with such a touch, 0 otherwise.\n"
        "\n"
        "When to use it: to teach wall play: taking the ball on the wall, wall dribbles and wall setups for aerials. Best for a bot that already drives on walls.\n"
        "\n"
        "How to use it:\n"
        "- No settings of its own; the weight is all there is.\n"
        "- It pays at most 1 per touch. With a goal at 150, a weight of about 5 to 30 is a touch-sized reward.\n"
        "\n"
        "Watch out for: it checks the car's height and its wheels, not that the surface is a wall, so the upper part of the curved corners and the ceiling count too, and ceiling touches pay the most. It pays the car's height, not where the ball goes, so the bot may learn to carry the ball up the wall for nothing. A long wall dribble pays on every step it touches.\n"
        "\n"
        "Measured: not measured."
    ),
    "touch_ball_scaled_by_hit_force": (
        "What it does: on a step with a touch that changes the ball's velocity by more than 500 units/s, it pays that change divided by 3,000: just over 0.17 at the least, 1 for a 3,000 change, 2 for 6,000. There is no cap. 0 otherwise.\n"
        "\n"
        "When to use it: to teach hitting the ball hard: power shots and strong clears. Early to mid training.\n"
        "\n"
        "How to use it:\n"
        "- No settings of its own; the weight is all there is.\n"
        "- Weight: about 0.17 to 2 per touch. With a goal at 150, about 10 to 50 (the starter's strong_touch, which is capped at 1, is 60).\n"
        "\n"
        "Watch out for: the change counts direction as well as speed, so sending a fast ball straight back the way it came pays a lot even if it goes nowhere useful; nothing checks the goal direction. The change is over the whole step, so a bounce in the same step adds to it, and every car touching that step is paid.\n"
        "\n"
        "Measured: not measured."
    ),
    "touch_ball_speed_gain": (
        "What it does: pays on a touch of the ball. With beginner on (default), 1 per touch. With it off, how much faster the ball is just after this touch than it was after this car's previous touch this game, in units/s (0 to 6,000, never negative; the first touch pays the ball's whole speed). Once per step with a touch.\n"
        "\n"
        "When to use it: beginner mode for a fresh bot learning to reach and hit the ball; the other mode later, to favour harder hits.\n"
        "\n"
        "How to use it:\n"
        "- beginner: on (default) pays 1 per touch; off pays the speed gained, in units/s.\n"
        "- Weight: with beginner on it is a touch reward: a tenth to a half of your goal reward's weight. Off, it pays in the thousands, so use a tiny weight, around 0.0005 to 0.002.\n"
        "\n"
        "Watch out for: in beginner mode a bot can farm it with many small touches, such as pushing or dribbling, and it says nothing about where the ball goes. With beginner off it compares with the speed after the car's own last touch, so touching a ball someone else made fast pays too, and a soft touch lowers the bar for the next.\n"
        "\n"
        "Measured: not measured."
    ),
    "velocity_ball_to_goal": (
        "What it does: pays, every step, how fast the ball moves toward the back of the opponents' net, as a share of the ball's top speed (6,000 units/s): 1 at full speed straight at their net, negative when it moves away. It depends on the ball only, so every player on the team gets it, whoever hit the ball. Paid about 15 times a second at tick skip 8.\n"
        "\n"
        "When to use it: to teach sending the ball toward the opponents' goal rather than anywhere. Useful once the bot can hit the ball, and kept for the whole run in most mixes. The default mix uses it at 2, with zero-sum and full team spirit.\n"
        "\n"
        "How to use it:\n"
        "- No settings of its own; the weight is all there is.\n"
        "- Weight: 2 to 5 next to a goal of 150 (the shot and save practice tasks use 5 next to a goal of 50). The other team's value is roughly the opposite of yours, so adding zero-sum, as the default mix does, roughly doubles it.\n"
        "\n"
        "Watch out for: it keeps paying while the ball rolls toward their goal on its own, so it rewards where the ball goes, not how the bot got it there. Distance does not matter: a ball moving toward their net at a given speed pays the same from midfield as from right in front of it.\n"
        "\n"
        "Measured: not measured."
    ),
    "velocity_ball_to_goal_aligned": (
        "What it does: every step, the ball's speed toward the back of the opponents' net (1 at 6,000 units/s) times how straight at the net it flies: all of it exactly on line, falling to none at about 41 degrees off (about half at 20 degrees). 0 to about 1; never negative.\n"
        "\n"
        "When to use it: to reward shots on target rather than any ball movement toward the other half. Suits bots that already hit the ball toward the goal.\n"
        "\n"
        "How to use it:\n"
        "- own_goal: default off. On measures the ball flying at the car's own net instead; use that copy with a negative weight to punish shots at your own goal.\n"
        "- Weight: up to 1 a step, about 15 times a second. Try 1 to 5; the preset's velocity_ball_to_goal uses 2 next to a goal at 150.\n"
        "\n"
        "Watch out for: it pays whoever hit the ball, including an opponent's mistake. Because it is never negative, a ball heading the wrong way costs nothing, unlike velocity_ball_to_goal; add the own_goal copy if that matters. With own_goal on and a positive weight, the bot learns to shoot at its own net.\n"
        "\n"
        "Measured: not measured."
    ),
    "velocity_player_to_ball": (
        "What it does: pays, every step, how fast the car moves toward the ball, as a share of a car's top speed (2,300 units/s): 1 at full speed straight at it, 0 sideways or standing still, negative when moving away. Paid about 15 times a second at tick skip 8.\n"
        "\n"
        "When to use it: from the start of training, to teach a new bot to go to the ball. It stays useful in most mixes; the default mix uses it at 4. Tie it to alignment with all_of so it pays only when the car approaches from the right side.\n"
        "\n"
        "How to use it:\n"
        "- No settings of its own; the weight is all there is.\n"
        "- Weight: 1 to 4 next to a goal of 150 (the practice tasks use 1 next to a goal of 50).\n"
        "\n"
        "Watch out for: it charges the car for moving away from the ball, for example to rotate back, defend or grab boost, so a big weight makes ball-chasers. It pays the approach, not the touch: what the car does on arrival is up to other rewards. Because it pays on almost every step, it can outweigh the rewards that pay on events.\n"
        "\n"
        "Measured: not measured on its own. In a run with the default mix it paid about 70% of all reward from start to end, which is why the warning about one reward dominating starts only at 90%."
    ),
    "wall_dash": (
        "What it does: pays for a wall dash: a short jump off a wall, then a dodge that puts the wheels straight back on it. Rules from samuelkassi's Wall Dash Coach. A dash is made when 3 or more wheels are back on the wall within a quarter second, with the car flat against it. It pays the speed gained along the wall out of full_gain, at most 1, scaled by three grades: the time between the two jump presses, how long the first jump was held, and how far up the wall the dodge points (best 50 to 70 degrees). Green keeps the pay, yellow multiplies it by yellow_factor, red by red_factor. A miss pays nothing.\n"
        "\n"
        "When to use it: a mechanics bot learning to gain speed on walls.\n"
        "\n"
        "How to use it:\n"
        "- full_gain: speed gain for the full pay (default 500 units/s).\n"
        "- min_speed_gain: the gain a dash must beat (default 0).\n"
        "- yellow_factor, red_factor: 0.5 and 0.25; set both to 1 to pay the gain alone.\n"
        "- chain_bonus: extra pay for a dash within a second of the last one's landing (0.5 = half again; default 0, off).\n"
        "- curves, corners: also pay dashes there. Off: flat walls only.\n"
        "- miss_cost: a charge for a dodge off the wall that misses (default 0, off).\n"
        "- Weight: 1 to 10 next to a goal at 150.\n"
        "\n"
        "Watch out for: the bot presses only on whole steps (every 4 ticks by default), so timing bands are rounded up to whole steps. The bot's diagonal dodge is at 45 degrees: yellow at best. Four or more jump presses pay nothing. It is not tied to the ball, and speed from going down the wall counts.\n"
        "\n"
        "Measured: not in training. In a scripted simulator test (tick skip 4, 1,500 units/s along the wall) a dash diagonally up the wall was paid and one down the wall was not."
    ),
    "wall_positioning": (
        "What it does: pays, every step, 1 while the car is on a wall (wheels down, more than 200 units up) with the ball higher than the car and within 464 units of it (five ball radii), holding more than 50 boost; 4 if the car also touches the ball on that step; 0 otherwise. Paid about 15 times a second at tick skip 8.\n"
        "\n"
        "When to use it: to teach waiting on the wall under the ball with boost, ready to start an air dribble off the wall. For air dribble bots, as a setup reward.\n"
        "\n"
        "How to use it:\n"
        "- No settings of its own; the weight is all there is.\n"
        "- Weight: small, about 0.1 to 1 next to a goal of 150. At weight 1, a second in position earns about 15, and pushing the ball every step about 60.\n"
        "\n"
        "Watch out for: it does not scale with height or check the ball's speed and direction, so it pays for any ball near and above the car on a wall, even one that is going nowhere. A bot can farm it by carrying the ball up the wall and holding it there, especially with the 4 for touches. It stops paying the moment the car jumps off, so it does not teach the take-off itself.\n"
        "\n"
        "Measured: not measured."
    ),
    "wavedash": (
        "What it does: pays when a dodge lands the car's wheels on the floor within 0.2 seconds of the dodge starting (a wavedash). It pays the speed gained since just before the dodge, out of 500 units/s (a dodge's own push): 0 to 1, once, on the landing. The floor means the roof points mostly up, within about 45 degrees of upright. A dodge and landing inside one step counts too.\n"
        "\n"
        "When to use it: a mechanics bot learning to gain speed with wavedashes, for recoveries after landing or quick speed off the ground.\n"
        "\n"
        "How to use it:\n"
        "- No settings of its own; the weight is all there is.\n"
        "- Weight: about 1 to 10 with a goal at 150. It pays once per dash, but a bot can dash every second or so, so a large weight adds up.\n"
        "\n"
        "Watch out for: it is not tied to the ball, so a heavy weight can teach dashing for its own sake. Speed counts in every direction, so falling speed lost on landing counts against the gain, and a car already at top speed (2,300) gains nothing. A dodge that lands later than 0.2 seconds pays nothing.\n"
        "\n"
        "Measured: not measured."
    ),
    "wizard_flick": (
        "What it does: pays 1 for a wizard flick: after a pop off the roof, a forward-diagonal dodge (10 to 80 degrees off straight) that does not hit the ball early but turns on and meets it late, with a bottom front corner (left or right). At the touch the roof must be tilted past 90 degrees (underside up) and the nose above level, the car must have been off the floor for 0.25 s, the ball must not have touched the floor in the game, and the ball's upward speed must rise by 200+ units/s. Only the first touch after the dodge counts; otherwise 0.\n"
        "\n"
        "When to use it: a wizard flick bot. The checks come from reviewing scripted flicks in RocketSim: the corner, the ball still in the air, the car not landing first.\n"
        "\n"
        "How to use it:\n"
        "- No settings of its own; the weight is all there is.\n"
        "- An event reward paid once per flick: a weight of about 100, as a success reward with min_pay 1.\n"
        "\n"
        "Watch out for: the ball touching the floor at any time in the game rules the flick out, so it suits short practice games that start from a dribble. A flick with the bottom front edge or face, not a corner, pays nothing.\n"
        "\n"
        "Measured: unit tests."
    ),
}


# The "Nerd explanation" of each reward (keys as MORE): its formula and a settings-file example between
# lines of three backticks. Plain text; the Rewards tab shows it behind its Nerd explanation button.
NERD: dict[str, str] = {
    "air": (
        "Paid every step: v = 1 when no wheel touches the floor, a wall or the ceiling, else 0. Being an every-step reward it is multiplied by tick_skip / 8, so a weight pays the same per second at any tick skip: at weight w a car in the air earns about w x 15 a second.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "air"\n'
        "weight = 0.25\n"
        "```\n"
        "\n"
        "Kept small, it only nudges the bot to try jumping; larger, the bot learns to stay airborne for its own sake."
    ),
    "face_ball": (
        "Every step: v = cos(angle between the car's nose and the line from the car to the ball) = forward . (ball - car) / |ball - car|. 1 facing the ball, 0 side-on, -1 facing away.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "face_ball"\n'
        "weight = 0.25\n"
        "```\n"
        "\n"
        "Cheap to earn while doing nothing else, so keep its weight small next to rewards for progress."
    ),
    "velocity_player_to_ball": (
        "Every step: v = (car velocity) . (direction to the ball) / 2300, the speed toward the ball as a share of the car's top speed: 1 at full speed straight at the ball, negative when driving away.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "velocity_player_to_ball"\n'
        "weight = 4\n"
        "```\n"
        "\n"
        "It pays approach speed, not arrival, so it is usually paired with a touch reward."
    ),
    "velocity_ball_to_goal": (
        "Every step: v = (ball velocity) . (direction from the ball to the back of the opponents' net, half the goal's height up) / 6000. 1 with the ball at top speed straight at their net, -1 straight at your own.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "velocity_ball_to_goal"\n'
        "weight = 2\n"
        "zero_sum = { team_spirit = 1 }\n"
        "```\n"
        "\n"
        "Both teams see the same ball, so zero_sum makes it a contest: one side's gain is the other's loss."
    ),
    "touch_ball": (
        "v = 1 on a step the car touches the ball, else 0. An event reward: not scaled by tick skip.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "touch_ball"\n'
        "weight = 5\n"
        "```\n"
        "\n"
        "A bot can learn to keep the ball pinned against itself to collect it every step; strong_touch pays for hits that move the ball instead."
    ),
    "strong_touch": (
        "On a touch: let dv = |ball velocity after the step - before| in km/h. v = clamp((dv - min) / (max - min), 0, 1), with min = min_speed_kph and max = max_speed_kph. A soft push pays 0, a hit that changes the ball's speed by max or more pays 1.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "strong_touch"\n'
        "weight = 60\n"
        "min_speed_kph = 20\n"
        "max_speed_kph = 100\n"
        "```\n"
        "\n"
        "uu/s to km/h: x 0.036 (2300 uu/s = 83 km/h)."
    ),
    "pickup_boost": (
        "When the car's boost rises from b0 to b1 (0-100): v = sqrt(b1 / 100) - sqrt(b0 / 100). The square root makes the same pad worth more to an emptier tank: a big pad from empty pays 1, a small pad (12) from empty 0.35, from half full about 0.08.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "pickup_boost"\n'
        "weight = 10\n"
        "```"
    ),
    "save_boost": (
        "Every step: v = (boost / 100)^exponent. With 0.5 (a square root), the first boost held is worth the most: 25 boost pays 0.5, 100 pays 1.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "save_boost"\n'
        "weight = 0.2\n"
        "exponent = 0.5\n"
        "```\n"
        "\n"
        "Every-step, so scaled by tick_skip / 8. Too large, and the bot hoards boost instead of using it."
    ),
    "bump": (
        "v = 1 on a step the car bumps an opponent (touches it hard enough for the game to count a bump, without a demolition). An event reward.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "bump"\n'
        "weight = 20\n"
        "zero_sum = { team_spirit = 0.5 }\n"
        "```"
    ),
    "demo": (
        "v = 1 on a step the car demolishes an opponent. With zero_sum, the demolished car's team loses the same share.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "demo"\n'
        "weight = 80\n"
        "zero_sum = { team_spirit = 0.5 }\n"
        "```"
    ),
    "goal": (
        "On a goal: every car of the scoring team gets 1, every car of the other team gets concede_scale (default -1). The weight multiplies both.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "goal"\n'
        "weight = 150\n"
        "concede_scale = -1\n"
        "```\n"
        "\n"
        "For a pure +1 / -1 game, the GCO starter writes concede_scale = 0 with zero_sum = { team_spirit = 1, opponent_scale = 1 }: the conceders' -1 then comes from the zero-sum step."
    ),
    "alignment": (
        "Every step: v = cos(angle between car->ball and ball->opponents' net) = d1 . d2 with d1 = (ball - car) / |ball - car|, d2 = (net - ball) / |net - ball|. 1 with the ball straight between the car and their net.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "all_of"\n'
        "weight = 4\n"
        'parts = [{ name = "velocity_player_to_ball" }, { name = "alignment" }]\n'
        "```\n"
        "\n"
        "On its own the bot can farm it by sitting behind the ball; inside all_of it only counts while moving toward the ball."
    ),
    "all_of": (
        "An AND of its parts p_1..p_n (each computed as its own reward, without weights):\n"
        "- all p_i > 0: v = (p_1 x p_2 x ... x p_n)^(1/n), their geometric mean\n"
        "- any p_i = 0: v = 0\n"
        "- otherwise (some negative): v = -(|p_1| x ... x |p_n|)^(1/n)\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "all_of"\n'
        "weight = 4\n"
        "zero_sum = { team_spirit = 1 }\n"
        'parts = [{ name = "velocity_player_to_ball" }, { name = "face_ball" }]\n'
        "```\n"
        "\n"
        "A group is every-step (scaled by tick skip) only when every part is."
    ),
    "jump_touch": (
        "On a touch with the car in the air and the ball at least min_height up: h = min(ball height - 92.75, 929.25) / 929.25, v = h^exp, paid only when v > 0.05. After a paid touch, that car's touches pay 0 for the next 149 steps.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "jump_touch"\n'
        "weight = 20\n"
        "min_height = 300\n"
        "exp = 0.5          # pays lower touches more\n"
        "```"
    ),
    "aerial_fly_at_ball": (
        "Every step in the air while moving toward a ball at least 300 uu up: v = (speed toward the ball / 2300) x e^(-0.5 x d / 2300) + 0.1 after a double jump, with d the distance to the ball's surface. The exponential halves the pay about every 3,200 uu of distance.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "aerial_fly_at_ball"\n'
        "weight = 2\n"
        "```"
    ),
    "aerial_chain_distance": (
        "One car leads an aerial at a time. Its first touch above 256 uu pays height_scale x max(0, car height + ball height - 512) / 10240; each later touch while it stays above 256 uu pays distance_scale x (distance the car and the ball moved since its last touch) / 10240.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "aerial_chain_distance"\n'
        "weight = 10\n"
        "height_scale = 1.0\n"
        "distance_scale = 1.0\n"
        "```"
    ),
    "aerial_chain_distance_v2": (
        "First touch 256+ uu up: height_scale x max(0, car z + ball z - 512) / 10240. Each further touch before the car comes down (or another car touches in the air): distance_scale x (car travel + ball travel since the last touch) / 10240. Typically 0.03 to 0.15 a touch.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "aerial_chain_distance_v2"\n'
        "weight = 20\n"
        "```"
    ),
    "aerial_approach": (
        "Every step in the air, rising, below ball_height_min, with the ball above and nearer than 3 x its height:\n"
        "v = cos(flight, car->ball) x speed / 2300 (+ 0.5 x max(0, cos) with beginner), + 0.5 when nearer than on its last such step, then x 1.5 on a touch. 0 to 3.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "aerial_approach"\n'
        "weight = 1\n"
        "ball_height_min = 400\n"
        "beginner = true\n"
        "```"
    ),
    "aerial_navigation": (
        "Every step flying upward toward a ball above ball_height_min (car above player_height_min, within 3 x the ball's height across the field), with f = cos(car's flight, car->ball) and c = cos(nose, car->ball):\n"
        "- flight part: 0.5 x max(0, f) with beginner, else f x speed / 2300 (which can be negative);\n"
        "- plus 0.25 x c when c > 0, and 0.1 x c more while boosting.\n"
        "0 to 0.85 as a beginner.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "aerial_navigation"\n'
        "weight = 1\n"
        "beginner = false\n"
        "```"
    ),
    "aerial_navigation_strict": (
        "Every step flying upward with over 20 boost, under a ball above ball_height_min (car above player_height_min): with nearness n = 1 - min(1, distance / max_distance), v = n x (0.5 x flight term + 0.15 x facing term) + 0.85 on a touch. 0 to 1.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "aerial_navigation_strict"\n'
        "weight = 2\n"
        "max_distance = 1500\n"
        "```"
    ),
    "aerial_distance_to_ball": (
        "Every step in the air: v = max(0, 1 - distance to the ball / 600). 1 touching it, 0 from 600 uu away or on the ground.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "aerial_distance_to_ball"\n'
        "weight = 0.5\n"
        "```"
    ),
    "aerial_touch_airtime": (
        "On a touch: v = min(t / 1.75, 1, ball height / 2044), t being the car's time in the air before this step (counted in steps that ended in the air, reset on the ground).\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "aerial_touch_airtime"\n'
        "weight = 20\n"
        "```"
    ),
    "aerial_touch_scaled_by_height": (
        "On a touch with the car in the air: v = ball height / 2000 (0 to about 1).\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "aerial_touch_scaled_by_height"\n'
        "weight = 10\n"
        "```"
    ),
    "air_touch_long_flight": (
        "On a touch: v = min(min(1, t_air / 1.75), ball height / 2044), the smaller of the flight time and height shares. A high touch after a short hop pays little; so does a long flight to a low ball.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "air_touch_long_flight"\n'
        "weight = 20\n"
        "```"
    ),
    "high_touch": (
        "On a touch: v = min(min(1, t_air / 1.75), ball height / 2044); 0 for a touch from the ground. The same idea as air_touch_long_flight: both conditions must be met for full pay.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "high_touch"\n'
        "weight = 20\n"
        "```"
    ),
    "air_dribble": (
        "A state machine follows each car's chain of separate air touches. A chain opens from a floor pop, a roof dribble, a wall or an aerial catch, and needs a minimum number of touches (min_touches_*) to count. Pay:\n"
        "- shaping while the chain is open (touch quality, control, progress toward goal, air roll), capped at max_shaping_per_episode;\n"
        "- a goal off a valid chain: goal_reward x (1 + per_extra_touch x extra touches) x the bonuses (air roll, flip resets, cross-map distance);\n"
        "- an own goal off the chain: -own_goal_penalty.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "air_dribble"\n'
        "weight = 10\n"
        "goal_reward = 2\n"
        "max_shaping_per_episode = 1.5\n"
        "```\n"
        "\n"
        "Every setting is listed with its default in the Rewards tab's catalog."
    ),
    "air_dribble_setup_touch": (
        "On a touch in the air with the ball above 250 uu, above the car and further upfield: v = (1 - |ball x| / 2200) x n, n the number of such touches since the car left the ground. The x term pays a ball near the middle of the field; it turns negative beyond 2,200 uu to the side.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "air_dribble_setup_touch"\n'
        "weight = 2\n"
        "```"
    ),
    "air_dribble_under_ball": (
        "Every step in the air: +1 when the ball is within 300 uu, above the car and the car's nose points up (if both fall, only while both fall slower than 650 uu/s); +0.5 when car and ball move the same general way. 0 to 1.5.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "air_dribble_under_ball"\n'
        "weight = 0.5\n"
        "```"
    ),
    "air_dribble_upfield": (
        "Every step in the air under a ball above 250 uu in the opponent's half, within 400 uu and facing it (cosine above 0.74): a base of up to 53.5 (40 for a touch, 5 for being behind the ball) with more than 15 boost, half with less; halved when the car or ball is not rising; then x 2 x (1 - |car x| / 4096). Large numbers: use a small weight.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "air_dribble_upfield"\n'
        "weight = 0.02\n"
        "```"
    ),
    "air_roll": (
        "Every step above 200 uu: v = |roll rate| in rad/s (0 to 5.5). With in_air_with_flip: v = |roll| - |pitch| / 5 - |yaw| / 5, only while in the air with a flip left.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "air_roll"\n'
        "weight = 0.05\n"
        "in_air_with_flip = true\n"
        "```"
    ),
    "air_roll_on_kickoff": (
        "Every step of a kickoff (the ball still on the centre spot): v = roll rate / 5.5, -1 to 1, its sign the roll's direction. 0 once the ball has moved.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "air_roll_on_kickoff"\n'
        "weight = 0.5\n"
        "```"
    ),
    "align_ball_goal": (
        "Every step: v = defense x cos(car->ball, own net->car) + offense x cos(car->ball, car->opponents' net). The first is 1 with the car between its net and the ball; the second 1 with the ball straight between the car and their net.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "align_ball_goal"\n'
        "weight = 1\n"
        "defense = 0.1\n"
        "offense = 1.0\n"
        "```"
    ),
    "ball_y_coordinate": (
        "Every step: v = (ball y / 5212.75)^exponent, y as the car's team sees it (attacking +y): about -1 at its own back wall, 1 at the opponents'.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "ball_y_coordinate"\n'
        "weight = 0.5\n"
        "exponent = 1     # keep it odd so your own half stays negative\n"
        "```"
    ),
    "ball_touch_scaled_by_height": (
        "On a touch: v = ball height / 2000 (about 0.05 on the ground, 1 at the ceiling). With high_touch_penalty: only above 200 uu, v = 1 - sqrt(height / 20), always negative, a penalty that grows with height.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "ball_touch_scaled_by_height"\n'
        "weight = 5\n"
        "```"
    ),
    "behind_ball": (
        "Every step: v = 1 while the car is nearer its own back wall than the ball is (y_car < y_ball from its team's view), else 0.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "behind_ball"\n'
        "weight = 0.1\n"
        "```"
    ),
    "behind_ball_alignment": (
        "Every step: v = align_w x cos(car->ball, car->back of the opponents' net): align_w with car, ball and net in one line, -align_w the other way round.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "behind_ball_alignment"\n'
        "weight = 1\n"
        "align_w = 0.75\n"
        "```"
    ),
    "boost_gain": (
        "When boost rises by g (0-100): v = g / 100, or sqrt(g / 100) with sqrt = true (a small pad's 12 then pays 0.35).\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "boost_gain"\n'
        "weight = 5\n"
        "sqrt = true\n"
        "```"
    ),
    "boost_lose": (
        "When boost falls from b0 to b1: v = boost_lose_w x (sqrt(b1 / 100) - sqrt(b0 / 100)) x (1 - car height / 642.775). The middle term is negative (boost went down), so below the goal's height (642.775 uu) spending is charged, at that height it is free, and above it spending pays: boost belongs in the air.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "boost_lose"\n'
        "weight = 1\n"
        "boost_lose_w = 1\n"
        "```"
    ),
    "boost_pad_pickup": (
        "When boost rises by g (as a share of a full tank): v = g x big, plus g x small when the boost ends below 98 having started below 88 (a small pad). With the defaults a small pad pays 1.56 and a big pad from empty 10.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "boost_pad_pickup"\n'
        "weight = 1\n"
        "big = 10\n"
        "small = 3\n"
        "```"
    ),
    "boost_pickup_sqrt": (
        "When boost rises by g (0-100): v = sqrt(g / 100). A big pad from empty pays 1, a small pad 0.35.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "boost_pickup_sqrt"\n'
        "weight = 10\n"
        "```"
    ),
    "boost_use": (
        "When boost falls by s (0-100): v = sqrt(s / 100). A step of full boosting spends about 2.2 at tick skip 8 and pays 0.15. Positive: it pays for using boost.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "boost_use"\n'
        "weight = 0.5\n"
        "```"
    ),
    "small_boost_pickup": (
        "v = 1 when boost rises by less than 15 in a step (a small pad, or a big pad when almost full), else 0.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "small_boost_pickup"\n'
        "weight = 2\n"
        "```"
    ),
    "breezi_flick": (
        "The move: a pop off the roof, a tornado spin (roll and yaw turning the same way), then a musty, the ball in the air throughout. The musty is judged exactly as musty_flick (with the same settings). Pay on the step the musty is judged:\n"
        "\n"
        "v = breezi_scale x musty credit x spin fullness + musty_share x musty credit\n"
        "\n"
        "spin fullness rising from 0 at spin_zero_deg of turning to 1 at spin_full_deg.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "breezi_flick"\n'
        "weight = 10\n"
        "breezi_scale = 1.0\n"
        "musty_share = 0.25\n"
        "```"
    ),
    "distance_to_ball": (
        "Every step: v = distance from the car to the ball's centre / 2000: about 0.05 touching it, about 6.6 across the field. Positive weight pays being far; a negative weight pays being near.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "distance_to_ball"\n'
        "weight = -0.05\n"
        "```"
    ),
    "dont_boost_while_supersonic": (
        "v = -1 on each step the car's boost went down while it moved faster than 2,200 uu/s (supersonic), else 0. Boosting past supersonic adds almost no speed (the top speed is 2,300 uu/s), so this charges wasted boost.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "dont_boost_while_supersonic"\n'
        "weight = 0.5\n"
        "```"
    ),
    "dribble_carry": (
        "Every step the ball is 112.75 to 292.75 uu up and within 150 uu of the car in x and in y: v = 0.2, plus 0.8 if the car touches it that step (0 to 1).\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "dribble_carry"\n'
        "weight = 1\n"
        "```\n"
        "\n"
        "The x and y tests compare distances from the field's centre lines, so a car mirrored across a centre line from the ball also passes."
    ),
    "dribble_cradle": (
        "Every step the ball sits in the cradle (112.75 to 292.75 uu up, within 170 uu of the car, away from the walls) and no opponent is within minimum_barrier of the ball: v = 1, else 0.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "dribble_cradle"\n'
        "weight = 1\n"
        "minimum_barrier = 600     # 0 turns the opponent check off\n"
        "```"
    ),
    "dribble_flick": (
        "Every step in the cradle: 0.5 (with training on). If the ball is 112.75 to 172.75 uu up and moving within max_vel_diff of the car: +1 when no opponent is within minimum_barrier; challenged, -0.5 on the ground, +2 in the air with a flip left, +5 once the flip is used (the flick). -0.5 to 5.5.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "dribble_flick"\n'
        "weight = 1\n"
        "minimum_barrier = 800\n"
        "max_vel_diff = 200\n"
        "```"
    ),
    "dribble_match_speed": (
        "Every step on the ground within 197 uu of a ball 109 to 180 uu high: v = speed / 2300 + 2 x (1 - |speed_car - speed_ball| / (speed_car + speed_ball)). The second term is 1 when the speeds match. 0 to 3.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "dribble_match_speed"\n'
        "weight = 0.5\n"
        "```"
    ),
    "dribble_roof": (
        "Every step the ball is 112.75 to 292.75 uu high and within 150 uu of the car in x and y (as distances from the centre lines), away from the walls: v = 0.2, or 1 with a touch.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "dribble_roof"\n'
        "weight = 1\n"
        "```"
    ),
    "dribble_upfield": (
        "dribble_cradle's condition, paid by how far up the field the car is: v = sqrt((y + 5000) / 100) / 10 with y toward the opponents' net: 0 behind y = -5000, 0.71 at the centre line, 1 at y = 5000.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "dribble_upfield"\n'
        "weight = 1\n"
        "minimum_barrier = 600\n"
        "```"
    ),
    "example_ball_height": (
        "The example of a reward of your own (crates/nitro_core/src/custom/example.rs): every step, v = (ball height / ceiling height)^power, 0 on the ground to 1 at the ceiling.\n"
        "\n"
        "```rust\n"
        "fn value(&self, _prev: &GameSnapshot, cur: &GameSnapshot, _events: &StepEvents, _car: usize) -> f32 {\n"
        "    (cur.ball.pos.z / CEILING_Z).clamp(0.0, 1.0).powf(self.power)\n"
        "}\n"
        "```\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "example_ball_height"\n'
        "weight = 0.1\n"
        "power = 2\n"
        "```\n"
        "\n"
        "Copy the file, rename the struct, change the maths, add it on the Resources page and press Build."
    ),
    "flip_reset": (
        "Flip resets off the ball in the air, as a sequence of paid stages:\n"
        "- approach: a potential (closeness, not too far above the ball; lower while the flip is still unused, since a wheel touch then would not be a reset) paid as it rises, at most max_setup_per_episode;\n"
        "- the reset: reset_reward x style, style = (1 + flip_into_reset_bonus or backflip bonus) x (1 + spin bonus) x (1 + multi-reset factor) x (1 + need_bonus x need);\n"
        "- follow-through: follow_reward x style for staying near the ball for follow_time;\n"
        "- use: use_reward x quality x style for a dodge (or another reset) into the ball soon after;\n"
        "- a goal off a used reset: goal_reward.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "flip_reset"\n'
        "weight = 5\n"
        "reset_reward = 10\n"
        "use_reward = 8\n"
        "goal_reward = 20\n"
        "```\n"
        "\n"
        "Every setting is in the Rewards tab's catalog with its default."
    ),
    "flip_reset_approach": (
        "Every step: v = flip_reset_help_w x 2 cos(car->ball, car's underside) x min(1, |ball x| / 1300) x min(1, ball height / 1700) x cos(car->opponents' net, nose) x min(1, 40 x u / (d + 1)), with u how far the wheels face the ceiling and d the distance to the ball with height doubled. Up to about 2; negative when one factor is.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "flip_reset_approach"\n'
        "weight = 1\n"
        "flip_reset_help_w = 1\n"
        "```"
    ),
    "flip_reset_height": (
        "When the car, in the air, touches the ball and has a flip it did not have the step before: v = min(1, car height / 1500).\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "flip_reset_height"\n'
        "weight = 20\n"
        "```"
    ),
    "flip_reset_height_v2": (
        "When the car gets its flip back while higher than minimum_height (off the ball, or landing on a wall or the ceiling up there): v = height x height_scaling (250 at 1,000 uu with the defaults: use a small weight).\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "flip_reset_height_v2"\n'
        "weight = 0.02\n"
        "minimum_height = 500\n"
        "```"
    ),
    "flip_reset_simple": (
        "On a flip reset (the jump comes back in the air with the ball above 200 uu, within 110 uu and under the wheels): flip_reset_w for the first of each flight (every one with prevent_chain_reset off), + 0.5 for the car's first of the episode, + up to inc_flip_reset_w from the second in a row.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "flip_reset_simple"\n'
        "weight = 1\n"
        "flip_reset_w = 10\n"
        "inc_flip_reset_w = 2\n"
        "```"
    ),
    "goal_in_mouth": (
        "v = +1 on the step the ball enters the goal mouth this car attacks, -1 the one it defends.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "goal_in_mouth"\n'
        "weight = 50\n"
        "```"
    ),
    "goal_speed": (
        "On the step the car's team scores: v = ball speed (uu/s) x reward_factor x 2: 600 for a 3,000 uu/s goal with the default. Use a small weight.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "goal_speed"\n'
        "weight = 0.05\n"
        "```"
    ),
    "goal_speed_and_placement": (
        "On the step the car's team scores: v = ball speed on the step before / 6000, x 1.75 when the ball was then off the floor (above 102.75 uu). 0 to 1.75.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "goal_speed_and_placement"\n'
        "weight = 50\n"
        "```"
    ),
    "goal_speed_bonus": (
        "On a goal: v = goal_speed_bonus_w x ball speed on the step before / 6000, paid to the scoring team and charged to the other: -1 to 1 by default.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "goal_speed_bonus"\n'
        "weight = 30\n"
        "```"
    ),
    "goal_speed_top_corner": (
        "goal_speed_bonus times |ball height / 640| x |ball x / 640| where the ball went in: about 0 along the floor or through the middle, up to about 1.4 in a top corner (the goal is 1,786 uu wide and 642.775 uu high).\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "goal_speed_top_corner"\n'
        "weight = 30\n"
        "```"
    ),
    "kickoff": (
        "Every step while the ball waits on the centre spot: v = 1 when this car is nearer the ball than every opponent, else -0.8 (also with no opponent). 0 once the ball has moved.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "kickoff"\n'
        "weight = 1\n"
        "```"
    ),
    "kickoff_nearest": (
        "Every step of a kickoff: v = 1 while the car is nearer the ball than every opponent, else -0.8. Teammates do not count, so in 2v2 both cars of a team are paid to go.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "kickoff_nearest"\n'
        "weight = 1\n"
        "```"
    ),
    "kickoff_nearest_team": (
        "Every step the ball stands still: v = 1 when any car of this car's team (itself included) is nearer the ball than every opponent, else -1. Only one car of a team needs to win the race.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "kickoff_nearest_team"\n'
        "weight = 1\n"
        "```"
    ),
    "kickoff_nearest_v2": (
        "Every step of a kickoff while the car is more than min_distance from the ball: v = 1 when nearer than every opponent, else -1. 0 inside min_distance, so the last stretch, where the challenge is decided, is left to other rewards.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "kickoff_nearest_v2"\n'
        "weight = 1\n"
        "min_distance = 1200\n"
        "```"
    ),
    "kickoff_speed_no_boost": (
        "Every step the ball stands still while the car's boost is below 2: v = max(0, speed toward the ball / 2300). It pays keeping speed after the kickoff boost runs out.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "kickoff_speed_no_boost"\n'
        "weight = 1\n"
        "```"
    ),
    "kickoff_speed_no_boost_v2": (
        "Every step of a kickoff while boost is below max_boost: v = goal_speed x max(speed toward the ball / 2300, 0.025); 0.0125 to 0.5 with the defaults, even moving away.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "kickoff_speed_no_boost_v2"\n'
        "weight = 1\n"
        "max_boost = 2\n"
        "goal_speed = 0.5\n"
        "```"
    ),
    "landing_recovery": (
        "Every step the car falls in the air above 250 uu: v = (cos(nose, flight) seen from above + roof's upward share) / 2, -1 to 1. 1 with the nose along the direction of travel and the wheels straight down.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "landing_recovery"\n'
        "weight = 0.5\n"
        "```"
    ),
    "mawkzy_flick": (
        "Stages, each paid at most once per possession of the ball:\n"
        "- 0.1 for jumping off a controlled hood dribble;\n"
        "- up to 0.3 for turning into the Mawkzy's pose after the pop (kept only if a Mawkzy dodge follows);\n"
        "- up to 0.15 for the dodge's shape;\n"
        "- the flick itself, judged on the ball's launch (power, lift, aim at the net): -0.5 backward to 1.5 for a full-power Mawkzy.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "mawkzy_flick"\n'
        "weight = 10\n"
        "flick_weight = 1.0\n"
        "```\n"
        "\n"
        "No dribble pay unless dribble_rate is set."
    ),
    "musty_flick": (
        "Judged once when the ball leaves the car after a musty (a backflip-style dodge whose car hits the ball with its roof's rear, nose down). Credit adds goal-directed power, lift, aim and a shot on net, up to flick_weight; x 1.5 off a wall, the ceiling or a flip reset; half for a plain roof push; down to -backward_penalty for a ball sent toward your own goal.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "musty_flick"\n'
        "weight = 10\n"
        "flick_weight = 1.0\n"
        "backward_penalty = 0.5\n"
        "```"
    ),
    "musty_pose_shaping": (
        "Potential-based shaping toward the musty's pose: Phi in 0..1 grows as the nose turns down toward the ball, the roof faces it and the ball is close. Each step in the air with a dodge left: v = gamma x Phi(now) - Phi(before). If the dodge that follows is a musty the pay is kept, otherwise it is paid back; at most 1 per dodge.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "musty_pose_shaping"\n'
        "weight = 2\n"
        "gamma = 0.995     # the training's discount per step\n"
        "```"
    ),
    "on_ground": (
        "Every step: v = 1 while the car's wheels touch the floor, a wall or the ceiling, else 0. The opposite of air.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "on_ground"\n'
        "weight = 0.1\n"
        "```"
    ),
    "out_of_position": (
        "Every step an opponent is within near_radii ball radii of the ball and this car is further toward the opponents' net than the ball and every such opponent (by more than margin): v = -1, else 0.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "out_of_position"\n'
        "weight = 1\n"
        "near_radii = 8\n"
        "```"
    ),
    "player_on_wall": (
        "Every step: v = 1 while the car is on its wheels higher than 300 uu (a wall or the ceiling), else 0.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "player_on_wall"\n'
        "weight = 0.2\n"
        "```"
    ),
    "possession": (
        "Every step the car's team touched the ball last and the car is within min_dist of it with no opponent nearer: v = n / 10, where n counts such steps (of any such car of the team) and restarts when the other team touches. It grows without end: 0.1, 0.2, 0.3...\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "possession"\n'
        "weight = 0.05\n"
        "min_dist = 500\n"
        "```"
    ),
    "raze_flick": (
        "Judged once when the ball leaves the car after the whole move: hood dribble, a dodge with the roof tilted, the car's top bouncing off the floor (the pogo), then its rear popping the ball toward the goal. Credit for the pop, the pogo, the shot and its power, up to flick_weight; down to -backward_penalty for a ball sent toward your own goal.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "raze_flick"\n'
        "weight = 10\n"
        "flick_weight = 1.0\n"
        "```"
    ),
    "raze_flick_setup": (
        "From the pop jump off a dribble to the dodge, at most once per possession: each step pays setup_rate x q x step seconds, q in 0..1 measuring how closely the car and ball match reference Raze flicks (each feature's squared mismatch capped at max_z2). About 0.87 x setup_rate in all at best.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "raze_flick_setup"\n'
        "weight = 2\n"
        "setup_rate = 1.0\n"
        "```"
    ),
    "ball_to_roof_shaping": (
        "Until the ball is first under control on the roof this episode, each step pays how much closer it came to the roof spot, as a potential running from -1 (d_max or further) to 0 (on it): the steps add up to at most 1 an episode.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "ball_to_roof_shaping"\n'
        "weight = 1\n"
        "d_max = 1500\n"
        "```"
    ),
    "reset_musty": (
        "A musty flick off a flip reset the car still held when it dodged: musty_flick's pay (up to about 2.3 for an aerial musty on net) x a factor from 0.5 for a smack to 1 for a sling (the car riding the ball before the flick). Any other musty pays nothing; a backward shot can cost up to 0.5.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "reset_musty"\n'
        "weight = 15\n"
        "```"
    ),
    "reset_musty_pose_shaping": (
        "While the car holds a counted flip reset: v = gamma x Phi(now) - Phi(before), Phi in 0..1 for the musty pose (nose down past vertical, roof to the ball, close). Kept when the dodge is a musty, paid back otherwise; at most 1 per dodge.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "reset_musty_pose_shaping"\n'
        "weight = 5\n"
        "gamma = 0.995\n"
        "```"
    ),
    "speed_flip": (
        "Once per jump, for a forward-diagonal dodge cancelled so the nose stays level: credit = how much farther along its line the car got than boosting straight would have, up to max_credit (full at 800 uu ahead). Any other dodge pays nothing.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "speed_flip"\n'
        "weight = 10\n"
        "max_credit = 1\n"
        "```"
    ),
    "speed_forward": (
        "Every step: v = speed / 2300, positive moving nose first, negative in reverse (-1 to 1).\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "speed_forward"\n'
        "weight = 0.5\n"
        "```"
    ),
    "speed_linear": (
        "Every step: v = speed / 1000: 0 at rest to 2.3 at top speed, not capped at 1.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "speed_linear"\n'
        "weight = 0.2\n"
        "```"
    ),
    "speed_sqrt": (
        "Every step: v = min(1, sqrt(speed / 1800)): full from 1,800 uu/s.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "speed_sqrt"\n'
        "weight = 0.5\n"
        "```"
    ),
    "speed_sqrt_v2": (
        "Every step: v = sqrt(speed / 2300): 0.5 at a quarter of top speed, 1 at top speed. The square root pays the first bit of speed the most.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "speed_sqrt_v2"\n'
        "weight = 0.5\n"
        "```"
    ),
    "speed_toward_ball": (
        "Every step: v = max(0, car velocity . direction to the ball) / 2300. Like velocity_player_to_ball without the penalty for moving away.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "speed_toward_ball"\n'
        "weight = 4\n"
        "```"
    ),
    "speedy_touch": (
        "On a touch: v = min(1, |change in the ball's speed over the step| / 2777), 2,777 uu/s being 100 km/h. A hard brake counts as much as a hard hit.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "speedy_touch"\n'
        "weight = 20\n"
        "```"
    ),
    "supersonic": (
        "Every step: v = 1 while the game marks the car supersonic (from 2,200 uu/s, kept a moment while slowing), else 0.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "supersonic"\n'
        "weight = 0.2\n"
        "```"
    ),
    "tornado_shaping": (
        "As musty_pose_shaping, with the potential Phi = (|roll| + |yaw| turned since the pop, both the same way) / full_deg, capped at 1, while the ball is close and in the air. Kept only when the dodge is one musty_flick judges; at most 1 per dodge.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "tornado_shaping"\n'
        "weight = 2\n"
        "full_deg = 360\n"
        "```"
    ),
    "touch_ball_scaled_by_hit_force": (
        "On a touch that changes the ball's velocity by more than 500 uu/s over the step: v = |dv| / 3000, from about 0.17, not capped (a 6,000 uu/s change pays 2).\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "touch_ball_scaled_by_hit_force"\n'
        "weight = 20\n"
        "```"
    ),
    "touch_ball_on_wall_scaled_by_height": (
        "On a touch with the car's wheels on a wall above 200 uu: v = car height / 2044 (0.1 to 1).\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "touch_ball_on_wall_scaled_by_height"\n'
        "weight = 10\n"
        "```"
    ),
    "touch_ball_speed_gain": (
        "On a touch: v = 1 with beginner; otherwise v = max(0, ball speed now - ball speed after this car's previous touch) in uu/s (the first touch pays the ball's whole speed). Use a small weight without beginner.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "touch_ball_speed_gain"\n'
        "weight = 0.005\n"
        "beginner = false\n"
        "```"
    ),
    "velocity_ball_to_goal_aligned": (
        "Every step: the ball's speed toward the back of the opponents' net / 6000, times how straight at it the ball flies: full exactly at the net, nothing from about 41 degrees off. Never negative.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "velocity_ball_to_goal_aligned"\n'
        "weight = 2\n"
        "```"
    ),
    "wall_dash": (
        "A first jump off a wall (roof up.z between -0.7 and 0.98) starts an attempt. Made: the second press dodges, the wheels leave the wall, then 3+ wheels touch it within 30 ticks of the first touch back (not below z = 60) with dot(roof, wall normal) > 0.9. gain = the highest speed in the wall's plane over the 18+ ticks after, minus the speed in that plane at the first press. v = min(1, gain / full_gain) x g(press to press) x g(hold) x g(flip angle) x (1 + chain_bonus for a make whose first jump came within 120 ticks of the last make's landing, without 6+ ticks off every surface or a touch of the floor between); g = 1 green, yellow_factor yellow, red_factor red. Bands in ticks (green up to / yellow up to), each edge rounded up to a whole step: press to press 12/14 under 1300 uu/s along the wall at the first press, 13/13 to 1500, 13/14 to 1700, 14/15 to 1900, 15/16 above; hold 6/9. Flip angle = atan2(the dodge's sideways part toward the ceiling, its forward part): green 50-70 degrees, yellow 45-85, red otherwise; not graded with the car pointing up or down the wall (|right.z| < 0.2). Surfaces from the wall's normal n at the first press: |n.z| > 0.1 a curve, else |n.x| or |n.y| >= 0.98 a flat wall, else a corner. v = 0 with 4+ jump presses, gain <= min_speed_gain or a surface not allowed; v = -miss_cost when a dodge from an allowed surface is not made (no return within 120 ticks, back with under 3 wheels, twisted, or on the floor). Rules from walldashcoach by samuelkassi (MIT).\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "wall_dash"\n'
        "weight = 5\n"
        "chain_bonus = 0.5\n"
        "```"
    ),
    "wall_positioning": (
        "Every step the car is on a wall above 200 uu with the ball above it within 463.75 uu (5 ball radii) and more than 50 boost: v = 1, or 4 with a touch.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "wall_positioning"\n'
        "weight = 0.5\n"
        "```"
    ),
    "wavedash": (
        "When a dodge lands the wheels on the floor right after it starts: v = min(1, speed gained since before the dodge / 500).\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "wavedash"\n'
        "weight = 5\n"
        "```"
    ),
    "wizard_flick": (
        "v = 1 for a wizard flick: the first touch after a forward-diagonal dodge meets the ball with a bottom front corner, the roof tilted past 90 degrees and the nose above level, the car well off the floor for 0.25 s, the ball never on the floor this game, and the ball's upward speed rising by 200+ uu/s. Else 0.\n"
        "\n"
        "```toml\n"
        "[[rewards]]\n"
        'name = "wizard_flick"\n'
        "weight = 20\n"
        "```"
    ),
}
