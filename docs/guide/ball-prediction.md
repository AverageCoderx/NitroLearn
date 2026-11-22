# Ball prediction

Rocket League players read where the ball is going. With ball prediction on, a bot is told too: at each
decision, where the ball will be a moment ahead. It is off by default and not proven yet: it costs some
training speed, and whether it makes a better bot is not measured.

## Turning it on

In the dashboard: when you make a new bot, open its **Settings** tab before the first start, **Show
advanced**, and look under **Games** for ball prediction (the search box finds it too). It cannot be
switched on or off after the bot's first save, because it changes what the bot sees.

## What it does

In the settings file, list a few times in
seconds and the bot is told, at each decision, where the ball will be that far ahead and how fast it will
move, also compared with its own car:
```toml
[env]
ball_prediction = [0.5, 1, 2, 3]    # seconds ahead; each time adds 12 numbers to what the bot sees
```
The prediction follows the ball alone (bounces off the floor, walls and ceiling included): it cannot know that
a car is about to hit the ball. Each time is rounded to a whole decision, and `nitro check-settings` shows the
rounded times. Choose it before the first start: it is fixed for a run, like the observation. It costs some
training speed, as every game also works out the ball's path (about 15% on one test PC); whether it helps the
bot learn is not measured yet. Copying replays and replays in training work with it. An exported bot reads RLBot's own
ball prediction at the same times; its README says how to let it predict the ball itself when RLBot sends
none.
