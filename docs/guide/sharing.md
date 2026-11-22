# Sharing bots

There are three ways to pass a bot on, depending on what the other person (or computer) will do with it.

## To play it: share its RLBot folder

Make the bot's RLBot bot (**Make RLBot bot** on its page, or `.\nitro export <bot>`; see
[Playing your bot in RLBot](rlbot.md)). The folder `bots\<bot>` holds everything RLBot v5 needs: zip it and
send it. The other person does not need NitroLearn, Python or a graphics card; they add the folder to RLBot
v5 as a bot folder. Keep `NOTICES.md` in the folder when you share it: it holds the licence notices that go
with the bot.

Bots you train are yours to share, as long as they are never used online against real players
(`BOTS-LICENSE.md` in NitroLearn's folder has the exact terms). Bot tournaments and RLBot matches are fine.

## To keep training it: pack it

**Pack for another computer** on the bot's page (or `.\nitro pack <bot>`) puts the bot, its settings,
its newest save and the data files its settings read, in one `.nitropack.zip` file in the `packs` folder. On the other
computer, **Add a packed bot…** on the Garage page (or `.\nitro unpack <file>`) adds it, and it carries on
from its newest save. The file names no folder of your computer: a path in the bot's settings or saves
goes in relative to NitroLearn's folder or the bot's, or by its name alone. The arena files stay out (they
are Rocket League's game files): the other computer downloads its own. Nothing is overwritten: a bot of the
same name there is replaced only if you say so (the old one is kept aside), or added under a free name.

## To share its settings only

A bot's settings file (`runs\<bot>\settings.toml`) or a reward set (the Rewards tab's **Save as a reward
set**) is a small text file anyone can add on the **Resources** page with **Add a file…**, as a starting
point for a new bot of their own.

## Training on a rented computer
A bot can move to another machine with a bigger graphics card (a rented cloud machine, say), train there and
come back, carrying on from its newest save. Pack it as above; on the machine, `bash tools/cloud_setup.sh`
sets NitroLearn up, `nitro unpack <file>` adds the bot and `nitro resume <bot>` trains it; back home, **Add a
packed bot…** on the dashboard's Garage page brings it back. W&B's own files stay home.
