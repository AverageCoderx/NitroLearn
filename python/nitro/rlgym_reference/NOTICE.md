# RLGym reference files

Copied for `nitro import-rlgym`'s parity check: the import builds each observation with this code, exactly as an
RLGym-PPO bot saw it, and holds NitroLearn's version of it to the same numbers.

| File | From | Changed |
|---|---|---|
| `common_values.py`, `math.py`, `physics_object.py`, `player_data.py`, `game_state.py`, `obs_builder.py`, `default_obs.py`, `advanced_obs.py` | rlgym-sim (github.com/AechPro/rocket-league-gym-sim, `rlgym_sim/utils/...`), Apache-2.0, `LICENSE-rlgym-sim` | imports made relative to this folder; `obs_builder.py`: the `gym` import dropped (only a type hint), `from __future__ import annotations` added |
| `advanced_padder.py`, `lookup_act.py` | rlgym-tools v1.8.3 (github.com/RLGym/rlgym-tools, `rlgym_tools/extra_obs/advanced_padder.py`, `rlgym_tools/extra_action_parsers/lookup_act.py`), Apache-2.0, `LICENSE-rlgym-tools` | imports made relative; `lookup_act.py`: the `gym` and `ActionParser` imports dropped (LookupAction no longer subclasses ActionParser), `from __future__ import annotations` added |
