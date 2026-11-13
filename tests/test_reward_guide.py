"""Every reward's "See more" explanation in the Rewards tab."""

from __future__ import annotations

from nitro.dashboard import rewards
from nitro.dashboard.reward_guide import MORE


def test_every_reward_that_comes_with_nitrolearn_explains_how_it_works() -> None:
    catalog = rewards.catalog()
    shipped = [k for k in catalog if not k.get("own") or k.get("source") in ("library", "example")]
    missing = [k["name"] for k in shipped if not k["more"]]
    assert not missing, missing
    for name, text in MORE.items():
        assert text.startswith("What it does:"), name
        assert 40 <= len(text.split()) <= 320, name
        assert "Watch out for:" in text or "When to use it:" in text, name
    assert {k["name"] for k in shipped} <= set(MORE)
