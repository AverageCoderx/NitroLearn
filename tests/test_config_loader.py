from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest

from nitro.config import NitroConfig
from nitro.config.loader import ConfigError, KeyLocator, load_toml, parse_toml, section, setting


@dataclass(frozen=True)
class Item:
    name: str
    weight: float = setting(1.0, "how much it counts", minimum=0.0)


@dataclass(frozen=True)
class Inner:
    tick_skip: int = setting(8, "physics ticks per decision", minimum=1, maximum=16)
    mode: Literal["fast", "careful"] = setting("fast", "how to run")
    enabled: bool = setting(True, "on or off")
    folder: Path = setting(Path("meshes"), "a folder")


@dataclass(frozen=True)
class Schema:
    inner: Inner = section(Inner, "inner settings")
    items: tuple[Item, ...] = ()
    rate: float = setting(0.5, "a rate", minimum=0.0, maximum=1.0)


# Absolute paths on this OS ("C:\" on Windows, "/" elsewhere), so the tests pass on both.
ROOT = Path(Path.cwd().anchor)
BASE = ROOT / "base"
ABSOLUTE_FOLDER = ROOT / "abs" / "meshes"


def parse(text: str, base: Path = BASE) -> Schema:
    return parse_toml(text, Schema, file=Path("test.toml"), base_dir=base)


def issues_of(text: str) -> list[tuple[str, int | None, str]]:
    with pytest.raises(ConfigError) as info:
        parse(text)
    return [(issue.path, issue.line, issue.message) for issue in info.value.issues]


def test_empty_file_gives_defaults_with_paths_resolved() -> None:
    config = parse("")
    assert config.inner.tick_skip == 8
    assert config.inner.mode == "fast"
    assert config.items == ()
    assert config.inner.folder == (BASE / "meshes").resolve()


def test_values_are_converted_to_declared_types() -> None:
    config = parse(
        f"""
rate = 1          # an int is accepted where a number is expected
[inner]
tick_skip = 4
mode = "careful"
enabled = false
folder = "{ABSOLUTE_FOLDER.as_posix()}"

[[items]]
name = "touch"
weight = 2

[[items]]
name = "goal"
"""
    )
    assert config.rate == 1.0 and isinstance(config.rate, float)
    assert config.inner == Inner(4, "careful", False, ABSOLUTE_FOLDER)
    assert config.items == (Item("touch", 2.0), Item("goal", 1.0))


def test_unknown_setting_suggests_the_closest_name_and_line() -> None:
    with pytest.raises(ConfigError) as info:
        parse("[inner]\ntick_skp = 4\n")
    (issue,) = info.value.issues
    assert issue.path == "inner.tick_skp"
    assert issue.line == 2
    assert issue.hint == "did you mean 'tick_skip'?"


def test_wrong_type_is_explained_in_plain_words() -> None:
    assert issues_of('[inner]\ntick_skip = "eight"\n') == [
        ("inner.tick_skip", 2, "must be a whole number (like 8), but is 'eight'")
    ]
    assert issues_of("[inner]\ntick_skip = 8.0\n")[0][2].startswith("must be a whole number")
    assert issues_of("rate = true\n")[0][2] == "must be a number (like 0.5), but is True"


def test_ranges_and_choices_are_enforced() -> None:
    assert issues_of("[inner]\ntick_skip = 0\n") == [("inner.tick_skip", 2, "must be at least 1, but is 0")]
    assert issues_of("rate = 1.5\n") == [("rate", 1, "must be at most 1, but is 1.5")]
    assert issues_of('[inner]\nmode = "slow"\n') == [
        ("inner.mode", 2, "must be one of 'fast', 'careful', but is 'slow'")
    ]


def test_infinity_and_nan_are_refused() -> None:
    assert issues_of("rate = inf\n") == [("rate", 1, "must be an ordinary number, but is inf")]
    assert issues_of("rate = nan\n") == [("rate", 1, "must be an ordinary number, but is nan")]


def test_every_problem_is_reported_at_once_with_list_positions() -> None:
    text = """
rate = -1

[[items]]
name = "touch"

[[items]]
name = "goal"
weight = -3
colour = "red"
"""
    assert issues_of(text) == [
        ("rate", 2, "must be at least 0, but is -1"),
        ("items[1].weight", 9, "must be at least 0, but is -3"),
        ("items[1].colour", 10, "'colour' is not a known setting"),
    ]


def test_missing_required_setting_is_reported() -> None:
    assert issues_of("[[items]]\nweight = 2\n") == [
        ("items[0].name", 1, "this setting is required but missing")
    ]


def test_section_given_as_a_value_is_reported() -> None:
    assert issues_of("inner = 3\n") == [("inner", 1, "must be a section [inner], not a single value")]


def test_invalid_toml_reports_the_line() -> None:
    with pytest.raises(ConfigError) as info:
        parse("[inner]\ntick_skip = = 3\n")
    (issue,) = info.value.issues
    assert issue.line == 2
    assert issue.message.startswith("this line cannot be read: it is not written the way a settings file is")
    assert "(TOML says: " in issue.message, "the technical detail follows"


def test_problems_are_listed_in_the_order_of_the_files_lines() -> None:
    """Mistakes found section by section are still listed top to bottom."""
    text = "[ppo]\nepochs = -1\n[run]\nseed = -2\n[env]\nteam_size = 9\n"
    with pytest.raises(ConfigError) as info:
        parse_toml(text, NitroConfig, file=Path("s.toml"), base_dir=Path("."))
    lines = [issue.line for issue in info.value.issues]
    assert len(lines) == 3 and lines == sorted(lines), lines


def test_missing_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as info:
        load_toml(tmp_path / "nope.toml", Schema)
    assert info.value.issues[0].message == "the settings file does not exist"


def test_files_saved_with_a_byte_order_mark_load(tmp_path: Path) -> None:
    file = tmp_path / "bom.toml"
    file.write_bytes(b"\xef\xbb\xbf[inner]\ntick_skip = 4\n")
    assert load_toml(file, Schema).inner.tick_skip == 4


def test_relative_paths_are_relative_to_the_settings_file(tmp_path: Path) -> None:
    file = tmp_path / "bots" / "my_bot.toml"
    file.parent.mkdir()
    file.write_text('[inner]\nfolder = "../shared/meshes"\n', encoding="utf-8")
    assert load_toml(file, Schema).inner.folder == (tmp_path / "shared" / "meshes").resolve()


def test_error_text_lists_each_issue_with_hint() -> None:
    with pytest.raises(ConfigError) as info:
        parse("[inner]\ntick_skp = 4\n")
    assert str(info.value) == (
        "1 problem in test.toml:\n"
        "  - line 2, inner.tick_skp: 'tick_skp' is not a known setting\n"
        "      hint: did you mean 'tick_skip'?"
    )


def test_key_locator_handles_comments_quotes_and_nesting() -> None:
    locator = KeyLocator(
        '# top\n"rate" = 1\n[inner]\n  tick_skip = 3 # note\n[[items]]\n[[items]]\nname = "x"\n'
    )
    assert locator.line_of("rate") == 2
    assert locator.line_of("inner.tick_skip") == 4
    assert locator.line_of("items[1].name") == 7
    assert locator.line_of("items[1].weight") == 6
    assert locator.line_of("inner.unwritten") == 3
    assert locator.line_of("nowhere") is None
