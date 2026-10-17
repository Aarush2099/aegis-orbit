from pathlib import Path

import pytest

from benchmarks.oracle_reference import get, load_config


def test_config_is_nested_and_tab_free():
    config = load_config()
    assert config["environment"]["initial_offset_m"]["low"] == 67.0
    assert config["environment"]["initial_offset_m"]["high"] == 67.5
    assert "\t" not in Path("configs/sac_unshielded.yaml").read_text(encoding="utf-8")


def test_missing_required_config_key_raises_named_key():
    with pytest.raises(KeyError, match="missing"):
        get({}, "environment", "n")


def test_tab_indentation_is_rejected():
    path = Path("configs/_tab_config_test.yaml")
    path.write_text("root:\n\tchild: 1\n", encoding="utf-8")
    try:
        with pytest.raises(Exception):
            load_config(path)
    finally:
        path.unlink()
