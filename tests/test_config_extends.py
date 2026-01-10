from __future__ import annotations

from pathlib import Path

import pytest

from predator_prey_sbi.config import load_config


def test_load_config_extends_merges_nested_dicts(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"

    base.write_text(
        "\n".join(
            [
                "observation_operator: annual_mean",
                "inference:",
                "  num_simulations: 2000",
                "  embedding:",
                "    hidden_features: 64",
                "    num_transforms: 5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    child.write_text(
        "\n".join(
            [
                "extends: base.yaml",
                "inference:",
                "  num_simulations: 4000",
                "  embedding:",
                "    hidden_features: 128",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = load_config(str(child))
    assert cfg["observation_operator"] == "annual_mean"
    assert cfg["inference"]["num_simulations"] == 4000
    assert cfg["inference"]["embedding"]["hidden_features"] == 128
    assert cfg["inference"]["embedding"]["num_transforms"] == 5


def test_load_config_extends_cycle_raises(tmp_path: Path) -> None:
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text("extends: b.yaml\n", encoding="utf-8")
    b.write_text("extends: a.yaml\n", encoding="utf-8")
    with pytest.raises(ValueError, match="extends cycle"):
        load_config(str(a))
