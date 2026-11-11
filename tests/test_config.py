import pytest

from holux_metric.config import ConfigError, load_config, resolve_parameters


def test_defaults_load():
    cfg = load_config(None)
    assert cfg["outlier"]["selected"] == "statistical"


def test_invalid_method_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("outlier:\n  selected: magic\n")
    with pytest.raises(ConfigError):
        load_config(bad)


def test_unknown_key_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("voxel: 1\n")
    with pytest.raises(ConfigError):
        load_config(bad)


def test_relative_lengths_scale_with_diagonal():
    cfg = load_config(None)
    params = resolve_parameters(cfg, diagonal=10.0)
    assert params["voxel_size"] == pytest.approx(0.05)
    assert params["outlier_radius"] == pytest.approx(0.2)
