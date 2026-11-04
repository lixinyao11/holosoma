"""One upstream configuration object, explicit presets, and safe composition."""
from dataclasses import asdict
import json
import sys

import pytest
import yaml

from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_values.wbt.g1 import m15, hdmi_goal043
from holosoma.config_values.wbt.g1.observation import obj_future_terms, object_bbox_terms
from test_m15_multi40 import prepared


@pytest.mark.parametrize("group,kwargs", [
    ("A", dict(extra_dr=False, motion_set="single043", track_phase=False)),
    ("B", dict(extra_dr=True, motion_set="single043", track_phase=False)),
    ("C", dict(extra_dr=False, motion_set="multi40", track_phase=False)),
    ("D", dict(extra_dr=False, motion_set="multi40", track_phase=True)),
])
def test_named_group_preserves_explicit_configuration(prepared, group, kwargs):
    root, _, _ = prepared
    cfg = m15.make_config(root, group=group)
    assert isinstance(cfg, ExperimentConfig)
    assert asdict(cfg) == asdict(m15.make_config(root, **kwargs))
    assert hdmi_goal043.make_config is m15.make_config
    assert hdmi_goal043.make_eval_config is m15.make_eval_config


@pytest.mark.parametrize("kwargs", [
    dict(group="B", extra_dr=False), dict(group="C", motion_set="single043"),
    dict(group="D", track_phase=False), dict(group="A", track_phase=True),
    dict(group="unknown"), dict(rewind_prob=-.1), dict(rewind_prob=1.1),
    dict(rewind_prob=float("nan")), dict(object_persist_steps=0),
    dict(object_persist_steps=1.5), dict(object_persist_steps=True),
])
def test_conflicts_fail_before_asset_loading(tmp_path, kwargs):
    with pytest.raises(ValueError):
        m15.make_config(tmp_path, **kwargs)


def test_upstream_options_preserve_custom_profile_and_dataset(prepared):
    root, _, _ = prepared
    base = m15.make_config(root, group="D")
    changed = m15.make_config(root, group="D", rewind_prob=.8, ego_camera=True,
                              future_object_poses=True, object_bbox=True,
                              object_persist_steps=5, body_pos_z_only=True)
    for field in ("robot", "action", "randomization", "algo"):
        assert asdict(getattr(changed, field)) == asdict(getattr(base, field))
    motion = changed.command.setup_terms["motion_command"].params["motion_config"]
    assert motion.rewind_prob == .8
    assert motion.motion_dir == base.command.setup_terms["motion_command"].params["motion_config"].motion_dir
    assert changed.simulator.config.ego_camera.enabled
    assert changed.termination.terms["bad_tracking"].params["object_persist_steps"] == 5
    assert changed.termination.terms["bad_tracking"].params["body_pos_z_only"]
    for group in ("actor_obs", "critic_obs"):
        terms = changed.observation.groups[group].terms
        for key, term in base.observation.groups[group].terms.items():
            assert asdict(terms[key]) == asdict(term)
        for key, term in {**obj_future_terms, **object_bbox_terms}.items():
            assert asdict(terms[key]) == asdict(term)
    assert not base.simulator.config.ego_camera.enabled
    assert base.command.setup_terms["motion_command"].params["motion_config"].rewind_prob == 0
    assert changed.training.name.endswith("_track_phase")


def test_hardware_switches_reach_the_single_builder(prepared):
    root, _, _ = prepared
    cfg = m15.make_config(root, collision_exclusions=False, motor_dr=False, delay_filter=False)
    assert cfg.robot.asset.collision_exclusions == ()
    assert "hdmi_armature" not in cfg.randomization.setup_terms
    assert not cfg.randomization.setup_terms["actuator_randomizer_state"].params.get("enable_pd_gain", False)
    assert "HDMIJointPositionActionTerm" not in cfg.action.terms["joint_control"].func


@pytest.mark.parametrize("group", ["C", "D"])
def test_eval_uses_heldout_split_without_changing_group(prepared, group):
    root, _, _ = prepared
    cfg = m15.make_eval_config(root, group=group)
    assert cfg.command.setup_terms["motion_command"].params["motion_config"].motion_dir.endswith("motions_eval")
    assert ("track_phase" in cfg.observation.groups["actor_obs"].terms) == (group == "D")


@pytest.mark.parametrize("suffix", [".json", ".yaml"])
def test_cli_exports_upstream_representation_without_overwrite(prepared, tmp_path, monkeypatch, suffix):
    root, _, _ = prepared
    output = tmp_path / ("experiment" + suffix)
    monkeypatch.setattr(sys, "argv", ["m15", "--asset-root", str(root), "--group", "D", "--output", str(output)])
    m15.main()
    parsed = json.loads(output.read_text()) if suffix == ".json" else yaml.safe_load(output.read_text())
    assert parsed == m15.make_config(root, group="D").to_serializable_dict()
    previous = output.read_bytes()
    with pytest.raises(FileExistsError):
        m15.main()
    assert output.read_bytes() == previous
