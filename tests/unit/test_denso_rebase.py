"""CPU contracts for the denso-hil integration; no Isaac or GPU launch."""
import json
import subprocess

import cv2
import numpy as np
import pytest

from holosoma.config_values.wbt.g1.hdmi_goal import make_legacy_m15_config
from holosoma.config_values.experiment import DEFAULTS
from holosoma.utils import video_utils


@pytest.mark.parametrize("extra_dr", [False, True])
def test_upstream_features_are_opt_in_for_legacy(extra_dr):
    cfg = make_legacy_m15_config(extra_dr=extra_dr)
    motion = cfg.command.setup_terms["motion_command"].params["motion_config"]
    assert motion.rewind_prob == 0
    assert not cfg.robot.object.object_per_env
    assert not cfg.simulator.config.ego_camera.enabled
    assert "object_bbox" not in cfg.observation.groups["actor_obs"].terms
    assert not any(k.startswith("obj_future_") for k in cfg.observation.groups["actor_obs"].terms)
    assert cfg.termination.terms["bad_tracking"].params.get("object_persist_steps", 1) == 1


def test_upstream_presets_remain_available():
    names = tuple(DEFAULTS)
    assert any("bbox" in name for name in names)
    assert any("rewind" in name for name in names)
    assert any("future" in name for name in names)


def test_same_second_videos_are_distinct_and_decodable(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(video_utils.time, "time", lambda: 1000)
    frames = np.zeros((3, 32, 32, 3), dtype=np.uint8)
    paths = [video_utils.create_video(frames, 25, tmp_path, wandb_logging=False) for _ in range(2)]
    assert paths[0] != paths[1]
    for path in paths:
        cap = cv2.VideoCapture(str(path))
        count = 0
        while cap.read()[0]:
            count += 1
        cap.release()
        assert count == len(frames)
    assert not list(tmp_path.glob("*temp_raw*"))


def test_nonzero_rank_does_not_create_video(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "1")
    assert video_utils.create_video(None, 25, tmp_path / "unused", wandb_logging=False) is None
    assert not (tmp_path / "unused").exists()


def test_encoder_failure_retains_diagnostics(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "0")
    def fail(command, **kwargs):
        raise subprocess.CalledProcessError(1, command, output="", stderr="test encoder failure")
    monkeypatch.setattr(video_utils.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="Video creation failed"):
        video_utils.create_video(np.zeros((2, 32, 32, 3), dtype=np.uint8), 25, tmp_path,
                                 output_format="h264", wandb_logging=False)
    reports = list(tmp_path.glob("*.ffmpeg-error.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text())
    assert report["rank"] == "0"
    assert report["stderr"] == "test encoder failure"
    assert report["returncode"] == 1
    assert not list(tmp_path.glob("*temp_raw*"))
