"""Unit tests for shared video recorder scheduling."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from holosoma.config_types.video import FixedCameraConfig, VideoConfig
from holosoma.simulator.shared.video_recorder import VideoRecorderInterface


class DummyVideoRecorder(VideoRecorderInterface):
    """Minimal concrete recorder for testing shared scheduling logic."""

    def _capture_frame_impl(self) -> None:
        self._add_frame(np.zeros((2, 2, 3), dtype=np.uint8))


def _make_simulator() -> SimpleNamespace:
    return SimpleNamespace(
        num_envs=1,
        simulator_config=SimpleNamespace(sim=SimpleNamespace(fps=200, control_decimation=4)),
        robot_root_states=np.zeros((1, 13), dtype=np.float32),
    )


def test_training_step_trigger_starts_fixed_duration_clip():
    recorder = DummyVideoRecorder(
        VideoConfig(
            enabled=True,
            trigger="training_step",
            interval=1000,
            clip_duration_s=5.0,
            camera=FixedCameraConfig(),
        ),
        _make_simulator(),
    )
    recorder._encode_and_save_video = Mock()  # type: ignore[method-assign]

    recorder.on_training_step(0)

    assert recorder.is_recording is True

    for _ in range(249):
        recorder.on_control_step()

    assert recorder.is_recording is True

    recorder.on_control_step()

    assert recorder.is_recording is False
    recorder._encode_and_save_video.assert_called_once()


def test_training_step_trigger_repeats_on_interval_after_startup_clip():
    recorder = DummyVideoRecorder(
        VideoConfig(
            enabled=True,
            trigger="training_step",
            interval=1000,
            clip_duration_s=5.0,
            camera=FixedCameraConfig(),
        ),
        _make_simulator(),
    )
    recorder._encode_and_save_video = Mock()  # type: ignore[method-assign]

    recorder.on_training_step(0)
    assert recorder.is_recording is True

    for _ in range(250):
        recorder.on_control_step()

    assert recorder.is_recording is False

    recorder.on_training_step(1000)
    assert recorder.is_recording is True


def test_video_config_defaults_use_fixed_training_clips():
    config = VideoConfig()

    assert config.trigger == "training_step"
    assert config.interval == 500
    assert config.clip_duration_s == 5.0
    assert isinstance(config.camera, FixedCameraConfig)


def test_training_step_trigger_ignores_episode_boundaries():
    recorder = DummyVideoRecorder(
        VideoConfig(
            enabled=True,
            trigger="training_step",
            interval=1000,
            clip_duration_s=1.0,
            camera=FixedCameraConfig(),
        ),
        _make_simulator(),
    )
    recorder._encode_and_save_video = Mock()  # type: ignore[method-assign]

    recorder.on_training_step(1000)
    recorder.on_episode_end(0)
    recorder.on_episode_start(0)

    assert recorder.is_recording is True


def test_episode_trigger_preserves_legacy_interval_behavior():
    recorder = DummyVideoRecorder(
        VideoConfig(
            enabled=True,
            trigger="episode",
            interval=2,
            camera=FixedCameraConfig(),
        ),
        _make_simulator(),
    )

    recorder.on_episode_start(0)
    assert recorder.is_recording is False

    recorder.on_episode_start(0)
    assert recorder.is_recording is True
