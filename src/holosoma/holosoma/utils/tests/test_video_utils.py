from types import SimpleNamespace

import torch

from holosoma.utils.video_utils import format_command_labels, format_video_overlay_text


def test_format_command_labels_handles_missing_commands():
    assert format_command_labels(None) == "Commands: Not available"


def test_format_video_overlay_text_prefers_task_specific_overlay():
    simulator = SimpleNamespace(
        commands=None,
        task=SimpleNamespace(get_video_overlay_text=lambda env_id: f"WBT beta=0.50 env={env_id}"),
    )

    assert format_video_overlay_text(simulator, env_id=3) == "WBT beta=0.50 env=3"


def test_format_video_overlay_text_falls_back_to_generic_commands():
    simulator = SimpleNamespace(
        commands=torch.tensor([[0.5, -0.25, 0.1, 0.0, 0.0, 0.2, 0.0, 0.0, 1.1]], dtype=torch.float32),
        task=SimpleNamespace(get_video_overlay_text=lambda env_id: ""),
    )

    overlay = format_video_overlay_text(simulator, env_id=0)

    assert "vx=0.50" in overlay
    assert "vy=-0.25" in overlay
    assert "ang_vel=0.10" in overlay
