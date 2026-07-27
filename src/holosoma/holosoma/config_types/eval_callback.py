"""Config types for eval callbacks."""

from __future__ import annotations

import dataclasses

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class RecordingConfig:
    """Settings for trajectory recording during evaluation."""

    enabled: bool = False
    """Whether to enable trajectory recording."""

    output_path: str = "eval_recording.npz"
    """Path to save NPZ recording."""

    env_id: int = 0
    """Environment ID to record."""


@dataclass(frozen=True)
class RecordingCallbackConfig:
    """Instantiation config for EvalRecordingCallback."""

    _target_: str = "holosoma.agents.callbacks.recording.EvalRecordingCallback"
    """Class to instantiate."""

    config: RecordingConfig = RecordingConfig()
    """Recording settings."""


@dataclass(frozen=True)
class PushConfig:
    """Settings for push perturbation during evaluation."""

    enabled: bool = False
    """Enable push perturbations."""

    force_range: tuple[float, float] = (50.0, 200.0)
    """Min and max force magnitude in Newtons."""

    duration_s: tuple[float, float] = (0.1, 0.3)
    """Min and max push duration in seconds."""

    interval_s: tuple[float, float] = (3.0, 8.0)
    """Min and max interval between pushes in seconds."""

    body_names: str = "torso_link,pelvis"
    """Comma-separated body names to push."""

    env_id: int = 0
    """Environment ID to apply pushes."""


@dataclass(frozen=True)
class PushCallbackConfig:
    """Instantiation config for EvalPushCallback."""

    _target_: str = "holosoma.agents.callbacks.push.EvalPushCallback"
    """Class to instantiate."""

    config: PushConfig = PushConfig()
    """Push perturbation settings."""


@dataclass(frozen=True)
class PayloadConfig:
    """Settings for wrist payload simulation during evaluation.

    Applies a constant downward force on wrist/elbow links to simulate
    the robot holding a payload (force = mass * 9.81).
    """

    enabled: bool = False
    """Enable wrist payload forces."""

    mass_kg: float = 1.0
    """Payload mass in kg. Force is split evenly across resolved bodies."""

    body_names: str = "left_wrist_yaw_link,right_wrist_yaw_link"
    """Comma-separated wrist body names."""

    env_id: int = 0
    """Environment ID to apply payload forces."""


@dataclass(frozen=True)
class PayloadCallbackConfig:
    """Instantiation config for EvalPayloadCallback."""

    _target_: str = "holosoma.agents.callbacks.payload.EvalPayloadCallback"
    """Class to instantiate."""

    config: PayloadConfig = PayloadConfig()
    """Payload simulation settings."""


@dataclass(frozen=True)
class ViserConfig:
    """Settings for viser web visualization during evaluation."""

    enabled: bool = False
    """Whether to enable viser visualization."""

    port: int = 8012
    """Port for the viser web server."""

    num_envs: int = 16
    """Maximum number of environments to visualize, starting from environment 0."""

    show_reference: bool = True
    """Show reference motion (WBT only, uses a second robot with color override)."""

    reference_color: tuple[float, float, float] = (0.3, 0.6, 1.0)
    """RGB color for reference robot mesh (0-1 range)."""

    reference_offset: tuple[float, float, float] = (0.0, 1.0, 0.0)
    """Spatial offset for reference robot to avoid overlap."""

    ground_width: float = 50.0
    """Width of the ground grid plane."""

    ground_depth: float = 50.0
    """Depth of the ground grid plane."""

    ground_spacing: float = 1.0
    """Spacing between grid lines."""

    realtime: bool = True
    """Pace visualization to not exceed real-time (sleep if sim is faster)."""

    camera_position_offset: tuple[float, float, float] = (3.5, -4.25, 2.3)
    """Default camera position relative to environment 0's initial robot root."""

    camera_look_at_offset: tuple[float, float, float] = (0.0, 0.5, -0.25)
    """Default camera target relative to environment 0's initial robot root."""

    camera_fov_degrees: float = 55.0
    """Default vertical camera field of view in degrees."""

    recording_width: int = 1280
    """Requested video width, clamped to the connected browser viewport."""

    recording_height: int = 720
    """Requested video height, clamped to the connected browser viewport."""


@dataclass(frozen=True)
class ViserCallbackConfig:
    """Instantiation config for EvalViserCallback."""

    _target_: str = "holosoma.agents.callbacks.viser_callback.EvalViserCallback"
    """Class to instantiate."""

    config: ViserConfig = ViserConfig()
    """Viser visualization settings."""


@dataclass(frozen=True)
class EvalCallbacksConfig:
    """Container for all eval callback configs.

    To add a new callback, add a field here with its config type.
    Each field's value is passed to instantiate() if it has a _target_.
    """

    recording: RecordingCallbackConfig = RecordingCallbackConfig()
    """Trajectory recording callback."""

    push: PushCallbackConfig = PushCallbackConfig()
    """Push perturbation callback."""

    payload: PayloadCallbackConfig = PayloadCallbackConfig()
    """Wrist payload simulation callback."""

    viser: ViserCallbackConfig = ViserCallbackConfig()
    """Viser web visualization callback."""

    def collect_active_callbacks(self) -> dict:
        """Collect callback configs where config.enabled is True."""
        cb_configs = {}
        for f in dataclasses.fields(self):
            cfg = getattr(self, f.name)
            if not hasattr(cfg, "_target_"):
                raise ValueError(f"Callback config '{f.name}' missing _target_ field")
            if not hasattr(cfg.config, "enabled"):
                raise ValueError(f"Callback config '{f.name}' missing config.enabled field")
            if cfg.config.enabled:
                cb_configs[f.name] = cfg
        return cb_configs
