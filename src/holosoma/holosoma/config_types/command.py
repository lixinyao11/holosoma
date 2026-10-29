"""Configuration types for the command & curriculum manager."""

from __future__ import annotations

from dataclasses import field
from typing import Any, Literal

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class CommandTermCfg:
    """Configuration for a single command or curriculum hook."""

    func: str
    """Import path for the command hook (function or callable class)."""

    params: dict[str, Any] = field(default_factory=dict)
    """Additional parameters forwarded to the hook."""


@dataclass(frozen=True)
class CommandManagerCfg:
    """Configuration for the command manager."""

    params: dict[str, Any] = field(default_factory=dict)
    """Global parameters shared across command hooks."""

    setup_terms: dict[str, CommandTermCfg] = field(default_factory=dict)
    """Hooks invoked during environment setup."""

    reset_terms: dict[str, CommandTermCfg] = field(default_factory=dict)
    """Hooks invoked on environment reset."""

    step_terms: dict[str, CommandTermCfg] = field(default_factory=dict)


########################################################################################################################
# Motion command configuration
########################################################################################################################
@dataclass(frozen=True)
class StepZeroRootNoiseConfig:
    """Gaussian-clipped root-pose noise override used when resetting at the first frame of a motion clip."""

    pos_std_xyz: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """Standard deviation of world-frame root position noise on x, y, z."""

    pos_clip_xyz: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """Absolute clip range of world-frame root position noise on x, y, z."""

    rpy_std: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """Standard deviation of root roll, pitch, yaw noise in radians."""

    rpy_clip: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """Absolute clip range of root roll, pitch, yaw noise in radians."""


@dataclass(frozen=True)
class NoiseToInitialPoseConfig:
    """Initial pose of the robot and object to those in the motion file."""

    overall_noise_scale: float = 0.0
    """Overall noise scale for the initial pose."""

    dof_pos: float = 0.0
    """Noise scale for the initial dof position."""

    root_pos: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """noise scale for root position x, y, z."""

    root_rot: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """noise scale for root rotation roll, pitch, yaw."""

    root_lin_vel: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """noise scale for root linear velocity vx, vy, vz."""

    root_ang_vel: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """noise scale for root angular velocity wx, wy, wz."""

    object_pos: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """noise scale for object position x, y, z."""

    step_zero_root_noise: StepZeroRootNoiseConfig | None = None
    """Optional root position/orientation noise override used when reset timestep equals the clip start."""


@dataclass(frozen=True)
class GoalPoseNoiseConfig:
    """Axis-wise Gaussian noise with independent clipping for a goal pose."""

    pos_std_xyz: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """Standard deviation of world-frame position noise on x, y, z."""

    pos_clip_xyz: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """Absolute clip range of world-frame position noise on x, y, z."""

    rpy_std: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """Standard deviation of world-frame roll, pitch, yaw noise in radians."""

    rpy_clip: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """Absolute clip range of world-frame roll, pitch, yaw noise in radians."""


@dataclass(frozen=True)
class GoalOrientationConfig:
    """Optional yaw-only handling for goal-conditioned orientation logic."""

    robot_yaw_only: bool = False
    """If True, goal-conditioned robot orientation uses yaw-only torso orientation."""

    object_yaw_only: bool = False
    """If True, goal-conditioned object orientation uses yaw-only object orientation."""


@dataclass(frozen=True)
class AdaptiveTimestepSamplerConfig:
    """Adaptive reset sampler settings for WBT motion/time-step selection."""

    bin_width_s: float = 1.0
    """Bin width in seconds used to bucket each motion clip locally."""

    ema_alpha: float = 0.01
    """EMA update rate for per-bin fail-EMA statistics."""

    bin_uniform_mix: float = 0.1
    """Uniform mixing coefficient for per-motion bin sampling probabilities."""

    motion_uniform_mix: float = 0.1
    """Uniform mixing coefficient for per-group motion sampling probabilities."""

    score_power: float = 1.0
    """Exponent applied to difficulty scores before probability normalization."""

    min_segment_steps: int = 10
    """Minimum segment length in control steps required before updating statistics."""

    update_interval: int = 10
    """Training-step interval for applying real adaptive-sampler EMA updates.

    When greater than 1, WBT collects segment-end counts only on the most recent
    rollout of each interval and applies one EMA update at the end of that rollout.
    """

    dump_interval: int = 100
    """Training-step interval for appending a human-readable sampler snapshot file."""

    dump_file: str = "sampleweight.txt"
    """Relative filename written inside the experiment log directory for sampler snapshots."""


@dataclass(frozen=True)
class EvalFixedResetConfig:
    """Optional eval reset override for inspecting a fixed motion segment."""

    enabled: bool = False
    """If True, evaluation resets use this fixed motion/bin/frame instead of the default eval reset sampler."""

    motion_id: int = 0
    """Global motion id to evaluate."""

    bin_id: int | None = None
    """Motion-local adaptive-sampler bin id to sample from. Mutually exclusive with local_frame."""

    local_frame: int | None = None
    """Exact motion-local frame to evaluate. Mutually exclusive with bin_id."""

    randomize_within_bin: bool = True
    """If True with bin_id set, sample uniformly inside the bin; otherwise use the bin start frame."""

    def __post_init__(self) -> None:
        if self.motion_id < 0:
            raise ValueError(f"EvalFixedResetConfig.motion_id must be >= 0, got {self.motion_id}.")
        if self.bin_id is not None and self.bin_id < 0:
            raise ValueError(f"EvalFixedResetConfig.bin_id must be >= 0, got {self.bin_id}.")
        if self.local_frame is not None and self.local_frame < 0:
            raise ValueError(f"EvalFixedResetConfig.local_frame must be >= 0, got {self.local_frame}.")
        if self.bin_id is not None and self.local_frame is not None:
            raise ValueError("EvalFixedResetConfig.bin_id and local_frame are mutually exclusive.")


@dataclass(frozen=True)
class HybridSizeRandomizationConfig:
    """Fixed per-environment geometry randomization for supported hybrid WBT tasks."""

    enabled: bool = False
    """Whether fixed size randomization is enabled."""

    track_scale_range: list[float] = field(default_factory=lambda: [0.95, 1.05])
    """Uniform scalar sampling bounds for tracking environments."""

    gen_scale_range: list[float] = field(default_factory=lambda: [0.85, 1.15])
    """Uniform scalar sampling bounds for goal-generation environments."""

    mode: Literal["uniform_xyz", "height_only"] = "uniform_xyz"
    """Scale all axes for objects, or only the height axis for climb obstacles."""

    sampling_seed: int = 0
    """Seed for the dedicated size-sampling stream."""

    ground_clearance: float = 0.001
    """Minimum vertical clearance between scaled collision geometry and the ground."""

    ground_contact_tolerance: float = 0.02
    """Maximum canonical bottom height treated as a ground-contact reference pose."""

    def __post_init__(self) -> None:
        for field_name, scale_range in (
            ("track_scale_range", self.track_scale_range),
            ("gen_scale_range", self.gen_scale_range),
        ):
            if len(scale_range) != 2:
                raise ValueError(f"HybridSizeRandomizationConfig.{field_name} must contain [minimum, maximum].")
            lower, upper = (float(value) for value in scale_range)
            if lower <= 0.0 or upper <= 0.0:
                raise ValueError(f"HybridSizeRandomizationConfig.{field_name} values must be positive.")
            if lower > upper:
                raise ValueError(
                    f"HybridSizeRandomizationConfig.{field_name} minimum must be <= maximum, "
                    f"got {lower} > {upper}."
                )
        if self.ground_clearance < 0.0:
            raise ValueError("HybridSizeRandomizationConfig.ground_clearance must be non-negative.")
        if self.ground_contact_tolerance < 0.0:
            raise ValueError("HybridSizeRandomizationConfig.ground_contact_tolerance must be non-negative.")


@dataclass(frozen=True)
class HybridMotionConfig:
    """Hybrid WBT settings for mixing tracking and goal-generation environments."""

    track_ratio: float = 0.5
    """Fraction of environments that stay in tracking mode for the whole run."""

    env_partition_seed: int = 0
    """Seed used to create the fixed track/gen environment partition."""

    gen_start_at_timestep_zero_prob: float | None = 0.1
    """Override for starting gen environments at timestep zero.

    By default, gen environments use 0.1. If set to None explicitly, gen
    environments inherit MotionConfig.start_at_timestep_zero_prob. Track
    environments always use MotionConfig.start_at_timestep_zero_prob.
    """

    gen_step_zero_root_noise: StepZeroRootNoiseConfig | None = None
    """Optional step-zero root position/orientation noise override for gen environments."""

    robot_goal_noise: GoalPoseNoiseConfig = field(default_factory=GoalPoseNoiseConfig)
    """Noise applied to the terminal robot goal in gen environments."""

    object_goal_noise: GoalPoseNoiseConfig = field(default_factory=GoalPoseNoiseConfig)
    """Noise applied to the terminal object goal in gen environments."""

    size_randomization: HybridSizeRandomizationConfig = field(default_factory=HybridSizeRandomizationConfig)
    """Optional fixed-per-environment object or obstacle size randomization."""


@dataclass(frozen=True)
class MotionConfig:
    """Motion related configuration for Whole Body Tracking.

    NOTE:
    - Motion file is assumed to be in the format of:
      - joint_pos: (T, J)
      - joint_vel: (T, J)

      - body_pos_w: (T, B, 3)
      - body_quat_w: (T, B, 4) # wxyz -> xyzw
      - body_lin_vel_w: (T, B, 3)
      - body_ang_vel_w: (T, B, 3)

      If object is present in the motion file, it is assumed to be in the format of:
      - object_pos_w: (T, 3)
      - object_quat_w: (T, 4)
      - object_lin_vel_w: (T, 3)
      - object_ang_vel_w: (T, 3)

      Fixed scene geometry can also be embedded in the motion file using:
      - fixed_scene_object_names: (N,)
      - fixed_scene_object_geom_types: (N,)
      - fixed_scene_object_pos_w: (N, 3)
      - fixed_scene_object_quat_w: (N, 4)
      - fixed_scene_object_size: (N, 3)

      For object-tracking clips, the tracked object's simulator asset can be
      embedded alongside the object trajectory using:
      - object_usd_path: scalar string path to the tracked object's USD asset
      - object_urdf_path: scalar string path to the tracked object's URDF asset
      - object_mesh_path: scalar string path to a .usd/.urdf fallback asset
      - object_box_size: (3,) when object_geom_type is "box"

      If the motion clip assumes a terrain, the terrain has to be specified in holosoma/config/terrain/terrain_wbt.yaml
    """

    motion_file: str
    """Motion file (.npz) that contains motion_clips to track. """

    body_name_ref: list[str]
    """Body name of the reference frame (in general, torso_link). """
    body_names_to_track: list[str]
    """Key body names to track, used for reward/termination computation."""

    motion_dir: str = ""
    """Directory (or comma-separated directories) of .npz motion files.
    When non-empty, takes precedence over motion_file."""

    motion_dir_object_groups: bool = False
    """Treat each comma-separated entry of ``motion_dir`` as the clips of one object: entry i goes with
    ``robot.object.object_urdf_paths[i]``. Environment e tracks object variant ``e % N`` (the others stay parked) and
    only samples clips of that object."""

    load_fixed_scene_from_motion: bool = False
    """Whether to load scene/object geometry metadata from motion_file before simulator creation.

    This supports static obstacles embedded in robot-only clips and tracked object
    assets embedded in object-tracking clips.
    """

    # motion sampling related
    use_adaptive_timesteps_sampler: bool = False
    """During training, whether to prioritize training on motion segments where the robot fails often."""

    adaptive_timestep_sampler: AdaptiveTimestepSamplerConfig = field(default_factory=AdaptiveTimestepSamplerConfig)
    """Parameters for the adaptive reset sampler used when adaptive sampling is enabled."""

    start_at_timestep_zero_prob: float = 0.1
    """Probability of starting at timestep zero."""

    eval_sample_random_timestep: bool = False
    """During evaluation, sample a random motion timestep on reset instead of forcing the first frame."""

    eval_fixed_reset: EvalFixedResetConfig = field(default_factory=EvalFixedResetConfig)
    """Optional evaluation reset override for inspecting a specific motion bin or frame."""

    freeze_at_timestep_zero_prob: float = 0.0
    """When starting at timestep 0, probability of freezing motion counter at 0 (not advancing).
    This makes the robot practice holding the initial pose. Only applies when episode starts at timestep 0.
    Sampled independently each policy step; expected wait is roughly 1 / (1 - p) steps before unfreezing."""

    enable_default_pose_prepend: bool = False
    """If True, pre-append interpolated frames from default pose to the motion's first pose.
    This provides a smooth transition trajectory that the policy can track."""

    default_pose_prepend_duration_s: float = 2.0
    """Duration in seconds of the pre-appended interpolation phase.
    Only used if enable_default_pose_prepend is True."""

    enable_default_pose_append: bool = False
    """If True, post-append interpolated frames from the motion's last pose back to default pose.
    This provides a smooth return trajectory that the policy can track."""

    default_pose_append_duration_s: float = 2.0
    """Duration in seconds of the post-appended interpolation phase.
    Only used if enable_default_pose_append is True."""

    enable_terminal_pose_hold: bool = False
    """If True, append fixed hold frames to the end of each motion clip.
    The last pose is repeated while all velocity targets are set to zero."""

    terminal_pose_hold_duration_s: float = 1.0
    """Duration in seconds of the terminal pose hold.
    Only used if enable_terminal_pose_hold is True."""

    goal_orientation: GoalOrientationConfig = field(default_factory=GoalOrientationConfig)
    """Optional yaw-only handling for goal-conditioned observation/reward orientation terms."""

    hybrid: HybridMotionConfig | None = None
    """Optional hybrid WBT settings. When unset, the command behaves as standard WBT."""

    # noise related
    noise_to_initial_pose: NoiseToInitialPoseConfig = field(default_factory=NoiseToInitialPoseConfig)
