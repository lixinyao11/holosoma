import pytest

from holosoma.config_types.command import (
    EvalFixedResetConfig,
    GoalOrientationConfig,
    MotionConfig,
)


def test_goal_orientation_defaults_to_full_orientation():
    goal_orientation = GoalOrientationConfig()

    assert goal_orientation.robot_yaw_only is False
    assert goal_orientation.object_yaw_only is False


def test_motion_config_embeds_goal_orientation_config():
    motion_cfg = MotionConfig(
        motion_file="holosoma/data/motions/dummy.npz",
        body_name_ref=["torso_link"],
        body_names_to_track=["torso_link"],
    )

    assert motion_cfg.goal_orientation.robot_yaw_only is False
    assert motion_cfg.goal_orientation.object_yaw_only is False
    assert motion_cfg.load_fixed_scene_from_motion is False
    assert motion_cfg.eval_sample_random_timestep is False
    assert motion_cfg.eval_fixed_reset.enabled is False


def test_eval_fixed_reset_config_rejects_ambiguous_frame_selection():
    with pytest.raises(ValueError):
        EvalFixedResetConfig(
            enabled=True,
            motion_id=0,
            bin_id=1,
            local_frame=10,
        )


def test_motion_config_can_enable_fixed_scene_loading():
    motion_cfg = MotionConfig(
        motion_file="holosoma/data/motions/dummy.npz",
        body_name_ref=["torso_link"],
        body_names_to_track=["torso_link"],
        load_fixed_scene_from_motion=True,
    )

    assert motion_cfg.load_fixed_scene_from_motion is True
