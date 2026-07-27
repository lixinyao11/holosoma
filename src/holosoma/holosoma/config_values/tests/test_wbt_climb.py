from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_values import command as command_defaults_module
from holosoma.config_values import experiment as experiment_defaults_module
from holosoma.config_values import robot as robot_defaults_module
from holosoma.config_values.wbt.g1.command import g1_29dof_wbt_command_climb, motion_config_climb
from holosoma.config_values.wbt.g1.experiment import g1_29dof_wbt, g1_29dof_wbt_climb


def test_wbt_climb_defaults_are_registered():
    assert command_defaults_module.DEFAULTS["g1_29dof_wbt_climb"] is g1_29dof_wbt_command_climb
    assert experiment_defaults_module.DEFAULTS["g1_29dof_wbt_climb"] is g1_29dof_wbt_climb

    aliases = {f"exp:{key.replace('_', '-')}": value for key, value in experiment_defaults_module.DEFAULTS.items()}
    assert aliases["exp:g1-29dof-wbt-climb"] is g1_29dof_wbt_climb


def test_wbt_climb_experiment_inherits_plain_wbt_stack():
    assert isinstance(g1_29dof_wbt_climb, ExperimentConfig)

    assert g1_29dof_wbt_climb.env_class == g1_29dof_wbt.env_class
    assert g1_29dof_wbt_climb.algo == g1_29dof_wbt.algo
    assert g1_29dof_wbt_climb.simulator == g1_29dof_wbt.simulator
    assert g1_29dof_wbt_climb.robot.asset.urdf_file == robot_defaults_module.g1_29dof_w_object.asset.urdf_file
    assert g1_29dof_wbt_climb.robot.object.object_urdf_path is None
    assert g1_29dof_wbt_climb.terrain == g1_29dof_wbt.terrain
    assert g1_29dof_wbt_climb.observation == g1_29dof_wbt.observation
    assert g1_29dof_wbt_climb.action == g1_29dof_wbt.action
    assert g1_29dof_wbt_climb.reward == g1_29dof_wbt.reward
    assert g1_29dof_wbt_climb.termination == g1_29dof_wbt.termination
    assert g1_29dof_wbt_climb.randomization == g1_29dof_wbt.randomization
    assert g1_29dof_wbt_climb.curriculum == g1_29dof_wbt.curriculum
    assert g1_29dof_wbt_climb.command is g1_29dof_wbt_command_climb
    assert g1_29dof_wbt_climb.training.name == "g1_29dof_wbt_manager_climb"


def test_wbt_climb_motion_config_points_to_reframed_clip_with_fixed_scene_loading():
    assert (
        motion_config_climb.motion_file
        == "holosoma/data/motions/g1_29dof/whole_body_tracking/climbox1/walk_climb1_reframed.npz"
    )
    assert motion_config_climb.motion_dir == ""
    assert motion_config_climb.load_fixed_scene_from_motion is True
    assert g1_29dof_wbt_command_climb.setup_terms["motion_command"].func.endswith(":MotionCommand")
