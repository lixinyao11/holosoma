"""Current physical interaction features; no motion-reference inputs."""
import torch
from holosoma.managers.observation.terms import wbt
from holosoma.utils.rotations import quat_rotate_inverse


def object_ang_vel_b(env):
    cmd = wbt._get_motion_command_and_assert_type(env)
    return quat_rotate_inverse(wbt._get_goal_conditioning_ref_quat_w(cmd), cmd.simulator_object_ang_vel_w, w_last=True)


def object_to_hands_b(env, body_names=("left_wrist_yaw_link", "right_wrist_yaw_link")):
    """Object-origin to wrist displacement, expressed in the current torso conditioning frame.

    Wrists are explicit: this asset does not have HDMI's palm landmarks. This
    feature is neither a contact sensor nor a grasp/release detector.
    """
    cmd = wbt._get_motion_command_and_assert_type(env)
    names = env.simulator._body_list
    indices = [names.index(name) for name in body_names]  # fail on missing landmarks
    delta = env.simulator._rigid_body_pos[:, indices] - cmd.simulator_object_pos_w[:, None]
    quat = wbt._get_goal_conditioning_ref_quat_w(cmd)[:, None].expand(-1, len(indices), -1)
    return quat_rotate_inverse(quat.reshape(-1, 4), delta.reshape(-1, 3), w_last=True).reshape(env.num_envs, -1)
