"""Pinned HDMI mode-15 hardware; keep Holosoma's controller and action scaling."""
from dataclasses import replace
from pathlib import Path
import json


def with_mode15(robot):
    root = Path(__file__).resolve().parents[3] / 'data/robots/g1_mode15'
    spec = json.loads((root / 'motor_specs.json').read_text())
    motors = spec['joints']
    if set(motors) != set(robot.dof_names):
        raise ValueError('Mode-15 motor names do not match the policy joint contract')
    ordered = [motors[name] for name in robot.dof_names]
    kp, kd = {}, {}
    for key in robot.control.stiffness:
        matched = [motors[n] for n in robot.dof_names if key in n.replace('_joint', '')]
        if not matched or len({(m['stiffness'], m['damping']) for m in matched}) != 1:
            raise ValueError(f'Ambiguous mode-15 PD group: {key}')
        kp[key], kd[key] = matched[0]['stiffness'], matched[0]['damping']
    return replace(
        robot,
        dof_effort_limit_list=[m['effort_limit'] for m in ordered],
        dof_vel_limit_list=[m['velocity_limit'] for m in ordered],
        dof_armature_list=[m['armature'] for m in ordered],
        dof_pos_lower_limit_list=[m['lower'] for m in ordered],
        dof_pos_upper_limit_list=[m['upper'] for m in ordered],
        control=replace(robot.control, stiffness=kp, damping=kd),
        asset=replace(robot.asset, asset_root='@holosoma/data/robots/g1_mode15',
                      urdf_file='g1_mode15_holosoma.urdf', usd_file=None,
                      xml_file='source/g1-mode_13_15.xml'),
    )
