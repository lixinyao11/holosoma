from __future__ import annotations

from dataclasses import replace
from typing import Any

WBT_MANAGER_CLASS = "holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager"
WBT_ROBOT_INIT_POS = [0.0, 0.0, 0.76]


def apply_wbt_robot_defaults(config: Any) -> Any:
    """Reapply WBT robot-profile defaults after compositional robot overrides."""
    if getattr(config, "env_class", None) != WBT_MANAGER_CLASS:
        return config

    robot_cfg = getattr(config, "robot", None)
    if robot_cfg is None:
        return config

    updated_robot = replace(
        robot_cfg,
        asset=replace(robot_cfg.asset, enable_self_collisions=True),
        control=replace(robot_cfg.control, action_scales_by_effort_limit_over_p_gain=True),
        init_state=replace(robot_cfg.init_state, pos=list(WBT_ROBOT_INIT_POS)),
    )
    return replace(config, robot=updated_robot)
