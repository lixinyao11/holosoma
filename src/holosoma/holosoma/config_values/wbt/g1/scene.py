"""Whole Body Tracking scene preset for the G1 robot."""

from __future__ import annotations

from holosoma.config_types.scene import RigidObjectConfig, SceneConfig

# WBT object-tracking spawns a single free box, addressed by the actor name "object"
# (see managers/command/terms/wbt.py).
g1_29dof_wbt_object_scene = SceneConfig(
    env_spacing=0.0,
    rigid_objects={
        "object": RigidObjectConfig(
            urdf_file="holosoma/data/scene_objects/boxes/large_box.urdf",
        )
    },
)

# Several objects, one per environment group: environment e spawns urdf_files[e % N] (set on the CLI with
# --scene.rigid-objects.object.urdf-files ...). Heterogeneous assets need physics replication off.
g1_29dof_wbt_multi_object_scene = SceneConfig(
    env_spacing=0.0,
    replicate_physics=False,
    rigid_objects={
        "object": RigidObjectConfig(urdf_files=[]),
    },
)
