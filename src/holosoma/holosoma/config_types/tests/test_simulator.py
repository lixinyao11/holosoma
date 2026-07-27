import pytest

from holosoma.config_types.simulator import RigidObjectConfig


def test_rigid_object_config_accepts_primitive_box_source():
    cfg = RigidObjectConfig(
        name="climb_box",
        geom_type="box",
        size=[0.8, 2.0, 0.4],
        position=[2.7, 0.0, 0.2],
        orientation=[1.0, 0.0, 0.0, 0.0],
    )

    assert cfg.geom_type == "box"
    assert cfg.size == [0.8, 2.0, 0.4]


def test_rigid_object_config_rejects_multiple_sources():
    with pytest.raises(ValueError):
        RigidObjectConfig(
            name="climb_box",
            usd_path="box.usd",
            geom_type="box",
            size=[0.8, 2.0, 0.4],
        )


def test_rigid_object_config_rejects_box_without_size():
    with pytest.raises(ValueError):
        RigidObjectConfig(
            name="climb_box",
            geom_type="box",
        )


def test_rigid_object_config_accepts_matching_per_environment_transforms():
    cfg = RigidObjectConfig(
        name="climb_box",
        geom_type="box",
        size=[0.8, 2.0, 0.5],
        per_env_scales=[[1.0, 1.0, 0.9], [1.0, 1.0, 1.1]],
        per_env_positions=[[2.7, 0.0, 0.226], [2.7, 0.0, 0.276]],
        local_collision_bounds=[[-0.4, -1.0, -0.25], [0.4, 1.0, 0.25]],
    )

    assert len(cfg.per_env_scales) == 2
    assert len(cfg.per_env_positions) == 2


def test_rigid_object_config_rejects_invalid_per_environment_transforms():
    with pytest.raises(ValueError):
        RigidObjectConfig(
            name="climb_box",
            geom_type="box",
            size=[0.8, 2.0, 0.5],
            per_env_scales=[[1.0, 1.0, 0.0]],
        )

    with pytest.raises(ValueError):
        RigidObjectConfig(
            name="climb_box",
            geom_type="box",
            size=[0.8, 2.0, 0.5],
            per_env_scales=[[1.0, 1.0, 0.9]],
            per_env_positions=[[2.7, 0.0, 0.226], [2.7, 0.0, 0.276]],
        )
