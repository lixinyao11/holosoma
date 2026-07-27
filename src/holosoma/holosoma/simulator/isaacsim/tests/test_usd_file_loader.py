import pytest

pytest.importorskip("pxr")

from holosoma.config_types.simulator import IsaacSimPhysicsConfig, PhysicsConfig, RigidObjectConfig
from holosoma.simulator.isaacsim import usd_file_loader as usd_file_loader_module


class _DummyRigidObject:
    def __init__(self, cfg):
        self.cfg = cfg


def test_usd_file_loader_supports_box_geom(monkeypatch):
    monkeypatch.setattr(usd_file_loader_module, "RigidObject", _DummyRigidObject)

    loader = usd_file_loader_module.USDFileLoader(sim_context=None, scene=None, device="cpu")
    rigid_obj = RigidObjectConfig(
        name="climb_box",
        geom_type="box",
        size=[0.8, 2.0, 0.4],
        position=[2.7, 0.0, 0.2],
        orientation=[1.0, 0.0, 0.0, 0.0],
        physics=PhysicsConfig(
            kinematic_enabled=True,
            isaacsim=IsaacSimPhysicsConfig(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
        ),
    )

    rigid_objects = loader.load_rigid_objects([rigid_obj], asset_root=None)

    climb_box = rigid_objects["climb_box"]
    cfg = climb_box.cfg
    assert cfg.prim_path == "/World/envs/env_.*/climb_box"
    assert tuple(cfg.spawn.size) == (0.8, 2.0, 0.4)
    assert cfg.spawn.rigid_props.kinematic_enabled is True
    assert cfg.spawn.collision_props.collision_enabled is True
    assert cfg.spawn.physics_material.static_friction == 1.0
    assert cfg.spawn.physics_material.dynamic_friction == 1.0
    assert cfg.spawn.physics_material.restitution == 0.0
    assert cfg.init_state.pos == (2.7, 0.0, 0.2)
    assert cfg.init_state.rot == (1.0, 0.0, 0.0, 0.0)


def test_usd_rigid_object_config_resolves_package_data_paths(monkeypatch, tmp_path):
    resolved_usd = tmp_path / "object.usd"
    resolved_usd.write_text("#usda 1.0\n")
    requested_paths = {}

    def fake_resolve_object_file_path(path, asset_root):
        requested_paths["path"] = path
        requested_paths["asset_root"] = asset_root
        return str(resolved_usd)

    def fake_create_rigid_object_cfg_with_physics(self, **kwargs):
        return kwargs

    monkeypatch.setattr(
        usd_file_loader_module.USDFileLoader,
        "_resolve_object_file_path",
        staticmethod(fake_resolve_object_file_path),
    )
    monkeypatch.setattr(
        usd_file_loader_module.USDFileLoader,
        "_create_rigid_object_cfg_with_physics",
        fake_create_rigid_object_cfg_with_physics,
    )

    loader = usd_file_loader_module.USDFileLoader(sim_context=None, scene=None, device="cpu")
    cfg = loader._create_rigid_object_config(
        "object",
        "holosoma/data/objects/grail/pickup_ground/object_usd/object.usd",
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        physics_config=None,
        assets_root=None,
    )

    assert requested_paths == {
        "path": "holosoma/data/objects/grail/pickup_ground/object_usd/object.usd",
        "asset_root": None,
    }
    assert cfg["usd_path"] == str(resolved_usd)
    assert cfg["source_path"] is None
    assert cfg["target_prim_path"] == "/World/envs/env_.*/object"
