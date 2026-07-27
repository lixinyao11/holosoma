from types import MethodType, SimpleNamespace

import torch

from holosoma.simulator.isaacsim.isaacsim import IsaacSim


def test_get_actor_initial_poses_prefers_rigid_object_cfg_init_state():
    sim = object.__new__(IsaacSim)
    sim.sim_device = "cpu"
    sim.scene = SimpleNamespace(
        num_envs=2,
        rigid_objects={
            "climb_box": SimpleNamespace(
                cfg=SimpleNamespace(
                    init_state=SimpleNamespace(
                        pos=(2.7, 0.0, 0.25),
                        rot=(1.0, 0.0, 0.0, 0.0),
                    )
                ),
                data=SimpleNamespace(default_root_state=torch.zeros(1, 13, dtype=torch.float32)),
            )
        },
    )
    sim._is_scene_object = MethodType(lambda self, _name: False, sim)

    env_ids = torch.tensor([0, 1], dtype=torch.long)

    poses = IsaacSim.get_actor_initial_poses(sim, ["climb_box"], env_ids)

    expected = torch.tensor(
        [
            [2.7, 0.0, 0.25, 0.0, 0.0, 0.0, 1.0],
            [2.7, 0.0, 0.25, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(poses, expected)


def test_get_actor_states_by_index_reads_from_unified_root_state_proxy():
    sim = object.__new__(IsaacSim)
    sim.all_root_states = torch.tensor(
        [
            [2.7, 0.0, 0.25, 0.0, 0.0, 0.0, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            [7.7, 0.0, 0.25, 0.0, 0.0, 0.0, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
        ],
        dtype=torch.float32,
    )

    states = IsaacSim.get_actor_states_by_index(sim, torch.tensor([1], dtype=torch.long))

    assert torch.allclose(states, sim.all_root_states[[1], :13])


def test_set_actor_states_by_index_writes_to_unified_root_state_proxy():
    sim = object.__new__(IsaacSim)
    sim.all_root_states = torch.zeros(2, 13, dtype=torch.float32)
    writes: list[str] = []
    sim.write_state_updates = MethodType(lambda self: writes.append("write"), sim)

    new_states = torch.tensor([[2.7, 0.0, 0.25, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)

    IsaacSim.set_actor_states_by_index(sim, torch.tensor([0], dtype=torch.long), new_states)

    assert torch.allclose(sim.all_root_states[[0], :13], new_states)
    assert writes == ["write"]
