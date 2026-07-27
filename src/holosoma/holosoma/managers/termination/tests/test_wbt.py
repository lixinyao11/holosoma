from types import SimpleNamespace

import torch

from holosoma.config_types.termination import TerminationTermCfg
from holosoma.managers.termination.terms.wbt import BodyGroupProximity


def test_body_group_proximity_uses_default_left_right_foot_pair():
    simulator = SimpleNamespace(
        body_names=[
            "pelvis",
            "left_foot_contact_point",
            "right_foot_contact_point",
        ],
        _rigid_body_pos=torch.tensor(
            [
                [[0.0, 0.0, 0.8], [0.0, 0.0, 0.0], [0.04, 0.0, 0.0]],
                [[0.0, 0.0, 0.8], [0.0, 0.0, 0.0], [0.08, 0.0, 0.0]],
            ],
            dtype=torch.float32,
        ),
    )
    env = SimpleNamespace(device="cpu", num_envs=2, simulator=simulator)
    cfg = TerminationTermCfg(
        func="holosoma.managers.termination.terms.wbt:BodyGroupProximity",
        params={
            "min_distance": 0.05,
            "body_groups": [["left_foot_contact_point", "right_foot_contact_point"]],
        },
    )

    term = BodyGroupProximity(cfg, env)
    terminated = term(env)
    breakdown = term.get_last_breakdown()

    assert torch.equal(terminated, torch.tensor([True, False]))
    assert torch.equal(breakdown["group_0"], torch.tensor([True, False]))


def test_body_group_proximity_checks_all_pairs_within_each_group():
    simulator = SimpleNamespace(
        body_names=[
            "left_foot_contact_point",
            "right_foot_contact_point",
            "left_wrist_yaw_link",
            "right_wrist_yaw_link",
        ],
        _rigid_body_pos=torch.tensor(
            [
                [[0.00, 0.0, 0.0], [0.20, 0.0, 0.0], [0.03, 0.0, 0.0], [0.20, 0.2, 0.0]],
                [[0.00, 0.0, 0.0], [0.20, 0.0, 0.0], [0.09, 0.0, 0.0], [0.20, 0.2, 0.0]],
            ],
            dtype=torch.float32,
        ),
    )
    env = SimpleNamespace(device="cpu", num_envs=2, simulator=simulator)
    cfg = TerminationTermCfg(
        func="holosoma.managers.termination.terms.wbt:BodyGroupProximity",
        params={
            "min_distance": 0.05,
            "body_groups": [["left_foot_contact_point", "right_foot_contact_point", "left_wrist_yaw_link"]],
        },
    )

    term = BodyGroupProximity(cfg, env)
    terminated = term(env)
    breakdown = term.get_last_breakdown()

    assert torch.equal(terminated, torch.tensor([True, False]))
    assert torch.equal(breakdown["group_0"], torch.tensor([True, False]))
