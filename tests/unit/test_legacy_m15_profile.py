"""CPU regression checks for the adopted legacy M15 profile."""
from dataclasses import asdict
from itertools import product

import pytest
import torch

from holosoma.config_values.wbt.g1.hdmi_goal import make_legacy_m15_config
from holosoma.utils.hdmi_actuator_dr import HDMISubstepFilter, grouped_bounds, log_uniform_scales


@pytest.mark.parametrize('collision,motor,delay', list(product((False, True), repeat=3)))
def test_switches_preserve_the_common_experiment(collision, motor, delay):
    baseline = asdict(make_legacy_m15_config(collision_exclusions=False, motor_dr=False, delay_filter=False))
    actual = asdict(make_legacy_m15_config(collision_exclusions=collision, motor_dr=motor, delay_filter=delay))
    for key in baseline.keys() - {'robot', 'action', 'randomization'}:
        assert actual[key] == baseline[key], key
    for key in baseline['robot'].keys() - {'asset'}:
        assert actual['robot'][key] == baseline['robot'][key], key
    for key in baseline['robot']['asset'].keys() - {'collision_exclusions'}:
        assert actual['robot']['asset'][key] == baseline['robot']['asset'][key], key
    assert len(actual['robot']['asset']['collision_exclusions']) == (4 if collision else 0)
    for phase, exceptions in [('setup_terms', {'actuator_randomizer_state', 'hdmi_armature'}),
                              ('reset_terms', {'actuator_randomizer_state'})]:
        old = baseline['randomization'][phase]
        new = actual['randomization'][phase]
        assert {k: v for k, v in new.items() if k not in exceptions} == {
            k: v for k, v in old.items() if k not in exceptions}
    assert ('hdmi_armature' in actual['randomization']['setup_terms']) is motor
    assert actual['randomization']['setup_terms']['actuator_randomizer_state']['params']['enable_pd_gain'] is motor
    assert ('HDMIJointPositionActionTerm' in actual['action']['terms']['joint_control']['func']) is delay
    # Building another combination cannot contaminate the explicit reference.
    assert baseline == asdict(make_legacy_m15_config(collision_exclusions=False, motor_dr=False, delay_filter=False))


def test_registry_uses_combined_profile_and_keeps_reference():
    from holosoma.config_values.experiment import DEFAULTS
    assert DEFAULTS['g1_29dof_wbt_hybrid_object_hdmi'] == make_legacy_m15_config()
    assert DEFAULTS['g1_29dof_wbt_hybrid_object_hdmi_m15_reference'] == make_legacy_m15_config(
        collision_exclusions=False, motor_dr=False, delay_filter=False)


def test_pd_groups_cover_all_29_joints_without_overlap():
    cfg = make_legacy_m15_config()
    params = cfg.randomization.setup_terms['actuator_randomizer_state'].params
    names = cfg.robot.dof_names
    bounds = grouped_bounds(names, params['kp_groups'], device='cpu')
    assert len(names) == 29 and bounds.shape == (29, 2)
    assert params['kp_groups'] == params['kd_groups']
    for name, bound in zip(names, bounds):
        expected = (.9, 1.1) if any(part in name for part in ('shoulder', 'elbow', 'wrist')) else (.8, 1.2)
        torch.testing.assert_close(bound, torch.tensor(expected))
    scales = log_uniform_scales(bounds, 512)
    assert torch.all(scales >= bounds[:, 0]) and torch.all(scales <= bounds[:, 1])
    assert torch.all(scales.std(dim=0) > 0)
    assert cfg.randomization.setup_terms['hdmi_armature'].params['scale_range'] == (.75, 1.25)


def test_substep_delay_and_lpf_have_expected_impulse_response():
    f = HDMISubstepFilter(2, 1, 4, 'cpu', min_delay=1, max_delay=3)
    f.reset(); f.delay[:, 0] = torch.tensor([1, 3]); f.alpha.fill_(1.)
    f.process_action(torch.ones(2, 1))
    observed = torch.stack([f.apply_action(i).clone()[:, 0] for i in range(4)])
    torch.testing.assert_close(observed, torch.tensor([[0., 0.], [1., 0.], [1., 0.], [1., 1.]]))
    f.reset(); f.delay.fill_(1); f.alpha.fill_(.8); f.process_action(torch.ones(2, 1))
    observed = torch.stack([f.apply_action(i).clone()[0, 0] for i in range(4)])
    torch.testing.assert_close(observed, torch.tensor([0., .8, .96, .992]))
    other = f.applied_action[1].clone(); f.reset(torch.tensor([0]))
    assert f.applied_action[0].item() == 0
    torch.testing.assert_close(f.applied_action[1], other)
