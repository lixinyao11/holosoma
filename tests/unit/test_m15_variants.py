"""CPU regression checks for the B/C configuration boundary and fixed dataset gate."""
from copy import deepcopy
from dataclasses import asdict
import json

import numpy as np
import pytest

from holosoma.config_values.wbt.g1.hdmi_goal import make_legacy_m15_config
from holosoma.config_values.wbt.g1.hdmi_goal043 import make_config
from holosoma.utils.m15_motion_sets import MULTI50_IDS, REFERENCE_NAME, audit_motion, sha256, validate_multi50


def test_extra_dr_exactly_four_changes_and_no_mutation():
    before = asdict(make_legacy_m15_config())
    actual = asdict(make_legacy_m15_config(extra_dr=True))
    expected = deepcopy(before)
    random = expected['randomization']
    random['setup_terms']['hdmi_body'] = deepcopy(actual['randomization']['setup_terms']['hdmi_body'])
    assert random['setup_terms']['hdmi_body']['params'] == {'mass_range': (0.9, 1.1), 'com_range': (-0.03, 0.03)}
    random['setup_terms']['push_randomizer_state']['params'].update(
        max_push_vel=[.2, .2, .1, .2, .2, .4], push_interval_s=[4., 6.])
    for stage in ('setup_terms', 'reset_terms'):
        term = deepcopy(actual['randomization'][stage]['hdmi_joint_friction'])
        assert term['params'] == {'effort_range_nm': (.05, .25)}
        random[stage]['hdmi_joint_friction'] = term
    for stage in ('setup_terms', 'reset_terms', 'step_terms'):
        term = expected['command'][stage]['motion_command']
        term['func'] = 'holosoma.managers.command.terms.hdmi_body:HDMIBodyMotionCommand'
        if stage == 'setup_terms':
            term['params']['joint_velocity_noise_range'] = (-.1, .1)
    assert actual == expected
    assert asdict(make_legacy_m15_config()) == before
    assert asdict(make_legacy_m15_config(extra_dr=False)) == before


@pytest.fixture
def multi50(tmp_path):
    (tmp_path / 'motions').mkdir()
    directory = tmp_path / 'motions_multi50'
    directory.mkdir()
    (tmp_path / 'suitcase.urdf').write_text('<robot name="fixture"/>')
    frames = 3
    robot = make_legacy_m15_config().robot
    bodies = len(robot.body_names)
    data = dict(joint_names=np.array(robot.dof_names), body_names=np.array(robot.body_names), fps=np.array([50]),
                joint_pos=np.zeros((frames, 36)), joint_vel=np.zeros((frames, 35)),
                body_pos_w=np.zeros((frames, bodies, 3)), body_quat_w=np.tile([0., 0, 0, 1], (frames, bodies, 1)),
                body_lin_vel_w=np.zeros((frames, bodies, 3)), body_ang_vel_w=np.zeros((frames, bodies, 3)),
                object_pos_w=np.zeros((frames, 3)), object_quat_w=np.tile([0., 0, 0, 1], (frames, 1)),
                object_lin_vel_w=np.zeros((frames, 3)), object_ang_vel_w=np.zeros((frames, 3)))
    reference = tmp_path / 'motions' / REFERENCE_NAME
    np.savez(reference, **data)
    motions = {}
    for i, name in enumerate(MULTI50_IDS):
        path = directory / f'omomo__{name}.npz'
        data['joint_pos'][:, -1] = i if path.name != REFERENCE_NAME else 0
        np.savez(path, **data)
        if path.name == REFERENCE_NAME:
            # Exact bytes, independent of ZIP timestamp metadata.
            path.write_bytes(reference.read_bytes())
        motions[path.name] = audit_motion(path, reference)
    manifest = dict(schema_version=1, motion_set='multi50', reference043_sha256=sha256(reference),
                    geometry={'suitcase.urdf': sha256(tmp_path/'suitcase.urdf')}, motions=motions)
    (tmp_path/'multi50_manifest.json').write_text(json.dumps(manifest))
    return tmp_path


@pytest.mark.parametrize('extra_dr', [False, True])
def test_motion_selection_only_changes_directory_and_name(multi50, extra_dr):
    a = asdict(make_config(multi50, extra_dr=extra_dr))
    c = asdict(make_config(multi50, extra_dr=extra_dr, motion_set='multi50'))
    c['training']['name'] = a['training']['name']
    c['command']['setup_terms']['motion_command']['params']['motion_config']['motion_dir'] = a['command']['setup_terms']['motion_command']['params']['motion_config']['motion_dir']
    assert c == a


@pytest.mark.parametrize('damage', ['missing', 'extra', 'changed', 'geometry', 'nan', 'joint_order'])
def test_multi50_rejects_partial_or_changed_data(multi50, damage):
    path = multi50/'motions_multi50'/f'omomo__{MULTI50_IDS[0]}.npz'
    if damage == 'missing':
        path.unlink()
    elif damage == 'extra':
        (path.parent/'extra.npz').write_bytes(path.read_bytes())
    elif damage == 'geometry':
        (multi50/'suitcase.urdf').write_text('changed')
    else:
        with np.load(path) as f:
            data = dict(f)
        if damage == 'joint_order':
            data['joint_names'] = np.array(['wrong'])
        else:
            data['joint_pos'][0, 0] = float('nan') if damage == 'nan' else 1
        np.savez(path, **data)
    with pytest.raises((ValueError, FileNotFoundError)):
        validate_multi50(multi50)


def test_invalid_motion_selector_fails(tmp_path):
    with pytest.raises(ValueError, match='motion_set'):
        make_config(tmp_path, motion_set='typo')


def test_native_multi_loader_loads_all_fifty(multi50):
    from holosoma.managers.command.terms.wbt import MultiMotionLoader
    directory = validate_multi50(multi50)
    robot = make_legacy_m15_config().robot
    loader = MultiMotionLoader(str(directory), robot.body_names, robot.dof_names, device='cpu')
    assert loader.num_motions == 50
    assert len(loader.motion_start_idx) == 50


def test_prepare_complete_cache_and_refuse_overwrite(multi50):
    import runpy
    import shutil
    from pathlib import Path
    helper = runpy.run_path(str(Path(__file__).parents[2] / "scripts/prepare_m15_multi50.py"))
    source = multi50.parent / (multi50.name + "_source")
    shutil.copytree(multi50, source)
    for p in (multi50 / "motions_multi50").glob("*.npz"):
        shutil.copy2(p, source / "motions" / p.name)
    destination = multi50.parent / (multi50.name + "_prepared")
    assert helper["prepare"](source, multi50, destination)["ready"]
    assert validate_multi50(destination) == destination / "motions_multi50"
    assert (destination / "motions" / REFERENCE_NAME).stat().st_ino != (multi50 / "motions" / REFERENCE_NAME).stat().st_ino
    with pytest.raises(FileExistsError):
        helper["prepare"](source, multi50, destination)


def test_prepare_incomplete_source_never_publishes(multi50):
    import runpy
    from pathlib import Path
    helper = runpy.run_path(str(Path(__file__).parents[2] / "scripts/prepare_m15_multi50.py"))
    destination = multi50.parent / (multi50.name + "_rejected")
    with pytest.raises(ValueError, match="missing_ids"):
        helper["prepare"](multi50, multi50, destination)
    assert not destination.exists()
    assert not destination.with_name(destination.name + ".preparing").exists()
