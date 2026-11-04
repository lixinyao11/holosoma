"""Only the new 043 configuration adapter is tested here; no simulator imports."""
from dataclasses import asdict
from copy import deepcopy
import pytest
from holosoma.config_values.wbt.g1.hdmi_goal import g1_29dof_wbt_hybrid_object_hdmi as base
from holosoma.config_values.wbt.g1.hdmi_goal043 import make_config


def test_043_changes_only_paths_and_run_name(tmp_path):
    (tmp_path/'motions').mkdir()
    motion=tmp_path/'motions/omomo__sub4_suitcase_043.npz'
    motion.touch();(tmp_path/'suitcase.urdf').touch()
    before=deepcopy(asdict(base))
    actual=asdict(make_config(tmp_path))
    expected=deepcopy(before)
    expected['training']['name']='g1_29dof_wbt_hdmi_goal043_upstream_hands'
    expected['command']['setup_terms']['motion_command']['params']['motion_config']['motion_file']=str(motion)
    expected['robot']['object']['object_urdf_path']=str(tmp_path/'suitcase.urdf')
    expected['robot']['object']['object_urdf_paths']=[]
    assert actual==expected
    assert asdict(base)==before


def test_043_missing_assets_fail_before_simulator(tmp_path):
    with pytest.raises(FileNotFoundError):make_config(tmp_path)
