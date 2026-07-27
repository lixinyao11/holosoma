import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from holosoma.utils.viser_robot import _resolve_urdf_link_attachments


def test_fixed_child_head_link_attaches_to_torso_body():
    urdf_path = Path("src/holosoma/holosoma/data/robots/g1/main_mesh_collision_halfspherehand.urdf")
    root = ET.parse(urdf_path).getroot()

    name_to_idx = {
        "pelvis": 0,
        "torso_link": 1,
    }

    attachments = _resolve_urdf_link_attachments(root, name_to_idx)

    assert "head_link" in attachments
    body_idx, offset_pos, offset_quat = attachments["head_link"]
    assert body_idx == name_to_idx["torso_link"]
    assert np.linalg.norm(offset_pos) > 0.0
    assert np.allclose(np.linalg.norm(offset_quat), 1.0, atol=1e-5)


def test_non_fixed_untracked_link_is_not_attached():
    urdf_path = Path("src/holosoma/holosoma/data/robots/g1/main_mesh_collision_halfspherehand.urdf")
    root = ET.parse(urdf_path).getroot()

    name_to_idx = {
        "pelvis": 0,
    }

    attachments = _resolve_urdf_link_attachments(root, name_to_idx)

    assert "left_hip_pitch_link" not in attachments
