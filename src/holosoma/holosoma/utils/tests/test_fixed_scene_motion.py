from types import SimpleNamespace

import numpy as np
import pytest

from holosoma.utils.fixed_scene_motion import (
    load_fixed_scene_objects_from_motion_config,
    load_fixed_scene_objects_from_motion_npz,
    load_fixed_scene_objects_from_motion_source,
)


def test_load_fixed_scene_objects_from_motion_npz_parses_one_box(tmp_path):
    motion_path = tmp_path / "motion_with_box.npz"
    np.savez(
        motion_path,
        fixed_scene_object_names=np.array(["climb_box"], dtype="<U32"),
        fixed_scene_object_geom_types=np.array(["box"], dtype="<U16"),
        fixed_scene_object_pos_w=np.array([[2.7, 0.0, 0.2]], dtype=np.float32),
        fixed_scene_object_quat_w=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        fixed_scene_object_size=np.array([[0.8, 2.0, 0.4]], dtype=np.float32),
    )

    specs = load_fixed_scene_objects_from_motion_npz(motion_path)

    assert len(specs) == 1
    assert specs[0].name == "climb_box"
    assert specs[0].geom_type == "box"
    assert np.allclose(specs[0].position_w, [2.7, 0.0, 0.2])
    assert np.allclose(specs[0].quat_wxyz, [1.0, 0.0, 0.0, 0.0])
    assert np.allclose(specs[0].size, [0.8, 2.0, 0.4])
    assert specs[0].is_motion_object is False


def test_load_fixed_scene_objects_from_motion_npz_parses_motion_object_usd(tmp_path):
    motion_path = tmp_path / "motion_with_usd_object.npz"
    object_usd_path = tmp_path / "alcohol.usd"
    vertices = np.array(
        [
            [-0.1, -0.1, 0.0],
            [0.1, -0.1, 0.0],
            [0.0, 0.1, 0.0],
            [0.0, 0.0, 0.2],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int32)
    np.savez(
        motion_path,
        object_pos_w=np.array([[0.1, 0.2, 0.3], [0.2, 0.2, 0.3]], dtype=np.float32),
        object_quat_w=np.array([[1.0, 0.0, 0.0, 0.0], [0.99, 0.01, 0.0, 0.0]], dtype=np.float32),
        object_lin_vel_w=np.zeros((2, 3), dtype=np.float32),
        object_ang_vel_w=np.zeros((2, 3), dtype=np.float32),
        object_geom_type=np.array("mesh"),
        object_box_size=np.array([0.08, 0.08, 0.18], dtype=np.float32),
        object_mesh_vertices=vertices,
        object_mesh_faces=faces,
        object_usd_path=np.array(str(object_usd_path)),
    )

    specs = load_fixed_scene_objects_from_motion_npz(motion_path)

    assert len(specs) == 1
    assert specs[0].name == "object"
    assert specs[0].geom_type == "usd"
    assert specs[0].usd_path == str(object_usd_path)
    assert specs[0].urdf_path is None
    assert specs[0].mesh_path is None
    assert np.allclose(specs[0].size, [0.08, 0.08, 0.18])
    assert np.allclose(specs[0].mesh_vertices, vertices)
    assert np.array_equal(specs[0].mesh_faces, faces)
    assert specs[0].is_motion_object is True
    assert np.allclose(specs[0].position_w, [0.1, 0.2, 0.3])
    assert np.allclose(specs[0].quat_wxyz, [1.0, 0.0, 0.0, 0.0])


def test_load_fixed_scene_objects_from_motion_npz_returns_empty_when_absent(tmp_path):
    motion_path = tmp_path / "motion_without_box.npz"
    np.savez(motion_path, fps=np.array(50.0, dtype=np.float32))

    assert load_fixed_scene_objects_from_motion_npz(motion_path) == []


def test_load_fixed_scene_objects_from_motion_npz_rejects_shape_mismatch(tmp_path):
    motion_path = tmp_path / "bad_motion.npz"
    np.savez(
        motion_path,
        fixed_scene_object_names=np.array(["climb_box"], dtype="<U32"),
        fixed_scene_object_geom_types=np.array(["box"], dtype="<U16"),
        fixed_scene_object_pos_w=np.array([[2.7, 0.0, 0.2], [3.0, 0.0, 0.2]], dtype=np.float32),
        fixed_scene_object_quat_w=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        fixed_scene_object_size=np.array([[0.8, 2.0, 0.4]], dtype=np.float32),
    )

    with pytest.raises(ValueError):
        load_fixed_scene_objects_from_motion_npz(motion_path)


def test_load_fixed_scene_objects_from_motion_source_accepts_matching_motion_dir(tmp_path):
    motion_dir = tmp_path / "motions"
    motion_dir.mkdir()
    fixed_scene_kwargs = dict(
        fixed_scene_object_names=np.array(["climb_box"], dtype="<U32"),
        fixed_scene_object_geom_types=np.array(["box"], dtype="<U16"),
        fixed_scene_object_pos_w=np.array([[2.7, 0.0, 0.25]], dtype=np.float32),
        fixed_scene_object_quat_w=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        fixed_scene_object_size=np.array([[0.8, 2.0, 0.5]], dtype=np.float32),
    )
    np.savez(motion_dir / "clip_a.npz", **fixed_scene_kwargs)
    np.savez(motion_dir / "clip_b.npz", **fixed_scene_kwargs)

    specs = load_fixed_scene_objects_from_motion_source(motion_dir=motion_dir)

    assert len(specs) == 1
    assert specs[0].name == "climb_box"
    assert np.allclose(specs[0].position_w, [2.7, 0.0, 0.25])
    assert np.allclose(specs[0].size, [0.8, 2.0, 0.5])


def test_load_fixed_scene_objects_from_motion_source_rejects_mismatched_motion_dir(tmp_path):
    motion_dir = tmp_path / "motions"
    motion_dir.mkdir()
    np.savez(
        motion_dir / "clip_a.npz",
        fixed_scene_object_names=np.array(["climb_box"], dtype="<U32"),
        fixed_scene_object_geom_types=np.array(["box"], dtype="<U16"),
        fixed_scene_object_pos_w=np.array([[2.7, 0.0, 0.25]], dtype=np.float32),
        fixed_scene_object_quat_w=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        fixed_scene_object_size=np.array([[0.8, 2.0, 0.5]], dtype=np.float32),
    )
    np.savez(
        motion_dir / "clip_b.npz",
        fixed_scene_object_names=np.array(["climb_box"], dtype="<U32"),
        fixed_scene_object_geom_types=np.array(["box"], dtype="<U16"),
        fixed_scene_object_pos_w=np.array([[3.1, 0.0, 0.25]], dtype=np.float32),
        fixed_scene_object_quat_w=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        fixed_scene_object_size=np.array([[0.8, 2.0, 0.5]], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="Fixed scene metadata must match across all motion files"):
        load_fixed_scene_objects_from_motion_source(motion_dir=motion_dir)


def test_load_fixed_scene_objects_from_motion_source_rejects_mismatched_motion_object_assets(tmp_path):
    motion_dir = tmp_path / "motions"
    motion_dir.mkdir()
    common_kwargs = dict(
        object_pos_w=np.zeros((2, 3), dtype=np.float32),
        object_quat_w=np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (2, 1)),
        object_lin_vel_w=np.zeros((2, 3), dtype=np.float32),
        object_ang_vel_w=np.zeros((2, 3), dtype=np.float32),
        object_geom_type=np.array("mesh"),
    )
    np.savez(motion_dir / "clip_a.npz", **common_kwargs, object_usd_path=np.array("/tmp/object_a.usd"))
    np.savez(motion_dir / "clip_b.npz", **common_kwargs, object_usd_path=np.array("/tmp/object_b.usd"))

    with pytest.raises(ValueError, match="Fixed scene metadata must match across all motion files"):
        load_fixed_scene_objects_from_motion_source(motion_dir=motion_dir)


def test_load_fixed_scene_objects_from_motion_source_accepts_same_motion_object_asset_with_different_pose(tmp_path):
    motion_dir = tmp_path / "motions"
    motion_dir.mkdir()
    common_kwargs = dict(
        object_quat_w=np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (2, 1)),
        object_lin_vel_w=np.zeros((2, 3), dtype=np.float32),
        object_ang_vel_w=np.zeros((2, 3), dtype=np.float32),
        object_geom_type=np.array("mesh"),
        object_usd_path=np.array("/tmp/alcohol.usd"),
    )
    np.savez(
        motion_dir / "clip_a.npz",
        **common_kwargs,
        object_pos_w=np.array([[0.0, 0.0, 0.2], [0.1, 0.0, 0.2]], dtype=np.float32),
    )
    np.savez(
        motion_dir / "clip_b.npz",
        **common_kwargs,
        object_pos_w=np.array([[0.5, 0.0, 0.2], [0.6, 0.0, 0.2]], dtype=np.float32),
    )

    specs = load_fixed_scene_objects_from_motion_source(motion_dir=motion_dir)

    assert len(specs) == 1
    assert specs[0].name == "object"
    assert specs[0].usd_path == "/tmp/alcohol.usd"
