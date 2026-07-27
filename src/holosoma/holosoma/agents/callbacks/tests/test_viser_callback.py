from types import SimpleNamespace

import numpy as np
import torch

import holosoma.agents.callbacks.viser_callback as viser_callback_module
from holosoma.agents.callbacks.viser_callback import EvalViserCallback
from holosoma.config_types.eval_callback import ViserConfig
from holosoma.utils.fixed_scene_motion import FixedSceneObjectSpec
from holosoma.utils.viser_scene import RigidMeshViser, StaticBoxViser


class _FakeViser:
    def __init__(self):
        self.meshes = {}
        self.transforms = {}

    def ok(self):
        return True

    def add_mesh_simple(self, name, vertices, faces, **kwargs):
        self.meshes[name] = (vertices, faces, kwargs)

    def add_line_segments(self, name, segments, **kwargs):
        self.meshes[name] = (segments, None, kwargs)

    def set_transform(self, name, position, wxyz):
        self.transforms[name] = (position, wxyz)


class _FakeGuiHandle:
    def __init__(self, *, disabled=False, value=None):
        self.disabled = disabled
        self.value = value
        self.callback = None

    def on_click(self, callback):
        self.callback = callback
        return callback


class _FakeGui:
    def __init__(self):
        self.buttons = {}
        self.texts = {}

    def add_button(self, label, **kwargs):
        handle = _FakeGuiHandle(disabled=kwargs.get("disabled", False))
        self.buttons[label] = handle
        return handle

    def add_text(self, label, initial_value, **_kwargs):
        handle = _FakeGuiHandle(value=initial_value)
        self.texts[label] = handle
        return handle


class _FakeRenderClient:
    def __init__(self, width=64, height=48):
        self.camera = SimpleNamespace(image_width=width, image_height=height)
        self.render_requests = []

    def get_render(self, *, height, width, transport_format):
        self.render_requests.append((width, height, transport_format))
        return np.full((height, width, 3), 127, dtype=np.uint8)


class _FakeRenderServer:
    def __init__(self):
        self.gui = _FakeGui()
        self.flush_count = 0
        self.initial_camera = SimpleNamespace(position=None, look_at=None, up=None, fov=None)
        self.clients = {}

    def flush(self):
        self.flush_count += 1

    def get_clients(self):
        return self.clients


class _FakeVideoWriter:
    instances = []

    def __init__(self, path, _fourcc, fps, resolution):
        self.path = path
        self.fps = fps
        self.resolution = resolution
        self.frames = []
        self.released = False
        self.instances.append(self)

    def isOpened(self):
        return True

    def write(self, frame):
        self.frames.append(frame.copy())

    def release(self):
        self.released = True
        if self.frames:
            with open(self.path, "wb") as video_file:
                video_file.write(b"fake mp4")


class _FakeVideoFileClip:
    instances = []

    def __init__(self, path, *, audio):
        self.path = path
        self.audio = audio
        self.write_kwargs = None
        self.closed = False
        self.instances.append(self)

    def write_videofile(self, output_path, **kwargs):
        self.write_kwargs = kwargs
        with open(output_path, "wb") as video_file:
            video_file.write(b"fake h264 mp4")

    def close(self):
        self.closed = True


def test_default_camera_is_root_relative_and_applied_once_to_connected_clients():
    training_loop = SimpleNamespace(device="cpu", log_dir="/tmp")
    callback = EvalViserCallback(
        ViserConfig(
            camera_position_offset=(4.0, -5.0, 2.5),
            camera_look_at_offset=(0.0, 0.5, -0.25),
            camera_fov_degrees=55.0,
        ),
        training_loop,
    )
    server = _FakeRenderServer()
    client_camera = SimpleNamespace(position=None, look_at=None, up=None, fov=None)
    server.clients[1] = SimpleNamespace(camera=client_camera)
    callback._viser = SimpleNamespace(server=server)

    callback._set_default_camera(
        np.array([2.0, 1.0, 1.25], dtype=np.float32),
        apply_to_clients=True,
    )

    expected_position = [6.0, -4.0, 3.75]
    expected_look_at = [2.0, 1.5, 1.0]
    assert np.allclose(server.initial_camera.position, expected_position)
    assert np.allclose(server.initial_camera.look_at, expected_look_at)
    assert np.allclose(server.initial_camera.up, [0.0, 0.0, 1.0])
    assert np.isclose(server.initial_camera.fov, np.deg2rad(55.0))
    assert np.allclose(client_camera.position, expected_position)
    assert np.allclose(client_camera.look_at, expected_look_at)
    assert np.allclose(client_camera.up, [0.0, 0.0, 1.0])
    assert np.isclose(client_camera.fov, np.deg2rad(55.0))


def test_reset_camera_button_restores_default_view_for_clicking_client():
    training_loop = SimpleNamespace(device="cpu", log_dir="/tmp")
    callback = EvalViserCallback(ViserConfig(), training_loop)
    server = _FakeRenderServer()
    client_camera = SimpleNamespace(position=None, look_at=None, up=None, fov=None)
    client = SimpleNamespace(camera=client_camera)
    callback._viser = SimpleNamespace(server=server)
    callback._set_default_camera(
        np.array([2.0, 1.0, 1.25], dtype=np.float32),
        apply_to_clients=False,
    )
    callback._setup_camera_controls()

    client_camera.position = np.zeros(3)
    client_camera.look_at = np.zeros(3)
    client_camera.up = np.array([0.0, 1.0, 0.0])
    client_camera.fov = 1.0
    reset_button = server.gui.buttons["Reset camera view"]
    assert reset_button.callback is not None
    reset_button.callback(SimpleNamespace(client=client))

    assert np.allclose(client_camera.position, server.initial_camera.position)
    assert np.allclose(client_camera.look_at, server.initial_camera.look_at)
    assert np.allclose(client_camera.up, server.initial_camera.up)
    assert np.isclose(client_camera.fov, server.initial_camera.fov)


def test_viser_recording_buttons_save_multiple_videos_to_eval_log(tmp_path, monkeypatch):
    _FakeVideoWriter.instances = []
    _FakeVideoFileClip.instances = []
    monkeypatch.setattr(viser_callback_module.cv2, "VideoWriter", _FakeVideoWriter)
    monkeypatch.setattr(viser_callback_module.cv2, "VideoWriter_fourcc", lambda *_args: 0)
    monkeypatch.setattr(viser_callback_module, "VideoFileClip", _FakeVideoFileClip)

    training_loop = SimpleNamespace(device="cpu", log_dir=str(tmp_path))
    callback = EvalViserCallback(
        ViserConfig(recording_width=64, recording_height=48),
        training_loop,
    )
    callback._sim_dt = 1.0 / 30.0
    server = _FakeRenderServer()
    callback._viser = SimpleNamespace(server=server)
    callback._setup_recording_controls()
    client = _FakeRenderClient()

    start_button = server.gui.buttons["Start recording"]
    stop_button = server.gui.buttons["Stop recording"]
    assert start_button.callback is not None
    assert stop_button.callback is not None

    start_button.callback(SimpleNamespace(client=client))
    callback._capture_recording_frame()
    callback._capture_recording_frame()
    first_path = stop_button.callback(SimpleNamespace(client=client))

    start_button.callback(SimpleNamespace(client=client))
    callback._capture_recording_frame()
    second_path = stop_button.callback(SimpleNamespace(client=client))

    assert first_path is None  # GUI callbacks intentionally do not return the saved path.
    assert second_path is None
    saved_paths = sorted(tmp_path.glob("viser_*.mp4"))
    assert len(saved_paths) == 2
    assert saved_paths[0].name != saved_paths[1].name
    assert server.flush_count == 3
    assert client.render_requests == [(64, 48, "jpeg")] * 3
    assert [len(writer.frames) for writer in _FakeVideoWriter.instances] == [2, 1]
    assert all(writer.released for writer in _FakeVideoWriter.instances)
    assert all(clip.write_kwargs["codec"] == "libx264" for clip in _FakeVideoFileClip.instances)
    assert all(clip.write_kwargs["pixel_format"] == "yuv420p" for clip in _FakeVideoFileClip.instances)
    assert all(clip.closed for clip in _FakeVideoFileClip.instances)
    assert list(tmp_path.glob("viser_*.raw.mp4")) == []
    assert server.gui.texts["Recording"].value.startswith("Saved viser_")


def test_hybrid_object_visibility_is_independent_of_track_gen_masks():
    motion_cmd = SimpleNamespace(
        motion=SimpleNamespace(has_object=True),
        track_env_mask=torch.tensor([True, True, False, False], dtype=torch.bool),
        gen_env_mask=torch.tensor([False, False, True, True], dtype=torch.bool),
    )

    for env_id in range(4):
        assert EvalViserCallback._should_show_object_goal(motion_cmd, env_id)
        assert not EvalViserCallback._should_show_torso_goal(motion_cmd, env_id)


def test_hybrid_robot_only_visibility_keeps_torso_goal_enabled():
    motion_cmd = SimpleNamespace(
        motion=SimpleNamespace(has_object=False),
        track_env_mask=torch.tensor([True, False], dtype=torch.bool),
        gen_env_mask=torch.tensor([False, True], dtype=torch.bool),
    )

    assert EvalViserCallback._should_show_torso_goal(motion_cmd, 0)
    assert EvalViserCallback._should_show_torso_goal(motion_cmd, 1)
    assert not EvalViserCallback._should_show_object_goal(motion_cmd, 0)
    assert not EvalViserCallback._should_show_object_goal(motion_cmd, 1)


def test_load_fixed_scene_box_specs_reads_box_metadata(tmp_path):
    motion_path = tmp_path / "climb_motion.npz"
    np.savez(
        motion_path,
        fixed_scene_object_names=np.array(["climb_box"], dtype="<U32"),
        fixed_scene_object_geom_types=np.array(["box"], dtype="<U16"),
        fixed_scene_object_pos_w=np.array([[2.7, 0.0, 0.25]], dtype=np.float32),
        fixed_scene_object_quat_w=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        fixed_scene_object_size=np.array([[0.8, 2.0, 0.5]], dtype=np.float32),
    )

    motion_cmd = SimpleNamespace(motion_cfg=SimpleNamespace(motion_file=str(motion_path)))

    specs = EvalViserCallback._load_fixed_scene_box_specs(motion_cmd)

    assert len(specs) == 1
    assert specs[0].name == "climb_box"
    assert np.allclose(specs[0].size, [0.8, 2.0, 0.5])


def test_load_motion_object_specs_reads_embedded_mesh_metadata(tmp_path):
    motion_path = tmp_path / "object_motion.npz"
    vertices = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    np.savez(
        motion_path,
        object_pos_w=np.array([[0.1, 0.2, 0.3]], dtype=np.float32),
        object_quat_w=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        object_geom_type=np.array("mesh"),
        object_usd_path=np.array("/tmp/object.usd"),
        object_mesh_vertices=vertices,
        object_mesh_faces=faces,
    )

    motion_cmd = SimpleNamespace(motion_cfg=SimpleNamespace(motion_file=str(motion_path)))

    specs = EvalViserCallback._load_motion_object_specs(motion_cmd)

    assert len(specs) == 1
    assert specs[0].name == "object"
    assert np.allclose(specs[0].mesh_vertices, vertices)
    assert np.array_equal(specs[0].mesh_faces, faces)


def test_create_motion_object_visualizer_uses_embedded_mesh():
    viser = _FakeViser()
    vertices = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    spec = FixedSceneObjectSpec(
        name="object",
        geom_type="usd",
        position_w=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        usd_path="/tmp/object.usd",
        mesh_vertices=vertices,
        mesh_faces=faces,
        is_motion_object=True,
    )

    viz = EvalViserCallback._create_motion_object_visualizer(viser, spec, prefix="/env_0/object_policy")

    assert isinstance(viz, RigidMeshViser)
    assert "/env_0/object_policy/mesh" in viser.meshes
    added_vertices, added_faces, _kwargs = viser.meshes["/env_0/object_policy/mesh"]
    assert np.allclose(added_vertices, vertices)
    assert np.array_equal(added_faces, faces)


def test_create_motion_object_visualizer_applies_goal_color_override():
    viser = _FakeViser()
    vertices = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    spec = FixedSceneObjectSpec(
        name="object",
        geom_type="usd",
        position_w=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        usd_path="/tmp/object.usd",
        mesh_vertices=vertices,
        mesh_faces=faces,
        is_motion_object=True,
    )

    EvalViserCallback._create_motion_object_visualizer(
        viser,
        spec,
        prefix="/env_0/object_target",
        color_override=EvalViserCallback._TARGET_OBJECT_COLOR,
    )

    assert "/env_0/object_target/mesh" in viser.meshes
    _vertices, _faces, kwargs = viser.meshes["/env_0/object_target/mesh"]
    assert kwargs["color"] == EvalViserCallback._TARGET_OBJECT_COLOR


def test_climb_box_uses_cool_gray_material_and_subtle_edge_accents():
    viser = _FakeViser()
    box = StaticBoxViser.from_fixed_scene(
        viser,
        object_name="climb_box",
        size_xyz=np.array([0.8, 2.0, 0.5], dtype=np.float32),
        prefix="/env_0/fixed_scene_policy/climb_box",
    )

    _vertices, _faces, material = viser.meshes["/env_0/fixed_scene_policy/climb_box/box"]
    edge_segments, _faces, edge_material = viser.meshes["/env_0/fixed_scene_policy/climb_box/box_edges"]
    assert material["color"] == StaticBoxViser._CLIMB_BOX_COLOR
    assert material["material"] == "standard"
    assert material["cast_shadow"] is True
    assert material["receive_shadow"] is True
    assert edge_segments.shape == (12, 2, 3)
    assert edge_material["color"] == StaticBoxViser._CLIMB_BOX_EDGE_COLOR
    assert edge_material["line_width"] == 1.25

    box.update(
        np.array([2.7, 0.0, 0.25], dtype=np.float32),
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    )
    assert "/env_0/fixed_scene_policy/climb_box/box" in viser.transforms
    assert "/env_0/fixed_scene_policy/climb_box/box_edges" in viser.transforms


def test_load_fixed_scene_box_specs_returns_empty_without_motion_file():
    motion_cmd = SimpleNamespace(motion_cfg=SimpleNamespace(motion_file=None))

    assert EvalViserCallback._load_fixed_scene_box_specs(motion_cmd) == []


def test_load_fixed_scene_box_specs_reads_serialized_motion_config_directory(tmp_path):
    motion_dir = tmp_path / "climb_dir"
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

    motion_cmd = SimpleNamespace(motion_cfg={"motion_file": "", "motion_dir": str(motion_dir)})

    specs = EvalViserCallback._load_fixed_scene_box_specs(motion_cmd)

    assert len(specs) == 1
    assert specs[0].name == "climb_box"
    assert np.allclose(specs[0].position_w, [2.7, 0.0, 0.25])


def test_get_fixed_scene_root_states_reads_rigid_object_world_states():
    root_state_w = torch.tensor(
        [
            [2.7, 0.0, 0.25, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [7.7, 0.0, 0.25, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    sim = SimpleNamespace(
        scene=SimpleNamespace(
            rigid_objects={
                "climb_box": SimpleNamespace(
                    data=SimpleNamespace(root_state_w=root_state_w),
                )
            }
        )
    )

    states = EvalViserCallback._get_fixed_scene_root_states(sim, ["climb_box"], [0, 1])

    assert len(states) == 1
    assert torch.allclose(states[0][:, :3], root_state_w[:, :3])
    assert torch.allclose(states[0][:, 3:7], torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]))
