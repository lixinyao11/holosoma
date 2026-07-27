"""Viser scene setup utilities: ground grid, lighting, static URDF loading.

Ported from MimicKit's ``mimickit/util/viser_scene.py`` and
``char_env._setup_viser_scene_assets``.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .viser_helper import ViserHelper


def setup_scene_lighting(viser: ViserHelper) -> None:
    """Set up three-point lighting matching MimicKit style."""
    if not viser.ok():
        return
    viser.add_ambient_light("/lights/ambient", color=(235, 240, 255), intensity=0.3)
    viser.add_directional_light(
        "/lights/key",
        color=(255, 255, 255),
        intensity=1.2,
        cast_shadow=True,
        wxyz=(0.9659258, -0.2588190, 0.0, 0.0),
    )
    viser.add_directional_light(
        "/lights/fill",
        color=(220, 230, 250),
        intensity=0.5,
        cast_shadow=False,
        wxyz=(0.9659258, 0.0, -0.2588190, 0.0),
    )


def add_ground_grid(
    viser: ViserHelper,
    width: float = 50.0,
    depth: float = 50.0,
    spacing: float = 1.0,
    height: float = 0.0,
    plane_color: Tuple[float, float, float] = (0.34, 0.34, 0.37),
    grid_color: Tuple[float, float, float] = (0.62, 0.62, 0.65),
) -> None:
    """Add a double-sided plane plus a simple grid overlay."""
    if not viser.ok():
        return

    half_w = float(width) * 0.5
    half_d = float(depth) * 0.5

    plane_verts = np.array(
        [
            [-half_w, -half_d, height],
            [half_w, -half_d, height],
            [half_w, half_d, height],
            [-half_w, half_d, height],
        ],
        dtype=np.float32,
    )
    plane_faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    viser.add_mesh_simple(
        "/ground/plane",
        plane_verts,
        plane_faces,
        color=plane_color,
        side="double",
        flat_shading=True,
        cast_shadow=False,
        receive_shadow=True,
    )

    num_x = int(np.floor(width / max(spacing, 1e-4))) + 1
    num_y = int(np.floor(depth / max(spacing, 1e-4))) + 1
    segments = []
    z = height + 1e-4
    for ix in range(num_x):
        x = -half_w + ix * spacing
        segments.append([[x, -half_d, z], [x, half_d, z]])
    for iy in range(num_y):
        y = -half_d + iy * spacing
        segments.append([[-half_w, y, z], [half_w, y, z]])

    viser.add_line_segments(
        "/ground/grid",
        np.array(segments, dtype=np.float32),
        color=grid_color,
        line_width=1.0,
    )


def load_static_urdf(
    viser: ViserHelper,
    urdf_path: str,
    prefix: str = "/scene",
    offset: Optional[np.ndarray] = None,
) -> None:
    """Load URDF visual geometry as static meshes (no articulation)."""
    if not viser.ok():
        return
    path = Path(os.path.expanduser(urdf_path))
    if not path.exists():
        print(f"[ViserScene] URDF not found: {path}")
        return

    try:
        import trimesh
    except Exception as exc:
        print(f"[ViserScene] trimesh unavailable, skipping URDF scene: {exc}")
        return

    try:
        from scipy.spatial.transform import Rotation as sRot
    except Exception as exc:
        print(f"[ViserScene] scipy unavailable, cannot parse URDF rotations: {exc}")
        return

    offset_vec = np.zeros(3, dtype=np.float32) if offset is None else np.asarray(offset, dtype=np.float32)

    tree = ET.parse(str(path))
    root = tree.getroot()

    material_colors: Dict[str, Tuple[float, float, float]] = {}
    for material in root.findall("material"):
        mat_name = material.attrib.get("name")
        if not mat_name:
            continue
        color = _parse_rgba(material.attrib.get("rgba"))
        if color is not None:
            material_colors[mat_name] = color
            continue
        color_tag = material.find("color")
        if color_tag is not None:
            color = _parse_rgba(color_tag.attrib.get("rgba"))
            if color is not None:
                material_colors[mat_name] = color

    visuals = root.findall(".//visual")
    for idx, visual in enumerate(visuals):
        origin = visual.find("origin")
        xyz = np.zeros(3, dtype=np.float32)
        quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        if origin is not None:
            xyz_attr = origin.attrib.get("xyz")
            if xyz_attr:
                xyz = np.array([float(v) for v in xyz_attr.split()], dtype=np.float32)
            rpy_attr = origin.attrib.get("rpy")
            if rpy_attr:
                roll, pitch, yaw = [float(v) for v in rpy_attr.split()]
                quat_scipy = sRot.from_euler("xyz", [roll, pitch, yaw]).as_quat()
                quat = np.array([quat_scipy[0], quat_scipy[1], quat_scipy[2], quat_scipy[3]], dtype=np.float32)
        xyz = xyz + offset_vec

        color = (0.72, 0.72, 0.75)
        mat = visual.find("material")
        if mat is not None:
            color = _extract_material_color(mat, material_colors, default=color)

        geometry = visual.find("geometry")
        if geometry is None:
            continue
        mesh = _build_urdf_geometry_mesh(geometry, path.parent, trimesh)
        if mesh is None:
            continue

        name = f"{prefix}/object_{idx}"
        viser.add_mesh_simple(
            name,
            mesh.vertices,
            mesh.faces,
            color=color,
            flat_shading=True,
            cast_shadow=True,
            receive_shadow=True,
        )
        viser.set_transform(name, xyz, _xyzw_to_wxyz(quat))


class StaticUrdfViser:
    """Render a rigid URDF asset in Viser and update it with a base pose."""

    _HIDDEN_POS = np.array([0.0, 0.0, -1000.0], dtype=np.float32)

    def __init__(
        self,
        viser: ViserHelper,
        urdf_path: str,
        prefix: str = "/object",
        color_override: Optional[Tuple[float, float, float]] = None,
    ):
        self._viser = viser
        self._urdf_path = str(Path(os.path.expanduser(urdf_path)))
        self._prefix = prefix
        self._color_override = color_override
        self._geom_specs: list[tuple[str, np.ndarray, np.ndarray]] = []
        self._load()

    def _load(self) -> None:
        if not self._viser.ok():
            return

        path = Path(self._urdf_path)
        if not path.exists():
            print(f"[ViserScene] URDF not found: {path}")
            return

        try:
            import trimesh
        except Exception as exc:
            print(f"[ViserScene] trimesh unavailable, skipping URDF scene: {exc}")
            return

        try:
            from scipy.spatial.transform import Rotation as sRot
        except Exception as exc:
            print(f"[ViserScene] scipy unavailable, cannot parse URDF rotations: {exc}")
            return

        tree = ET.parse(str(path))
        root = tree.getroot()

        material_colors: Dict[str, Tuple[float, float, float]] = {}
        for material in root.findall("material"):
            mat_name = material.attrib.get("name")
            if not mat_name:
                continue
            color = _parse_rgba(material.attrib.get("rgba"))
            if color is not None:
                material_colors[mat_name] = color
                continue
            color_tag = material.find("color")
            if color_tag is not None:
                color = _parse_rgba(color_tag.attrib.get("rgba"))
                if color is not None:
                    material_colors[mat_name] = color

        visuals = root.findall(".//visual")
        for idx, visual in enumerate(visuals):
            origin = visual.find("origin")
            xyz = np.zeros(3, dtype=np.float32)
            quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            if origin is not None:
                xyz_attr = origin.attrib.get("xyz")
                if xyz_attr:
                    xyz = np.array([float(v) for v in xyz_attr.split()], dtype=np.float32)
                rpy_attr = origin.attrib.get("rpy")
                if rpy_attr:
                    roll, pitch, yaw = [float(v) for v in rpy_attr.split()]
                    quat_scipy = sRot.from_euler("xyz", [roll, pitch, yaw]).as_quat()
                    quat = np.array(
                        [quat_scipy[0], quat_scipy[1], quat_scipy[2], quat_scipy[3]],
                        dtype=np.float32,
                    )

            color = (0.72, 0.72, 0.75)
            mat = visual.find("material")
            if mat is not None:
                color = _extract_material_color(mat, material_colors, default=color)
            if self._color_override is not None:
                color = self._color_override

            geometry = visual.find("geometry")
            if geometry is None:
                continue
            mesh = _build_urdf_geometry_mesh(geometry, path.parent, trimesh)
            if mesh is None:
                continue

            name = f"{self._prefix}/object_{idx}"
            self._viser.add_mesh_simple(
                name,
                mesh.vertices,
                mesh.faces,
                color=color,
                flat_shading=True,
                cast_shadow=True,
                receive_shadow=True,
            )
            self._geom_specs.append((name, xyz, quat))

    def update(
        self,
        base_pos_xyzw: np.ndarray,
        base_quat_xyzw: np.ndarray,
        world_offset: Optional[np.ndarray] = None,
    ) -> None:
        """Update the rigid object pose in world coordinates."""
        if not self._geom_specs or not self._viser.ok():
            return

        base_pos = np.asarray(base_pos_xyzw, dtype=np.float32)
        base_quat = np.asarray(base_quat_xyzw, dtype=np.float32)
        offset = np.zeros(3, dtype=np.float32) if world_offset is None else np.asarray(world_offset, dtype=np.float32)

        for name, local_pos, local_quat in self._geom_specs:
            world_quat = _quat_mul_xyzw(base_quat, local_quat)
            world_pos = base_pos + _quat_rotate_vec_xyzw(base_quat, local_pos) + offset
            self._viser.set_transform(name, world_pos, _xyzw_to_wxyz(world_quat))

    def hide(self) -> None:
        """Move the rigid object far below the scene so it is not visible."""
        self.update(self._HIDDEN_POS, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))


class StaticBoxViser:
    """Render a rigid box in Viser and update it with a base pose."""

    _HIDDEN_POS = np.array([0.0, 0.0, -1000.0], dtype=np.float32)
    _FIXED_SCENE_COLOR = (0.64, 0.50, 0.35)
    _CLIMB_BOX_COLOR = (0.62, 0.68, 0.74)
    _CLIMB_BOX_EDGE_COLOR = (0.42, 0.46, 0.52)

    def __init__(
        self,
        viser: ViserHelper,
        size_xyz: np.ndarray,
        prefix: str = "/object",
        color: Tuple[float, float, float] = (0.95, 0.55, 0.18),
        material: str = "standard",
        edge_color: Optional[Tuple[float, float, float]] = None,
        edge_width: float = 1.5,
    ):
        self._viser = viser
        self._name = f"{prefix.rstrip('/')}/box"
        self._edge_name = f"{prefix.rstrip('/')}/box_edges"
        self._size_xyz = np.asarray(size_xyz, dtype=np.float32).reshape(3)
        self._color = color
        self._material = material
        self._edge_color = edge_color
        self._edge_width = float(edge_width)
        self._load()

    @classmethod
    def from_fixed_scene(
        cls,
        viser: ViserHelper,
        *,
        object_name: str,
        size_xyz: np.ndarray,
        prefix: str,
        color_override: Optional[Tuple[float, float, float]] = None,
    ) -> "StaticBoxViser":
        if color_override is not None:
            return cls(
                viser,
                size_xyz,
                prefix=prefix,
                color=color_override,
            )
        if object_name == "climb_box":
            return cls(
                viser,
                size_xyz,
                prefix=prefix,
                color=cls._CLIMB_BOX_COLOR,
                material="standard",
                edge_color=cls._CLIMB_BOX_EDGE_COLOR,
                edge_width=1.25,
            )
        return cls(
            viser,
            size_xyz,
            prefix=prefix,
            color=cls._FIXED_SCENE_COLOR,
        )

    def _load(self) -> None:
        if not self._viser.ok():
            return

        half = 0.5 * self._size_xyz
        vertices = np.array(
            [
                [-half[0], -half[1], -half[2]],
                [half[0], -half[1], -half[2]],
                [half[0], half[1], -half[2]],
                [-half[0], half[1], -half[2]],
                [-half[0], -half[1], half[2]],
                [half[0], -half[1], half[2]],
                [half[0], half[1], half[2]],
                [-half[0], half[1], half[2]],
            ],
            dtype=np.float32,
        )
        faces = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],
                [4, 6, 5],
                [4, 7, 6],
                [0, 4, 5],
                [0, 5, 1],
                [1, 5, 6],
                [1, 6, 2],
                [2, 6, 7],
                [2, 7, 3],
                [3, 7, 4],
                [3, 4, 0],
            ],
            dtype=np.int32,
        )
        self._viser.add_mesh_simple(
            self._name,
            vertices,
            faces,
            color=self._color,
            material=self._material,
            flat_shading=True,
            cast_shadow=True,
            receive_shadow=True,
        )
        if self._edge_color is not None:
            edge_indices = np.array(
                [
                    [0, 1],
                    [1, 2],
                    [2, 3],
                    [3, 0],
                    [4, 5],
                    [5, 6],
                    [6, 7],
                    [7, 4],
                    [0, 4],
                    [1, 5],
                    [2, 6],
                    [3, 7],
                ],
                dtype=np.int32,
            )
            self._viser.add_line_segments(
                self._edge_name,
                vertices[edge_indices],
                color=self._edge_color,
                line_width=self._edge_width,
            )

    def update(
        self,
        base_pos_xyzw: np.ndarray,
        base_quat_xyzw: np.ndarray,
        world_offset: Optional[np.ndarray] = None,
    ) -> None:
        if not self._viser.ok():
            return
        base_pos = np.asarray(base_pos_xyzw, dtype=np.float32)
        offset = np.zeros(3, dtype=np.float32) if world_offset is None else np.asarray(world_offset, dtype=np.float32)
        base_quat = np.asarray(base_quat_xyzw, dtype=np.float32)
        position = base_pos + offset
        wxyz = _xyzw_to_wxyz(base_quat)
        self._viser.set_transform(self._name, position, wxyz)
        if self._edge_color is not None:
            self._viser.set_transform(self._edge_name, position, wxyz)

    def hide(self) -> None:
        self.update(self._HIDDEN_POS, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))


class RigidMeshViser:
    """Render a rigid mesh in Viser and update it with a base pose."""

    _HIDDEN_POS = np.array([0.0, 0.0, -1000.0], dtype=np.float32)
    _DEFAULT_COLOR = (0.92, 0.90, 0.82)

    def __init__(
        self,
        viser: ViserHelper,
        vertices: np.ndarray,
        faces: np.ndarray,
        prefix: str = "/object",
        color: Tuple[float, float, float] = _DEFAULT_COLOR,
    ):
        self._viser = viser
        self._name = f"{prefix.rstrip('/')}/mesh"
        self._vertices = np.asarray(vertices, dtype=np.float32)
        self._faces = np.asarray(faces, dtype=np.int32)
        self._color = color
        self._load()

    @classmethod
    def from_mesh_path(
        cls,
        viser: ViserHelper,
        mesh_path: str,
        prefix: str = "/object",
        color: Tuple[float, float, float] = _DEFAULT_COLOR,
    ) -> "RigidMeshViser":
        try:
            import trimesh
        except Exception as exc:
            raise RuntimeError(f"trimesh unavailable, cannot load object mesh '{mesh_path}': {exc}") from exc

        mesh = trimesh.load(str(Path(os.path.expanduser(mesh_path))), force="mesh")
        if hasattr(trimesh, "Scene") and isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        if getattr(mesh, "is_empty", True):
            raise ValueError(f"Object mesh is empty: {mesh_path}")
        return cls(
            viser,
            np.asarray(mesh.vertices, dtype=np.float32),
            np.asarray(mesh.faces, dtype=np.int32),
            prefix=prefix,
            color=color,
        )

    def _load(self) -> None:
        if not self._viser.ok():
            return
        self._viser.add_mesh_simple(
            self._name,
            self._vertices,
            self._faces,
            color=self._color,
            flat_shading=False,
            cast_shadow=True,
            receive_shadow=True,
        )

    def update(
        self,
        base_pos_xyzw: np.ndarray,
        base_quat_xyzw: np.ndarray,
        world_offset: Optional[np.ndarray] = None,
    ) -> None:
        if not self._viser.ok():
            return
        base_pos = np.asarray(base_pos_xyzw, dtype=np.float32)
        offset = np.zeros(3, dtype=np.float32) if world_offset is None else np.asarray(world_offset, dtype=np.float32)
        base_quat = np.asarray(base_quat_xyzw, dtype=np.float32)
        self._viser.set_transform(self._name, base_pos + offset, _xyzw_to_wxyz(base_quat))

    def hide(self) -> None:
        self.update(self._HIDDEN_POS, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))


# ---------------------------------------------------------------------------
# Helpers (shared with viser_robot.py but kept local to avoid circular deps)
# ---------------------------------------------------------------------------


def _parse_rgba(text: Optional[str]) -> Optional[Tuple[float, float, float]]:
    if not text:
        return None
    vals = [float(v) for v in text.split()]
    if len(vals) >= 3:
        return tuple(vals[:3])
    return None


def _extract_material_color(
    material_node: ET.Element,
    material_map: Dict[str, Tuple[float, float, float]],
    default: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    rgba = _parse_rgba(material_node.attrib.get("rgba"))
    if rgba is not None:
        return rgba
    color_tag = material_node.find("color")
    if color_tag is not None:
        rgba = _parse_rgba(color_tag.attrib.get("rgba"))
        if rgba is not None:
            return rgba
    mat_name = material_node.attrib.get("name")
    if mat_name and mat_name in material_map:
        return material_map[mat_name]
    return default


def _build_urdf_geometry_mesh(geometry: ET.Element, asset_dir: Path, trimesh_module):
    if geometry.find("box") is not None:
        size_attr = geometry.find("box").attrib.get("size", "")
        dims = np.array([float(v) for v in size_attr.split()], dtype=np.float32)
        return trimesh_module.creation.box(extents=dims)
    if geometry.find("cylinder") is not None:
        cyl = geometry.find("cylinder")
        radius = float(cyl.attrib.get("radius", 0.5))
        length = float(cyl.attrib.get("length", 0.5))
        return trimesh_module.creation.cylinder(radius=radius, height=length, sections=48)
    if geometry.find("sphere") is not None:
        radius = float(geometry.find("sphere").attrib.get("radius", 0.5))
        return trimesh_module.creation.icosphere(subdivisions=2, radius=radius)
    mesh_tag = geometry.find("mesh")
    if mesh_tag is not None:
        filename = mesh_tag.attrib.get("filename")
        if not filename:
            return None
        scale_attr = mesh_tag.attrib.get("scale")
        scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        if scale_attr:
            scale = np.array([float(v) for v in scale_attr.split()], dtype=np.float32)
        mesh_path = (asset_dir / filename).expanduser()
        if not mesh_path.exists():
            return None
        try:
            mesh = trimesh_module.load(str(mesh_path), force="mesh")
        except Exception:
            return None
        if mesh.is_empty:
            return None
        mesh.apply_scale(scale)
        return mesh
    return None


def _xyzw_to_wxyz(quat: np.ndarray) -> np.ndarray:
    return np.array([quat[3], quat[0], quat[1], quat[2]], dtype=np.float32)


def _quat_mul_xyzw(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return np.array([x, y, z, w], dtype=np.float32)


def _quat_rotate_vec_xyzw(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    rot = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )
    return (rot @ v.reshape(3, 1)).reshape(3)
