"""Render articulated assets (MJCF XML or URDF) in Viser.

Ported from MimicKit's ``mimickit/util/robot_viser.py``.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .viser_helper import ViserHelper


class RobotAssetViser:
    """Render articulated assets (MJCF XML or URDF) in Viser."""

    def __init__(
        self,
        viser: ViserHelper,
        asset_path: str,
        body_names: Optional[List[str]],
        name_prefix: str = "/robot",
        color_override: Optional[Tuple[float, float, float]] = None,
    ):
        self._viser = viser
        self._asset_path = os.path.abspath(os.path.expanduser(asset_path))
        self._body_names = body_names
        self._name_prefix = name_prefix.rstrip("/")
        self._color_override = color_override
        self._geom_specs: Dict[int, List[Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = {}
        self._geom_default_colors: Dict[str, Tuple[float, float, float]] = {}
        self._geom_colors: Dict[str, Tuple[float, float, float]] = {}
        self._load_asset()

    # ------------------------------------------------------------------
    # Asset loading
    # ------------------------------------------------------------------

    def _load_asset(self):
        if not self._viser.ok():
            return
        if not os.path.exists(self._asset_path):
            print(f"[RobotViser] Asset not found: {self._asset_path}")
            return
        ext = Path(self._asset_path).suffix.lower()
        if ext == ".urdf":
            self._load_urdf_asset()
        else:
            self._load_mjcf_asset()

    def _load_mjcf_asset(self):
        try:
            import trimesh
        except Exception as exc:
            print(f"[RobotViser] trimesh unavailable: {exc}")
            return

        tree = ET.parse(self._asset_path)
        root = tree.getroot()

        bodies: List[ET.Element] = []

        def visit(node: ET.Element) -> None:
            if node.tag == "body":
                bodies.append(node)
            for child in list(node):
                visit(child)

        world = root.find("worldbody")
        visit(world if world is not None else root)

        name_to_idx = {n: i for i, n in enumerate(self._body_names)} if self._body_names else None

        # Build mesh name -> file path map from <asset><mesh> elements.
        asset_dir = Path(self._asset_path).parent
        mesh_assets: Dict[str, str] = {}
        asset_node = root.find("asset")
        if asset_node is not None:
            for mesh_el in asset_node.findall("mesh"):
                mname = mesh_el.attrib.get("name")
                mfile = mesh_el.attrib.get("file")
                if mname and mfile:
                    mesh_assets[mname] = str(asset_dir / mfile)

        for enum_idx, body in enumerate(bodies):
            body_name = body.attrib.get("name", f"body_{enum_idx}")
            if name_to_idx is not None:
                if body_name not in name_to_idx:
                    continue
                body_idx = name_to_idx[body_name]
            else:
                body_idx = enum_idx

            for geom in body.findall("geom"):
                mesh, local_pos, local_quat, color = self._build_mjcf_mesh(trimesh, geom, mesh_assets)
                if mesh is None:
                    continue
                entry_name = f"{self._name_prefix}/{body_idx}/{len(self._geom_specs.get(body_idx, []))}"
                final_color = self._color_override if self._color_override is not None else color
                vertices = np.array(mesh.vertices, dtype=np.float32)
                faces = np.array(mesh.faces, dtype=np.int32)
                self._viser.add_mesh_simple(
                    entry_name, vertices, faces, color=final_color, flat_shading=True, cast_shadow=True, receive_shadow=True
                )
                self._geom_specs.setdefault(body_idx, []).append((entry_name, np.array(local_pos, dtype=np.float32), np.array(local_quat, dtype=np.float32), vertices, faces))
                self._geom_default_colors[entry_name] = tuple(final_color)
                self._geom_colors[entry_name] = tuple(final_color)

    def _build_mjcf_mesh(self, trimesh_module, geom: ET.Element, mesh_assets: Dict[str, str] = {}):
        gtype = geom.attrib.get("type", "capsule")
        size = _parse_floats(geom.attrib.get("size"))
        fromto = _parse_floats(geom.attrib.get("fromto")) if geom.attrib.get("fromto") else []
        lpos = (
            np.array(_parse_floats(geom.attrib.get("pos")), dtype=np.float32)
            if geom.attrib.get("pos")
            else np.zeros(3, dtype=np.float32)
        )
        lquat = _parse_quat_wxyz(geom.attrib.get("quat"))

        rgba = _parse_floats(geom.attrib.get("rgba"))
        color = tuple(rgba[:3]) if rgba and len(rgba) >= 3 else (0.48, 0.68, 0.98)

        mesh = None
        try:
            if gtype == "sphere" and len(size) >= 1:
                mesh = trimesh_module.creation.icosphere(subdivisions=2, radius=float(size[0]))
            elif gtype == "box" and len(size) == 3:
                mesh = trimesh_module.creation.box(extents=2 * np.array(size, dtype=np.float32))
            elif gtype in {"capsule", "cylinder"}:
                mesh, lpos, lquat = _make_segment_mesh(trimesh_module, fromto, size, lpos, lquat, gtype)
            elif gtype == "mesh":
                mesh_name = geom.attrib.get("mesh")
                if mesh_name and mesh_name in mesh_assets:
                    mesh_path = mesh_assets[mesh_name]
                    if os.path.exists(mesh_path):
                        mesh = trimesh_module.load(mesh_path, force="mesh")
                        if mesh.is_empty:
                            mesh = None
        except Exception:
            mesh = None

        return mesh, lpos, lquat, color

    def _load_urdf_asset(self):
        try:
            import trimesh
        except Exception as exc:
            print(f"[RobotViser] trimesh unavailable: {exc}")
            return

        try:
            from scipy.spatial.transform import Rotation as sRot
        except Exception as exc:
            print(f"[RobotViser] scipy unavailable for URDF parsing: {exc}")
            return

        tree = ET.parse(self._asset_path)
        root = tree.getroot()

        material_colors: Dict[str, Tuple[float, float, float]] = {}
        for material in root.findall("material"):
            mat_name = material.attrib.get("name")
            if not mat_name:
                continue
            simple = _parse_rgba(material.attrib.get("rgba"))
            if simple is not None:
                material_colors[mat_name] = simple
                continue
            color_tag = material.find("color")
            if color_tag is not None:
                simple = _parse_rgba(color_tag.attrib.get("rgba"))
                if simple is not None:
                    material_colors[mat_name] = simple

        name_to_idx = {n: i for i, n in enumerate(self._body_names)} if self._body_names else None
        link_attach_specs = _resolve_urdf_link_attachments(root, name_to_idx)

        for link_idx, link in enumerate(root.findall(".//link")):
            body_name = link.attrib.get("name", f"link_{link_idx}")
            if name_to_idx is not None:
                attach_spec = link_attach_specs.get(body_name)
                if attach_spec is None:
                    continue
                body_idx, link_offset_pos, link_offset_quat = attach_spec
            else:
                body_idx = link_idx
                link_offset_pos = np.zeros(3, dtype=np.float32)
                link_offset_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

            visuals = link.findall("visual")
            for vis_idx, visual in enumerate(visuals):
                origin = visual.find("origin")
                xyz = np.zeros(3, dtype=np.float32)
                quat = np.array([0, 0, 0, 1], dtype=np.float32)
                if origin is not None:
                    xyz_attr = origin.attrib.get("xyz")
                    if xyz_attr:
                        xyz = np.array([float(v) for v in xyz_attr.split()], dtype=np.float32)
                    rpy_attr = origin.attrib.get("rpy")
                    if rpy_attr:
                        roll, pitch, yaw = [float(v) for v in rpy_attr.split()]
                        quat = sRot.from_euler("xyz", [roll, pitch, yaw]).as_quat().astype(np.float32)

                xyz = link_offset_pos + _quat_rotate_vec_xyzw(link_offset_quat, xyz)
                quat = _quat_mul_xyzw(link_offset_quat, quat)

                color = (0.72, 0.72, 0.75)
                mat = visual.find("material")
                if mat is not None:
                    color = _extract_material_color(mat, material_colors, default=color)

                geometry = visual.find("geometry")
                if geometry is None:
                    continue

                mesh = _build_urdf_geometry_mesh(geometry, Path(self._asset_path).parent, trimesh)
                if mesh is None:
                    continue

                entry_name = f"{self._name_prefix}/{body_idx}/urdf_{vis_idx}_{len(self._geom_specs.get(body_idx, []))}"
                final_color = self._color_override if self._color_override is not None else color
                vertices = np.array(mesh.vertices, dtype=np.float32)
                faces = np.array(mesh.faces, dtype=np.int32)
                self._viser.add_mesh_simple(
                    entry_name, vertices, faces, color=final_color, flat_shading=True, cast_shadow=True, receive_shadow=True
                )
                self._geom_specs.setdefault(body_idx, []).append(
                    (entry_name, xyz.astype(np.float32), np.array([quat[0], quat[1], quat[2], quat[3]], dtype=np.float32), vertices, faces)
                )
                self._geom_default_colors[entry_name] = tuple(final_color)
                self._geom_colors[entry_name] = tuple(final_color)

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------

    def update(self, body_pos_xy: np.ndarray, body_quat_xyzw: np.ndarray, world_offset: Optional[np.ndarray] = None):
        """Update all body transforms.

        Parameters
        ----------
        body_pos_xy : np.ndarray, shape [num_bodies, 3]
        body_quat_xyzw : np.ndarray, shape [num_bodies, 4] (xyzw convention)
        world_offset : optional np.ndarray, shape [3]
        """
        if not self._geom_specs or not self._viser.ok():
            return
        if world_offset is None:
            world_offset = np.zeros(3, dtype=np.float32)
        for body_idx, geoms in self._geom_specs.items():
            if body_idx >= body_pos_xy.shape[0]:
                continue
            p_body = body_pos_xy[body_idx].astype(np.float32)
            q_body = body_quat_xyzw[body_idx].astype(np.float32)
            for name, local_pos, local_quat, _, _ in geoms:
                q_world = _quat_mul_xyzw(q_body, local_quat)
                p_world = p_body + _quat_rotate_vec_xyzw(q_body, local_pos) + world_offset
                wxyz = np.array([q_world[3], q_world[0], q_world[1], q_world[2]], dtype=np.float32)
                self._viser.set_transform(name, p_world, wxyz)

    def set_color(self, color: Optional[Tuple[float, float, float]]):
        if not self._viser.ok():
            return
        if color is not None:
            color_tuple = tuple(float(c) for c in color)
        else:
            color_tuple = None

        for geoms in self._geom_specs.values():
            for name, _, _, vertices, faces in geoms:
                target = color_tuple or self._geom_default_colors.get(name)
                if target is None:
                    continue
                if self._geom_colors.get(name) == target:
                    continue
                self._viser.replace_mesh(name, vertices, faces, target)
                self._geom_colors[name] = target


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _parse_floats(text: Optional[str]) -> List[float]:
    return [float(x) for x in text.split()] if text else []


def _parse_quat_wxyz(text: Optional[str]) -> np.ndarray:
    """Parse MuJoCo wxyz quaternion string and return xyzw array."""
    if not text:
        return np.array([0, 0, 0, 1], dtype=np.float32)
    vals = _parse_floats(text)
    if len(vals) != 4:
        return np.array([0, 0, 0, 1], dtype=np.float32)
    # MuJoCo stores wxyz, convert to xyzw
    return np.array([vals[1], vals[2], vals[3], vals[0]], dtype=np.float32)


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


def _resolve_urdf_link_attachments(
    root: ET.Element,
    name_to_idx: Optional[Dict[str, int]],
) -> Dict[str, Tuple[int, np.ndarray, np.ndarray]]:
    """Map URDF links to the tracked body they should follow in Viser.

    If a link is a fixed child of a tracked rigid body, attach its visuals to the
    nearest tracked ancestor and fold the fixed-joint transform into the visual pose.
    """
    if name_to_idx is None:
        return {}

    joint_by_child: Dict[str, Tuple[str, np.ndarray, np.ndarray, str]] = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue

        origin = joint.find("origin")
        xyz = np.zeros(3, dtype=np.float32)
        quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        if origin is not None:
            xyz_attr = origin.attrib.get("xyz")
            if xyz_attr:
                xyz = np.array([float(v) for v in xyz_attr.split()], dtype=np.float32)
            rpy_attr = origin.attrib.get("rpy")
            if rpy_attr:
                roll, pitch, yaw = [float(v) for v in rpy_attr.split()]
                quat = _quat_from_rpy_xyz(roll, pitch, yaw)

        joint_by_child[child.attrib["link"]] = (
            parent.attrib["link"],
            xyz,
            quat,
            joint.attrib.get("type", ""),
        )

    attachments: Dict[str, Tuple[int, np.ndarray, np.ndarray]] = {}
    for link in root.findall(".//link"):
        link_name = link.attrib.get("name")
        if not link_name:
            continue
        attachment = _resolve_link_attachment(link_name, name_to_idx, joint_by_child)
        if attachment is not None:
            attachments[link_name] = attachment
    return attachments


def _resolve_link_attachment(
    link_name: str,
    name_to_idx: Dict[str, int],
    joint_by_child: Dict[str, Tuple[str, np.ndarray, np.ndarray, str]],
) -> Optional[Tuple[int, np.ndarray, np.ndarray]]:
    if link_name in name_to_idx:
        return (
            name_to_idx[link_name],
            np.zeros(3, dtype=np.float32),
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        )

    chain_pos = np.zeros(3, dtype=np.float32)
    chain_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    visited: set[str] = set()
    current = link_name

    while current in joint_by_child and current not in visited:
        visited.add(current)
        parent_name, joint_pos, joint_quat, joint_type = joint_by_child[current]
        if joint_type != "fixed":
            return None

        chain_pos = joint_pos + _quat_rotate_vec_xyzw(joint_quat, chain_pos)
        chain_quat = _quat_mul_xyzw(joint_quat, chain_quat)

        if parent_name in name_to_idx:
            return name_to_idx[parent_name], chain_pos, chain_quat

        current = parent_name

    return None


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


def _make_segment_mesh(trimesh_module, fromto, size, local_pos, local_quat, shape: str):
    if len(fromto) == 6 and len(size) >= 1:
        p1 = np.array(fromto[:3], dtype=np.float32)
        p2 = np.array(fromto[3:], dtype=np.float32)
        vec = p2 - p1
        length = float(np.linalg.norm(vec))
        if length < 1e-6:
            return None, local_pos, local_quat

        radius = float(size[0])
        if shape == "capsule":
            mesh = trimesh_module.creation.capsule(radius=radius, height=max(length, 1e-4))
        else:
            mesh = trimesh_module.creation.cylinder(radius=radius, height=max(length, 1e-4), sections=32)

        direction = vec / length
        axis = np.cross([0.0, 0.0, 1.0], direction)
        norm_axis = np.linalg.norm(axis)
        if norm_axis < 1e-8:
            if np.dot([0.0, 0.0, 1.0], direction) > 0:
                local_quat = np.array([0, 0, 0, 1], dtype=np.float32)
            else:
                local_quat = np.array([1, 0, 0, 0], dtype=np.float32)
        else:
            axis = axis / norm_axis
            angle = float(np.arccos(np.clip(np.dot([0.0, 0.0, 1.0], direction), -1.0, 1.0)))
            s = np.sin(angle / 2.0)
            local_quat = np.array([axis[0] * s, axis[1] * s, axis[2] * s, np.cos(angle / 2.0)], dtype=np.float32)
        local_pos = (p1 + p2) * 0.5
        return mesh, local_pos, local_quat

    if len(size) >= 2:
        radius, half = float(size[0]), float(size[1])
        length = 2.0 * half
        if shape == "capsule":
            mesh = trimesh_module.creation.capsule(radius=radius, height=max(length, 1e-4))
        else:
            mesh = trimesh_module.creation.cylinder(radius=radius, height=max(length, 1e-4), sections=32)
        return mesh, local_pos, local_quat

    return None, local_pos, local_quat


def _quat_from_rpy_xyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)

    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    w = cr * cp * cy + sr * sp * sy
    return np.array([x, y, z, w], dtype=np.float32)


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
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )
    return (R @ v.reshape(3, 1)).reshape(3)
