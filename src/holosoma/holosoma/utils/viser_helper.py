"""Thin wrapper over ``viser`` with lights, materials, and mesh management.

Ported from MimicKit's ``mimickit/util/viser_visualizer.py``.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple, Union

import numpy as np

# Suppress noisy websocket debug logs from viser
logging.getLogger("websockets").setLevel(logging.WARNING)
logging.getLogger("viser").setLevel(logging.WARNING)


class ViserHelper:
    """Thin wrapper over ``viser`` with a handful of extras (lights, materials, etc.)."""

    _global_servers: Dict[int, "ViserHelper"] = {}
    _unit_sphere_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}

    def __init__(self, port: int = 8012):
        self.port = int(port)
        self._server = None
        self._ok = False
        self._handles: dict = {}
        self._init_server()

    @property
    def server(self):
        return self._server

    def ok(self) -> bool:
        return self._ok

    def _init_server(self):
        try:
            import viser
        except Exception as exc:
            print(f"[Viser] Not available ({exc})")
            return

        prev = ViserHelper._global_servers.get(self.port)
        if prev is not None and prev._server is not None:
            try:
                prev._server.stop()
            except Exception:
                pass
            ViserHelper._global_servers.pop(self.port, None)

        self._server = viser.ViserServer(port=self.port)
        ViserHelper._global_servers[self.port] = self
        self._ok = True

    # ------------------------------------------------------------------
    # Mesh operations
    # ------------------------------------------------------------------

    def add_mesh_simple(
        self,
        name: str,
        vertices: np.ndarray,
        faces: np.ndarray,
        color: Tuple[float, float, float] = (0.6, 0.7, 0.9),
        side: str = "double",
        material: str = "standard",
        flat_shading: bool = False,
        cast_shadow: bool = True,
        receive_shadow: Union[bool, float] = True,
    ):
        if not self._ok:
            return
        handle = self._server.scene.add_mesh_simple(
            name,
            vertices.astype(np.float32),
            faces.astype(np.int32),
            color=color,
            side=side,
            material=material,
            flat_shading=flat_shading,
            cast_shadow=cast_shadow,
            receive_shadow=receive_shadow,
        )
        self._handles[name] = handle

    def replace_mesh(
        self,
        name: str,
        vertices: np.ndarray,
        faces: np.ndarray,
        color: Tuple[float, float, float],
    ):
        if not self._ok:
            return
        handle = self._handles.pop(name, None)
        if handle is not None and hasattr(handle, "remove"):
            try:
                handle.remove()
            except Exception:
                pass
        self.add_mesh_simple(name, vertices, faces, color=color)

    def remove(self, name: str):
        if not self._ok:
            return
        handle = self._handles.pop(name, None)
        if handle is not None and hasattr(handle, "remove"):
            try:
                handle.remove()
            except Exception:
                pass

    def set_transform(self, name: str, position: np.ndarray, wxyz: np.ndarray):
        if not self._ok:
            return
        handle = self._handles.get(name)
        if handle is None:
            return
        handle.position = position.astype(np.float32)
        handle.wxyz = wxyz.astype(np.float32)

    # ------------------------------------------------------------------
    # Sphere helpers
    # ------------------------------------------------------------------

    @classmethod
    def _get_unit_sphere_mesh(
        cls,
        rings: int = 8,
        sectors: int = 16,
    ) -> Tuple[np.ndarray, np.ndarray]:
        key = (int(rings), int(sectors))
        cached = cls._unit_sphere_cache.get(key)
        if cached is not None:
            return cached

        vertices = []
        for ring in range(rings + 1):
            phi = np.pi * ring / rings
            sin_phi = np.sin(phi)
            cos_phi = np.cos(phi)
            for sector in range(sectors):
                theta = 2.0 * np.pi * sector / sectors
                vertices.append([sin_phi * np.cos(theta), sin_phi * np.sin(theta), cos_phi])

        faces = []
        for ring in range(rings):
            row0 = ring * sectors
            row1 = (ring + 1) * sectors
            for sector in range(sectors):
                next_sector = (sector + 1) % sectors
                a, b, c, d = row0 + sector, row1 + sector, row1 + next_sector, row0 + next_sector
                if ring > 0:
                    faces.append([a, b, d])
                if ring < rings - 1:
                    faces.append([d, b, c])

        verts_np = np.asarray(vertices, dtype=np.float32)
        faces_np = np.asarray(faces, dtype=np.int32)
        cls._unit_sphere_cache[key] = (verts_np, faces_np)
        return verts_np, faces_np

    def add_sphere(
        self,
        name: str,
        radius: float = 0.05,
        color: Tuple[float, float, float] = (0.9, 0.2, 0.2),
        position: Optional[np.ndarray] = None,
    ):
        if not self._ok:
            return
        if name not in self._handles:
            vertices, faces = self._get_unit_sphere_mesh()
            self.add_mesh_simple(
                name,
                vertices * float(radius),
                faces,
                color=color,
                flat_shading=False,
                cast_shadow=True,
                receive_shadow=True,
            )
        if position is None:
            position = np.zeros(3, dtype=np.float32)
        self.set_transform(name, np.asarray(position, dtype=np.float32), np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32))

    @staticmethod
    def _quat_rotate_vec_xyzw(quat_xyzw: np.ndarray, vec_xyz: np.ndarray) -> np.ndarray:
        quat = np.asarray(quat_xyzw, dtype=np.float32)
        vec = np.asarray(vec_xyz, dtype=np.float32)
        quat_xyz = quat[:3]
        quat_w = quat[3]
        twice_cross = 2.0 * np.cross(quat_xyz, vec)
        return vec + quat_w * twice_cross + np.cross(quat_xyz, twice_cross)

    @classmethod
    def _build_heading_arrow_segments(
        cls,
        position: np.ndarray,
        quat_xyzw: np.ndarray,
        shaft_length: float = 0.24,
        head_length: float = 0.08,
        head_width: float = 0.06,
        vertical_offset: float = 0.06,
    ) -> np.ndarray:
        origin = np.asarray(position, dtype=np.float32).reshape(3).copy()
        origin[2] += float(vertical_offset)

        forward_world = cls._quat_rotate_vec_xyzw(quat_xyzw, np.array([1.0, 0.0, 0.0], dtype=np.float32))
        heading_xy = forward_world[:2]
        heading_norm = float(np.linalg.norm(heading_xy))
        if heading_norm <= 1.0e-6:
            heading_xy = np.array([1.0, 0.0], dtype=np.float32)
        else:
            heading_xy = heading_xy / heading_norm

        direction = np.array([heading_xy[0], heading_xy[1], 0.0], dtype=np.float32)
        perpendicular = np.array([-heading_xy[1], heading_xy[0], 0.0], dtype=np.float32)

        tip = origin + float(shaft_length) * direction
        head_base = tip - float(head_length) * direction
        left = head_base + float(head_width) * perpendicular
        right = head_base - float(head_width) * perpendicular

        return np.asarray(
            [
                [origin, tip],
                [tip, left],
                [tip, right],
            ],
            dtype=np.float32,
        )

    def update_heading_arrow(
        self,
        name: str,
        position: np.ndarray,
        quat_xyzw: np.ndarray,
        color: Tuple[float, float, float] = (0.9, 0.2, 0.2),
        shaft_length: float = 0.24,
        head_length: float = 0.08,
        head_width: float = 0.06,
        vertical_offset: float = 0.06,
        line_width: float = 3.0,
    ):
        if not self._ok:
            return
        segments = self._build_heading_arrow_segments(
            position=position,
            quat_xyzw=quat_xyzw,
            shaft_length=shaft_length,
            head_length=head_length,
            head_width=head_width,
            vertical_offset=vertical_offset,
        )
        col = np.asarray(color, dtype=np.float32)
        colors = np.tile(col.reshape(1, 1, 3), (segments.shape[0], 2, 1))
        self.update_line_segments(name, segments, colors=colors, line_width=line_width)

    def update_spheres(
        self,
        name_prefix: str,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        radius: float = 0.05,
    ):
        if not self._ok:
            return
        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 3:
            return
        if colors is not None:
            colors = np.asarray(colors, dtype=np.float32)
        for idx, point in enumerate(pts):
            if colors is None:
                color = (0.9, 0.2, 0.2)
            else:
                c = colors[idx]
                color = (float(c[0]), float(c[1]), float(c[2]))
            self.add_sphere(f"{name_prefix}/{idx}", radius=radius, color=color, position=point)

    # ------------------------------------------------------------------
    # Point cloud and line segments
    # ------------------------------------------------------------------

    def update_point_cloud(
        self,
        name: str,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        point_size: float = 0.02,
    ):
        if not self._ok:
            return
        pts = points.astype(np.float32)
        handle = self._handles.get(name)
        if handle is None:
            color_arr = np.asarray(colors, dtype=np.float32) if colors is not None else None
            try:
                handle = self._server.scene.add_point_cloud(
                    name, points=pts, colors=color_arr, point_size=point_size, precision="float32"
                )
            except TypeError:
                handle = self._server.scene.add_point_cloud(name, points=pts, colors=color_arr, point_size=point_size)
            self._handles[name] = handle
        else:
            handle.points = pts
            if colors is not None:
                handle.colors = np.asarray(colors, dtype=np.float32)
            handle.point_size = point_size

    def update_line_segments(
        self,
        name: str,
        segments: np.ndarray,
        colors: Optional[np.ndarray] = None,
        line_width: float = 1.5,
    ):
        if not self._ok:
            return
        pts = segments.astype(np.float32)
        if name not in self._handles:
            handle = self._server.scene.add_line_segments(name, points=pts, colors=colors, line_width=line_width)
            self._handles[name] = handle
        else:
            handle = self._handles[name]
            handle.points = pts
            if colors is not None:
                handle.colors = colors
            handle.line_width = line_width

    def add_line_segments(
        self,
        name: str,
        segments: np.ndarray,
        color: Tuple[float, float, float] = (0.6, 0.6, 0.6),
        line_width: float = 1.0,
    ):
        if not self._ok:
            return
        pts = segments.astype(np.float32)
        col = np.array(color, dtype=np.float32)
        colors = np.tile(col.reshape(1, 1, 3), (pts.shape[0], 2, 1))
        handle = self._server.scene.add_line_segments(name, points=pts, colors=colors, line_width=line_width)
        self._handles[name] = handle

    # ------------------------------------------------------------------
    # Lighting
    # ------------------------------------------------------------------

    def add_ambient_light(
        self,
        name: str,
        color: Tuple[int, int, int] = (255, 255, 255),
        intensity: float = 0.3,
    ):
        if not self._ok:
            return
        self._server.scene.add_light_ambient(name, color=color, intensity=float(intensity))

    def add_directional_light(
        self,
        name: str,
        color: Tuple[int, int, int] = (255, 255, 255),
        intensity: float = 1.0,
        cast_shadow: bool = True,
        wxyz: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    ):
        if not self._ok:
            return
        self._server.scene.add_light_directional(
            name, color=color, intensity=float(intensity), cast_shadow=cast_shadow, wxyz=wxyz
        )

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------

    def set_camera(self, position: np.ndarray, lookat: np.ndarray):
        if not self._ok or self._server is None:
            return
        for _, client in self._server.get_clients().items():
            client.camera.position = position.astype(np.float32)
            client.camera.look_at = lookat.astype(np.float32)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def stop(self):
        if self._server is not None:
            try:
                self._server.stop()
            except Exception:
                pass
            ViserHelper._global_servers.pop(self.port, None)
            self._server = None
            self._ok = False
