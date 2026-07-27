"""Convert a GMR motion export into a holosoma-format motion NPZ.

Supported inputs:

1. GMR retargeting pickle (`scripts/bvh_to_robot.py` / `smplx_to_robot.py`):
    fps: int
    root_pos:  (N, 3)
    root_rot:  (N, 4) in xyzw
    dof_pos:   (N, 29)  (G1 29-DOF joint order matches holosoma's)

2. Robot trajectory HDF5 from another source:
    robot_data/meta_data/quat_order
    robot_data/trajectory_data/time
    robot_data/trajectory_data/bodies/{labels,pos_w,quat_w}
    robot_data/trajectory_data/joints/{labels,pos}

The holosoma training NPZ (e.g. `sub3_largebox_003_mj.npz`) has:
    fps:             (1,)      int
    joint_pos:       (N, 36)   [base_xyz, base_quat_wxyz, dof_pos]
    joint_vel:       (N, 35)   [base_lin_vel, base_ang_vel, dof_vel]
    body_pos_w:      (N, 51, 3)
    body_quat_w:     (N, 51, 4) wxyz
    body_lin_vel_w:  (N, 51, 3)
    body_ang_vel_w:  (N, 51, 3)
    joint_names:     (29,)
    body_names:      (51,)
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import mujoco  # type: ignore[import-not-found]
import numpy as np

DEFAULT_ROBOT_XML = (
    Path(__file__).resolve().parents[1] / "models" / "g1" / "g1_29dof.xml"
)
H5_ROOT = "robot_data/trajectory_data"


def _xyzw_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    return q_xyzw[..., [3, 0, 1, 2]]


def _quat_conjugate_wxyz(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., 1:] = -out[..., 1:]
    return out


def _quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def _quat_to_rotvec_wxyz(q: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    q = q / np.linalg.norm(q, axis=-1, keepdims=True).clip(min=eps)
    q = np.where(q[..., :1] < 0.0, -q, q)
    w = np.clip(q[..., 0], -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    s = np.sqrt(np.clip(1.0 - w * w, 0.0, None))
    axis = np.where(s[..., None] > eps, q[..., 1:] / s[..., None], np.zeros_like(q[..., 1:]))
    return axis * angle[..., None]


def _slerp_wxyz(q0: np.ndarray, q1: np.ndarray, t: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    q0 = q0 / np.linalg.norm(q0, axis=-1, keepdims=True).clip(min=eps)
    q1 = q1 / np.linalg.norm(q1, axis=-1, keepdims=True).clip(min=eps)
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.clip(np.sum(q0 * q1, axis=-1, keepdims=True), -1.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    close = np.abs(sin_theta) < eps
    t_b = t[..., None]
    s0 = np.sin((1.0 - t_b) * theta) / (sin_theta + eps)
    s1 = np.sin(t_b * theta) / (sin_theta + eps)
    slerp = s0 * q0 + s1 * q1
    lerp = (1.0 - t_b) * q0 + t_b * q1
    out = np.where(close, lerp, slerp)
    return out / np.linalg.norm(out, axis=-1, keepdims=True).clip(min=eps)


def _resample(
    root_pos: np.ndarray,
    root_quat_wxyz: np.ndarray,
    dof_pos: np.ndarray,
    input_fps: int,
    output_fps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample a clip to `output_fps` via lerp (positions, DOF) and slerp (root quat)."""
    if input_fps == output_fps:
        return root_pos, root_quat_wxyz, dof_pos
    n_in = root_pos.shape[0]
    duration = (n_in - 1) / float(input_fps)
    output_dt = 1.0 / float(output_fps)
    times = np.arange(0.0, duration, output_dt)
    phase = times * float(input_fps)
    idx0 = np.minimum(np.floor(phase).astype(np.int64), n_in - 2)
    idx1 = idx0 + 1
    blend = (phase - idx0.astype(np.float64))[:, None]

    root_pos_out = root_pos[idx0] * (1.0 - blend) + root_pos[idx1] * blend
    dof_pos_out = dof_pos[idx0] * (1.0 - blend) + dof_pos[idx1] * blend
    root_quat_out = _slerp_wxyz(root_quat_wxyz[idx0], root_quat_wxyz[idx1], blend[:, 0])
    return root_pos_out, root_quat_out, dof_pos_out


def _angular_velocity_world(rotations_wxyz: np.ndarray, dt: float) -> np.ndarray:
    """Central-difference angular velocity in world frame for a sequence of wxyz quats."""
    q_prev = rotations_wxyz[:-2]
    q_next = rotations_wxyz[2:]
    q_rel = _quat_mul_wxyz(q_next, _quat_conjugate_wxyz(q_prev))
    rotvec = _quat_to_rotvec_wxyz(q_rel)
    omega = rotvec / (2.0 * dt)
    return np.concatenate([omega[:1], omega, omega[-1:]], axis=0)


def _world_body_velocities(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    v = np.zeros((model.nbody, 6))
    for b in range(model.nbody):
        mujoco.mj_objectVelocity(
            model,
            data,
            mujoco.mjtObj.mjOBJ_BODY,
            b,
            v[b],
            0,
        )
    return v[:, 3:6].copy(), v[:, 0:3].copy()


def _as_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray) and value.shape == ():
        return _as_text(value.item())
    return str(value)


def _quat_dataset_to_wxyz(quat: np.ndarray, quat_order_text: str) -> np.ndarray:
    normalized = quat_order_text.replace(" ", "").lower()
    if "[w,x,y,z]" in normalized:
        return quat
    if "[x,y,z,w]" in normalized:
        return _xyzw_to_wxyz(quat)
    raise ValueError(f"Unsupported quaternion order metadata: {quat_order_text!r}")


def _infer_input_fps_from_time(time_s: np.ndarray) -> int:
    if time_s.ndim != 1 or time_s.shape[0] < 2:
        raise ValueError(f"Expected time array with at least 2 frames, got shape {time_s.shape}.")
    dt = np.diff(time_s)
    if np.any(dt <= 0.0):
        raise ValueError("Time array must be strictly increasing.")
    return int(round(1.0 / float(np.mean(dt))))


def _model_joint_names(model: mujoco.MjModel) -> list[str]:
    model_joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
    ]
    return [name for name in model_joint_names if name is not None]


def _load_motion_from_pkl(input_pkl: Path) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    with open(input_pkl, "rb") as f:
        gmr = pickle.load(f)

    input_fps = int(gmr["fps"])
    root_pos_all = np.asarray(gmr["root_pos"], dtype=np.float64)
    root_quat_wxyz_all = _xyzw_to_wxyz(np.asarray(gmr["root_rot"], dtype=np.float64))
    dof_pos_all = np.asarray(gmr["dof_pos"], dtype=np.float64)
    return input_fps, root_pos_all, root_quat_wxyz_all, dof_pos_all


def _load_motion_from_h5(input_h5: Path, model: mujoco.MjModel) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    try:
        import h5py  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on env setup
        raise ImportError(
            "Reading .h5 inputs requires h5py. Use an environment with h5py installed "
            "or convert the HDF5 data to an intermediate format first."
        ) from exc

    with h5py.File(input_h5, "r") as f:
        quat_order_text = _as_text(f["robot_data/meta_data/quat_order"][0])
        time_s = np.asarray(f[f"{H5_ROOT}/time"], dtype=np.float64)
        body_labels = [_as_text(v) for v in f[f"{H5_ROOT}/bodies/labels"][()]]
        joint_labels = [_as_text(v) for v in f[f"{H5_ROOT}/joints/labels"][()]]

        if "base" in body_labels:
            base_body_name = "base"
        elif "pelvis" in body_labels:
            base_body_name = "pelvis"
        else:
            raise ValueError(
                f"Could not find a floating-base body in {input_h5}. Expected 'base' or 'pelvis' in body labels."
            )

        base_body_index = body_labels.index(base_body_name)
        root_pos_all = np.asarray(f[f"{H5_ROOT}/bodies/pos_w"][:, base_body_index, :], dtype=np.float64)
        root_quat_raw = np.asarray(f[f"{H5_ROOT}/bodies/quat_w"][:, base_body_index, :], dtype=np.float64)
        root_quat_wxyz_all = _quat_dataset_to_wxyz(root_quat_raw, quat_order_text)

        dof_pos_by_label = np.asarray(f[f"{H5_ROOT}/joints/pos"], dtype=np.float64)
        joint_index_by_name = {name: idx for idx, name in enumerate(joint_labels)}
        target_joint_names = _model_joint_names(model)
        missing_joint_names = [name for name in target_joint_names if name not in joint_index_by_name]
        if missing_joint_names:
            raise ValueError(
                f"HDF5 joint labels do not cover the MuJoCo model order. Missing joints: {missing_joint_names}"
            )
        reorder = np.array([joint_index_by_name[name] for name in target_joint_names], dtype=np.int64)
        dof_pos_all = dof_pos_by_label[:, reorder]

    input_fps = _infer_input_fps_from_time(time_s)
    return input_fps, root_pos_all, root_quat_wxyz_all, dof_pos_all


def _load_motion_source(
    input_path: Path,
    model: mujoco.MjModel,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    suffix = input_path.suffix.lower()
    if suffix == ".pkl":
        return _load_motion_from_pkl(input_path)
    if suffix in {".h5", ".hdf5"}:
        return _load_motion_from_h5(input_path, model)
    raise ValueError(f"Unsupported input format: {input_path.suffix}. Expected .pkl, .h5, or .hdf5.")


def convert(
    input_path: Path,
    output_npz: Path,
    robot_xml: Path,
    frame_start: int,
    frame_end: int | None,
    output_fps: int,
) -> None:
    model = mujoco.MjModel.from_xml_path(str(robot_xml))
    data = mujoco.MjData(model)
    input_fps, root_pos_all, root_quat_wxyz_all, dof_pos_all = _load_motion_source(input_path, model)

    total_frames = root_pos_all.shape[0]
    if frame_end is None:
        frame_end = total_frames - 1
    if not (0 <= frame_start <= frame_end <= total_frames - 1):
        raise ValueError(
            f"Invalid trim range [{frame_start}, {frame_end}] for a clip with {total_frames} frames."
        )

    sl = slice(frame_start, frame_end + 1)
    root_pos = root_pos_all[sl]
    root_quat_wxyz = root_quat_wxyz_all[sl]
    dof_pos = dof_pos_all[sl]

    if root_pos.shape[0] < 2:
        raise ValueError(f"Need at least 2 frames after trimming, got {root_pos.shape[0]}.")

    root_pos, root_quat_wxyz, dof_pos = _resample(
        root_pos, root_quat_wxyz, dof_pos, input_fps=input_fps, output_fps=output_fps
    )
    n_frames = root_pos.shape[0]
    if n_frames < 2:
        raise ValueError(f"Need at least 2 frames after resampling, got {n_frames}.")

    if model.nq != 7 + dof_pos.shape[1]:
        raise ValueError(
            f"Robot model nq={model.nq} does not match expected 7 + {dof_pos.shape[1]} DOFs."
        )

    model_joint_names = _model_joint_names(model)
    body_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)
    ]
    joint_names = model_joint_names
    assert len(joint_names) == dof_pos.shape[1], (
        f"Joint count mismatch: model has {len(joint_names)} hinge joints, "
        f"input has {dof_pos.shape[1]} DOFs."
    )

    dt = 1.0 / output_fps
    base_lin_vel = np.gradient(root_pos, dt, axis=0)
    base_ang_vel = _angular_velocity_world(root_quat_wxyz, dt)
    dof_vel = np.gradient(dof_pos, dt, axis=0)

    joint_pos = np.concatenate([root_pos, root_quat_wxyz, dof_pos], axis=1)
    joint_vel = np.concatenate([base_lin_vel, base_ang_vel, dof_vel], axis=1)

    body_pos_w = np.empty((n_frames, model.nbody, 3), dtype=np.float64)
    body_quat_w = np.empty((n_frames, model.nbody, 4), dtype=np.float64)
    body_lin_vel_w = np.empty((n_frames, model.nbody, 3), dtype=np.float64)
    body_ang_vel_w = np.empty((n_frames, model.nbody, 3), dtype=np.float64)

    for i in range(n_frames):
        data.qpos[:] = joint_pos[i]
        data.qvel[:] = joint_vel[i]
        mujoco.mj_forward(model, data)
        body_pos_w[i] = data.xpos
        body_quat_w[i] = data.xquat
        lin_w, ang_w = _world_body_velocities(model, data)
        body_lin_vel_w[i] = lin_w
        body_ang_vel_w[i] = ang_w

    out = {
        "fps": np.array([output_fps], dtype=np.int64),
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,
        "body_lin_vel_w": body_lin_vel_w,
        "body_ang_vel_w": body_ang_vel_w,
        "joint_names": np.array(joint_names),
        "body_names": np.array(body_names),
    }

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_npz, **out)
    print(
        f"Input : {input_path}  (frames {frame_start}..{frame_end} of {total_frames}, input_fps={input_fps})"
    )
    print(
        f"Output: {output_npz}  (nbody={model.nbody}, ndof={dof_pos.shape[1]}, "
        f"frames={n_frames}, output_fps={output_fps})"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        help="Input motion file (.pkl from GMR or .h5/.hdf5 with robot trajectory data).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output NPZ path. Defaults to the input path with suffix replaced by '.npz' next to it.",
    )
    parser.add_argument(
        "--robot-xml",
        default=str(DEFAULT_ROBOT_XML),
        help="MuJoCo XML for the G1 29-DOF model (must produce 51 bodies to match existing training data).",
    )
    parser.add_argument("--frame-start", type=int, default=0, help="First frame to keep (inclusive).")
    parser.add_argument(
        "--frame-end",
        type=int,
        default=None,
        help="Last frame to keep (inclusive). Defaults to the final frame.",
    )
    parser.add_argument(
        "--output-fps",
        type=int,
        default=50,
        help="Target output FPS. The clip is resampled via lerp (pos/DOF) + slerp (root quat) before "
        "velocities are derived. Must match the RL env policy rate (50 Hz here).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting an existing output file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    if args.output is None:
        output_npz = input_path.with_suffix(".npz")
    else:
        output_npz = Path(args.output).expanduser().resolve()

    if output_npz.exists() and not args.overwrite:
        raise FileExistsError(f"Output file already exists: {output_npz}. Use --overwrite to replace it.")

    convert(
        input_path=input_path,
        output_npz=output_npz,
        robot_xml=Path(args.robot_xml).expanduser().resolve(),
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        output_fps=args.output_fps,
    )


if __name__ == "__main__":
    main()
