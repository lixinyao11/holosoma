"""Fixed multi50 membership and fail-closed CPU validation, before Isaac starts.

The legacy C32 list fixes motion IDs, not retargeting or file-format equivalence.
Only a complete Holosoma export from the same source as the single043 reference
may be used. Missing exports must not be replaced by raw qpos or duplicate clips.
"""
from pathlib import Path
import hashlib
import json

import numpy as np

# Replaced at implementation time with the previously frozen 50-ID list.
MULTI50_IDS = (
    'sub14_suitcase_005',
    'sub14_suitcase_007',
    'sub14_suitcase_008',
    'sub14_suitcase_010',
    'sub17_suitcase_005',
    'sub17_suitcase_006',
    'sub17_suitcase_007',
    'sub17_suitcase_009',
    'sub17_suitcase_010',
    'sub1_suitcase_010',
    'sub1_suitcase_011',
    'sub1_suitcase_012',
    'sub1_suitcase_013',
    'sub1_suitcase_014',
    'sub1_suitcase_015',
    'sub1_suitcase_017',
    'sub1_suitcase_018',
    'sub1_suitcase_019',
    'sub1_suitcase_070',
    'sub1_suitcase_071',
    'sub1_suitcase_072',
    'sub1_suitcase_073',
    'sub1_suitcase_075',
    'sub1_suitcase_076',
    'sub1_suitcase_077',
    'sub1_suitcase_079',
    'sub4_suitcase_006',
    'sub4_suitcase_007',
    'sub4_suitcase_008',
    'sub4_suitcase_009',
    'sub4_suitcase_010',
    'sub4_suitcase_042',
    'sub4_suitcase_043',
    'sub4_suitcase_044',
    'sub4_suitcase_047',
    'sub5_suitcase_008',
    'sub5_suitcase_009',
    'sub5_suitcase_010',
    'sub5_suitcase_016',
    'sub5_suitcase_051',
    'sub8_suitcase_007',
    'sub8_suitcase_008',
    'sub8_suitcase_009',
    'sub8_suitcase_010',
    'sub8_suitcase_011',
    'sub8_suitcase_012',
    'sub8_suitcase_013',
    'sub8_suitcase_042',
    'sub8_suitcase_043',
    'sub8_suitcase_044',
)
HELD_OUT_IDS = ("sub14_suitcase_006", "sub1_suitcase_016", "sub17_suitcase_008", "sub8_suitcase_006")
SELECTION_MANIFEST_SHA256 = "14c2bd30db437a0ba40adb4afdd2ec6e136eff0c2dfda7b77e78c14f8f484dc3"
REFERENCE_NAME = "omomo__sub4_suitcase_043.npz"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def audit_motion(path, reference):
    """Validate every tensor used by the existing multi-motion loader."""
    with np.load(path, allow_pickle=False) as data, np.load(reference, allow_pickle=False) as ref:
        for key in ("joint_names", "body_names"):
            if not np.array_equal(data[key], ref[key]):
                raise ValueError(f"{path}: {key} differs from single043")
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        if fps != 50.0 or fps != float(np.asarray(ref["fps"]).reshape(-1)[0]):
            raise ValueError(f"{path}: expected the same 50 Hz export as single043")
        frames = len(data["joint_pos"])
        if frames < 2:
            raise ValueError(f"{path}: fewer than two frames")
        for key in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w",
                    "body_ang_vel_w", "object_pos_w", "object_quat_w", "object_lin_vel_w", "object_ang_vel_w"):
            a = data[key]
            if a.shape != (frames, *ref[key].shape[1:]) or not np.isfinite(a).all():
                raise ValueError(f"{path}: invalid shape or nonfinite {key}")
        for key in ("body_quat_w", "object_quat_w"):
            if not np.allclose(np.linalg.norm(data[key], axis=-1), 1.0, atol=1e-3):
                raise ValueError(f"{path}: non-unit {key}")
        return {"frames": frames, "fps": fps, "sha256": sha256(path)}


def validate_multi50(root, *, robot_joint_names=None, robot_body_names=None):
    """Reject a missing/changed/partial dataset instead of letting the loader skip it."""
    root = Path(root).expanduser().resolve()
    directory = root / "motions_multi50"
    expected = {f"omomo__{name}.npz" for name in MULTI50_IDS}
    actual = {p.name for p in directory.glob("*.npz")}
    if len(MULTI50_IDS) != 50 or len(expected) != 50 or set(MULTI50_IDS) & set(HELD_OUT_IDS):
        raise ValueError("Invalid frozen multi50 membership")
    if actual != expected:
        raise ValueError(f"multi50 requires exactly 50 fixed files: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
    manifest = json.loads((root / "multi50_manifest.json").read_text())
    if manifest.get("schema_version") != 1 or manifest.get("motion_set") != "multi50":
        raise ValueError("Invalid multi50 manifest schema")
    if set(manifest.get("motions", {})) != expected:
        raise ValueError("Manifest does not contain the fixed 50 motions")
    reference = root / "motions" / REFERENCE_NAME
    with np.load(reference, allow_pickle=False) as data:
        for key, required in (("joint_names", robot_joint_names), ("body_names", robot_body_names)):
            names = data[key].tolist()
            if len(names) != len(set(names)) or (required is not None and not set(required) <= set(names)):
                raise ValueError(f"Reference {key} do not cover the configured robot")
    if sha256(reference) != manifest["reference043_sha256"] or sha256(directory / REFERENCE_NAME) != sha256(reference):
        raise ValueError("multi50's 043 must exactly match the single043 reference")
    if manifest.get("geometry", {}).get("suitcase.urdf") is None:
        raise ValueError("Missing shared suitcase geometry manifest")
    for rel, expected_sha in manifest["geometry"].items():
        path = (root / rel).resolve()
        if not path.is_relative_to(root) or sha256(path) != expected_sha:
            raise ValueError(f"Changed/invalid shared geometry: {rel}")
    for name in sorted(expected):
        evidence = audit_motion(directory / name, reference)
        if evidence != manifest["motions"][name]:
            raise ValueError(f"Motion differs from prepared manifest: {name}")
    return directory


# User-reviewed UMR 40 train / 2 eval / 1 reserve selection. Absolute source
# paths in the JSON are provenance only; deployment always resolves below root.
MULTI40_SELECTION_PATH = Path(__file__).with_name("m15_multi40_selection.json")
MULTI40_SELECTION_SHA256 = "7807207556ea92fa7eeca891c961c1fceb1338ccd65846eae2a451d57db2b22c"


def load_multi40_selection():
    if sha256(MULTI40_SELECTION_PATH) != MULTI40_SELECTION_SHA256:
        raise ValueError("Changed frozen multi40 selection")
    selection = json.loads(MULTI40_SELECTION_PATH.read_text())
    seen = set()
    for split, count in (("train", 40), ("eval", 2), ("reserve", 1), ("exclude_for_now", 17)):
        records = selection[split]
        ids = {x["sequence"] for x in records}
        if len(records) != count or len(ids) != count or seen & ids:
            raise ValueError(f"Invalid/disjoint multi40 {split} membership")
        for item in records:
            if item["motion"] != f"omomo__{item['sequence']}.npz" or Path(item["motion"]).name != item["motion"]:
                raise ValueError("Invalid selected motion filename")
        seen.update(ids)
    return selection


def motion_geometry(root):
    root = Path(root).resolve()
    paths = [root / "suitcase.urdf", *sorted((root / "meshes").rglob("*"))]
    result = {}
    for path in paths:
        if path.is_file():
            if not path.resolve().is_relative_to(root):
                raise ValueError(f"Geometry escapes cache: {path}")
            result[str(path.relative_to(root))] = sha256(path)
    return result


def validate_multi40(root, *, split="train", robot_joint_names=None, robot_body_names=None):
    """Validate both exact, disjoint directories before returning one split.

    The native loader skips malformed files; reject them here before Isaac starts.
    The reviewed manifest is pinned in source, not inferred from directory names.
    """
    if split not in ("train", "eval"):
        raise ValueError(f"Unknown multi40 split: {split}")
    root = Path(root).expanduser().resolve()
    selection = load_multi40_selection()
    if sha256(root / "multi40_selection.json") != MULTI40_SELECTION_SHA256:
        raise ValueError("Cache selection differs from frozen multi40 list")
    manifest = json.loads((root / "multi40_manifest.json").read_text())
    if (manifest.get("schema_version") != 1 or manifest.get("motion_set") != "multi40"
            or manifest.get("selection_sha256") != MULTI40_SELECTION_SHA256
            or manifest.get("selection_id") != selection["selection_id"]):
        raise ValueError("Invalid multi40 cache manifest")
    reference = root / "motions" / REFERENCE_NAME
    reference_record = next(x for x in selection["train"] if x["motion"] == REFERENCE_NAME)
    if sha256(reference) != reference_record["sha256"] or manifest.get("reference043_sha256") != reference_record["sha256"]:
        raise ValueError("multi40 reference043 changed")
    with np.load(reference, allow_pickle=False) as data:
        for key, required in (("joint_names", robot_joint_names), ("body_names", robot_body_names)):
            names = data[key].tolist()
            if len(names) != len(set(names)) or (required is not None and not set(required) <= set(names)):
                raise ValueError(f"Reference {key} do not cover the configured robot")
    geometry = motion_geometry(root)
    if geometry != manifest.get("geometry") or any(geometry.get(k) != v for k, v in selection["geometry"].items()):
        raise ValueError("Changed multi40 suitcase geometry")
    for role in ("train", "eval"):
        directory = root / f"motions_{role}"
        expected = {x["motion"]: x for x in selection[role]}
        actual = {p.name for p in directory.glob("*.npz")}
        if actual != set(expected):
            raise ValueError(f"multi40 {role} membership: missing={sorted(set(expected)-actual)}, extra={sorted(actual-set(expected))}")
        if set(manifest.get("motions", {}).get(role, {})) != set(expected):
            raise ValueError(f"multi40 {role} manifest membership differs")
        for name, record in expected.items():
            path = directory / name
            if not path.resolve().is_relative_to(root):
                raise ValueError(f"Motion escapes cache: {path}")
            evidence = audit_motion(path, reference)
            expected_evidence = {k: record[k] for k in ("frames", "fps", "sha256")}
            if evidence != expected_evidence or evidence != manifest["motions"][role][name]:
                raise ValueError(f"Changed multi40 {role} motion: {name}")
    return root / f"motions_{split}"
