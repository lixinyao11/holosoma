"""Prepare the fixed 50 Holosoma exports in a NEW desktop cache; no retargeting/GPU."""
import argparse
import json
from pathlib import Path
import shutil

from holosoma.utils.m15_motion_sets import (
    MULTI50_IDS, REFERENCE_NAME, SELECTION_MANIFEST_SHA256,
    audit_motion, sha256, validate_multi50,
)


def geometry(root):
    paths = [root / "suitcase.urdf", *sorted((root / "meshes").rglob("*"))]
    return {str(p.relative_to(root)): sha256(p) for p in paths if p.is_file()}


def audit(source, reference):
    source, reference = Path(source).resolve(), Path(reference).resolve()
    missing, errors, motions = [], [], {}
    anchor = reference / "motions" / REFERENCE_NAME
    if not anchor.is_file() or not (reference / "suitcase.urdf").is_file():
        raise ValueError("Reference root must contain the existing single043 and suitcase URDF")
    source_geometry, reference_geometry = geometry(source), geometry(reference)
    if source_geometry != reference_geometry:
        errors.append("Source suitcase geometry differs from the single043 reference")
    for motion_id in MULTI50_IDS:
        name = f"omomo__{motion_id}.npz"
        path = source / "motions" / name
        if not path.is_file():
            missing.append(motion_id)
            continue
        try:
            motions[name] = audit_motion(path, anchor)
        except (ValueError, KeyError, OSError) as exc:
            errors.append(str(exc))
    if REFERENCE_NAME in motions and motions[REFERENCE_NAME]["sha256"] != sha256(anchor):
        errors.append("Source 043 differs from the existing single043 reference")
    return {
        "schema_version": 1, "motion_set": "multi50", "ready": not missing and not errors,
        "selection_manifest_sha256": SELECTION_MANIFEST_SHA256,
        "source_root": str(source), "reference_root": str(reference),
        "reference043_sha256": sha256(anchor), "geometry": reference_geometry,
        "motions": motions, "missing_ids": missing, "errors": errors,
        "provenance_scope": "Fixed legacy C32 IDs; existing Holosoma exports. No raw-qpos conversion or trajectory-equivalence claim.",
    }


def prepare(source, reference, destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to write an existing cache: {destination}")
    report = audit(source, reference)
    if not report["ready"]:
        raise ValueError(json.dumps(report, indent=2))
    # A failed preparation retains its separate partial directory for diagnosis.
    partial = destination.with_name(destination.name + ".preparing")
    partial.mkdir(parents=True, exist_ok=False)
    source, reference = Path(source), Path(reference)
    for rel in report["geometry"]:
        target = partial / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(reference / rel, target)
    (partial / "motions").mkdir()
    shutil.copy2(reference / "motions" / REFERENCE_NAME, partial / "motions" / REFERENCE_NAME)
    (partial / "motions_multi50").mkdir()
    for name in report["motions"]:
        shutil.copy2(source / "motions" / name, partial / "motions_multi50" / name)
    (partial / "multi50_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    validate_multi50(partial)
    if destination.exists():
        raise FileExistsError(destination)
    partial.rename(destination)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--destination")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--report")
    args = parser.parse_args()
    if not args.audit_only and not args.destination:
        parser.error("--destination is required unless --audit-only")
    report = audit(args.source_root, args.reference_root) if args.audit_only else prepare(
        args.source_root, args.reference_root, args.destination)
    rendered = json.dumps(report, indent=2) + "\n"
    if args.report:
        Path(args.report).write_text(rendered)
    print(json.dumps({k: report[k] for k in ("ready", "missing_ids", "errors")}))
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
