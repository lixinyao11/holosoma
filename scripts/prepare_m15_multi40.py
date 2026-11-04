"""Prepare reviewed UMR train40/eval2 in a NEW desktop cache. CPU only."""
import argparse
import json
from pathlib import Path
import shutil
from holosoma.utils.m15_motion_sets import (
    REFERENCE_NAME, MULTI40_SELECTION_PATH, MULTI40_SELECTION_SHA256,
    load_multi40_selection, motion_geometry, audit_motion, sha256, validate_multi40,
)


def audit(source, reference):
    source, reference = Path(source).resolve(), Path(reference).resolve()
    selection = load_multi40_selection()
    anchor = reference / "motions" / REFERENCE_NAME
    geometry = motion_geometry(reference)
    if geometry != motion_geometry(source) or any(geometry.get(k) != v for k,v in selection["geometry"].items()):
        raise ValueError("Source/reference geometry differs from reviewed UMR suitcase")
    expected043 = next(r["sha256"] for r in selection["train"] if r["motion"] == REFERENCE_NAME)
    if sha256(anchor) != expected043:
        raise ValueError("Reference043 differs from reviewed UMR")
    motions = {}
    for split in ("train", "eval"):
        motions[split] = {}
        for record in selection[split]:
            path = source / "motions" / record["motion"]
            evidence = audit_motion(path, anchor)
            if evidence != {k:record[k] for k in ("frames","fps","sha256")}:
                raise ValueError(f"Source differs from reviewed list: {path}")
            motions[split][record["motion"]] = evidence
    return dict(schema_version=1, motion_set="multi40", selection_id=selection["selection_id"],
        selection_sha256=MULTI40_SELECTION_SHA256, reference043_sha256=expected043,
        geometry=geometry, motions=motions, ready=True,
        source_root=str(source), reference_root=str(reference),
        note="Reviewed UMR only. No trajectory edits, retargeting, GPU smoke or physics acceptance.")


def prepare(source, reference, destination):
    source, reference, destination = Path(source), Path(reference), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite cache: {destination}")
    report = audit(source, reference)
    partial = destination.with_name(destination.name + ".preparing")
    partial.mkdir(parents=True, exist_ok=False)
    for rel in report["geometry"]:
        target = partial / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(reference / rel, target)
    (partial / "motions").mkdir()
    shutil.copy2(reference / "motions" / REFERENCE_NAME, partial / "motions" / REFERENCE_NAME)
    for split in ("train", "eval"):
        directory = partial / f"motions_{split}"
        directory.mkdir()
        for name in report["motions"][split]:
            shutil.copy2(source / "motions" / name, directory / name)
    shutil.copy2(MULTI40_SELECTION_PATH, partial / "multi40_selection.json")
    (partial / "multi40_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    validate_multi40(partial)
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
    report = audit(args.source_root, args.reference_root) if args.audit_only else prepare(args.source_root, args.reference_root, args.destination)
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"ready":report["ready"],"counts":{k:len(v) for k,v in report["motions"].items()}}))


if __name__ == "__main__":
    main()
