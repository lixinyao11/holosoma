"""Portable 043 configuration factory; importing or CLI inspection never starts Isaac/PPO."""
from dataclasses import asdict, replace
from pathlib import Path
import argparse
import json
from holosoma.config_values.wbt.g1.hdmi_goal import make_legacy_m15_config
from holosoma.utils.m15_motion_sets import REFERENCE_NAME


def make_config(asset_root, *, extra_dr=False, motion_set="single043", motion_split="train", track_phase=False):
    if motion_set not in ("single043", "multi50", "multi40"):
        raise ValueError(f"Unknown motion_set: {motion_set}")
    if motion_split not in ("train", "eval") or (motion_set != "multi40" and motion_split != "train"):
        raise ValueError(f"Invalid motion_split={motion_split} for motion_set={motion_set}")
    root = Path(asset_root).expanduser().resolve()
    motion = root / "motions/omomo__sub4_suitcase_043.npz"
    urdf = root / "suitcase.urdf"
    for path in (motion, urdf):
        if not path.is_file():
            raise FileNotFoundError(path)
    cfg = make_legacy_m15_config(extra_dr=extra_dr)
    terms = dict(cfg.command.setup_terms)
    term = terms["motion_command"]
    params = dict(term.params)
    params["motion_config"] = replace(params["motion_config"], motion_file=str(motion))
    if motion_set == "multi50":
        from holosoma.utils.m15_motion_sets import validate_multi50
        motion_dir = validate_multi50(root, robot_joint_names=cfg.robot.dof_names, robot_body_names=cfg.robot.body_names)
        params["motion_config"] = replace(params["motion_config"], motion_dir=str(motion_dir))
    elif motion_set == "multi40":
        from holosoma.utils.m15_motion_sets import load_multi40_selection, validate_multi40
        motion_dir = validate_multi40(root, split=motion_split,
            robot_joint_names=cfg.robot.dof_names, robot_body_names=cfg.robot.body_names)
        # An eval config must not fall back to the training anchor even if a caller
        # later disables directory loading. The explicit split contains only its IDs.
        fallback = motion_dir / (REFERENCE_NAME if motion_split == "train" else load_multi40_selection()["eval"][0]["motion"])
        params["motion_config"] = replace(params["motion_config"], motion_dir=str(motion_dir), motion_file=str(fallback))
    names = {"single043": "g1_29dof_wbt_hdmi_goal043", "multi50": "g1_29dof_wbt_hdmi_multi50", "multi40": "g1_29dof_wbt_hdmi_multi40"}
    name = names[motion_set] + ("_eval2" if motion_split == "eval" else "")
    terms["motion_command"] = replace(term, params=params)
    cfg = replace(cfg,
        training=replace(cfg.training, name=name + "_upstream_hands" + ("_extra_dr" if extra_dr else "")),
        command=replace(cfg.command, setup_terms=terms),
        robot=replace(cfg.robot, object=replace(cfg.robot.object,
            object_urdf_path=str(urdf), object_urdf_paths=[])))
    if track_phase:
        from holosoma.config_values.wbt.g1.legacy_track_phase import with_track_phase
        cfg = with_track_phase(cfg)
    return cfg


def make_eval_config(asset_root, *, extra_dr=False, track_phase=False):
    """Select only the two held-out motions; the runner still sets evaluation mode."""
    return make_config(asset_root, extra_dr=extra_dr, motion_set="multi40", motion_split="eval", track_phase=track_phase)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True, help="Directory containing suitcase.urdf and motions/")
    parser.add_argument("--extra-dr", action="store_true", help="Enable the four-item B bundle")
    parser.add_argument("--motion-set", choices=("single043", "multi50", "multi40"), default="single043")
    parser.add_argument("--motion-split", choices=("train", "eval"), default="train")
    parser.add_argument("--track-phase", action="store_true", help="D: add one current actor phase scalar")
    args = parser.parse_args()
    print(json.dumps(asdict(make_config(args.asset_root, extra_dr=args.extra_dr, motion_set=args.motion_set, motion_split=args.motion_split, track_phase=args.track_phase)), indent=2, default=str))


if __name__ == "__main__":
    main()
