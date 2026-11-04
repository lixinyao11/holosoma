"""Single M15 experiment entry point, returning the upstream ExperimentConfig."""
from dataclasses import replace
from pathlib import Path
import argparse
import json
from holosoma.config_values.wbt.g1.hdmi_goal import make_legacy_m15_config
from holosoma.utils.m15_motion_sets import REFERENCE_NAME
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_values.wbt.g1.observation import obj_future_terms, object_bbox_terms

# Parameter presets, not independent copies of ExperimentConfig.
GROUP_PRESETS = {
    "A": dict(extra_dr=False, motion_set="single043", track_phase=False),
    "B": dict(extra_dr=True, motion_set="single043", track_phase=False),
    "C": dict(extra_dr=False, motion_set="multi40", track_phase=False),
    "D": dict(extra_dr=False, motion_set="multi40", track_phase=True),
}


def _resolve_group(group, **overrides):
    if group is not None and group not in GROUP_PRESETS:
        raise ValueError(f"Unknown group: {group}")
    values = dict(GROUP_PRESETS[group or "A"])
    for name, value in overrides.items():
        if value is not None:
            if group is not None and value != values[name]:
                raise ValueError(f"group {group} requires {name}={values[name]!r}; omit group for a custom configuration")
            values[name] = value
    return values



def make_config(asset_root, *, group=None, extra_dr=None, motion_set=None, motion_split="train",
                track_phase=None, collision_exclusions=True, motor_dr=True, delay_filter=True,
                rewind_prob=0.0, ego_camera=False, future_object_poses=False, object_bbox=False,
                object_persist_steps=1, body_pos_z_only=False) -> ExperimentConfig:
    """Build once from upstream definitions; never overlay an entire experiment preset.

    Explicit group and contradictory group-defining flags are rejected. Omitting
    group allows custom combinations. Other upstream options are independent.
    This factory supports the pinned single-suitcase dataset, not multi-object assets.
    """
    selected = _resolve_group(group, extra_dr=extra_dr, motion_set=motion_set, track_phase=track_phase)
    extra_dr, motion_set, track_phase = (selected[k] for k in ("extra_dr", "motion_set", "track_phase"))
    if not 0 <= rewind_prob <= 1:
        raise ValueError("rewind_prob must be in [0, 1]")
    if type(object_persist_steps) is not int or object_persist_steps < 1:
        raise ValueError("object_persist_steps must be a positive integer")
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
    cfg = make_legacy_m15_config(extra_dr=extra_dr, collision_exclusions=collision_exclusions,
                                 motor_dr=motor_dr, delay_filter=delay_filter)
    terms = dict(cfg.command.setup_terms)
    term = terms["motion_command"]
    params = dict(term.params)
    params["motion_config"] = replace(params["motion_config"], motion_file=str(motion), rewind_prob=rewind_prob)
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
    # Extend individual upstream fields/terms, preserving M15/history/dataset choices.
    additions = {}
    if future_object_poses:
        additions.update(obj_future_terms)
    if object_bbox:
        additions.update(object_bbox_terms)
    if additions:
        groups = {key: replace(value, terms={**value.terms, **additions})
                  for key, value in cfg.observation.groups.items()}
        cfg = replace(cfg, observation=replace(cfg.observation, groups=groups))
    if ego_camera:
        cfg = replace(cfg, simulator=replace(cfg.simulator, config=replace(
            cfg.simulator.config, ego_camera=replace(cfg.simulator.config.ego_camera, enabled=True))))
    if object_persist_steps != 1 or body_pos_z_only:
        termination = dict(cfg.termination.terms)
        term = termination["bad_tracking"]
        values = dict(term.params)
        if object_persist_steps != 1:
            values["object_persist_steps"] = object_persist_steps
        if body_pos_z_only:
            values["body_pos_z_only"] = True
        termination["bad_tracking"] = replace(term, params=values)
        cfg = replace(cfg, termination=replace(cfg.termination, terms=termination))
    suffixes = []
    for enabled, label in [(not collision_exclusions, "no_collision_filter"),
                           (not motor_dr, "no_motor_dr"), (not delay_filter, "no_delay_filter"),
                           (ego_camera, "ego_camera"), (future_object_poses, "future_object"),
                           (object_bbox, "bbox"), (body_pos_z_only, "body_z_only")]:
        if enabled:
            suffixes.append(label)
    if rewind_prob:
        suffixes.append(f"rewind{rewind_prob:g}")
    if object_persist_steps != 1:
        suffixes.append(f"persist{object_persist_steps}")
    if suffixes:
        cfg = replace(cfg, training=replace(cfg.training, name=cfg.training.name + "_" + "_".join(suffixes)))
    if track_phase:
        from holosoma.config_values.wbt.g1.legacy_track_phase import with_track_phase
        cfg = with_track_phase(cfg)
    return cfg


def make_eval_config(asset_root, *, extra_dr=None, track_phase=None, group=None, **options):
    """Select only the two held-out motions; the runner still sets evaluation mode."""
    return make_config(asset_root, group=group, extra_dr=extra_dr, motion_set="multi40",
                       motion_split="eval", track_phase=track_phase, **options)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True)
    parser.add_argument("--group", choices=tuple(GROUP_PRESETS))
    parser.add_argument("--motion-set", choices=("single043", "multi50", "multi40"), default=None)
    parser.add_argument("--motion-split", choices=("train", "eval"), default="train")
    for name in ("extra-dr", "track-phase"):
        parser.add_argument("--" + name, action=argparse.BooleanOptionalAction, default=None)
    for name in ("collision-exclusions", "motor-dr", "delay-filter"):
        parser.add_argument("--" + name, action=argparse.BooleanOptionalAction, default=True)
    for name in ("ego-camera", "future-object-poses", "object-bbox", "body-pos-z-only"):
        parser.add_argument("--" + name, action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--rewind-prob", type=float, default=0.0)
    parser.add_argument("--object-persist-steps", type=int, default=1)
    parser.add_argument("--output", type=Path, help="Write expanded upstream config as .json or .yaml; never starts training")
    args = vars(parser.parse_args())
    output = args.pop("output")
    if output is not None and output.suffix not in (".json", ".yaml", ".yml"):
        parser.error("--output must end in .json, .yaml or .yml")
    try:
        cfg = make_config(**args)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    data = cfg.to_serializable_dict()
    if output is None:
        print(json.dumps(data, indent=2))
    else:
        # The export uses the existing upstream representation, not a new schema.
        import yaml
        payload = (json.dumps(data, indent=2) + "\n" if output.suffix == ".json"
                   else yaml.safe_dump(data))
        with output.open("x") as stream:
            stream.write(payload)


if __name__ == "__main__":
    main()
