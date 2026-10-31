"""Portable 043 configuration factory; importing or CLI inspection never starts Isaac/PPO."""
from dataclasses import asdict, replace
from pathlib import Path
import argparse
import json
from holosoma.config_values.wbt.g1.hdmi_goal import g1_29dof_wbt_hybrid_object_hdmi


def make_config(asset_root):
    root = Path(asset_root).expanduser().resolve()
    motion = root / "motions/omomo__sub4_suitcase_043.npz"
    urdf = root / "suitcase.urdf"
    for path in (motion, urdf):
        if not path.is_file():
            raise FileNotFoundError(path)
    cfg = g1_29dof_wbt_hybrid_object_hdmi
    terms = dict(cfg.command.setup_terms)
    term = terms["motion_command"]
    params = dict(term.params)
    params["motion_config"] = replace(params["motion_config"], motion_file=str(motion))
    terms["motion_command"] = replace(term, params=params)
    return replace(cfg,
        training=replace(cfg.training, name="g1_29dof_wbt_hdmi_goal043"),
        command=replace(cfg.command, setup_terms=terms),
        robot=replace(cfg.robot, object=replace(cfg.robot.object,
            object_urdf_path=str(urdf), object_urdf_paths=[])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True, help="Directory containing suitcase.urdf and motions/")
    args = parser.parse_args()
    print(json.dumps(asdict(make_config(args.asset_root)), indent=2, default=str))


if __name__ == "__main__":
    main()
