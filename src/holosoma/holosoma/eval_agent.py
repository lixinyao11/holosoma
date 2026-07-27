from __future__ import annotations

import os

import tyro
from loguru import logger

from holosoma.agents.base_algo.base_algo import BaseAlgo
from holosoma.config_types.command import EvalFixedResetConfig
from holosoma.config_types.eval_callback import EvalCallbacksConfig
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.utils.config_utils import CONFIG_NAME
from holosoma.utils.eval_utils import (
    CheckpointConfig,
    init_eval_logging,
    load_checkpoint,
    load_saved_experiment_config,
)
from holosoma.utils.experiment_paths import get_experiment_dir, get_timestamp
from holosoma.utils.helpers import get_class
from holosoma.utils.sim_utils import (
    close_simulation_app,
    setup_simulation_environment,
)
from holosoma.utils.tyro_utils import TYRO_CONIFG

_RETIRED_29DOF_HAND_REWARD_TERMS = {
    "object_frame_gripper_position_error_exp",
    "wrist_frame_gripper_position_error_exp",
}
_RETIRED_29DOF_HAND_TERMINATION_TERMS = {
    "bad_hand_wrist_tracking",
    "bad_object_frame_gripper_tracking",
}


def _backfill_eval_randomization_flags(saved_config: ExperimentConfig) -> None:
    """Populate recently added randomization flags for older checkpoint configs.

    Older checkpoints may serialize randomization term params without newly added
    keys such as ``enabled``. Tyro derives the override schema from the loaded
    config, so missing keys make those CLI overrides unavailable at eval time.
    """
    randomization_cfg = saved_config.randomization
    if randomization_cfg is None:
        return

    setup_terms = randomization_cfg.setup_terms
    for term_name in (
        "randomize_robot_rigid_body_material_startup",
        "randomize_object_rigid_body_material_startup",
        "randomize_object_rigid_body_mass_startup",
        "randomize_object_rigid_body_inertia_startup",
    ):
        term_cfg = setup_terms.get(term_name)
        if term_cfg is None:
            continue
        term_cfg.params.setdefault("enabled", True)


def _drop_retired_29dof_hand_terms(saved_config: ExperimentConfig) -> None:
    """Keep recent 29-DoF checkpoints evaluable after removing unused hand terms."""
    if saved_config.robot.actions_dim != 29:
        return

    for term_name in _RETIRED_29DOF_HAND_REWARD_TERMS:
        saved_config.reward.terms.pop(term_name, None)
    for term_name in _RETIRED_29DOF_HAND_TERMINATION_TERMS:
        saved_config.termination.terms.pop(term_name, None)


def _backfill_eval_fixed_reset_type(saved_config: ExperimentConfig) -> None:
    """Restore typed fixed-reset config so Tyro accepts bin/local-frame eval overrides."""
    command_cfg = saved_config.command
    if command_cfg is None:
        return

    motion_term = command_cfg.setup_terms.get("motion_command")
    if motion_term is None:
        return

    motion_config = motion_term.params.get("motion_config")
    if not isinstance(motion_config, dict):
        return

    eval_fixed_reset = motion_config.get("eval_fixed_reset")
    if isinstance(eval_fixed_reset, EvalFixedResetConfig):
        return
    if not isinstance(eval_fixed_reset, dict):
        return

    eval_fixed_reset = dict(eval_fixed_reset)
    if eval_fixed_reset.get("bin_id") is None and eval_fixed_reset.get("local_frame") is None:
        eval_fixed_reset["bin_id"] = 0
        if eval_fixed_reset.get("enabled", False):
            eval_fixed_reset["randomize_within_bin"] = False
    motion_config["eval_fixed_reset"] = EvalFixedResetConfig(**eval_fixed_reset)


def run_eval_with_tyro(
    tyro_config: ExperimentConfig,
    checkpoint_cfg: CheckpointConfig,
    saved_config: ExperimentConfig,
    saved_wandb_path: str | None,
    eval_cbs_cfg: EvalCallbacksConfig | None = None,
):
    # Use shared simulation environment setup
    env, device, simulation_app = setup_simulation_environment(tyro_config)

    eval_log_dir = get_experiment_dir(tyro_config.logger, tyro_config.training, get_timestamp(), task_name="eval")
    eval_log_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving eval logs to {eval_log_dir}")
    tyro_config.save_config(str(eval_log_dir / CONFIG_NAME))

    # Inject eval callbacks into algo config
    if eval_cbs_cfg is not None:
        cb_configs = eval_cbs_cfg.collect_active_callbacks()
        if cb_configs:
            object.__setattr__(tyro_config.algo.config, "eval_callbacks", cb_configs)

    assert checkpoint_cfg.checkpoint is not None
    checkpoint = load_checkpoint(checkpoint_cfg.checkpoint, str(eval_log_dir))
    checkpoint_path = str(checkpoint)

    algo_class = get_class(tyro_config.algo._target_)
    algo: BaseAlgo = algo_class(
        device=device,
        env=env,
        config=tyro_config.algo.config,
        log_dir=str(eval_log_dir),
        multi_gpu_cfg=None,
    )
    algo.setup()
    algo.attach_checkpoint_metadata(saved_config, saved_wandb_path)
    algo.load(checkpoint_path)

    checkpoint_dir = os.path.dirname(checkpoint_path)

    exported_policy_dir_path = os.path.join(checkpoint_dir, "exported")
    os.makedirs(exported_policy_dir_path, exist_ok=True)
    exported_policy_name = checkpoint_path.split("/")[-1]  # example: model_5000.pt
    exported_onnx_name = exported_policy_name.replace(".pt", ".onnx")  # example: model_5000.onnx

    if tyro_config.training.export_onnx:
        exported_onnx_path = os.path.join(exported_policy_dir_path, exported_onnx_name)
        if not hasattr(algo, "export"):
            raise AttributeError(
                f"{algo_class.__name__} is missing an `export` method required for ONNX export during evaluation."
            )

        algo.export(onnx_file_path=exported_onnx_path)  # type: ignore[attr-defined]
        logger.info(f"Exported policy as onnx to: {exported_onnx_path}")

    algo.evaluate_policy(
        max_eval_steps=tyro_config.training.max_eval_steps,
    )

    # Cleanup simulation app
    if simulation_app:
        close_simulation_app(simulation_app)


def main() -> None:
    init_eval_logging()
    checkpoint_cfg, remaining_args = tyro.cli(CheckpointConfig, return_unknown_args=True, add_help=False)
    eval_cbs_cfg, remaining_args = tyro.cli(
        EvalCallbacksConfig, return_unknown_args=True, add_help=False, args=remaining_args
    )
    saved_cfg, saved_wandb_path = load_saved_experiment_config(checkpoint_cfg)
    _drop_retired_29dof_hand_terms(saved_cfg)
    _backfill_eval_randomization_flags(saved_cfg)
    _backfill_eval_fixed_reset_type(saved_cfg)
    eval_cfg = saved_cfg.get_eval_config()
    overwritten_tyro_config = tyro.cli(
        ExperimentConfig,
        default=eval_cfg,
        args=remaining_args,
        description="Overriding config on top of what's loaded.",
        config=TYRO_CONIFG,
    )

    run_eval_with_tyro(overwritten_tyro_config, checkpoint_cfg, saved_cfg, saved_wandb_path, eval_cbs_cfg=eval_cbs_cfg)


if __name__ == "__main__":
    main()
