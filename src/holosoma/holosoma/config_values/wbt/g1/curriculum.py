"""Whole Body Tracking curriculum presets for the G1 robot."""

from dataclasses import replace

from holosoma.config_types.curriculum import CurriculumManagerCfg, CurriculumTermCfg

g1_29dof_wbt_curriculum = CurriculumManagerCfg(
    params={
        "num_compute_average_epl": 1000,
    },
    setup_terms={
        "average_episode_tracker": CurriculumTermCfg(
            func="holosoma.managers.curriculum.terms.locomotion:AverageEpisodeLengthTracker",
            params={},
        ),
    },
    reset_terms={},
    step_terms={},
)

g1_29dof_wbt_curriculum_hybrid_object = replace(
    g1_29dof_wbt_curriculum,
    setup_terms={
        **g1_29dof_wbt_curriculum.setup_terms,
        "hybrid_gen_noise_curriculum": CurriculumTermCfg(
            func="holosoma.managers.curriculum.terms.wbt:HybridGenNoiseCurriculum",
            params={
                "enabled": True,
                "initial_scale": 0.3,
                "min_scale": 0.3,
                "max_scale": 1.0,
                "scale_step": 0.01,
                "success_threshold_up": 0.90,
                "success_threshold_down": 0.80,
                "update_interval": 20,
                "ema_alpha": 0.5,
            },
        ),
    },
)

__all__ = [
    "g1_29dof_wbt_curriculum",
    "g1_29dof_wbt_curriculum_hybrid_object",
]
