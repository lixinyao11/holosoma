"""Track-only motion time, without reference poses or a separate phase clock."""
import torch


def track_phase(env):
    """Return [N, 1] local clip progress; terminal hold=1 and gen=0.

    Uses the command's sampled frame, not episode_length_buf. The current
    preset has no default-pose prepend/return transitions. Their phase
    semantics are intentionally not inferred here.
    """
    cmd = env.command_manager.get_state("motion_command")
    cfg = cmd.motion_cfg
    if cfg.enable_default_pose_prepend or cfg.enable_default_pose_append:
        raise ValueError("track_phase requires motion clips without default-pose transitions")
    hold = (max(0, round(cfg.terminal_pose_hold_duration_s / env.dt))
            if cfg.enable_terminal_pose_hold else 0)
    ids = cmd.motion_ids
    start = cmd.motion.motion_start_idx[ids]
    length = cmd.motion.motion_end_idx[ids] - start - hold
    local_frame = cmd.time_steps - start
    phase = (local_frame.float() / (length - 1).clamp_min(1).float()).clamp(0, 1)
    # A one-frame clip is already at its terminal pose.
    phase = torch.where(length <= 1, torch.ones_like(phase), phase)
    return torch.where(cmd.track_env_mask, phase, torch.zeros_like(phase)).unsqueeze(-1)
