# Track phase (opt-in)

Use `exp:g1-29dof-wbt-hybrid-object-hdmi-friction-dr-track-phase` with the usual backend and 043 motion/asset overrides. For track-only runs retain `--command.setup-terms.motion-command.params.motion-config.hybrid.track-ratio=1.0`.

This extends the accepted mode15 + all current HDMI DR preset with exactly one actor scalar (565 -> 566); the critic, rewards, PPO, reset sampler, physics and DR are unchanged. Old presets remain available.

Phase is the local current motion frame divided by original clip length minus one, clamped to [0,1]. The appended terminal hold is excluded from the denominator, so phase remains one throughout the hold. Random/intermediate and partial resets use the command's new sampled frame immediately, never the randomized episode-length counter. A one-frame clip is terminal (one). Gen rows are zero; this is not a gen phase mechanism. Default-pose prepend/return transitions are not supported by this phase term.

Only motion timing is provided, not reference joint/object poses or future actions. The critic still has its existing privileged reference inputs. The phase term has no noise and no added history. Existing empirical observation normalization is unchanged.

The actor input dimension changed: a 565D checkpoint cannot be resumed directly into this preset. The smoke uses fresh initialization; no formal training is launched.
