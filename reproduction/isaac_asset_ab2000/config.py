"""Exact asset-only switch used by the accepted Isaac A/B2000 experiment."""
from dataclasses import replace


def configure(cfg, group):
    assert group in ('legacy', 'mode15')
    if group == 'legacy':
        return cfg
    from holosoma.config_values.wbt.g1.robot_mode15 import with_mode15
    return replace(cfg, robot=with_mode15(cfg.robot))
