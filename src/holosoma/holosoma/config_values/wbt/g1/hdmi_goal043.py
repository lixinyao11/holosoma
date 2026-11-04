"""Compatibility alias. New callers use holosoma.config_values.wbt.g1.m15.

No configuration is constructed here; both Python and CLI delegate to the
single canonical factory, preserving existing experiment wrapper imports.
"""
from holosoma.config_values.wbt.g1.m15 import make_config, make_eval_config, main

__all__ = ["make_config", "make_eval_config"]

if __name__ == "__main__":
    main()
