import tyro

from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_values import algo as algo_defaults_module
from holosoma.utils.tyro_utils import TYRO_CONIFG


def test_experiment_config():
    assert isinstance(tyro.cli(ExperimentConfig, args=(), config=TYRO_CONIFG), ExperimentConfig)


def test_ppo_latest_save_interval_default():
    assert algo_defaults_module.ppo.config.latest_save_interval == 100
