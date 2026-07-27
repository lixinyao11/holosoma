from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.utils.tensorboard import SummaryWriter

from holosoma.agents.modules.logging_utils import LoggingHelper


@pytest.fixture
def mock_writer():
    """Fixture providing a mock SummaryWriter."""
    return MagicMock(spec=SummaryWriter)


@pytest.fixture
def mock_wandb():
    """Fixture providing mocked wandb module."""
    with patch("holosoma.agents.modules.logging_utils.wandb") as mock_wandb:
        yield mock_wandb


@pytest.fixture
def logging_helper(mock_writer):
    """Fixture providing a LoggingHelper instance with default parameters."""
    return LoggingHelper(
        writer=mock_writer,
        log_dir="/tmp/test_logs",
        num_envs=2,
        num_steps_per_env=10,
        num_learning_iterations=100,
        device="cpu",
    )


@pytest.fixture
def prefixed_logging_helper(mock_writer):
    """Fixture providing a LoggingHelper instance with a prefix."""
    return LoggingHelper(
        writer=mock_writer,
        log_dir="/tmp/test_logs",
        num_envs=2,
        num_steps_per_env=10,
        num_learning_iterations=100,
        device="cpu",
        prefix="test_prefix/",
    )


def test_prefix_in_logging(prefixed_logging_helper, mock_writer, mock_wandb):
    """Test that the prefix is properly added to all logged metrics."""
    # Add episode info
    prefixed_logging_helper.ep_infos = [{"test_metric": torch.tensor([1.0], device=prefixed_logging_helper.device)}]

    # Call post_epoch_logging with some test data
    prefixed_logging_helper.post_epoch_logging(
        it=0,
        loss_dict={"test_loss": 0.5},
        extra_log_dicts={"test_section": {"test_metric": 1.0}},
    )

    # Check that the prefix was added to all logged metrics
    expected_calls = [
        "test_prefix/Loss/test_loss",
        "test_prefix/Perf/total_fps",
        "test_prefix/Perf/collection_time",
        "test_prefix/Perf/learning_time",
        "test_prefix/Train/num_samples",
        "test_prefix/test_section/test_metric",
        "test_prefix/Episode/test_metric",
    ]

    # Verify all expected calls were made to writer.add_scalar
    actual_calls = [call[0][0] for call in mock_writer.add_scalar.call_args_list]
    for expected in expected_calls:
        assert expected in actual_calls


def test_no_prefix_logging(logging_helper, mock_writer, mock_wandb):
    """Test that logging works correctly without a prefix."""
    # Add episode info
    logging_helper.ep_infos = [{"test_metric": torch.tensor([1.0], device=logging_helper.device)}]

    # Call post_epoch_logging with some test data
    logging_helper.post_epoch_logging(
        it=0,
        loss_dict={"test_loss": 0.5},
        extra_log_dicts={"test_section": {"test_metric": 1.0}},
    )

    # Check that metrics were logged without prefix
    expected_calls = [
        "Loss/test_loss",
        "Perf/total_fps",
        "Perf/collection_time",
        "Perf/learning_time",
        "Train/num_samples",
        "test_section/test_metric",
        "Episode/test_metric",
    ]

    # Verify all expected calls were made to writer.add_scalar
    actual_calls = [call[0][0] for call in mock_writer.add_scalar.call_args_list]
    for expected in expected_calls:
        assert expected in actual_calls


def test_episode_stats_update(logging_helper):
    """Test that episode statistics are properly updated."""
    # Create test data
    rewards = torch.tensor([1.0, 2.0], device=logging_helper.device)
    dones = torch.tensor([1.0, 0.0], device=logging_helper.device)
    infos = {
        "episode": {
            "test_metric": torch.tensor([1.0], device=logging_helper.device),
        },
        "raw_episode": {
            "raw_test_metric": torch.tensor([2.0], device=logging_helper.device),
        },
        "to_log": {
            "env_metric": torch.tensor([2.0], device=logging_helper.device),
        },
    }

    # Update episode stats
    logging_helper.update_episode_stats(rewards, dones, infos)

    # Verify reward buffer was updated
    assert len(logging_helper.rewbuffer) == 1
    assert logging_helper.rewbuffer[0] == 1.0  # First environment's reward

    # Verify length buffer was updated
    assert len(logging_helper.lenbuffer) == 1
    assert logging_helper.lenbuffer[0] == 1.0  # First environment's length

    # Verify episode info was stored
    assert len(logging_helper.ep_infos) == 1
    assert logging_helper.ep_infos[0]["test_metric"].item() == 1.0

    # Verify raw episode info was stored
    assert len(logging_helper.raw_ep_infos) == 1
    assert logging_helper.raw_ep_infos[0]["raw_test_metric"].item() == 2.0


def test_wandb_logging(prefixed_logging_helper, mock_wandb):
    """Test that metrics are properly logged to wandb when available."""
    # Add some episode info to avoid empty list error
    prefixed_logging_helper.ep_infos = [{"test_metric": torch.tensor([1.0], device=prefixed_logging_helper.device)}]
    prefixed_logging_helper.raw_ep_infos = [
        {"raw_test_metric": torch.tensor([2.0], device=prefixed_logging_helper.device)}
    ]

    # Call post_epoch_logging with some test data
    prefixed_logging_helper.post_epoch_logging(
        it=0,
        loss_dict={"test_loss": 0.5},
        extra_log_dicts={"test_section": {"test_metric": 1.0}},
    )

    # Verify wandb.log was called with the correct data
    mock_wandb.log.assert_called_once()
    logged_data = mock_wandb.log.call_args[0][0]
    assert "test_prefix/Loss/test_loss" in logged_data
    assert "test_prefix/test_section/test_metric" in logged_data
    assert "test_prefix/Episode/test_metric" in logged_data
    assert "test_prefix/RawEpisode/raw_test_metric" in logged_data
    assert logged_data["global_step"] == 0


def test_wandb_logging_with_termination_metrics(prefixed_logging_helper, mock_wandb):
    """Test that termination metrics are logged under the dedicated Termination namespace."""
    prefixed_logging_helper.ep_infos = [
        {
            "termination/bad_tracking": torch.tensor([1.0], device=prefixed_logging_helper.device),
            "termination/bad_tracking/ref_pos": torch.tensor([1.0], device=prefixed_logging_helper.device),
        }
    ]
    prefixed_logging_helper.episode_env_tensors.add(
        {"termination/reset_frac": torch.tensor([0.5], device=prefixed_logging_helper.device)}
    )

    prefixed_logging_helper.post_epoch_logging(
        it=3,
        loss_dict={"test_loss": 0.5},
        extra_log_dicts={},
    )

    logged_data = mock_wandb.log.call_args[0][0]
    assert "test_prefix/Termination/Episode/bad_tracking" in logged_data
    assert "test_prefix/Termination/Episode/bad_tracking/ref_pos" in logged_data
    assert "test_prefix/Termination/Env/reset_frac" in logged_data


def test_wandb_logging_with_curriculum_metrics(prefixed_logging_helper, mock_wandb):
    """Curriculum metrics should be lifted out of the Env namespace."""
    prefixed_logging_helper.episode_env_tensors.add(
        {"curriculum/beta_mean": torch.tensor([0.5], device=prefixed_logging_helper.device)}
    )

    prefixed_logging_helper.post_epoch_logging(
        it=5,
        loss_dict={"test_loss": 0.5},
        extra_log_dicts={},
    )

    logged_data = mock_wandb.log.call_args[0][0]
    assert "test_prefix/Curriculum/Env/beta_mean" in logged_data
    assert "test_prefix/Env/curriculum/beta_mean" not in logged_data


def test_wandb_logging_with_hybrid_metrics(prefixed_logging_helper, mock_wandb):
    """Hybrid metrics should be grouped under the dedicated Hybrid namespace."""
    prefixed_logging_helper.ep_infos = [
        {
            "hybrid/termination/track/bad_tracking": torch.tensor([1.0], device=prefixed_logging_helper.device),
            "hybrid/reward/gen/rew_goal_ref_reached_once": torch.tensor(
                [0.25], device=prefixed_logging_helper.device
            ),
        }
    ]
    prefixed_logging_helper.raw_ep_infos = [
        {
            "hybrid/reward/track/raw_rew_hybrid_motion_global_ref_position_error_exp": torch.tensor(
                [0.75], device=prefixed_logging_helper.device
            ),
        }
    ]
    prefixed_logging_helper.episode_env_tensors.add(
        {"hybrid/termination/gen/reset_frac": torch.tensor([0.5], device=prefixed_logging_helper.device)}
    )

    prefixed_logging_helper.post_epoch_logging(
        it=6,
        loss_dict={"test_loss": 0.5},
        extra_log_dicts={},
    )

    logged_data = mock_wandb.log.call_args[0][0]
    assert "test_prefix/Hybrid/Episode/termination/track/bad_tracking" in logged_data
    assert "test_prefix/Hybrid/Episode/reward/gen/rew_goal_ref_reached_once" in logged_data
    assert "test_prefix/Hybrid/RawEpisode/reward/track/raw_rew_hybrid_motion_global_ref_position_error_exp" in logged_data
    assert "test_prefix/Hybrid/Env/termination/gen/reset_frac" in logged_data
    assert "test_prefix/Episode/hybrid/termination/track/bad_tracking" not in logged_data


def test_sparse_episode_metrics_do_not_crash_logging(prefixed_logging_helper, mock_wandb):
    """Missing optional episode keys across logging windows should be tolerated."""
    prefixed_logging_helper.ep_infos = [
        {
            "rew_test": torch.tensor([1.0], device=prefixed_logging_helper.device),
            "termination/timeout": torch.tensor([1.0], device=prefixed_logging_helper.device),
        },
        {
            "rew_test": torch.tensor([], device=prefixed_logging_helper.device),
        },
    ]

    prefixed_logging_helper.post_epoch_logging(
        it=4,
        loss_dict={"test_loss": 0.5},
        extra_log_dicts={},
    )

    logged_data = mock_wandb.log.call_args[0][0]
    assert "test_prefix/Episode/rew_test" in logged_data
    assert "test_prefix/Termination/Episode/timeout" in logged_data


def test_console_eta_uses_iterations_completed_since_resume(logging_helper):
    logging_helper.set_run_start_learning_iteration(14_500)
    logging_helper.num_learning_iterations = 20_000
    logging_helper.tot_time = 1821.87

    output = logging_helper._create_console_output(
        it=14_795,
        loss_dict={},
        env_log_dict={},
        extra_log_dicts={},
        ep_string="",
        width=80,
        pad=35,
        iteration_time=5.77,
        fps=1,
    )

    assert "Learning iteration 14795/34500" in output
    assert "ETA: 121277.5s" in output


def test_save_checkpoint_artifact(prefixed_logging_helper, mock_wandb, tmp_path):
    """Test that checkpoints are properly saved and logged to wandb."""
    # Create a temporary directory for the test
    log_dir = tmp_path / "test_logs"
    log_dir.mkdir()
    prefixed_logging_helper.log_dir = str(log_dir)

    # Create test state dict
    state_dict = {"test_param": torch.tensor([1.0])}
    checkpoint_path = log_dir / "checkpoint.pt"

    # Save checkpoint
    prefixed_logging_helper.save_checkpoint_artifact(state_dict, str(checkpoint_path))

    # Verify wandb.save was called with correct path
    mock_wandb.save.assert_called_once()
    assert mock_wandb.save.call_args[0][0] == str(checkpoint_path)
    assert mock_wandb.save.call_args[1]["base_path"] == str(log_dir)
    assert mock_wandb.save.call_args[1]["policy"] == "now"
