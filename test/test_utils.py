"""Test `xvfm.utils`."""

import os
import time
import shutil
import tempfile
import pytest
import torch
from unittest.mock import patch

# Import the items to test from utils.py
from xvfm.utils import (
    Velocity,
    generate,
    evaluate,
    plot_performance,
    clean_workspace
)

###############################################################################
# A simple mock model for testing
###############################################################################
class MockModel(torch.nn.Module):
    """A mock torch model that just returns x + t for simplicity."""
    def __init__(self, d_in=2):
        super().__init__()
        self.d_in = d_in

    def forward(self, x, t):
        # Simple: return x + t
        return x + t


###############################################################################
# Tests for Velocity
###############################################################################
def test_velocity_forward():
    d_in = 2
    mock_model = MockModel(d_in=d_in)
    velocity_fn = Velocity(mock_model)

    x = torch.randn(5, d_in)  # batch of 5
    t = torch.tensor(0.5)

    result = velocity_fn(t, x)

    # Check shape
    assert result.shape == x.shape, (
        "Output from Velocity.forward should match the shape of input x"
    )


###############################################################################
# Tests for generate
###############################################################################
def test_generate_output_shape():
    d_in = 3
    mock_model = MockModel(d_in=d_in)
    num_samples = 10

    dev = torch.device('cpu')
    output = generate(mock_model, num_samples, dev)

    # Expect shape (num_samples, d_in)
    assert output.shape == (num_samples, d_in), (
        "generate() should return a tensor of shape (num_samples, d_in)"
    )


###############################################################################
# Tests for evaluate
###############################################################################
@pytest.mark.parametrize("task_type", ["regression", "classification"])
@patch("utils.get_performance", return_value={"mock_score": 1.0})
def test_evaluate_runs(mock_get_perf, task_type):
    class Args:
        def __init__(self):
            self.num_feat = 3
            self.classes = []
            self.task_type = task_type
            self.dataset = "mock_dataset"

    args = Args()
    mock_model = MockModel(d_in=args.num_feat)

    # For regression, expect last column is a continuous target
    # For classification, the last column is a discrete label
    if task_type == "regression":
        test_data = torch.randn(5, args.num_feat + 1)
    else:
        # classification => last column is discrete
        test_data = torch.randint(0, 2, (5, args.num_feat + 1)).float()

    dev = torch.device('cpu')
    suffix = 1  # triggers both performance print and plot

    # Patch plot_performance so we don't create files
    with patch("utils.plot_performance") as mock_plot:
        scores = evaluate(args, mock_model, test_data, dev, suffix)
        mock_get_perf.assert_called_once()
        mock_plot.assert_called_once()

    assert "mock_score" in scores, (
        "evaluate() should return the dictionary from get_performance"
    )


###############################################################################
# Tests for plot_performance
###############################################################################
@patch("utils.PdfPages")
def test_plot_performance_no_crash(mock_pdf):
    class Args:
        def __init__(self):
            self.num_feat = 3
            self.classes = []
            self.task_type = "regression"
            self.dataset = "mock_dataset"

    args = Args()

    # shape: (batch_size, num_feat + 1)
    test_data = torch.randn(5, args.num_feat + 1)
    gen_data = torch.randn(5, args.num_feat + 1)

    # Just ensure it doesn't raise
    try:
        plot_performance(test_data, gen_data, args, suffix=1)
    except Exception as e:
        pytest.fail(f"plot_performance raised an exception: {e}")


###############################################################################
# Tests for clean_workspace
###############################################################################
def test_clean_workspace():
    test_dir = tempfile.mkdtemp()
    start_time = time.time() - 10  # slightly in the past
    bkp_file = os.path.join(test_dir, "test_file.bkp")

    # Create a .bkp file
    with open(bkp_file, 'w') as f:
        f.write("Backup file content")

    # Create a subdirectory
    sub_dir = os.path.join(test_dir, "subdir")
    os.mkdir(sub_dir)

    try:
        # We'll pretend that this directory is the `workspace` root
        # so we patch os.getcwd() to return test_dir's parent.
        with patch("utils.os.getcwd", return_value=os.path.dirname(test_dir)):

            # Force updated ctime by rewriting
            time.sleep(1)
            with open(bkp_file, 'w') as f:
                f.write("Recently updated to change ctime")

            # Create a new dir after start_time
            new_sub_dir = os.path.join(test_dir, "new_subdir")
            os.mkdir(new_sub_dir)

            # Now call clean_workspace
            clean_workspace(start_time)

            # The .bkp file and the new_sub_dir should be removed
            assert not os.path.exists(bkp_file), (
                ".bkp file updated after start_time should be removed"
            )
            assert not os.path.exists(new_sub_dir), (
                "Directory created after start_time should be removed"
            )
    finally:
        # Cleanup our temp dir
        shutil.rmtree(test_dir, ignore_errors=True)
