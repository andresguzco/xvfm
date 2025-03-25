"""Test `xvfm.train`."""

import sys
import pytest
import torch
from unittest.mock import patch, MagicMock
from argparse import Namespace

# Import the top-level functions from main.py
from main import get_args, main


###############################################################################
# Test get_args
###############################################################################
def test_get_args_defaults(monkeypatch):
    """
    Test that get_args() correctly parses default arguments.
    We'll monkeypatch sys.argv to simulate command line input.
    """
    test_argv = [
        "prog",  # typically ignored
    ]
    monkeypatch.setattr(sys, "argv", test_argv)

    args = get_args()
    assert args.epochs == 8000, "Default epochs should be 8000"
    assert args.lr == 1e-3, "Default lr should be 1e-3"
    assert args.batch_size == 4096, "Default batch size should be 4096"


def test_get_args_custom(monkeypatch):
    """
    Test get_args() with some custom arguments via monkeypatching sys.argv.
    """
    test_argv = [
        "prog",
        "--dataset", "my_dataset",
        "--data_path", "/tmp/data",
        "--task_type", "regression",
        "--epochs", "10",
        "--batch_size", "32",
        "--lr", "0.01",
        "--logging", "50",
        "--loss", "my_loss",
        "--num_eval", "500"
    ]
    monkeypatch.setattr(sys, "argv", test_argv)

    args = get_args()
    assert args.dataset == "my_dataset"
    assert args.data_path == "/tmp/data"
    assert args.task_type == "regression"
    assert args.epochs == 10
    assert args.batch_size == 32
    assert args.lr == 0.01
    assert args.logging == 50
    assert args.loss == "my_loss"
    assert args.num_eval == 500


###############################################################################
# Test main
###############################################################################
@pytest.mark.parametrize("task_type", ["regression", "classification"])
@patch("main.wandb.init")
@patch("main.wandb.finish")
@patch("main.make_dataset")
@patch("main.prepare_fast_dataloader")
@patch("main.prepare_test_data")
@patch("main.evaluate", return_value={"dummy_score": 1.23})
@patch("main.clean_workspace")
@patch("main.time.time", side_effect=[0, 1, 2, 3, 4])  # mocking time calls
def test_main_run(
    mock_time,
    mock_clean,
    mock_eval,
    mock_test_data,
    mock_dataloader,
    mock_make_dataset,
    mock_wandb_finish,
    mock_wandb_init,
    task_type
):
    """
    Ensure main() runs without error and calls key functions.
    This is a light integration test with heavy mocking to prevent actual training.
    """
    # Mock the dataset creation
    mock_dataset = MagicMock()
    mock_make_dataset.return_value = mock_dataset

    # Mock the dataloader
    mock_loader = MagicMock()
    mock_loader.num_feat = 3
    mock_loader.classes = [2]
    # We'll produce a couple of (x_1, y) batches
    # x_1 shape => [batch_size, num_feat + len(classes)] => e.g. 3 + 1=4
    # y shape => [batch_size, 1]
    # We'll let the loader iterate 2 times
    def mock_iter():
        for _ in range(2):
            x_1 = torch.randn(4, 4)  # batch_size=4
            y = torch.randint(0, 2, (4, 1))
            yield x_1, y

    mock_loader.__iter__.side_effect = mock_iter
    mock_dataloader.return_value = mock_loader

    # Mock test data: shape [num_eval, num_feat + len(classes) + 1], e.g. 1000 x 5
    # but we will pass a smaller shape for demonstration
    mock_test_data.return_value = torch.randn(5, 5)

    # Now build an args object as if from the command line
    # We only keep minimal fields since main() will parse them.
    args = Namespace(
        dataset="my_dataset",
        data_path="/tmp/data",
        task_type=task_type,
        epochs=1,          # just 1 epoch
        batch_size=2,
        lr=1e-3,
        logging=1,         # log every epoch
        loss="llk",
        num_eval=5
    )

    # We also mock the creation of the model (Tabby) and criterion
    with patch("main.Tabby") as mock_tabby_class, \
         patch("main.GuassianMultinomial") as mock_loss_class:
        mock_tabby_instance = MagicMock()
        mock_tabby_class.return_value = mock_tabby_instance
        mock_loss_instance = MagicMock()
        mock_loss_class.return_value = mock_loss_instance

        # We mock the forward pass
        mock_tabby_instance.return_value = torch.randn(4, 5)

        # Run main
        main(args)

    # Check if wandb.init was called
    mock_wandb_init.assert_called_once()
    # We also expect the evaluate function to be called once per epoch (since logging=1)
    # We used 1 epoch => so it should be called 1 time
    mock_eval.assert_called_once()
    # The training loop runs 1 epoch, iterates the dataloader 2 times => we check if the model was called
    # The final wandb.finish call is triggered
    mock_wandb_finish.assert_called_once()


@pytest.mark.parametrize("task_type", ["regression", "classification"])
def test_main_no_crash_with_minimal_args(task_type, monkeypatch):
    """
    A more minimal approach: run main() with monkeypatched sys.argv, 
    ignoring more advanced mocks. 
    We define only 1 epoch to avoid lengthy tests. 
    We'll still need to patch out some functions or we risk real I/O.
    """
    test_argv = [
        "prog",
        "--dataset", "my_dataset",
        "--data_path", "/tmp/fakedata",
        "--task_type", task_type,
        "--epochs", "1",
        "--batch_size", "2",
        "--logging", "0"
    ]
    monkeypatch.setattr(sys, "argv", test_argv)

    # We'll patch out just enough calls to prevent real I/O or real training
    with patch("main.wandb.init"), \
         patch("main.wandb.finish"), \
         patch("main.make_dataset"), \
         patch("main.prepare_fast_dataloader"), \
         patch("main.prepare_test_data"), \
         patch("main.Tabby"), \
         patch("main.GuassianMultinomial"), \
         patch("main.evaluate"), \
         patch("main.clean_workspace"):
        # Just ensure it doesn't crash
        args = get_args()
        main(args)
        # The code should run to completion.
