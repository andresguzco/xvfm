"""Test `xvfm.eval`."""

import pytest
import torch
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, mock_open

# Import items from eval.py
from xvfm.eval import (
    get_mle,
    get_results,
    get_performance,
    get_shape_trend_score,
    get_detection,
    get_quality
)

###############################################################################
# Fixtures for synthetic data and a mock "args" object
###############################################################################
@pytest.fixture
def mock_args(tmp_path):
    """
    Provides a mock 'args' object with minimal fields
    used by get_performance and get_mle.
    """
    class Args:
        def __init__(self):
            self.num_feat = 3
            self.classes = [2]  # e.g., 2 categories
            self.task_type = "regression"
            self.data_path = str(tmp_path)  # location for "info.json" or related

    return Args()


@pytest.fixture
def synthetic_data(mock_args):
    """Create a pair of synthetic DataFrames/Tensors for testing."""
    # Suppose each row has:
    #  num_feat = 3 continuous features
    #  classes = [2], meaning 1 categorical column with 2 possible categories
    #  + 1 target => total columns = 3 + 1 + 1 = 5
    # We'll produce PyTorch tensors and also DataFrames for certain tests.
    batch_size = 6
    # syn_data / test_data shape => [batch, 5]
    syn_np = np.random.rand(batch_size, 5)
    test_np = np.random.rand(batch_size, 5)

    # For the categorical column, let's pick integer classes in [0,2)
    syn_np[:, 3] = np.random.randint(0, 2, size=batch_size)
    test_np[:, 3] = np.random.randint(0, 2, size=batch_size)

    syn_torch = torch.tensor(syn_np, dtype=torch.float32)
    test_torch = torch.tensor(test_np, dtype=torch.float32)
    return syn_torch, test_torch

###############################################################################
# Test get_results
###############################################################################
def test_get_results():
    """
    Ensure get_results picks maximum metrics for each 'metric' across model entries.
    """
    sample_data = {
        "scoreA": {
            "model1": {"m1": 10, "m2": 7},
            "model2": {"m1": 12, "m2": 5}
        },
        "scoreB": {
            "model1": {"m1": 3, "m2": 20},
        }
    }
    # We call get_results, which returns the max of each metric encountered
    # across all model entries in each top-level key.
    out = get_results(sample_data)
    # For "scoreA" we have two models => model1 => m1=10,m2=7; model2 => m1=12,m2=5
    # So the max for 'm1' is 12, max for 'm2' is 7
    # For "scoreB" we only have model1 => m1=3, m2=20 => so max => m1=3, m2=20
    print(out["m1"], out["m2"])
    assert out["m1"] == 12
    assert out["m2"] == 20


###############################################################################
# Test get_mle
###############################################################################
@patch("builtins.open", new_callable=mock_open, read_data='{"task_type":"regression"}')
@patch("eval.get_evaluator")
def test_get_mle_regression(mock_get_eval, mock_file, tmp_path):
    """
    Test that get_mle calls get_evaluator for 'regression' and processes results correctly.
    We'll mock out the returned function from get_evaluator so we don't do real training.
    """
    # Suppose the evaluator function returns two lists, each containing dict of model results
    mock_eval_func = MagicMock(return_value=(
        [{"name": "XGBRegressor", "r2": 0.8}, {"name": "MyRegressor", "r2": 0.9}],
        [{"name": "XGBRegressor", "rmse": 1.2}, {"name": "MyRegressor", "rmse": 1.1}],
    ))
    mock_get_eval.return_value = mock_eval_func

    train = np.random.rand(4, 5)
    test = np.random.rand(2, 5)

    # We'll place a fake info.json in data_path
    data_path = str(tmp_path)

    # run get_mle
    scores = get_mle(train, test, data_path)
    # Expect a dict with "r2" and "rmse" top-level keys
    assert "r2" in scores
    assert "rmse" in scores
    # Each should have "XGBRegressor" and "MyRegressor"
    assert "XGBRegressor" in scores["r2"]
    assert "MyRegressor" in scores["r2"]
    assert "XGBRegressor" in scores["rmse"]
    assert "MyRegressor" in scores["rmse"]


@patch("builtins.open", new_callable=mock_open, read_data='{"task_type":"binclass"}')
@patch("eval.get_evaluator")
def test_get_mle_binclass(mock_get_eval, mock_file, tmp_path):
    """
    Similar test but for 'binclass'. 
    Mocks out the evaluator returning f1, auroc, acc, avg results.
    """
    mock_eval_func = MagicMock(return_value=(
        [{"name": "XGBClassifier", "f1": 0.8}],
        [{"name": "XGBClassifier", "roc_auc": 0.7}],
        [{"name": "XGBClassifier", "accuracy": 0.9}],
        [{"name": "XGBClassifier", "avg": 0.75}],
    ))
    mock_get_eval.return_value = mock_eval_func

    train = np.random.rand(5, 5)
    test = np.random.rand(5, 5)
    scores = get_mle(train, test, str(tmp_path))

    # Expect f1, auroc, acc, avg keys
    assert all(k in scores for k in ["f1", "auroc", "acc", "avg"]), \
        "Binary classification should produce f1, auroc, acc, avg metrics"


###############################################################################
# Test get_shape_trend_score
###############################################################################
@patch("eval.QualityReport")
def test_get_shape_trend_score(mock_qr):
    """
    Checks that get_shape_trend_score calls QualityReport and returns a tuple
    (Shape, Trend, Quality).
    """
    # Mock the QualityReport's 'get_properties' to return a dictionary that
    # has a "Score" key with shape, trend, etc.
    mock_report_instance = MagicMock()
    mock_report_instance.get_properties.return_value = {
        "Score": [0.9, 0.8, 0.95]  # typically shape/trend, maybe something else
    }
    mock_qr.return_value = mock_report_instance

    real = pd.DataFrame(np.random.rand(5, 3))
    syn = pd.DataFrame(np.random.rand(5, 3))
    shape, trend, quality = get_shape_trend_score(real, syn)

    assert shape == 0.9, "Should match the 1st score from 'Score'"
    assert trend == 0.8, "Should match the 2nd score"
    assert quality == (0.9 + 0.8) / 2, "Mean of shape & trend"


###############################################################################
# Test get_detection
###############################################################################
@patch("eval.LogisticDetection")
def test_get_detection(mock_ld):
    """
    Check that get_detection calls LogisticDetection.compute with the right arguments.
    """
    mock_ld.compute.return_value = 0.75

    real = pd.DataFrame(np.random.rand(5, 3))
    syn = pd.DataFrame(np.random.rand(5, 3))
    score = get_detection(real, syn)

    assert score == 0.75
    mock_ld.compute.assert_called_once()


###############################################################################
# Test get_quality
###############################################################################
@patch("eval.eval_statistical.AlphaPrecision")
def test_get_quality(mock_alpha_prec):
    """
    Check that get_quality calls the synthcity AlphaPrecision metric and returns (alpha, beta).
    """
    mock_alpha_instance = MagicMock()
    # Suppose it returns a dictionary with naive metrics
    mock_alpha_instance.evaluate.return_value = {
        "delta_precision_alpha_naive": 0.6,
        "delta_coverage_beta_naive": 0.3
    }
    mock_alpha_prec.return_value = mock_alpha_instance

    real_x = np.random.rand(5, 3)
    syn_x = np.random.rand(5, 3)

    alpha, beta = get_quality(real_x, syn_x)
    assert alpha == 0.6
    assert beta == 0.3


###############################################################################
# Test get_performance (the main wrapper)
###############################################################################
@patch("eval.get_mle", return_value={"r2":{"modelA":{"val":0.8}}})
@patch("eval.get_shape_trend_score", return_value=(0.9,0.95,0.925))
@patch("eval.get_detection", return_value=0.5)
@patch("eval.get_quality", return_value=(0.7,0.6))
def test_get_performance(
    mock_gq, mock_det, mock_sts, mock_mle, mock_args, synthetic_data
):
    """
    Test the main get_performance function. We will:
      - Provide synthetic torch data
      - Expect dictionary with shape/trend/detection/quality/alpha/beta
      - If mle=True, also expect items from get_mle
    """
    syn_data, test_data = synthetic_data

    # We'll set args.task_type = 'regression' for example
    mock_args.task_type = "regression"
    # Call get_performance with mle=True
    scores = get_performance(syn_data, test_data, mock_args, mle=True)

    # Check basic keys
    assert "shape" in scores
    assert "trend" in scores
    assert "detection" in scores
    assert "quality" in scores
    assert "alpha" in scores
    assert "beta" in scores

    # Also the MLE result for regression => "r2" top-level
    assert "r2" in scores, "With mle=True, get_mle result merges into final dict"
    # mock_mle should have been called once
    mock_mle.assert_called_once()
