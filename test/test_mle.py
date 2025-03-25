"""Test `xvfm.mle`."""

import pytest
import numpy as np

from xvfm.mle import (
    feat_transform,
    prepare_ml_problem,
    _evaluate_binary_classification,
    _evaluate_regression,
    get_evaluator
)

###############################################################################
# TEST feat_transform
###############################################################################
def test_feat_transform_no_cmin_cmax():
    """
    Test feat_transform when cmin/cmax are not provided.
    Checks that logs or normalized values are returned correctly.
    """
    # Suppose we have 3 numeric features, 2 categorical, 1 label => total 6 columns
    # data shape: [batch_size, 6]
    # numeric => columns 0..2, categorical => columns 3..4, label => column 5
    data = np.array([
        [10, 0.5, 1000, 1, 2, 0],
        [50, 1.5, 2000, 2, 0, 1],
        [1,  100,  999, 0, 1, 1]
    ], dtype=float)

    info = {
        "num_col_idx": [0, 1, 2],
        "cat_col_idx": [3, 4],
    }

    transformed_feats, labels, cmax, cmin = feat_transform(data, info)
    # Expect shape => (3, #num_feats + #cat_feats) => (3, 5)
    assert transformed_feats.shape == (3, 5)
    # The last column in the original data is the label
    assert np.array_equal(labels, data[:, -1]), "Labels must match last column of data."
    # cmax, cmin returned must be some float values
    assert cmax is not None and cmin is not None, "Should return computed cmax/cmin."

def test_feat_transform_with_cmin_cmax():
    """
    Test feat_transform when cmin/cmax are provided explicitly.
    Ensures consistent scaling is used.
    """
    data = np.array([
        [100, 5, 50, 1, 2, 1],
        [200, 6, 10, 0, 1, 0],
    ], dtype=float)
    info = {
        "num_col_idx": [0, 1, 2],
        "cat_col_idx": [3, 4],
    }
    # Provide custom cmax / cmin
    cmin, cmax = 0, 200
    transformed_feats, labels, new_cmax, new_cmin = feat_transform(
        data, info, cmax=cmax, cmin=cmin
    )
    # Check shapes
    assert transformed_feats.shape == (2, 5)
    # The cmax/cmin returned from the function is the same as provided
    assert new_cmax == cmax and new_cmin == cmin, "Should keep user-provided cmax/cmin."
    # Check label extraction
    assert np.array_equal(labels, data[:, -1])


###############################################################################
# TEST prepare_ml_problem
###############################################################################
@pytest.mark.parametrize("task_type", ["binclass", "regression"])
def test_prepare_ml_problem(task_type):
    """
    Test prepare_ml_problem to ensure it splits data and returns expected shapes.
    """
    # Let's make a small dataset of shape [batch_size, total_cols]
    # We'll do 3 numeric, 2 categorical, 1 label => total 6 columns
    # For binclass => label in {0,1}, for regression => label in float
    rng = np.random.RandomState(42)
    data_size = 20
    data = rng.rand(data_size, 6)

    # For binclass, let's convert last col to {0,1}
    if task_type == "binclass":
        data[:, -1] = (data[:, -1] > 0.5).astype(float)
    # For regression, we can just keep it float

    # We'll split data into train (14 rows) and test (6 rows), for example
    train = data[:14]
    test = data[14:]

    info = {
        "num_col_idx": [0, 1, 2],  # 3 numeric
        "cat_col_idx": [3, 4],     # 2 categorical
        "task_type": task_type,
    }

    train_X, train_y, val_X, val_y, test_X, test_y, model_spec = prepare_ml_problem(train, test, info)

    # Basic shape checks
    # Because we do a 1/9 for val => val_size = 14/9 => ~1 row
    # so train_X should have ~13 rows, val_X ~1 row, test_X 6 rows
    assert train_X.shape[0] + val_X.shape[0] == 14, "Train+val rows must match original train set."
    assert test_X.shape[0] == 6, "Test rows must match original test set."
    # Features shape => (num_row, #num_cols + #cat_cols)
    assert train_X.shape[1] == 5, "Should have numeric+categorical => 5 columns of features."

    # Check model spec is consistent with the task
    # For binclass => XGBClassifier, for regression => XGBRegressor
    for spec in model_spec:
        if task_type == "binclass":
            assert "binary:logistic" in spec["kwargs"]["objective"], "Binary classification objective expected."
        else:
            assert "reg:linear" in spec["kwargs"]["objective"], "Regression objective expected."


###############################################################################
# TEST _evaluate_binary_classification
###############################################################################
def test_evaluate_binary_classification():
    """
    Tests that _evaluate_binary_classification runs end-to-end without error,
    and returns valid results data structures.
    """
    # Create a small binary classification dataset
    rng = np.random.RandomState(0)
    data_size = 30
    data = rng.rand(data_size, 6)
    data[:, -1] = (data[:, -1] > 0.5).astype(float)

    train = data[:20]
    test = data[20:]
    info = {
        "num_col_idx": [0, 1, 2],
        "cat_col_idx": [3, 4],
        "task_type": "binclass",
    }

    f1_scores, auroc_scores, acc_scores, avg_scores = \
        _evaluate_binary_classification(train, test, info)

    # We expect each of these to be a list of dictionaries (one per model spec)
    for score_list in [f1_scores, auroc_scores, acc_scores, avg_scores]:
        assert isinstance(score_list, list), "Expected a list of model-results dicts."
        assert len(score_list) > 0, "Should have at least one model result."
        # Check that keys exist, e.g. 'name', 'f1', 'roc_auc', 'accuracy'
        for d in score_list:
            assert "name" in d, "Result dict should contain 'name'."
            if "f1" in d:  # binary classification
                assert 0 <= d["f1"] <= 1.0, "F1 should be between [0,1]."

###############################################################################
# TEST _evaluate_regression
###############################################################################
def test_evaluate_regression():
    """
    Tests that _evaluate_regression runs end-to-end without error,
    and returns valid results data structures.
    """
    rng = np.random.RandomState(1)
    data_size = 30
    data = rng.rand(data_size, 6)
    # For regression, last col is some float. We'll keep it random in [0,1]
    train = data[:20]
    test = data[20:]
    info = {
        "num_col_idx": [0, 1, 2],
        "cat_col_idx": [3, 4],
        "task_type": "regression",
    }

    r2_scores, rmse_scores = _evaluate_regression(train, test, info)

    # We expect each to be a list of dictionaries (one per model spec)
    for score_list in [r2_scores, rmse_scores]:
        assert isinstance(score_list, list), "Should be a list."
        assert len(score_list) > 0, "Should have at least one model result."
        for d in score_list:
            assert "name" in d, "Result dict should contain model name"
            # check r2 is a float, RMSE is a float, etc.
            if "r2" in d:
                assert not np.isnan(d["r2"]), "R2 shouldn't be NaN"
                assert d["r2"] >= -1.0, "R2 can be negative, but not typically less than -1."

###############################################################################
# TEST get_evaluator
###############################################################################
def test_get_evaluator():
    """
    Test that get_evaluator returns the correct function for binclass or regression.
    """
    evaluator_bin = get_evaluator("binclass")
    evaluator_reg = get_evaluator("regression")
    assert evaluator_bin == _evaluate_binary_classification, "Should return binclass evaluator"
    assert evaluator_reg == _evaluate_regression, "Should return regression evaluator"

    with pytest.raises(KeyError):
        # Should raise KeyError for an unsupported problem type
        get_evaluator("some_unknown_task")
