"""Test `data.utils`."""

import json
import pytest
import numpy as np
from unittest.mock import patch

import torch

from utils import (
    load_json,
    read_pure_data,
    Dataset,
    SeedsMetricsReport,
    TaskType,
    transform_dataset,
    cat_drop_rare,

    prepare_test_data,
    prepare_fast_dataloader,
    FastTensorDataLoader,
    dump_pickle,
    load_pickle,
)


###############################################################################
# Test load_json
###############################################################################
@patch("pathlib.Path.read_text", return_value='{"key": "value"}')
def test_load_json(mock_read_text):
    """Test load_json to ensure it reads JSON content via Path and returns a dict."""
    result = load_json("some_file.json")
    assert result == {"key": "value"}
    mock_read_text.assert_called_once()


###############################################################################
# Test read_pure_data
###############################################################################
@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_read_pure_data(split, tmp_path):
    """
    Test read_pure_data by writing out small .npy files and then verifying 
    we can read them back with the correct shapes.
    """
    # Make a directory for dummy data
    dir_ = tmp_path / "fake_data"
    dir_.mkdir()

    # Create small arrays
    X_num_path = dir_ / f"X_num_{split}.npy"
    X_cat_path = dir_ / f"X_cat_{split}.npy"
    y_path = dir_ / f"y_{split}.npy"

    X_num_array = np.random.randn(5, 3)
    X_cat_array = np.random.randint(0, 5, size=(5, 2))
    y_array = np.random.rand(5)

    np.save(str(X_num_path), X_num_array)
    np.save(str(X_cat_path), X_cat_array)
    np.save(str(y_path), y_array)

    X_num, X_cat, y = read_pure_data(dir_, split=split)

    assert np.allclose(X_num, X_num_array), "X_num should match the saved data"
    assert np.allclose(X_cat, X_cat_array), "X_cat should match the saved data"
    assert np.allclose(y, y_array), "y should match the saved data"


###############################################################################
# Test Dataset.from_dir
###############################################################################
@patch("utils.load_json", return_value={"task_type": "binclass", "n_classes": 2})
def test_dataset_from_dir(mock_info, tmp_path):
    """
    Test Dataset.from_dir ensuring it loads arrays for train, test, 
    and populates the Dataset object properly.
    """
    # We'll create train/test .npy files in tmp_path
    dir_ = tmp_path / "data_folder"
    dir_.mkdir()

    # If we want the dataset to have X_num or X_cat, let's create those for "train" / "test"
    X_num_train = np.random.randn(10, 3)
    X_num_test = np.random.randn(5, 3)
    y_train = np.random.randint(0, 2, size=10)
    y_test = np.random.randint(0, 2, size=5)

    # Save them
    np.save(dir_ / "X_num_train.npy", X_num_train)
    np.save(dir_ / "X_num_test.npy", X_num_test)
    np.save(dir_ / "y_train.npy", y_train)
    np.save(dir_ / "y_test.npy", y_test)
    # Usually we might not have "val", so we'll skip that

    # Create a minimal info.json to confirm we do see it
    info_path = dir_ / "info.json"
    info_path.write_text(json.dumps({"task_type": "binclass", "n_classes": 2}))

    # Now call from_dir
    ds = Dataset.from_dir(dir_)

    assert ds.task_type == TaskType.BINCLASS
    assert ds.n_classes == 2
    assert np.allclose(ds.X_num["train"], X_num_train)
    assert np.allclose(ds.X_num["test"], X_num_test)
    assert np.allclose(ds.y["train"], y_train)
    assert np.allclose(ds.y["test"], y_test)
    # Check that val doesn't exist => ds.y won't have 'val'
    assert "val" not in ds.y


###############################################################################
# Test SeedsMetricsReport
###############################################################################
def test_seeds_metrics_report():
    """
    Create multiple dummy MetricsReport-like objects and add them to SeedsMetricsReport,
    then verify we get aggregated stats.
    """
    from utils import MetricsReport

    class MockReport(MetricsReport):
        def __init__(self, rep, task_type):
            super().__init__(rep, task_type)

    # Suppose we have 2 "reports" for binclass with accuracy, f1, roc_auc
    rep1 = {
        "train": {"accuracy": 0.8, "macro avg": {"f1-score": 0.75}, "roc_auc": 0.7},
        "val": {"accuracy": 0.85, "macro avg": {"f1-score": 0.78}, "roc_auc": 0.73},
        "test": {"accuracy": 0.81, "macro avg": {"f1-score": 0.76}, "roc_auc": 0.71},
    }
    rep2 = {
        "train": {"accuracy": 0.88, "macro avg": {"f1-score": 0.85}, "roc_auc": 0.8},
        "val": {"accuracy": 0.86, "macro avg": {"f1-score": 0.8}, "roc_auc": 0.75},
        "test": {"accuracy": 0.82, "macro avg": {"f1-score": 0.78}, "roc_auc": 0.72},
    }

    m1 = MockReport(rep1, TaskType.BINCLASS)
    m2 = MockReport(rep2, TaskType.BINCLASS)

    seeds_report = SeedsMetricsReport()
    seeds_report.add_report(m1)
    seeds_report.add_report(m2)

    agg = seeds_report.get_mean_std()
    # We expect means, std for train/val/test => accuracy, f1, roc_auc
    # Check a few values:
    # For train accuracy => [0.8, 0.88] => mean=0.84, std ~ 0.04
    mean_train_acc = agg["train"]["accuracy-mean"]
    std_train_acc = agg["train"]["accuracy-std"]
    assert pytest.approx(mean_train_acc, 0.01) == 0.84
    assert pytest.approx(std_train_acc, 0.01) == 0.04


###############################################################################
# Test transform_dataset (basic)
###############################################################################
def test_transform_dataset_basic():
    """
    Provide a small Dataset with numeric and categorical data,
    call transform_dataset with a 'quantile' normalization,
    and check shapes or result for no crash.
    """
    # Setup a minimal dataset
    X_num = {
        "train": np.random.randn(5, 2),
        "test": np.random.randn(3, 2),
    }
    X_cat = {
        "train": np.array([["A","B","A","C","B"]]).T,
        "test": np.array([["C","B","A"]]).T,
    }
    y = {
        "train": np.array([0,1,1,0,1]),
        "test": np.array([1,0,0]),
    }
    ds = Dataset(
        X_num=X_num, 
        X_cat=X_cat,
        y=y,
        y_info={},
        task_type=TaskType.BINCLASS,
        n_classes=2
    )

    from utils import Transformations
    T = Transformations(
        normalization="quantile", 
        cat_min_frequency=None, 
        cat_encoding=None,
        y_policy=None
    )

    ds2 = transform_dataset(ds, T)
    # ds2 should have numeric data quantile-transformed and cat data ordinal-encoded
    # Because cat_encoding=None => it uses ordinal encoding (not "one-hot" or "counter")
    # We won't test the numeric transformations deeply, just ensure shapes match
    assert ds2.X_num is not None, "Should produce X_num (since cat is appended as numeric if cat is ordinal-encoded)."
    # Because encoding=None => cat becomes an OrdinalEncoder => still integer columns appended or replaced
    # Actually the code sets is_num=False if encoding=None, but let's see the logic:
    # If encoding is None => it uses an OrdinalEncoder, but it returns (X, False) => is_converted_to_numerical=False
    # So it doesn't merge with X_num. So X_cat is replaced with encoded ints, X_num is quantiled
    # Let's see the code for cat_encode(..., encoding=None) => it returns (X, False), so X_cat remains separate.
    # So ds2.X_cat is now integer-coded
    assert ds2.X_cat is not None
    assert ds2.X_cat["train"].shape == (5, 1), "Single cat column after ordinal encoding"
    assert ds2.X_num["train"].shape == (5, 2), "Still 2 numeric columns"
    # No crash => success


###############################################################################
# Test cat_drop_rare
###############################################################################
def test_cat_drop_rare():
    """
    Provide a small cat dictionary for X with some categories that are rare,
    check that cat_drop_rare replaces them with '__rare__'.
    """
    X_cat = {
        "train": np.array([
            ["A","B","A","C","D","A","D","D"],
            ["foo","foo","bar","foo","bar","foo","bar","baz"]
        ], dtype=object).T,  # shape(8,2)
        "test": np.array([
            ["A","D","X"],
            ["bar","baz","bar"]
        ], dtype=object).T
    }
    # min_frequency = 0.25 => with 8 train rows => min_count=2
    # So categories that appear <2 times in train => replaced with '__rare__'
    # Let's see "C" => appears once => replaced, "X" => not in train => replaced
    # We'll just verify the logic
    out = cat_drop_rare(X_cat, 0.25)
    # out["train"] => shape(8,2)
    # first col => "A"(3x), "B"(1x), "C"(1x), "D"(3x)
    # Actually B=1 => that also is <2, so B => '__rare__'
    # So final train col => [A, __rare__, A, __rare__, __rare__, A, __rare__, __rare__]
    train_col0 = out["train"][:,0]
    # 'B' => rare, 'C' => rare, 'D' appears 3 times => that is >=2 => so 'D' stays
    # Wait, we have 3 D => that's okay, it's not rare
    # Actually let's count properly: A=3, B=1, C=1, D=3 => so B and C are <2 => replaced
    # So the final train col => [A, __rare__, A, __rare__, D, A, D, D]
    assert train_col0.tolist() == ["A","__rare__","A","__rare__","D","A","D","D"], f"Train col0 is {train_col0}"

    # test col0 => ["A","D","X"] => "X" not in train => replaced
    test_col0 = out["test"][:,0]
    assert test_col0.tolist() == ["A","D","__rare__"], f"Test col0 is {test_col0}"


###############################################################################
# Test build_target
###############################################################################
def test_build_target_regression_default():
    """
    Provide a y dict with a train set that has mean/std, verify that 
    policy='default' for regression subtracts mean and divides by std.
    """
    y = {
        "train": np.array([10.0, 12.0, 14.0, 16.0]),
        "test": np.array([12.0, 18.0])
    }
    from utils import build_target, TaskType
    out, info = build_target(y, policy="default", task_type=TaskType.REGRESSION)
    mean = info["mean"]
    std = info["std"]
    assert mean == 13.0, f"Mean => {mean}"
    assert pytest.approx(std, 0.001) == 2.236, f"Std => {std}"
    # train => [10,12,14,16] => => [-3/2.236, -1/2.236, 1/2.236, 3/2.236]
    # We won't check every value precisely, but let's ensure the transformation happened
    assert abs(out["train"][0] - ((10-13)/std)) < 1e-6
    assert abs(out["test"][1] - ((18-13)/std)) < 1e-6


def test_build_target_bad_policy():
    """Check that build_target raises an error for unknown policy."""
    y = {
        "train": np.array([1.0, 2.0]),
        "test": np.array([3.0]),
    }
    from utils import build_target, TaskType, raise_unknown
    with pytest.raises(ValueError) as excinfo:
        build_target(y, policy="some_unknown_policy", task_type=TaskType.REGRESSION)
    assert "Unknown policy: some_unknown_policy" in str(excinfo.value)


###############################################################################
# Test calculate_metrics
###############################################################################
def test_calculate_metrics_regression():
    """Check the R2, RMSE logic for regression with a known mean/std."""
    y_true = np.array([10.0, 15.0])
    y_pred = np.array([12.0, 14.0])
    y_info = {"std": 2.0}
    from utils import calculate_metrics, TaskType
    result = calculate_metrics(
        y_true, y_pred, TaskType.REGRESSION, prediction_type=None, y_info=y_info
    )
    # RMSE => sqrt( ( (10-12)^2 + (15-14)^2 ) /2 ) => sqrt( (4 + 1)/2 )= sqrt(2.5)=1.581...
    # Then multiplied by std=2 => 3.162
    rmse = result["rmse"]
    r2 = result["r2"]
    assert pytest.approx(rmse, 0.001) == 3.162, f"RMSE => {rmse}"
    # R2 => baseline => if you want the formula => ~0.5 maybe
    # We can just ensure it's ~0.5
    assert 0.4 < r2 < 0.6, f"R2 => {r2}"


def test_calculate_metrics_binclass_probs():
    """Check binclass with probs => we do label = round(prob), plus check roc_auc."""
    y_true = np.array([0,1,1,0])
    y_pred_probs = np.array([0.2,0.7,0.8,0.3])  # shape(4,)
    from utils import calculate_metrics, TaskType, PredictionType
    res = calculate_metrics(
        y_true, y_pred_probs, TaskType.BINCLASS, PredictionType.PROBS, y_info={}
    )
    # We can do a quick check => label=round(prob)= [0,1,1,0], that matches y_true => perfect classification => f1=1.0
    assert pytest.approx(res["accuracy"], 1e-7) == 1.0
    assert pytest.approx(res["macro avg"]["f1-score"], 1e-7) == 1.0
    # roc_auc => 1.0 as well
    assert pytest.approx(res["roc_auc"], 1e-7) == 1.0


###############################################################################
# Test prepare_test_data and prepare_fast_dataloader
###############################################################################
def test_prepare_test_data():
    """
    Provide a minimal dataset with X_num["test"], X_cat=None, y["test"], 
    check that prepare_test_data merges X,y => shape=(*, n_num + 1)
    """
    ds = Dataset(
        X_num={"train":np.random.randn(4,3), "test":np.random.randn(2,3)},
        X_cat=None,
        y={"train": np.array([0,1,1,0]), "test":np.array([1,0])},
        y_info={},
        task_type=TaskType.BINCLASS,
        n_classes=2
    )
    out = prepare_test_data(ds)
    # out => shape(2, 4) => 3 numeric + 1 label
    assert out.shape == (2,4), f"Should have 2 rows, 4 columns => 3 + 1"


def test_prepare_fast_dataloader():
    """
    Provide a small dataset with numeric and cat for 'train', check the returned 
    FastTensorDataLoader has the correct shapes and metadata.
    """
    ds = Dataset(
        X_num={"train": np.random.randn(5,2)},
        X_cat={"train": np.array([["A","B","C","A","B"]]).T},
        y={"train": np.array([1,0,1,1,0])},
        y_info={},
        task_type=TaskType.BINCLASS,
        n_classes=2
    )
    loader = prepare_fast_dataloader(ds, "train", batch_size=2)
    # The final X => shape(5,3) => 2 num + 1 cat
    assert loader.X.shape == (5,3)
    assert loader.y.shape == (5,)
    assert loader.classes == [3], "One cat column with 3 unique categories => [3]"
    assert loader.num_feat == 2


###############################################################################
# Test FastTensorDataLoader iteration
###############################################################################
def test_fast_tensor_data_loader_iter():
    X = torch.randn(5, 3)
    y = torch.randint(0,2,(5,))
    loader = FastTensorDataLoader(X, y, batch_size=2, shuffle=False)

    it = iter(loader)
    batch1 = next(it)
    assert batch1[0].shape == (2,3) and batch1[1].shape == (2,)
    batch2 = next(it)
    # second batch => shape(2,3)/(2,) => total data=5 => last batch => shape(1,3)
    batch3 = next(it)
    assert batch3[0].shape == (1,3) and batch3[1].shape == (1,)
    with pytest.raises(StopIteration):
        next(it)


###############################################################################
# Test dump_pickle and load_pickle
###############################################################################
def test_dump_load_pickle(tmp_path):
    data = {"hello": "world", "numbers": [1,2,3]}
    path = tmp_path / "test.pkl"
    dump_pickle(data, path)

    loaded = load_pickle(path)
    assert loaded == data, "Loaded data should match original"
