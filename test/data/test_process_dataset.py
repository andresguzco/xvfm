"""Test `data.process_dataset`."""

import pytest
import numpy as np
import pandas as pd
import tempfile
import shutil
from unittest.mock import patch, mock_open, MagicMock

# Import relevant functions from process_dataset.py
from data.process_dataset import (
    preprocess_beijing,
    preprocess_news,
    get_column_name_mapping,
    train_val_test_split,
    process_data,
    INFO_PATH
)

###############################################################################
# Helpers
###############################################################################
@pytest.fixture
def temp_dir():
    """Fixture to create and remove a temporary directory."""
    dirpath = tempfile.mkdtemp()
    yield dirpath
    shutil.rmtree(dirpath, ignore_errors=True)

###############################################################################
# Tests for preprocess_beijing
###############################################################################
@patch("builtins.open", new_callable=mock_open, read_data='{"raw_data_path":"beijing_raw.csv","data_path":"beijing_clean.csv"}')
@patch("process_dataset.pd.read_csv")
@patch("process_dataset.pd.DataFrame.dropna")
@patch("process_dataset.pd.DataFrame.to_csv")
def test_preprocess_beijing(mock_to_csv, mock_dropna, mock_read_csv, mock_file, temp_dir):
    """
    Test that preprocess_beijing reads info.json, loads the CSV, drops the first column,
    calls dropna(), and writes out the cleaned CSV.
    """
    # Mock .columns => e.g. 4 columns => columns = [col0, col1, col2, col3]
    mock_df = MagicMock()
    mock_df.columns = ["col0","col1","col2","col3"]
    # So dropping columns[0] => we keep col1..col3
    mock_read_csv.return_value = mock_df
    # dropna() => returns self or some new df => we'll chain it
    mock_dropna.return_value = mock_df

    # We want preprocess_beijing to read Info/beijing.json => but we have a mock.
    # The function uses f"{INFO_PATH}/beijing.json", so let's confirm:
    # with open(f"{INFO_PATH}/beijing.json", "r") as f: ...
    # We'll let the mock_open handle that.

    preprocess_beijing()

    # We expect a read_csv for the raw_data_path from the JSON: "beijing_raw.csv"
    mock_read_csv.assert_called_once_with("beijing_raw.csv")

    # Then we expect .to_csv() with "beijing_clean.csv" from the JSON
    mock_to_csv.assert_called_once_with("beijing_clean.csv", index=False)

    # We also confirm that the columns are subset from the second column onward.
    # The code does: data_df = data_df[columns[1:]]
    # So we check that mock_df.__getitem__ got called with slice(1, None).
    mock_df.__getitem__.assert_called_once()
    args, kwargs = mock_df.__getitem__.call_args
    # The code uses columns[1:], so let's see if that's what's passed in
    # It should be something like mock_df.columns[1:], which we set to ["col1","col2","col3"]
    # We can't directly see the slice in the mock, but we see if we got that subset
    # We'll just confirm it's a list or slice containing "col1","col2","col3"
    subset = args[0]
    assert subset == ["col1","col2","col3"], "Should drop the first column"


###############################################################################
# Tests for preprocess_news
###############################################################################
@patch("builtins.open", new_callable=mock_open, read_data='{"raw_data_path":"news_raw.csv","data_path":"news_clean.csv"}')
@patch("process_dataset.pd.read_csv")
@patch("process_dataset.pd.DataFrame.to_csv")
def test_preprocess_news(mock_to_csv, mock_read_csv, mock_file, temp_dir):
    """
    Test that preprocess_news reads 'news.json', drops 'url' col, transforms cat_columns,
    and saves to CSV. We'll do partial checks with a mock DataFrame.
    """
    # Let's mock up a DataFrame with columns that can match the indexing done by the code
    # The code uses data_df.columns => we want at least 38 columns to cover cat_columns2 indexing
    all_cols = [f"col{i}" for i in range(40)]
    all_cols[0] = "url"  # so code can drop it
    mock_df = MagicMock()
    mock_df.columns = pd.Index(all_cols)

    # We'll also need .astype(int).to_numpy().argmax(axis=1) calls for cat_columns1, cat_columns2
    # We'll just mock them out so we don't do real numeric ops
    def side_effect_astype(*args, **kwargs):
        # Return self so we can chain .to_numpy() => let's do a submock
        astype_mock = MagicMock()
        astype_mock.to_numpy.return_value = np.random.randint(0,5,size=(10,6))  # shape(10,6) => .argmax => shape(10,)
        return astype_mock

    mock_df.__getitem__.return_value.astype.side_effect = side_effect_astype

    mock_read_csv.return_value = mock_df

    preprocess_news()

    # Check we dropped the 'url' column => data_df.drop('url', axis=1)
    mock_df.drop.assert_any_call("url", axis=1)

    # Finally check that .to_csv was called with "data/news/news.csv"
    mock_to_csv.assert_called_once_with("data/news/news.csv", index=False)


###############################################################################
# Test get_column_name_mapping
###############################################################################
def test_get_column_name_mapping():
    """
    Provide a synthetic DataFrame with known columns, 
    plus known num_col_idx, cat_col_idx, and target_col_idx. 
    Check that idx_mapping, inverse_idx_mapping, and idx_name_mapping are correct.
    """
    data = pd.DataFrame({
        "A": [1,2,3],
        "B": [4,5,6],
        "C": [7,8,9],
        "D": ["cat","dog","cat"]
    })
    num_col_idx = [0,1]
    cat_col_idx = [3]
    target_col_idx = [2]

    idx_map, inv_idx_map, idx_name_map = get_column_name_mapping(
        data_df=data,
        num_col_idx=num_col_idx,
        cat_col_idx=cat_col_idx,
        target_col_idx=target_col_idx
    )

    # We expect:
    #  columns => [A,B,C,D]
    #  num => col 0 -> new index 0, col 1 -> new index 1
    #  cat => col 3 -> new index 2
    #  target => col 2 -> new index 3
    # idx_map => {0: 0, 1: 1, 3: 2, 2: 3}
    assert idx_map == {0:0, 1:1, 3:2, 2:3}
    # inv_idx_map => {0:0, 1:1, 2:3, 3:2}
    assert inv_idx_map == {0:0, 1:1, 2:3, 3:2}
    # idx_name_map => {0:"A", 1:"B", 2:"C", 3:"D"}
    assert idx_name_map == {0:"A", 1:"B", 2:"C", 3:"D"}


###############################################################################
# Test train_val_test_split
###############################################################################
def test_train_val_test_split():
    """
    Provide a small data DataFrame with a known categorical column and check 
    that the function eventually finds a seed that includes all categories in the train set.
    """
    rng = np.random.RandomState(42)
    data_size = 8
    data = pd.DataFrame({
        "num_col": rng.randint(0, 100, data_size),
        "cat_col": ["A","B","B","A","B","A","C","A"],
        "val": rng.randn(data_size)
    })
    cat_cols = ["cat_col"]
    num_train = 5
    num_test = 3

    train_df, test_df, final_seed = train_val_test_split(
        data_df=data, cat_columns=cat_cols,
        num_train=num_train, num_test=num_test
    )

    # Check shapes
    assert train_df.shape[0] == num_train, "Train set should be size 5"
    assert test_df.shape[0] == num_test, "Test set should be size 3"

    # We also want to ensure train_df covers all categories in data_df for cat_col
    full_cat = set(data["cat_col"])
    train_cat = set(train_df["cat_col"])
    assert train_cat == full_cat, "Train set must contain all categories from data_df"


###############################################################################
# Test process_data
###############################################################################
@patch("builtins.open", new_callable=mock_open, read_data='{"data_path":"mock_data.csv","file_type":"csv","header":0,"column_names":[],"num_col_idx":[0,1],"cat_col_idx":[2],"target_col_idx":[3],"test_path":"","task_type":"regression"}')
@patch("process_dataset.pd.read_csv")
@patch("process_dataset.train_val_test_split", return_value=(pd.DataFrame(), pd.DataFrame(), 123))
@patch("process_dataset.np.save")
@patch("process_dataset.pd.DataFrame.to_csv")
def test_process_data_regression(
    mock_to_csv,
    mock_npsave,
    mock_tvts,
    mock_read_csv,
    mock_file,
    temp_dir
):
    """
    Tests process_data flow for a typical 'regression' dataset. 
    We'll pass a single name, patch reading/writing, and ensure we see the calls.
    """

    # Setup the mock data
    mock_df = pd.DataFrame({
        0: [1.0, 2.0, 3.0],
        1: [4.0, 5.0, 6.0],
        2: ["cat","cat","dog"],
        3: [0.5, 0.6, 0.7],
    })
    mock_read_csv.return_value = mock_df

    # We also want to ensure that process_data tries to create folders like data/<name>
    # We'll just patch 'os.path.exists' and 'os.makedirs' to control environment
    with patch("os.path.exists", return_value=False), \
         patch("os.makedirs"):
        # Now call process_data with a mock name
        from process_dataset import process_data
        process_data("mydataset")

    # Check that train_val_test_split was called once
    mock_tvts.assert_called_once()

    # The code saves multiple files => X_num_train.npy, X_cat_train.npy, y_train.npy, etc.
    # We confirm that np.save got called with these paths
    expected_saves = [
        "data/mydataset/X_num_train.npy",
        "data/mydataset/X_cat_train.npy",
        "data/mydataset/y_train.npy",
        "data/mydataset/X_num_test.npy",
        "data/mydataset/X_cat_test.npy",
        "data/mydataset/y_test.npy"
    ]
    calls = [call[0][0] for call in mock_npsave.call_args_list]  # each call_args looks like (('file.npy', data), {})
    for path in expected_saves:
        assert path in calls, f"{path} should be saved"

    # We also expect .to_csv calls for train.csv and test.csv, plus any earlier calls in the pipeline
    # 2 calls to DataFrame.to_csv in the final steps
    # The code also calls e.g. test_df.to_csv, etc. So let's check those calls
    final_calls = [c[0][0] for c in mock_to_csv.call_args_list]
    # Among them, we want "data/mydataset/train.csv" and "data/mydataset/test.csv"
    assert "data/mydataset/train.csv" in final_calls, "Train CSV output missing"
    assert "data/mydataset/test.csv" in final_calls, "Test CSV output missing"
    # The function also writes an info.json => we'll check open calls or ensure it's not missed
    # The function does: with open(f"{save_dir}/info.json","w") as file => ...
    # We can confirm that with mock_file if we want:
    # But we used the same mock for reading => we can check if we had a second call with "w" mode
    open_calls = mock_file.call_args_list
    wrote_info_json = any("info.json" in c[0][0] and "w" in c[0][1] for c in open_calls)
    assert wrote_info_json, "Should write info.json"

    # Since it's regression, we confirm that the code sets metadata for numerical target, etc. 
    # That logic is tested mostly by verifying the final dictionary gets dumped,
    # but we've verified it made a second open in "w" mode for that info.


@pytest.mark.parametrize("name", ["news", "beijing"])
@patch("process_dataset.preprocess_news")
@patch("process_dataset.preprocess_beijing")
@patch("builtins.open", new_callable=mock_open, read_data='{"data_path":"mock.csv","file_type":"csv","header":0,"column_names":[],"num_col_idx":[],"cat_col_idx":[],"target_col_idx":[],"test_path":"","task_type":"regression"}')
@patch("process_dataset.pd.read_csv")
@patch("process_dataset.train_val_test_split", return_value=(pd.DataFrame(), pd.DataFrame(), 123))
@patch("process_dataset.np.save")
@patch("process_dataset.pd.DataFrame.to_csv")
def test_process_data_news_beijing(
    mock_tocsv,
    mock_npsave,
    mock_tvts,
    mock_pd_read_csv,
    mock_file,
    mock_pbeijing,
    mock_pnews,
    name
):
    """
    Ensure that calling process_data with name="news" or "beijing" calls 
    the correct preprocessing function. Then continues with normal flow.
    """
    from process_dataset import process_data
    process_data(name)

    if name == "news":
        mock_pnews.assert_called_once()
        mock_pbeijing.assert_not_called()
    else:
        mock_pbeijing.assert_called_once()
        mock_pnews.assert_not_called()
