"""Test `data.download_dataset`."""

import os
import pytest
import tempfile
import zipfile
from unittest.mock import patch

# Import the functions from download_dataset.py
from data.download_dataset import (
    unzip_file,
    download_from_uci,
    NAME_URL_DICT_UCI
)

###############################################################################
# Test unzip_file
###############################################################################
def test_unzip_file():
    """Test that unzip_file correctly extracts a zip archive."""
    # Create a temporary directory in which we'll build our test .zip and extract it
    with tempfile.TemporaryDirectory() as tmpdir:
        # Path for our test zip file
        zip_path = os.path.join(tmpdir, "test.zip")
        # The directory where we will extract
        extract_dir = os.path.join(tmpdir, "extracted")

        # Create a small zipfile containing one small text file
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("test_file.txt", "Hello, zip!")

        # Now call unzip_file
        unzip_file(zip_path, extract_dir)

        # Check that test_file.txt was extracted
        extracted_file_path = os.path.join(extract_dir, "test_file.txt")
        assert os.path.isfile(extracted_file_path), "test_file.txt should be extracted"

        # Check file content
        with open(extracted_file_path, "r") as f:
            content = f.read()
        assert content == "Hello, zip!", "test_file.txt content should match"

###############################################################################
# Test download_from_uci
###############################################################################
@pytest.mark.parametrize("dataset_name", list(NAME_URL_DICT_UCI.keys()))
def test_download_from_uci_new_dataset(dataset_name):
    """
    Test download_from_uci for a new dataset folder that doesn't exist yet.
    We mock out the network call (request.urlretrieve) so no real download is done,
    and we also patch out the actual unzip_file call.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # We'll store the folder name in tmpdir/dataset_name
        dataset_path = os.path.join(tmpdir, dataset_name)

        # Make sure it doesn't exist
        assert not os.path.exists(dataset_path)

        with patch("download_dataset.request.urlretrieve") as mock_urlretrieve, \
             patch("download_dataset.unzip_file") as mock_unzip:
            # Patch out printing if we want to keep test logs clean
            # but it's optional, so let's skip mocking prints for now.

            # We also pass the name of the function from the global dictionary
            # but we can't directly control the dictionary path unless we adjust code:
            # We'll pass dataset_name. The code will build the path using the dict.

            # We have to temporarily patch NAME_URL_DICT_UCI if we want to test a custom URL
            # but let's just use the existing URL from the dictionary.

            # Now call the function
            os.chdir(tmpdir)  # So that the function tries to create the folder here
            download_from_uci(dataset_name)

            # The folder should now exist
            assert os.path.exists(dataset_path), f"{dataset_path} should be created"

            # The .zip file should be placed there
            zip_filepath = os.path.join(dataset_path, f"{dataset_name}.zip")
            assert os.path.exists(zip_filepath), f"{zip_filepath} should be created"

            # Check if urlretrieve was called once with the expected URL, 
            # and that the second argument is the correct local zip path.
            expected_url = NAME_URL_DICT_UCI[dataset_name]
            mock_urlretrieve.assert_called_once_with(
                expected_url,
                zip_filepath
            )

            # unzip_file should be called once to extract the same zip
            mock_unzip.assert_called_once_with(zip_filepath, dataset_path)


def test_download_from_uci_already_exists():
    """
    Test that if the dataset folder already exists, 
    'download_from_uci' prints 'Aready downloaded.' and doesn't try to download again.
    """
    dataset_name = "adult"  # example from the dict
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = os.path.join(tmpdir, dataset_name)
        os.makedirs(dataset_path)

        # Now call download_from_uci again with the same name,
        # but the folder already exists => it should skip.
        with patch("download_dataset.request.urlretrieve") as mock_urlretrieve, \
             patch("download_dataset.unzip_file") as mock_unzip:
            os.chdir(tmpdir)
            download_from_uci(dataset_name)

            # Should NOT download or unzip again
            mock_urlretrieve.assert_not_called()
            mock_unzip.assert_not_called()
