"""Test `xvfm.models`."""
import pytest
import torch

# Import everything from models.py for testing
from xvfm.models import (
    MLP,
    MultiMLP,
    Tabformer,
    MultiHeadAttention,
    ColumnEncoder,
    ColumnDecoder,
    Tabby
)


###############################################################################
# TEST MLP
###############################################################################
def test_mlp_init():
    """Test that MLP can be initialized without errors."""
    model = MLP(
        d_in=4,             # input dimension
        d_layers=[8, 8],    # hidden layers
        dropouts=0.1,       # dropout rate
        d_out=6,            # output dimension
        num_feat=3,
        classes=[],
        task="regression",
    )
    assert isinstance(model, MLP), "MLP should be created successfully"


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_mlp_forward(task):
    """Test MLP forward pass for both regression and classification tasks."""
    model = MLP(
        d_in=5,
        d_layers=[10, 10],
        dropouts=0.2,
        d_out=8,
        num_feat=3,
        classes=[3, 4] if task == "classification" else [],
        task=task,
    )
    x = torch.randn(2, 5)  # batch_size=2, d_in=5
    out = model(x)
    assert out.shape == (2, 8), "Output should have shape (batch_size, d_out)"

    # If classification, last columns should be exponentiated probabilities
    if task == "classification":
        # For the code as written, sum(classes) > 0 => modifies certain columns
        # We won't deeply check correctness of softmax, just shape or positivity
        assert torch.all(out[:, -1] > 0), "Last column in classification output must be > 0"
    else:
        # Regression => last column is just the raw value
        pass


def test_mlp_make_baseline():
    """Test the make_baseline classmethod to ensure it returns an MLP instance."""
    model = MLP.make_baseline(
        d_in=3, d_layers=[5, 5], dropout=0.1,
        d_out=4, num_feat=2, classes=[], task="regression"
    )
    assert isinstance(model, MLP), "make_baseline should return an MLP instance"


###############################################################################
# TEST MultiMLP
###############################################################################
def test_multi_mlp_forward():
    """Test that MultiMLP can run a forward pass without errors."""
    model = MultiMLP(d_in=6, num_feat=3, classes=[2], task="classification")
    x = torch.randn(4, 6)  # batch_size=4, d_in=6
    t = torch.tensor([0.1, 0.2, 0.3, 0.4]).unsqueeze(1)  # shape (4, 1)
    out = model(x, t)
    assert out.shape == (4, 6), "Output should have shape (batch_size, d_in)"
    # For classification, last columns should be exponentiated probabilities
    assert torch.all(out[:, -1] > 0), "Last column in classification output must be > 0"


###############################################################################
# TEST Tabformer
###############################################################################
def test_tabformer_init_and_forward():
    """Test that Tabformer can be created and runs forward without errors."""
    model = Tabformer(
        d_in=6,
        classes=[3, 4],
        num_feat=2,
        task="classification",
        d_layer=2,
        n_layer=2,
        dropout=0.1,
        dim_t=16
    )
    x = torch.randn(2, 6)    # batch_size=2, d_in=6
    t = torch.tensor([0.5, 0.7]).unsqueeze(1)  # (2,1)
    out = model(x, t)
    assert out.shape == (2, 6), "Output should match (batch_size, d_in)"
    # Check classification columns are > 0
    assert torch.all(out[:, -1] > 0), "Classification output last column must be > 0"


###############################################################################
# TEST MultiHeadAttention
###############################################################################
def test_multi_head_attention_forward():
    """Test that MultiHeadAttention can process data without errors."""
    d_model = 8
    batch_size = 2
    seq_len = 3  # We'll treat x as [B, seq_len, d_model]
    mha_block = MultiHeadAttention(d_model=d_model, n_heads=2, dropout=0.1, ff_factor=2)
    x = torch.randn(batch_size, seq_len, d_model)
    out = mha_block(x)
    assert out.shape == (batch_size, seq_len, d_model), (
        "Output should have shape [batch_size, seq_len, d_model]"
    )


###############################################################################
# TEST ColumnEncoder & ColumnDecoder
###############################################################################
def test_column_encoder_decoder():
    """Test that ColumnEncoder and ColumnDecoder can encode and decode without errors."""
    num_cols = 5
    d_col = 4
    encoder = ColumnEncoder(num_cols, d_col)
    decoder = ColumnDecoder(num_cols, d_col)
    x = torch.randn(2, num_cols)  # batch_size=2
    encoded = encoder(x)
    # encoded should have shape [2, num_cols * 4], but from the code:
    #   it returns cat of each linear, which is shape [2, num_cols*d_col]? 
    # Actually the code returns [2, 5, 4] => we need to check carefully.
    # ColumnEncoder: returns torch.cat of each col encoding on dim=1,
    # each col after linear => shape [2,1,d_col].
    # So cat(...) => shape [2, 5, d_col].
    # We'll check that shape.
    assert encoded.shape == (2, num_cols, d_col), (
        "Encoder should return [batch_size, num_cols, d_col]"
    )

    # The decoder expects a shape of [batch_size, num_cols*d_col], since it
    # slices along i*(d_col) : ...
    # But the code's forward loops "X[:, i * self.d : (i + 1) * self.d]" => shape [batch_size, d_col]
    # So let's flatten the encoded output for the decoder:
    encoded_flat = encoded.view(2, num_cols * d_col)
    decoded = decoder(encoded_flat)
    # The decoder merges them back into shape [2, num_cols], each col => 1 dimension
    assert decoded.shape == (2, num_cols), "ColumnDecoder should return [batch_size, num_cols]"


###############################################################################
# TEST Tabby
###############################################################################
@pytest.mark.parametrize("task", ["regression", "classification"])
def test_tabby_forward(task):
    """Test Tabby forward pass for regression or classification."""
    d_in = 6
    model = Tabby(
        d_in=d_in,
        num_feat=3,
        classes=[2, 3] if task == "classification" else [],
        task=task,
        n_layers=2,
        d_model=2,
        n_heads=2,
        dropout=0.1,
        ff_factor=2,
        d_t=32,  # must be > d_in * d_model = 6*2=12
    )
    x = torch.randn(3, d_in)  # batch_size=3
    t = torch.tensor([0.1, 0.2, 0.3])
    out = model(x, t)

    assert out.shape == (3, d_in), "Tabby output should match (batch_size, d_in)"
    if task == "classification":
        # Check that last columns are probabilities
        # Here, we see that code sets `res[:, -2:] = ...` for classification
        # We'll just check positivity as a sanity check
        assert torch.all(out[:, -1] > 0), "Last column in classification output must be > 0"
    else:
        # For regression, it's just the raw value in out[:, -1]
        pass
