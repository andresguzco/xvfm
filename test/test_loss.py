"""Test `xvfm.loss`."""
import pytest
import torch

from xvfm.loss import GuassianMultinomial

@pytest.mark.parametrize("task_type", ["regression", "classification"])
@pytest.mark.parametrize("classes", [[], [3], [3, 2]])
def test_gaussian_multinomial_forward(task_type, classes):
    """
    Test the GuassianMultinomial forward method for different configurations
    of 'task_type' and 'classes'.
    """
    num_feat = 4
    batch_size = 5

    # Instantiate the loss module
    loss_module = GuassianMultinomial(num_feat=num_feat, classes=classes, task=task_type)

    # Create synthetic inputs:
    #   'res' shape must at least have 'num_feat' + sum(classes) + ...
    #   If task=regression => 1 output for the target
    #   If task=classification => 2 outputs for the target (as coded in the final else clause).
    #
    # So total_out = num_feat + sum(classes) + (1 if regression else 2)
    total_out = num_feat + sum(classes)
    if task_type == "regression":
        total_out += 1
    else:
        total_out += 2

    # 'res' simulates the model output (logits or means).
    # shape: [batch_size, total_out]
    res = torch.randn(batch_size, total_out)

    # 'x' shape: [batch_size, num_feat + len(classes) + 1]
    #   the last column is the "target" (regression or classification label)
    #   so total x columns = num_feat (continuous) + len(classes) (categorical) + 1 (target).
    x_dim = num_feat + len(classes) + 1
    x = torch.randn(batch_size, x_dim)

    # If we have classes, let's pretend the categorical columns are smaller discrete integers
    # so that they are valid category indices in [0, class_size).
    idx = num_feat
    for c in classes:
        if c > 0:
            x[:, idx] = torch.randint(low=0, high=c, size=(batch_size,))
            idx += 1

    # For classification, the last column is a discrete label
    # for regression, it's just a float. We'll keep it random or clamp if we want discrete
    if task_type == "classification":
        x[:, -1] = torch.randint(low=0, high=2, size=(batch_size,))

    # 't' is presumably a time/step scalar or vector
    # The code expects shape [batch_size]. We'll pass a 1D tensor for convenience.
    t = torch.rand(batch_size)

    # Now call the loss
    output = loss_module(res, x, t)

    # Check that we get a scalar (empty shape or shape [])
    assert output.shape == (), "Loss should be a scalar."

    # Also check it's finite
    assert torch.isfinite(output), "Loss should be a finite scalar."


def test_gaussian_multinomial_no_classes_regression():
    """
    Simple test with no classes and a regression task.
    """
    num_feat = 3
    batch_size = 4
    classes = []
    task_type = "regression"

    loss_module = GuassianMultinomial(num_feat, classes, task_type)

    # If no classes, then total_out = num_feat + 1 for regression
    total_out = num_feat + 1
    res = torch.randn(batch_size, total_out)

    # x => shape [batch_size, num_feat + 1]
    # last column is the regression target
    x_dim = num_feat + 1
    x = torch.randn(batch_size, x_dim)

    # t => shape [batch_size]
    t = torch.rand(batch_size)

    out = loss_module(res, x, t)
    assert out.shape == (), "Loss should be scalar for no classes + regression"
    assert torch.isfinite(out), "Loss should be finite."


def test_gaussian_multinomial_with_classes_classification():
    """
    Simple test with some classes and classification.
    """
    num_feat = 3
    classes = [4, 2]  # two categorical columns
    task_type = "classification"
    batch_size = 4

    loss_module = GuassianMultinomial(num_feat, classes, task_type)

    # total_out = num_feat + sum(classes) + 2 for classification
    total_out = num_feat + sum(classes) + 2
    res = torch.randn(batch_size, total_out)

    # x => shape [batch_size, num_feat + len(classes) + 1]
    x_dim = num_feat + len(classes) + 1
    x = torch.randn(batch_size, x_dim)

    # make sure categorical columns are integer
    # cat columns => x[:, 3] and x[:, 4] if num_feat=3 => the next 2 columns
    x[:, 3] = torch.randint(0, 4, size=(batch_size,))
    x[:, 4] = torch.randint(0, 2, size=(batch_size,))
    # last column is classification label => let's say 0/1
    x[:, -1] = torch.randint(0, 2, size=(batch_size,))

    t = torch.rand(batch_size)
    out = loss_module(res, x, t)

    assert out.shape == (), "Loss should be scalar for classification"
    assert torch.isfinite(out), "Loss should be finite."
