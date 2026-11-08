import torch


def test_tensor_creation():
    """Verify that PyTorch creates tensors correctly."""
    tensor = torch.tensor([1.0, 2.0, 3.0])

    assert tensor.shape == (3,)
    assert tensor.dtype == torch.float32

def test_matrix_multiplication():
    """Verify basic matrix multiplication."""
    a = torch.tensor([[1.0, 2.0]])
    b = torch.tensor([[3.0], [4.0]])

    result = torch.matmul(a, b)

    assert result.item() == 11.0

def test_autograd():
    """Verify that PyTorch calculates gradients."""
    x = torch.tensor(2.0, requires_grad=True)

    y = x**2
    y.backward()

    assert x.grad is not None
    assert x.grad.item() == 4.0