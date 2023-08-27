import torch

def expectile_regression_loss(
    X: torch.Tensor, Y: torch.Tensor, tau: float
) -> torch.Tensor:
    assert 0 <= tau <= 1
    diff = Y - X
    return torch.mean(torch.max(tau * diff**2, (tau - 1) * diff**2))

def quantile_regression_loss(
    X: torch.Tensor, Y: torch.Tensor, tau: float
) -> torch.Tensor:
    assert 0 <= tau <= 1
    diff = Y - X
    return torch.mean(torch.max(tau * diff, (tau - 1) * diff))

def log_ratio_square_loss(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    return torch.mean((torch.log(X) - torch.log(Y)) ** 2)

def square_loss(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    return torch.mean((X - Y) ** 2)

