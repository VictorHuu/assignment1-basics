import torch
from torch import Tensor
from torch.nn import functional as F, __init__
from torch.nn import Module
from torch.nn.parameter import Parameter

class Linear(Module):

    __constants__ = ["in_features","out_features"]
    in_features: int
    out_features: int
    weights: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weights: Tensor | None=None,
        device=None,
        dtype=None,
    )->None:
        factory_kwargs = {"device":device,"dtype":dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(
            torch.empty((out_features,in_features),**factory_kwargs)
        )
        torch.nn.init.trunc_normal_(self.weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weights.T

