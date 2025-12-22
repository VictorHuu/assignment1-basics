import torch
import math
from torch import Tensor
from torch.nn import functional as F, __init__
from torch.nn import Module
from torch.nn.parameter import Parameter

class RMSNorm(Module):

    __constants__ = ["d_model","eps"]
    d_model: int
    eps: float
    weights: Tensor

    def __init__(
        self,
        d_model,
        eps = 1e-5,
        device: torch.device | None=None,
        dtype:torch.dtype| None=None
    )->None:
        factory_kwargs = {"device":device,"dtype":dtype}
        super().__init__()
        self.d_model= d_model
        self.eps = eps
        self.weights = Parameter(
            torch.empty(self.d_model,**factory_kwargs)
        )
        torch.nn.init.ones_(self.weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x**2,-1,keepdim=True)+self.eps)
        y = x / rms * self.weights
        return y.to(in_dtype)
