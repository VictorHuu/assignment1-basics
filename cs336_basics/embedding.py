import torch
from torch import Tensor
from torch.nn import functional as F, __init__
from torch.nn import Module
from torch.nn.parameter import Parameter

class Embedding(Module):

    __constants__ = ["num_embeddings","embedding_dim"]
    num_embeddings: int
    embedding_dim: int
    weights: Tensor

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        device=None,
        dtype=None
    )->None:
        factory_kwargs = {"device":device,"dtype":dtype}
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weights = Parameter(
            torch.empty((self.num_embeddings,self.embedding_dim),**factory_kwargs)
        )
        torch.nn.init.trunc_normal_(self.weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights[x]
            
