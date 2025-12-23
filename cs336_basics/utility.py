import torch
import math
from jaxtyping import Bool, Float, Int
from torch import Tensor

def softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    M = torch.max(in_features,dim,keepdim=True)[0]
    x_stable = in_features-M
    S = torch.exp(x_stable).sum(dim=dim,keepdim=True)
    return torch.div(torch.exp(x_stable),S)

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = Q.shape[-1]
    reciprocal = math.sqrt(d_k)
    x= Q @ K.mT/reciprocal
    if mask is not None:
        x= x.masked_fill(mask==False,float("-inf"))
    return softmax(x,-1) @ V