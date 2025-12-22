import torch
from torch import Tensor
from torch.nn import Module

class RoPE(Module):
    """
    Rotary Positional Embedding (RoPE) Implementation.
    
    This implementation uses the 'Half-Split' approach (standard in Llama):
    The feature vector is split in two, and the rotation is applied across 
    corresponding halves.
    """
    __constants__ = ["theta", "d_k"]
    theta: float
    d_k: int

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None) -> None:
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        factory_kwargs = {"device": device}
        
        i_range = torch.arange(max_seq_len,**factory_kwargs).float()
        k_range = torch.arange(d_k//2,**factory_kwargs).float()

        inv_freq = torch.pow(theta,torch.div(2*k_range,-d_k))

        self.freq = torch.outer(i_range,inv_freq)

        self.register_buffer("sin_cached",torch.sin(self.freq),persistent=True)
        self.register_buffer("cos_cached",torch.cos(self.freq),persistent=True)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos_u = self.cos_cached[token_positions].type_as(x)
        sin_u = self.sin_cached[token_positions].type_as(x)

        cos_diag = torch.repeat_interleave(cos_u,2,dim=-1)
        R = torch.diag_embed(cos_diag)

        off_diag = torch.zeros_like(R)
        off_diag[...,0::2,1::2]=sin_u.diag_embed()
        off_diag[...,1::2,0::2]=-sin_u.diag_embed()
        R=R+off_diag
        return torch.matmul(x.unsqueeze(-2),R).squeeze(-2)