import torch
from torch import Tensor
from torch.nn import functional as F, __init__
from torch.nn import Module
from torch.nn.parameter import Parameter
from jaxtyping import Bool, Float, Int
from . import utility
from . import rope
class MultiheadSelfAttention(Module):

    __constants__ = ["d_model","num_heads"]
    d_model: int
    num_heads: int

    def __init__(
        self,
        d_model: int,
        num_heads:int,
        d_in:int,
        theta:float=0,
        max_seq_len:int=0,
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
    )->None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_in = d_in
        self.theta= theta
        self.max_seq_len = max_seq_len
        self.token_positions = token_positions
        self.q_proj_weight = Parameter(torch.empty((d_model,d_in)))
        self.k_proj_weight = Parameter(torch.empty((d_model,d_in)))
        self.v_proj_weight = Parameter(torch.empty((d_model,d_in)))
        self.o_proj_weight = Parameter(torch.empty((d_model,d_model)))
        torch.nn.init.trunc_normal_(self.q_proj_weight)
        torch.nn.init.trunc_normal_(self.k_proj_weight)
        torch.nn.init.trunc_normal_(self.v_proj_weight)
        torch.nn.init.trunc_normal_(self.o_proj_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = x @ self.q_proj_weight.T
        k = x @ self.k_proj_weight.T
        v = x @ self.v_proj_weight.T
        q = q.view(*q.shape[:-1],self.num_heads,-1).transpose(-3,-2)
        k = k.view(*k.shape[:-1],self.num_heads,-1).transpose(-3,-2)
        v = v.view(*v.shape[:-1],self.num_heads,-1).transpose(-3,-2)
        if self.theta !=0 and self.max_seq_len!=0:
            my_rope = rope.RoPE(self.theta,q.shape[-1],self.max_seq_len)
            q = my_rope(q,self.token_positions)
            my_rope2 = rope.RoPE(self.theta,k.shape[-1],self.max_seq_len)
            k = my_rope(k,self.token_positions)
        seq_len = x.shape[-2]
        mask = torch.tril(torch.ones(seq_len,seq_len),0).to(torch.bool)
        context = utility.scaled_dot_product_attention(q,k,v,mask)
        context = context.transpose(-3,-2).contiguous()
        context = context.view(*context.shape[:-2],-1)
        return context @ self.o_proj_weight.T





