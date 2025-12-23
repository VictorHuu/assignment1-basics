import torch
import math
from jaxtyping import Bool, Float, Int
from torch import Tensor

from . import rmsnorm
from . import mhas_attention
from . import embedding
from . import linear

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

def run_rmsnorm(
    d_model: int,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
    eps: float=1e-5
) -> Float[Tensor, " ... d_model"]:
    my_rmsnorm = rmsnorm.RMSNorm(d_model=d_model,eps=eps)

    my_rmsnorm.load_state_dict({
        "weights":weights
    })
    return my_rmsnorm(in_features)

def multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    d_in = q_proj_weight.shape[1]
    mha_sa= mhas_attention.MultiheadSelfAttention(d_model=d_model,
    num_heads=num_heads,d_in=d_in)
    mha_sa.load_state_dict({
        "q_proj_weight":q_proj_weight,
        "k_proj_weight":k_proj_weight,
        "v_proj_weight":v_proj_weight,
        "o_proj_weight":o_proj_weight,
    })

    return mha_sa(in_features)

def multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    d_in = q_proj_weight.shape[1]
    mha_sa= mhas_attention.MultiheadSelfAttention(d_model,num_heads,d_in,
    theta=theta,max_seq_len=max_seq_len,token_positions=token_positions)
    mha_sa.load_state_dict({
        "q_proj_weight":q_proj_weight,
        "k_proj_weight":k_proj_weight,
        "v_proj_weight":v_proj_weight,
        "o_proj_weight":o_proj_weight,
    })

    return mha_sa(in_features)

def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:

    my_embedding = embedding.Embedding(num_embeddings=vocab_size,embedding_dim=d_model)

    my_embedding.load_state_dict({
        "weights":weights
    })
    return my_embedding(token_ids)

def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    d_ff = (d_ff +63) // 64 * 64
    y = in_features @ w1_weight.t()
    z = y * torch.sigmoid(y)
    w = in_features @ w3_weight.t()
    return (z*w) @ w2_weight.t()

def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    d_ff = (d_ff +63) // 64 * 64
    seq_len = in_features.shape[-2]
    token_positions = torch.arange(seq_len, device=in_features.device)
    n1 = run_rmsnorm(d_model=d_model,weights=weights['ln1.weight'],in_features=in_features)
    attn_output = multihead_self_attention_with_rope(d_model,num_heads,
        q_proj_weight=weights['attn.q_proj.weight'],
        k_proj_weight=weights['attn.k_proj.weight'],
        v_proj_weight=weights['attn.v_proj.weight'],
        o_proj_weight=weights['attn.output_proj.weight'],
        in_features=n1, theta=theta, max_seq_len=max_seq_len,token_positions=token_positions
    )
    tmp = attn_output + in_features
    n2 = run_rmsnorm(d_model=d_model,weights=weights['ln2.weight'],in_features=tmp)
    ffn_output = run_swiglu(d_model=d_model,d_ff=d_ff,
        w1_weight=weights['ffn.w1.weight'],
        w2_weight=weights['ffn.w2.weight'],
        w3_weight=weights['ffn.w3.weight'],
        in_features=n2)
    return ffn_output + tmp

def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:

    my_linear = linear.Linear(in_features=d_in,out_features=d_out)

    my_linear.load_state_dict({
        "weights":weights
    })
    return my_linear(in_features)