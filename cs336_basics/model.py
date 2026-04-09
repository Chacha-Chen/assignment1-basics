import torch 
import torch.nn as nn
from jaxtyping import Float
from einops import einsum, rearrange

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device= None, dtype=None):
        super().__init__()
        # in_features: int final dimension of the input
        # out_features: int final dimension of the output
        # device: torch.device | None = None Device to store the parameters on
        # dtype: torch.dtype | None = None Data type of the parameters
        
        std = (2/(in_features+out_features))**0.5 # do i need to convert to float type? 
        self.W = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype)) # how to specify size and init with zero? 
        nn.init.trunc_normal_(self.W, mean = 0, std = std, a=-3*std, b = 3*std)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # y = Wx (d_out, d_in) * (...., d_in)
        return einsum(self.W, x, "d_out d_in, ... d_in -> ... d_out ")

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        # num_embeddings: int Size of the vocabulary
        # embedding_dim: int Dimension of the embedding vectors, i.e., dmodel
        # device: torch.device | None = None Device to store the parameters on
        # dtype: torch.dtype | None = None Data type of the parameters
        
        # (vocab_size, d_model) 
        self.embedding_layer = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.embedding_layer, mean=0, std=1, a=-3, b=3)

        # forward：给你一堆 token ID（整数），你只需要从矩阵里按行取出对应的向量。想想怎么用 Python 索引做到这件事？                                                                
        # 比如一个矩阵 W shape (10, 4)，token_ids = [2, 5, 7]，那 W[token_ids] 返回什么？                                                                 

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # Lookup the embedding vectors for the given token IDs.
        # take embedding vector with indices from token_ids

        return self.embedding_layer[token_ids]
    
class RMSnorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weights = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.d_model = d_model

    def forward(self, a):
        # a size [..., d_model]
        in_dtype = a.dtype
        a = a.to(torch.float32)
        scaler = a/(torch.sqrt(torch.mean(a**2, dim=-1, keepdim=True) + self.eps))

        return (scaler*self.weights).to(in_dtype)
    

def SiLU(in_features):
    return in_features * torch.sigmoid(in_features)


class SwiGLU(nn.Module):
    def __init__(self,d_model, d_ff):
        super().__init__()
        # d_model (int): Dimensionality of the feedforward input and output.
        # d_ff (int): Dimensionality of the up-project happening internally to your swiglu.

        self.w1 = Linear(d_model, d_ff)
        self.w3 = Linear(d_model, d_ff)

        self.w2 = Linear(d_ff, d_model)
    
    def forward(self, x):
        x = SiLU(self.w1(x))*(self.w3(x)) # 点积是*吗？
        return self.w2(x)

class rope(nn.Module):
    def __init__(self,):
        pass

    def forward(self,):
        pass


def softmax(x, dim):
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x = x-x_max
    x_exp = torch.exp(x)
    return x_exp/torch.sum(x_exp, dim=dim, keepdim=True)

def scaled_dot_product_attention(Q,K,V,mask=None):
    # Q (Float[Tensor, " ... queries d_k"]): Query tensor
    # K (Float[Tensor, " ... keys d_k"]): Key tensor
    # V (Float[Tensor, " ... values d_v"]): Values tensor
    # mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor

    # softmax(Q@K.T/sqrt(d_k))@V
    # B T d_k, B d_k T -> B T T 
    attn = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
    d_k = Q.shape[-1]
    attn = attn/d_k**0.5

    # apply mask
    # mask = torch.triu(torch.ones()) # ones or ones_like here?
    if mask is not None:
        attn = attn.masked_fill(mask == False, float('-inf'))

    attn = softmax(attn, dim=-1)
    attn = einsum(attn, V, "... queries keys, ... keys d_v -> ... queries d_v")
    return attn

class MultiheadSelfAttention(nn.Module):
    def __init__(self,d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model// num_heads
        self.d_model = d_model
        self.Q = Linear(d_model, d_model)
        self.K = Linear(d_model, d_model)
        self.V = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)

    def forward(self,in_features):
        # in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        
        q = self.Q(in_features)
        k = self.K(in_features)
        v = self.V(in_features)

        q = rearrange(q, "... seq_len (heads d_k) -> ... heads seq_len d_k", heads = self.num_heads)
        k = rearrange(k, "... seq_len (heads d_k) -> ... heads seq_len d_k", heads = self.num_heads)
        v = rearrange(v, "... seq_len (heads d_k) -> ... heads seq_len d_k", heads = self.num_heads)

        seq_len = q.shape[-2]
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1) == 0 # upper triangle is 1 but i need to fill them with False?
        
        attn = scaled_dot_product_attention(q, k, v, mask)

        attn = rearrange(attn,"... heads seq_len d_k -> ... seq_len (heads d_k)")

        return self.output_proj(attn)