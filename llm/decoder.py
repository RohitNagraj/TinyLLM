import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, head_dim, dropout_val=0.1):
        super().__init__()
        # head_dim = embed_dim // num_heads
        # This is causal attention
        self.head_dim = head_dim
        self.embed_dim = embed_dim

        self.key = nn.Linear(self.embed_dim, self.head_dim, bias=False)
        self.query = nn.Linear(self.embed_dim, self.head_dim, bias=False)
        self.value = nn.Linear(self.embed_dim, self.head_dim, bias=False)
        self.activation = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout_val)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        k = self.key(x)  # (batch_size, seq_len, head_dim)
        q = self.query(x)  # (batch_size, seq_len, head_dim)

        # Note that attention scores are Q.KT and attention weights are normalized and softmaxed attention scores.
        attn_scores = q @ k.transpose(2, 1)  # (batch_size, seq_len, seq_len)
        attn_weights = attn_scores/self.head_dim**-0.5
        lower_triangular_matrix = torch.tril(torch.ones(seq_len, seq_len))
        attn_weights.masked_fill(lower_triangular_matrix==0, float('-inf'))
        attn_weights = self.activation(attn_weights)
        attn_weights = self.dropout(attn_weights)

        v = self.value(x) # (batch_size, seq_len, head_dim)
        out = attn_weights@v # (batch_size, seq_len, head_dim)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, embed_dim, head_dim, attn_dropout_rate=0.1):
        super().__init__()
        self.attn_heads = nn.ModuleList([SelfAttention(embed_dim=embed_dim, head_dim=head_dim) for _ in range(num_heads)])
        self.projection = nn.Linear(num_heads * head_dim, embed_dim)
        self.dropout = nn.Dropout(attn_dropout_rate)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        out = torch.cat([attn_head(x) for attn_head in self.attn_heads], dim=-1)
        out = self.projection(out)
        out = self.dropout(out)
        return out
    
class VanillaFFN(nn.Module):
    """
    Vanilla FFN uses ReLU activation. This was part of the original transformers paper. 
    However, modern LLMs don't use this anymore.
    Idea: Project embed_dim to ffn_dim (typically 4 x embed_dim). Use activation and then down-project.
    """
    def __init__(self, embed_dim, hidden_dim, dropout):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        self.up_proj = nn.Module(embed_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.down_proj = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        up = self.activation(self.up_proj(x))
        down = self.dropout(self.down_proj(up))
        return down

class SwiGLU(nn.Module):
    """
    SwiGLU stands was Swish activation function + Gated Linear Unit (GLU).
    Idea of Gated Linear Unit:
    Gate proj acts as a gate on how much of the up_proj is passed through
        gate = X @ W_gate
        up = X @ W_up
        down = X @ W_down
        activation_output = swish(up)
        gate_activated = elementwise_multiply(gate, activation_output)
        output = gate_activated @ down
    """
    def __init__(self, embed_dim, hidden_dim, dropout):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.gate_proj = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SiLU()

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.activation(self.self.up_proj(x))
        gated_activation = gate * up
        down = self.down_proj(gated_activation)
        return down
    
class DecoderBlock((nn.Module)):
    def __init__(self, embed_dim, num_heads, attn_dropout_rate=0.1):
        self().__init__()
        
        # Config
        head_dim = embed_dim // num_heads
        hidden_dim = embed_dim * 4


        self.mha = MultiHeadAttention(num_heads=num_heads, embed_dim=embed_dim, head_dim=head_dim, attn_dropout_rate=attn_dropout_rate)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = SwiGLU(embed_dim=embed_dim, hidden_dim=hidden_dim, dropout=attn_dropout_rate)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        out = self.mha(x)
        out = self.norm1(out + x) # Note the residual connection
        out = self.ffn(x)
        return out