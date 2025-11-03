import torch
import torch.nn as nn
import torch.nn.functional as F
from .decoder import DecoderBlock


class TinyLLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, block_size, attn_dropout_rate=0.1):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = nn.Embedding(block_size, embed_dim) # Notice how you pass block_size
        self.decoder_blocks = nn.Sequential(*[DecoderBlock(embed_dim, num_heads, attn_dropout_rate) for _ in range(num_layers)])
        self.ln = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    
    def forward(self, idx, targets=None):
        batch_size, seq_len = idx.shape
        tok_emb = self.token_embedding(idx) # (bs, seq) -> (bs, seq, embed_dim)
        pos_emb = self.positional_embedding(torch.arange(seq_len)) # (seq_len, embed_dim)
        x = tok_emb + pos_emb

        x = self.decoder_blocks(x)
        x = self.ln(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            bs, seq, vocab_size = logits.shape
            logits = logits.view(bs*seq, vocab_size)
            targets = targets.view(bs*seq)
            loss = F.cross_entropy(logits, targets)

        return logits, loss