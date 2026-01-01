import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.vocab import Vocab 
from src.models.utils import padding_mask, masked_mean_pool

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))  # (1,max_len,d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        assert d_model % num_heads == 0
        self.h = num_heads
        self.dh = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        # (B,L,D) -> (B,H,L,Dh)
        B, L, D = x.shape
        return x.view(B, L, self.h, self.dh).transpose(1, 2)

    def _combine(self, x: torch.Tensor) -> torch.Tensor:
        # (B,H,L,Dh) -> (B,L,D)
        B, H, L, Dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, H * Dh)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None):
        Q = self._split(self.wq(q))
        K = self._split(self.wk(k))
        V = self._split(self.wv(v))

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.dh)  # (B,H,Lq,Lk)

        if key_padding_mask is not None:
            scores = scores.masked_fill(~key_padding_mask, -1e4)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = attn @ V
        out = self._combine(out)
        out = self.wo(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, head: int, d_ff: int, dropout: float):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, head, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dp = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.ln1(x + self.dp(self.attn(x, x, x, key_padding_mask)))
        x = self.ln2(x + self.dp(self.ff(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, d_model: int, head: int, n_layers: int, d_ff: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, head, d_ff, dropout) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, key_padding_mask)
        return x


class TransformerBackbone(nn.Module):
    def __init__(self, vocab, d_model: int, head: int, n_layers: int, d_ff: int, dropout: float):
        super().__init__()
        self.vocab = vocab
        self.pad_id = vocab.pad_id
        self.max_len = vocab.max_len
        self.d_model = d_model

        vocab_size = len(vocab) if hasattr(vocab, "__len__") else vocab.vocab_size
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=self.pad_id)
        self.pos = PositionalEncoding(d_model, dropout, max_len=self.max_len)
        self.enc = TransformerEncoder(d_model, head, n_layers, d_ff, dropout)
        self.dp = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor):
        mask = padding_mask(input_ids, self.pad_id)                 # (B,1,1,L)
        x = self.embed(input_ids) * math.sqrt(self.d_model)         # (B,L,D)
        x = self.dp(self.pos(x))
        x = self.enc(x, key_padding_mask=mask)
        return x  # (B,L,D)


class TransformerForClassification(nn.Module):
    def __init__(self, backbone: TransformerBackbone, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.d_model, num_classes)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        feats = self.backbone(input_ids)  # (B,L,D)
        pooled = masked_mean_pool(feats, input_ids, self.backbone.pad_id)
        logits = self.head(pooled)        # (B,C)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return loss, logits


class TransformerForSeqLabeling(nn.Module):
    def __init__(self, backbone: TransformerBackbone, num_tags: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.d_model, num_tags)

    def forward(self, input_ids: torch.Tensor, tags: torch.Tensor | None = None):
        feats = self.backbone(input_ids)  # (B,L,D)
        logits = self.head(feats)         # (B,L,T)

        loss = None
        if tags is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                tags.view(-1),
                ignore_index=-100
            )
            
        return loss, logits