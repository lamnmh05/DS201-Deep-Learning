import torch
import torch.nn as nn

from src.data.vocab import Vocab


class Decoder(nn.Module):
    def __init__(
        self,
        vocab: Vocab,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.3
    ):
        super().__init__()
        self.vocab = vocab

        self.embedding = nn.Embedding(
            vocab.tar_vocab_size,
            embedding_dim,
            padding_idx=vocab.pad_id
        )

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.fc_out = nn.Linear(hidden_dim, vocab.tar_vocab_size)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden: torch.Tensor,
        cell: torch.Tensor,
        tar_id: torch.Tensor,
    ):

        # (batch, seq_len, emb_dim)
        embed = self.dropout(self.embedding(tar_id))

        # LSTM decoding
        output, (hidden, cell) = self.lstm(embed, (hidden, cell))

        # Project to vocab
        logits = self.fc_out(output)
        # logits: (batch, seq_len, vocab_size)

        return logits, hidden, cell

