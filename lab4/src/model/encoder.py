import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.data.vocab import Vocab

class Encoder(nn.Module):
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
            vocab.src_vocab_size,
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

        self.dropout = nn.Dropout(dropout)

    def forward(self, src_ids: torch.Tensor):
        max_id = src_ids.max().item()
        if max_id >= self.embedding.num_embeddings:
            raise ValueError(
                f"Token id {max_id} out of range "
                f"(vocab_size={self.embedding.num_embeddings})"
            )
        
        lengths = (src_ids != self.vocab.pad_id).sum(dim=1).long()

        embed = self.dropout(self.embedding(src_ids))

        packed = pack_padded_sequence(
            embed,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        packed_out, (hidden, cell) = self.lstm(packed)

        encoder_output, _ = pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=src_ids.size(1)
        )

        return encoder_output, hidden, cell

