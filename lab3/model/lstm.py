import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from data.vocab import Vocab

class LSTM(nn.Module):
    def __init__(self, vocab: Vocab, embed_dim = 256, hidden_dim =256, num_layers = 5):
        super().__init__()
        self.vocab = vocab

        self.embedding = nn.Embedding(
            num_embeddings=vocab.vocab_size,
            embedding_dim=embed_dim,
            padding_idx=vocab.pad_id
        )
        self.model = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.classifier = nn.Linear(
            in_features=hidden_dim,
            out_features=vocab.num_labels
        )

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids: torch.Tensor, labels=None):
        lengths = (input_ids != self.vocab.pad_id).sum(dim=1)
        embed = self.embedding(input_ids)

        packed = pack_padded_sequence(
            input=embed,
            lengths=lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        packed_out, (h_n, c_n) = self.model(packed)

        last_hidden = h_n[-1]

        logits = self.classifier(last_hidden)

        if labels is not None:
            loss = self.loss_fn(logits, labels.view(-1))
            return loss, logits
        
        return logits

    