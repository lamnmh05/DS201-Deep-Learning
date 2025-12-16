import torch
import torch.nn as nn

from src.data.vocab import Vocab
from .encoder import Encoder
from .decoder import Decoder


class Seq2Seq(nn.Module):
    def __init__(
            self,
            vocab: Vocab,
            embedding_dim: int,
            hidden_dim: int,
            num_layers: int,
            dropout: float,
            ):
        super().__init__()
        self.vocab = vocab

        self.encoder = Encoder(
            vocab,
            embedding_dim,
            hidden_dim,
            num_layers,
            dropout
        )

        self.decoder = Decoder(
            vocab,
            embedding_dim,
            hidden_dim,
            num_layers,
            dropout
        )

        self.output = nn.Linear(hidden_dim, vocab.tar_vocab_size)

        self.loss = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)

    def forward(self, src_ids, tar_ids):
        batch_size, tar_len = tar_ids.shape
        vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, tar_len, vocab_size)

        _, hidden, cell = self.encoder(src_ids)

        input = tar_ids[:, 0].unsqueeze(1)

        for t in range(1, tar_len):
            output, hidden, cell = self.decoder(hidden, cell, input)
            outputs[:, t, :] = output.squeeze(1)
            input = tar_ids[:, t].unsqueeze(1)
            
        loss = self.loss(
            outputs[:, 1:].reshape(-1, vocab_size),
            tar_ids[:, 1:].reshape(-1)
        )

        return loss
    

    def predict(self, src_ids):
        self.eval()
        batch_size = src_ids.size(0)

        with torch.no_grad():
            # encoder
            _, hidden, cell = self.encoder(src_ids)

            # init sos
            input = torch.full(
                (batch_size, 1),
                self.vocab.sos_id,
                dtype=torch.long,
            )

            predictions = []
            finished = torch.zeros(batch_size, dtype=torch.bool)

            for _ in range(self.vocab.max_len):
                logits, hidden, cell = self.decoder(hidden, cell, input)
                # logits: (batch, 1, vocab_size)

                next_token = logits.argmax(dim=-1)  # (batch, 1)
                predictions.append(next_token)

                # check EOS
                finished |= (next_token.squeeze(1) == self.vocab.eos_id)
                if finished.all():
                    break

                # next input = predicted token
                input = next_token

            # (batch_size, gen_len)
            predictions = torch.cat(predictions, dim=1)

        return predictions
