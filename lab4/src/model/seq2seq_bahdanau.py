import torch
import torch.nn as nn

from src.data.vocab import Vocab
from .encoder import Encoder
from .decoder import Decoder

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W_enc = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_dec = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, dec_hidden, enc_outputs, mask=None):
        src_len = enc_outputs.size(1)
        dec_hidden = dec_hidden.unsqueeze(1).repeat(1, src_len, 1)

        energy = torch.tanh(
            self.W_enc(enc_outputs) + self.W_dec(dec_hidden)
        )

        scores = self.v(energy).squeeze(-1)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = torch.softmax(scores, dim=1)

        context = torch.bmm(
            attn_weights.unsqueeze(1),
            enc_outputs
        ).squeeze(1)

        return context


class Seq2Seq_Bahdanau(nn.Module):
    def __init__(
        self,
        vocab,
        embedding_dim,
        hidden_dim,
        num_layers,
        dropout,
    ):
        super().__init__()
        self.vocab = vocab

        self.encoder = Encoder(
            vocab, embedding_dim, hidden_dim, num_layers, dropout
        )

        self.decoder = Decoder(
            vocab, embedding_dim, hidden_dim, num_layers, dropout
        )

        self.attention = BahdanauAttention(hidden_dim)

        self.fc_out = nn.Linear(
            hidden_dim * 2, vocab.tar_vocab_size
        )

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)

    def _create_mask(self, src_ids):
        return (src_ids != self.vocab.pad_id)

    def forward(self, src_ids, tar_ids):
        batch_size, tar_len = tar_ids.shape
        vocab_size = self.vocab.tar_vocab_size

        outputs = torch.zeros(
            batch_size, tar_len, vocab_size,
            device=src_ids.device
        )

        encoder_outputs, hidden, cell = self.encoder(src_ids)
        mask = self._create_mask(src_ids)

        input = tar_ids[:, 0].unsqueeze(1)

        for t in range(1, tar_len):
            decoder_output, hidden, cell = self.decoder(
                hidden, cell, input
            )  # (batch, 1, hidden_dim)

            dec_hidden = hidden[-1]

            context = self.attention(
                dec_hidden, encoder_outputs, mask
            )

            combined = torch.cat(
                [dec_hidden, context],  # hidden + context
                dim=1
            )

            logits = self.fc_out(combined)
            outputs[:, t, :] = logits

            input = tar_ids[:, t].unsqueeze(1)

        loss = self.loss_fn(
            outputs[:, 1:].reshape(-1, vocab_size),
            tar_ids[:, 1:].reshape(-1)
        )

        return loss
    
    def predict(self, src_ids):
        self.eval()
        batch_size = src_ids.size(0)
        device = src_ids.device

        with torch.no_grad():
            # Encode
            encoder_outputs, hidden, cell = self.encoder(src_ids)
            mask = (src_ids != self.vocab.pad_id)

            # Init with <sos>
            input = torch.full(
                (batch_size, 1),
                self.vocab.sos_id,
                dtype=torch.long,
                device=device
            )

            predictions = []
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

            for _ in range(self.vocab.max_len):
                decoder_output, hidden, cell = self.decoder(
                    hidden, cell, input
                )

                dec_hidden = hidden[-1]  # (batch, hidden_dim)

                context = self.attention(
                    dec_hidden, encoder_outputs, mask
                )

                combined = torch.cat(
                    [dec_hidden, context],
                    dim=1
                )

                logits = self.fc_out(combined)
                next_token = logits.argmax(dim=-1, keepdim=True)

                predictions.append(next_token)

                finished |= (next_token.squeeze(1) == self.vocab.eos_id)
                if finished.all():
                    break

                input = next_token

            predictions = torch.cat(predictions, dim=1)

        return predictions

