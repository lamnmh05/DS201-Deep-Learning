import torch
import torch.nn as nn

from src.data.vocab import Vocab
from .encoder import Encoder
from .decoder import Decoder

class LuongAttention(nn.Module):
    def __init__(self, hidden_dim, attention_type="general"):
        super().__init__()
        self.attention_type = attention_type

        if attention_type == "general":
            self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        elif attention_type == "dot":
            self.W = None
        else:
            raise ValueError("attention_type must be 'dot' or 'general'")

    def forward(self, dec_output, enc_outputs, mask=None):
        """
        dec_output: (batch, hidden_dim)
        enc_outputs: (batch, src_len, hidden_dim)
        mask: (batch, src_len)
        """

        if self.attention_type == "general":
            dec_output = self.W(dec_output)

        # score = ht · hs
        scores = torch.bmm(
            enc_outputs,
            dec_output.unsqueeze(2)
        ).squeeze(2)  # (batch, src_len)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = torch.softmax(scores, dim=1)

        context = torch.bmm(
            attn_weights.unsqueeze(1),
            enc_outputs
        ).squeeze(1)  # (batch, hidden_dim)

        return context, attn_weights


class Seq2Seq_Luong(nn.Module):
    def __init__(
        self,
        vocab: Vocab,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        attention_type: str = "general"
    ):
        super().__init__()
        self.vocab = vocab
        self.hidden_dim = hidden_dim

        self.encoder = Encoder(
            vocab, embedding_dim, hidden_dim, num_layers, dropout
        )

        self.decoder = Decoder(
            vocab, embedding_dim, hidden_dim, num_layers, dropout
        )

        self.attention = LuongAttention(
            hidden_dim=hidden_dim,
            attention_type=attention_type
        )

        # h~t = tanh(Wc [ht ; ct])
        self.concat = nn.Linear(
            hidden_dim * 2, hidden_dim
        )

        self.fc_out = nn.Linear(
            hidden_dim, vocab.tar_vocab_size
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
            )
            # decoder_output: (batch, 1, hidden_dim)

            ht = hidden[-1]
            
            context, _ = self.attention(
                ht, encoder_outputs, mask
            )

            combined = torch.cat([ht, context], dim=1)
            h_tilde = torch.tanh(self.concat(combined))

            logits = self.fc_out(h_tilde)
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
            encoder_outputs, hidden, cell = self.encoder(src_ids)
            mask = self._create_mask(src_ids)

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

                ht = hidden[-1]

                context, _ = self.attention(
                    ht, encoder_outputs, mask
                )

                combined = torch.cat([ht, context], dim=1)
                h_tilde = torch.tanh(self.concat(combined))

                logits = self.fc_out(h_tilde)
                next_token = logits.argmax(dim=-1, keepdim=True)

                predictions.append(next_token)

                finished |= (next_token.squeeze(1) == self.vocab.eos_id)
                if finished.all():
                    break

                input = next_token

            predictions = torch.cat(predictions, dim=1)

        return predictions
