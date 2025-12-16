import torch
import torch.nn as nn
from typing import Tuple

from lab4.data.vocab import Vocab

class Seq2seqLSTM(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 n_encoder: int, 
                 n_decoder: int,
                 dropout: int, 
                 vocab: Vocab
    ):
        super().__init__()

        self.vocab = vocab
        self.n_encoder = n_encoder
        self.n_decoder = n_decoder

        self.src_embedding = nn.Embedding(
            num_embeddings=vocab.total_src_tokens, 
            embedding_dim=d_model, 
            padding_idx=vocab.pad_idx
        )
        
        self.encoder = nn.LSTM(
            input_size = d_model, 
            hidden_size=d_model, 
            num_layers=n_encoder, 
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        self.tgt_embedding = nn.Embedding(
            num_embeddings=vocab.total_tgt_tokens, 
            embedding_dim=2*d_model, 
            padding_idx=vocab.pad_idx
        )

        self.attn_weights = nn.Linear(
            in_features=4*d_model,
            out_features=1
        )

        self.decoder = nn.LSTM(
            input_size=2*d_model,
            hidden_size=2*d_model,
            num_layers=n_decoder, 
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )
        
        self.output_head = nn.Linear(
            in_features=2*d_model,
            out_features=vocab.total_tgt_tokens
        )
        self.loss = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.train()
        embedded_x = self.src_embedding(x)
        bs, l, dim = embedded_x.shape
        
        hidden_states = []
        for ith in range(l):
            _, (hidden_state, _) = self.encoder(embedded_x[:, ith].unsqueeze(1))
            hidden_state = hidden_state.reshape(self.n_encoder, 2, bs, -1) # (layers, 2, bs, dim)
            hidden_state = hidden_state.permute(0, 2, 1, -1).reshape(self.n_encoder, bs, 1, -1) # (layers, bs, 2, dim) -> (layers, bs, 1, 2*dim)
            hidden_states.append(hidden_state)

        # teacher-forcing mechanism
        enc_hidden_states: torch.Tensor = torch.cat(hidden_states, dim=2) # (layers, bs, l, 2*dim)
        enc_hidden_states = enc_hidden_states[:self.n_decoder]
        _, tgt_len = y.shape
        dec_hidden_state = torch.zeros((self.n_decoder, bs, 2*dim)).to(x.device)
        logits = []
        for ith in range(tgt_len):
            y_ith = y[:, ith].unsqueeze(-1)
            dec_hidden_state = self.forward_step(y_ith, enc_hidden_states, dec_hidden_state)
            # get the last hidden states
            last_hidden_states = dec_hidden_state[-1] # (bs, dim)
            logit = self.output_head(last_hidden_states)
            logits.append(logit.unsqueeze(1))
        
        logits = torch.cat(logits, dim=1)
        
        loss = self.loss(logits.reshape(-1, self.vocab.total_tgt_tokens), y.reshape(-1, ))

        return loss
    
    def aligning(self, query: torch.Tensor, k_v: torch.Tensor):
        '''
            query: (layers, bs, dim)
            k_v: (layers, bs, len, dim)
        '''
        _, _, l, _ = k_v.shape
        query = query.unsqueeze(2).repeat(1, 1, l, 1) # (layers, bs, len, dim)

        a = self.attn_weights(
            torch.cat([query, k_v], dim=-1)
        ) # (layers, bs, len, 1)
        a = nn.functional.softmax(a, dim=-2) # (layers, bs, len, 1)
        cell_mem = (a * k_v).sum(-2) # (layers, bs, dim)

        return cell_mem
    
    def forward_step(
            self, 
            input_ids: torch.Tensor, 
            enc_hidden_states: torch.Tensor,
            dec_hidden_state: torch.Tensor
        ):

        embedded_input = self.tgt_embedding(input_ids)
        cell_mem = self.aligning(dec_hidden_state, enc_hidden_states) # (layers, bs, dim)
        _, (dec_hidden_state, cell_mem) = self.decoder(embedded_input, (dec_hidden_state, cell_mem))

        return dec_hidden_state.contiguous()

    def predict(self, x: torch.Tensor):
        self.eval()
        embedded_x = self.src_embedding(x)
        bs, l, _ = embedded_x.shape
        
        hidden_states = []
        for ith in range(l):
            _, (hidden_state, _) = self.encoder(embedded_x[:, ith].unsqueeze(1))
            hidden_state = hidden_state.reshape(self.n_encoder, 2, bs, -1) # (layers, 2, bs, dim)
            hidden_state = hidden_state.permute(0, 2, 1, -1).reshape(self.n_encoder, bs, 1, -1) # (layers, bs, 2, dim) -> (layers, bs, 1, 2*dim)
            hidden_states.append(hidden_state)

        enc_hidden_states: torch.Tensor = torch.cat(hidden_states, dim=2) # (layers, bs, l, 2*dim)
        enc_hidden_states = enc_hidden_states[:self.n_decoder]
        y_ith = torch.zeros(bs, ).fill_(self.vocab.bos_idx).long().to(x.device) # (bs, 1)
        mark_eos = torch.zeros_like(y_ith).bool()
        outputs = []
        while True:
            dec_hidden_state = self.forward_step(y_ith, enc_hidden_states, dec_hidden_state)
            # get the last hidden states
            last_hidden_states = dec_hidden_state[-1] # (bs, dim)
            logit = self.output_head(last_hidden_states)
            
            y_ith = logit.argmax(dim=-1).long() # (bs, )
            mark_eos = (y_ith == self.vocab.eos_idx)
            
            if all(mark_eos.tolist()):
                break

            outputs.append(y_ith.unsqueeze(-1))

        outputs = torch.cat(outputs, dim=-1)
        
        return outputs # (bs, length)

