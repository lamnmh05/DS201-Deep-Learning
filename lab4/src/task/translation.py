import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from typing import Optional
import os
from tqdm import tqdm
from rouge import Rouge

from src.utils.logging import setup_logger
from src.data.vocab import Vocab
from src.data.PhoMT import PhoMT

class Translation:
    def __init__(
        self,
        vocab: Vocab,
        model: nn.Module,
        train_path: str,
        dev_path: Optional[str] = None,
        test_path: Optional[str] = None,
        logger=None,
        checkpoint_path: str = "checkpoints",
        lr: float = 1e-3,
        batch_size: int = 32
    ):
        self.logger = logger if logger is not None else setup_logger(output=checkpoint_path)
        self.checkpoint_path = checkpoint_path
        os.makedirs(self.checkpoint_path, exist_ok=True)

        self.logger.info("Building vocabulary")
        self.vocab = vocab

        self.logger.info("Loading datasets")
        self.train_dataset = PhoMT(train_path, self.vocab)
        self.val_dataset = PhoMT(dev_path, self.vocab) if dev_path else None
        self.test_dataset = PhoMT(test_path, self.vocab) if test_path else None

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.train_dataset.collate_fn
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            collate_fn=self.val_dataset.collate_fn
        ) if self.val_dataset else None

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            collate_fn=self.test_dataset.collate_fn
        ) if self.test_dataset else None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        self.logger.info(f"Using device: {self.device}")

        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.vocab.pad_id)

        self.rouge = Rouge()


    def forward_batch(self, batch):
        """
        Forward one batch
        Returns:
            loss: torch.Tensor
            references: List[str]
            predictions: List[str]
        """
        src_ids = batch["src_ids"].to(self.device)
        tar_ids = batch["tar_ids"].to(self.device)

        loss = self.model(src_ids, tar_ids)

        # pred_ids = self.model.predict(src_ids)

        if not self.model.training:
            pred_ids = self.model.predict(src_ids)
        else:
            pred_ids = None

        references = []
        predictions = []

        for i in range(src_ids.size(0)):
            ref = self.vocab.decode_sentence(tar_ids[i].tolist(), self.vocab.tar_lang)
            # pred = self.vocab.decode_sentence(pred_ids[i].tolist(), self.vocab.tar_lang)
            pred = "" if pred_ids is None else \
            self.vocab.decode_sentence(pred_ids[i].tolist(), self.vocab.tar_lang)


            references.append(ref)
            predictions.append(pred)
            

        return loss, references, predictions


    def evaluate(self, dataloader, desc="Evaluating"):
        self.model.eval()
        total_loss = 0
        rouge_l_scores = []

        pbar = tqdm(dataloader, desc=desc, ncols=90)

        with torch.no_grad():
            for batch in pbar:
                loss, refs, preds = self.forward_batch(batch)
                total_loss += loss.item()

                scores = self.rouge.get_scores(preds, refs, avg=True)
                rouge_l = scores["rouge-l"]["f"]
                rouge_l_scores.append(rouge_l)

                avg_loss = total_loss / (pbar.n + 1)
                avg_rouge = sum(rouge_l_scores) / len(rouge_l_scores)

                pbar.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    rougeL=f"{avg_rouge:.4f}"
                )

        return avg_loss, sum(rouge_l_scores) / len(rouge_l_scores)


    def train(self, epochs=20, patience=5):
        best_rouge = 0
        patience_counter = 0
        
        model_name = self.model.__class__.__name__
        save_path = os.path.join(
            self.checkpoint_path,
            f"best_model_{model_name}.pt"
        )

        for epoch in range(1, epochs + 1):
            self.model.train()
            running_loss = 0

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{epochs}", ncols=90)

            for batch in pbar:
                self.optimizer.zero_grad()

                loss, _, _ = self.forward_batch(batch)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                avg_loss = running_loss / (pbar.n + 1)
                pbar.set_postfix(loss=f"{avg_loss:.4f}")

            val_loss, val_rouge = self.evaluate(self.val_loader, desc="Validating")

            self.logger.info(
                f"[Epoch {epoch}] Val Loss={val_loss:.4f} | ROUGE-L={val_rouge:.4f}"
            )

            if val_rouge > best_rouge:
                best_rouge = val_rouge
                patience_counter = 0
                torch.save(self.model.state_dict(), save_path)
                self.logger.info(f"New BEST model saved (ROUGE-L={best_rouge:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    self.logger.info("Early stopping triggered.")
                    break

        self.model.load_state_dict(torch.load(save_path, map_location=self.device))

    def test(self):
        return self.evaluate(self.test_loader, desc="Testing")
