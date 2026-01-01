import os

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.metrics import f1_score
from tqdm import tqdm

from src.utils.logging import setup_logger


class BaseTask:
    def __init__(self, vocab, model, checkpoint_path):

        self.logger = setup_logger()

        self.checkpoint_path = checkpoint_path
        os.makedirs(self.checkpoint_path, exist_ok=True)

        self.logger.info("Creating vocab")
        self.vocab = vocab
        self.vocab.make_vocab()

        self.logger.info("Loading datasets & dataloaders")
        self.load_datasets()
        self.create_dataloaders()

        self.model = model

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        self.optimizer = Adam(self.model.parameters(), lr=self.lr)

    def load_datasets(self):
        raise NotImplementedError

    def create_dataloaders(self):
        raise NotImplementedError

    def forward_batch(self, batch):
        """Return loss, y_true, y_pred"""
        raise NotImplementedError


    def evaluate_metrics(self, dataloader, desc="Evaluating"):
        self.model.eval()
        total_loss = 0
        y_true_all = []
        y_pred_all = []

        pbar = tqdm(dataloader, desc=desc, ncols=90)

        with torch.no_grad():
            for batch in pbar:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                loss, y_true, y_pred = self.forward_batch(batch)
                total_loss += loss.item()

                y_true_all.extend(y_true)
                y_pred_all.extend(y_pred)

                avg_loss = total_loss / (pbar.n + 1)
                pbar.set_postfix(loss=f"{avg_loss:.4f}")

        avg_loss = total_loss / len(dataloader)
        macro_f1 = f1_score(y_true_all, y_pred_all, average="macro")

        return avg_loss, macro_f1


    def train(self, epochs=20, patience=5):
        save_model_path = os.path.join(self.checkpoint_path, 'best_model.pt')

        best_f1 = 0
        patience_counter = 0

        for epoch in range(1, epochs + 1):

            self.model.train()
            running_loss = 0

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{epochs}", ncols=90)

            for batch in pbar:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                loss, _, _ = self.forward_batch(batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                avg_loss = running_loss / (pbar.n + 1)
                pbar.set_postfix(loss=f"{avg_loss:.4f}")

            val_loss, val_f1 = self.evaluate_metrics(self.val_loader, desc="Validating")

            self.logger.info(
                f"[Epoch {epoch}] Val_loss={val_loss:.4f} | Val_Macro-F1={val_f1:.4f}"
            )

            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                torch.save(self.model.state_dict(), save_model_path)
                self.logger.info(f"New BEST model saved (F1={best_f1:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    self.logger.info("Early stopping triggered.")
                    break

        # Load best model
        self.model.load_state_dict(torch.load(save_model_path, map_location=self.device))


    def test(self):
        return self.evaluate_metrics(self.test_loader, desc="Testing")


    # def get_predictions(self, dataset):
    #     loader = DataLoader(dataset, batch_size=32, collate_fn=dataset.collate_fn)

    #     self.model.eval()
    #     preds_out = []

    #     pbar = tqdm(loader, desc="Predicting", ncols=90)

    #     with torch.no_grad():
    #         for batch in pbar:
    #             batch = {k: v.to(self.device) for k, v in batch.items()}
    #             _, _, y_pred = self.forward_batch(batch)
    #             preds_out.extend(y_pred)

    #     return preds_out
