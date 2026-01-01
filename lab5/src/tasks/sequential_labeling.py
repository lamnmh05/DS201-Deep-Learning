import torch
from torch.utils.data import DataLoader

from src.data.phonert import PhoNert
from src.models.transformer import TransformerForSeqLabeling
from src.tasks.base_task import BaseTask


class SequentialLabelingTask(BaseTask):

    def __init__(self, vocab, train_path, val_path, test_path,
                 model, checkpoint_path, lr=1e-3):

        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.model = model
        self.checkpoint_path = checkpoint_path
        self.lr = lr

        super().__init__(vocab, model, checkpoint_path)

    def load_datasets(self):
        self.train_dataset = PhoNert(self.train_path, self.vocab)
        self.val_dataset = PhoNert(self.val_path, self.vocab)
        self.test_dataset = PhoNert(self.test_path, self.vocab)

    def create_dataloaders(self):
        self.train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True,
                                       collate_fn=self.train_dataset.collate_fn)
        self.val_loader = DataLoader(self.val_dataset, batch_size=32,
                                     collate_fn=self.val_dataset.collate_fn)
        self.test_loader = DataLoader(self.test_dataset, batch_size=32,
                                      collate_fn=self.test_dataset.collate_fn)

    def forward_batch(self, batch):
        input_ids = batch["input_ids"]
        labels = batch["labels"]

        loss, logits = self.model(input_ids, labels)
        preds = logits.argmax(dim=-1)

        mask = (labels != -100)
        true = labels[mask].tolist()
        pred = preds[mask].tolist()

        return loss, true, pred
