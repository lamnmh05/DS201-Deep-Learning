import torch
from torch.utils.data import DataLoader

from data.uit_vsfc import UIT_VSFC
from model import lstm
from tasks.base_task import BaseTask


class TextClassificationTask(BaseTask):

    def __init__(self, vocab, train_path, val_path, test_path,
                 model, checkpoint_path, lr=1e-3):

        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.model = model
        self.label_key = vocab.label_key
        self.checkpoint_path = checkpoint_path
        self.lr = lr

        super().__init__(vocab, model, checkpoint_path)

    def load_datasets(self):
        self.train_dataset = UIT_VSFC(self.train_path, self.label_key, self.vocab)
        self.val_dataset = UIT_VSFC(self.val_path, self.label_key, self.vocab)
        self.test_dataset = UIT_VSFC(self.test_path, self.label_key, self.vocab)

    def create_dataloaders(self):
        self.train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True,
                                       collate_fn=self.train_dataset.collate_fn)
        self.val_loader = DataLoader(self.val_dataset, batch_size=32,
                                     collate_fn=self.val_dataset.collate_fn)
        self.test_loader = DataLoader(self.test_dataset, batch_size=32,
                                      collate_fn=self.test_dataset.collate_fn)


    def forward_batch(self, batch):
        input_ids = batch["input_ids"]
        labels = batch["label"].view(-1)

        loss, logits = self.model(input_ids, labels)   

        preds = logits.argmax(dim=-1).tolist()  
        true = labels.tolist()                  

        return loss, true, preds

