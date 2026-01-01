import json
import torch
from torch.utils.data import Dataset, DataLoader
from src.data.vocab import Vocab


class PhoNert(Dataset):
    def __init__(self, path: str, vocab: Vocab):
        super().__init__()

        self.path = path
        self.vocab = vocab

        self.data = json.load(open(path, 'r', encoding='utf-8'))
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]

        words = item['words']
        label = item['tags']

        encoded_sentence = self.vocab.encode_sentence(words)
        encoded_label = self.vocab.encode_label(label)

        length = (encoded_sentence != self.vocab.pad_id).sum()

        return {
            "input_ids" : encoded_sentence,
            "label" : encoded_label,
            "length": length,
        }
    

    @staticmethod
    def collate_fn(samples: list[dict]) -> dict[dict]:
        samples = {
            "input_ids": torch.stack([sample['input_ids'] for sample in samples], dim=0),
            'labels': torch.stack([sample['label'] for sample in samples], dim=0),
            "lengths": torch.stack([sample['length'] for sample in samples], dim=0)
        }

        return samples
    

if __name__ == '__main__':
    path = r'src\data\PhoNERT\test.json'
    dataset = PhoNert(path, Vocab(path, 'words', 'tags'))
    loader = DataLoader(dataset, 16, collate_fn=dataset.collate_fn)

    for i, item in enumerate(loader):
        print(item)
        raise