import json
import torch
from torch.utils.data import Dataset, DataLoader

from data.vocab import Vocab


class UIT_VSFC(Dataset):
    def __init__(self, path:str, label_key: str, vocab: Vocab) -> None:
        super().__init__()

        self.path = path
        self.label_key = label_key
        self.vocab = vocab

        self.data = json.load(open(path, 'r', encoding='utf-8'))
    

    def __len__(self) -> int:
        return len(self.data)


    def __getitem__(self, index: int):
        item = self.data[index]
        sentence = item['sentence']
        label = item[self.label_key]

        encoded_sentence = self.vocab.encode_sentence(sentence)
        encoded_label = self.vocab.encode_label(label)

        return {
            "input_ids" : encoded_sentence,
            "label" : encoded_label
        }
    
    @staticmethod
    def collate_fn(samples: list[dict]) -> dict[dict]:
        samples = {
            "input_ids": torch.stack([sample['input_ids'] for sample in samples], dim=0),
            'label': torch.stack([sample['label'] for sample in samples], dim=0)
        }

        return samples

if __name__ == '__main__':
    path = r'UIT-VSFC\UIT-VSFC-test.json'
    dataset = UIT_VSFC(path, 'topic', Vocab(path, 'sentence', 'topic'))
    loader = DataLoader(dataset, 16, collate_fn=dataset.collate_fn)

    for i, item in enumerate(loader):
        print(item)
        raise
