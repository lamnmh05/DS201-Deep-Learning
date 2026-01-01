import json
import torch
from torch.utils.data import Dataset, DataLoader

from src.data.vocab import Vocab



class UIT_ViOCD_Domain(Dataset):
    def __init__(self, path: str, vocab: Vocab):
        super().__init__()

        self._data = json.load(open(path, 'r', encoding='utf-8'))
        self._vocab = vocab
    
    def __len__(self) -> int:
        return len(self._data)
    
    def __getitem__(self, index: int):
        item = self._data[index]
        
        sentence = item['review']
        label = item['domain']

        encoded_sentence = self._vocab.encode_sentence(sentence)
        encoded_label = self._vocab.encode_label(label)

        length = (encoded_sentence != self._vocab.pad_id).sum()

        return {
            "input_id": encoded_sentence,
            "label": encoded_label,
            "length": length
        }
    
    @staticmethod
    def collate_fn(samples: list[dict]) -> dict[dict]:
        samples = {
            "input_ids": torch.stack([sample['input_id'] for sample in samples], dim=0),
            'labels': torch.stack([sample['label'] for sample in samples], dim=0),
            "lengths": torch.stack([sample['length'] for sample in samples], dim=0)
        }

        return samples
    

if __name__ == "__main__":
    path = r'src\data\UIT_ViOCD\test_preprocessed.json'
    vocab = Vocab(path, "review", "domain")
    data = UIT_ViOCD_Domain(path, vocab)
    loader = DataLoader(data, 16, collate_fn=data.collate_fn)

    for i, item in enumerate(loader):
        print(item)
        raise
