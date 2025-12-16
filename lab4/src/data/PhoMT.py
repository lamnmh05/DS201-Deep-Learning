import torch
from torch.utils.data import Dataset, DataLoader
import json

from src.data.vocab import Vocab

class PhoMT(Dataset):
    def __init__(self, path: str, vocab: Vocab):
        self.data = json.load(open(path, 'r', encoding='utf-8'))
        self.vocab = vocab
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        src_sentences = item[self.vocab.src_lang]
        tar_sentences = item[self.vocab.tar_lang]

        src_id = self.vocab.encode_sentence(src_sentences, self.vocab.src_lang)
        tar_id = self.vocab.encode_sentence(tar_sentences, self.vocab.tar_lang)

        return {
            "src_id": src_id,
            "tar_id": tar_id
        }
    
    @staticmethod
    def collate_fn(samples: list[dict]) -> dict[dict]:
        samples = {
            "src_ids": torch.stack([sample['src_id'] for sample in samples], dim=0),
            'tar_ids': torch.stack([sample['tar_id'] for sample in samples], dim=0)
        }

        return samples

import numpy as np
if __name__ == '__main__':
    # path = r'small-PhoMT\small-test.json'
    # vocab = Vocab(path, 100, 'vietnamese', 'english')
    # dataset = PhoMT(path, vocab)
    # loader = DataLoader(dataset, 16, collate_fn=dataset.collate_fn)

    # for i, item in enumerate(loader):
    #     print(item)
    #     raise

    data = json.load(open(r'small-PhoMT\small-train.json', encoding='utf-8'))

    src_len = []
    tar_len = []

    for item in data:
        src_len.append(len(item['english'].split()))
        tar_len.append(len(item['vietnamese'].split()))

    print("SRC (english)")
    print("  min :", np.min(src_len))
    print("  max :", np.max(src_len))
    print("  mean:", np.mean(src_len))

    print("\nTAR (vietnamese)")
    print("  min :", np.min(tar_len))
    print("  max :", np.max(tar_len))
    print("  mean:", np.mean(tar_len))

    print("SRC 95%:", np.percentile(src_len, 95))
    print("TAR 95%:", np.percentile(tar_len, 95))
