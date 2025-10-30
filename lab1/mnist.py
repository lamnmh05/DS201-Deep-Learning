import torch
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import idx2numpy

def collate_fn(items: list[dict]) -> dict[dict]:
    items = [
        {
            "image": np.expand_dims(item["image"], axis=0),
            "label": np.array(item["label"]),
        }
        for item in items
    ]

    items = {
        "image": np.stack([item["image"] for item in items], axis=0),
        "label": np.stack([item["label"] for item in items], axis=0),
    }

    items = {
        "image": torch.tensor(items["image"]),
        "label": torch.tensor(items["label"]),
    }

    return items

class MNISTDataset(Dataset):
    def __init__(self, img_path: str, label_path: str) -> None:
        # images_file = open(img_path, "r")
        # labels_file = open(label_path, "r")
        images = idx2numpy.convert_from_file(img_path)
        labels = idx2numpy.convert_from_file(label_path)

        self._data = [
            {
                "image": np.array(image, dtype=np.float32),
                "label": label,
            }
            for image, label in zip(images, labels)
        ]
    
    def __len__(self) -> int:
        return len(self._data)
    
    def __getitem__(self, idx: int) -> dict:
        return self._data[idx]
    

if __name__ == "__main__":
    train_dataset = MNISTDataset(
        img_path= r"lab1\mnist\train-images.idx3-ubyte",
        label_path= r"lab1\mnist\train-labels.idx1-ubyte",
    )
    test_dataset = MNISTDataset(
        img_path= r"lab1\mnist\t10k-images.idx3-ubyte",
        label_path= r"lab1\mnist\t10k-labels.idx1-ubyte",
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn,)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn,)

    for item in train_loader:
        print(item)
        break

    for item in test_loader:
        print(item)
        break