import os

import numpy as np
import cv2 as cv
import torch
from torch.utils.data import Dataset, DataLoader

def vinafood_collate_fn(samples: list[dict]) -> dict[dict]:
    items =  [
        {
            'image': sample['image'],
            'label': np.array(sample['label']),
        }
        for sample in samples 
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
    

class VinaFood21(Dataset):
    def __init__(self, path: str, image_size: tuple = (224,224)) -> None:
        super().__init__()
        
        self.image_size = image_size
        self.label2id = {}
        self.id2label = {}

        self.data: list[dict] = self.load_data(path)

    def load_data(self, path: str) -> list[dict]:
        data = []
        label_id = 0

        for folder in os.listdir(path):
            label = folder
            if label not in self.label2id:
                self.label2id[label] = label_id
                label_id += 1
            
            for image_file in os.listdir(os.path.join(path, folder)):
                image = cv.imread(os.path.join(path, folder, image_file))
                data.append(
                    {
                        'image': image,
                        'label': label,
                    }
                )

        self.id2label = {
            id: label
            for label, id in self.label2id.items()
        }
        return data


    def __len__(self) -> int:
        return len(self.data)


    def __getitem__(self, index: int) -> dict:
        item = self.data[index]

        image = item['image']
        label = item['label']

        image = cv.resize(image, self.image_size) # (h, w, 3)
        image = np.transpose(image, (2, 0, 1)) # (3, h, w)

        label_id = self.label2id[label]

        return {
            'image': image,
            'label': label_id
        }
    


if __name__ == '__main__':
    dummy_path = 'lab2\VinaFood21\dummy'

    dummy_data = VinaFood21(path=dummy_path, image_size=(26,26))
    dummy_loader = DataLoader(dummy_data, batch_size=2, shuffle=True, collate_fn=vinafood_collate_fn)

    for item in dummy_loader:
        print(item)
        break
