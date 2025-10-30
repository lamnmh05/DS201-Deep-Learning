import os

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


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
    def __init__(self, root_path: str, image_size: tuple = (224,224)) -> None:
        super().__init__()
        
        self.image_size = image_size
        self.image_file_paths = []
        self.labels = []

        self.label2id = {}
        self.id2label = {}

        folders = os.listdir(root_path)

        for idx, folder in enumerate(folders):
            label = folder
            if label not in self.label2id:
                self.label2id[label] = idx
            
            folder_path = os.path.join(root_path, folder)
            image_files = os.listdir(folder_path)

            for image_file in image_files:
                image_path = os.path.join(folder_path, image_file)
                self.image_file_paths.append(image_path)
                self.labels.append(self.label2id[label])
                
        self.id2label = {
            id: label
            for label, id in self.label2id.items()
        }

        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),        # Convert to (C, H, W) with [0,1]
        ])

    def __len__(self) -> int:
        return len(self.image_file_paths)

    def __getitem__(self, index: int) -> dict:

        image_path = self.image_file_paths[index]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        label = self.labels[index]
        label = torch.tensor(self.labels[index], dtype=torch.long)


        return {
            'image': image,
            'label': label
        }
    


if __name__ == '__main__':
    dummy_path = 'lab2\VinaFood21\dummy'

    dummy_data = VinaFood21(root_path=dummy_path, image_size=(26,26))
    dummy_loader = DataLoader(dummy_data, batch_size=2, shuffle=True, collate_fn=vinafood_collate_fn)

    for item in dummy_loader:
        print(item)
        break
