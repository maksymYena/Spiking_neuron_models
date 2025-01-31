import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class HazardDataset(Dataset):
    def __init__(self, images_folder, annotations_file, transform=None):
        self.images_folder = images_folder
        with open(annotations_file, 'r') as f:
            data = json.load(f)

        # Создаём словарь {image_id: file_name}
        self.image_data = {img['id']: img['file_name'] for img in data['images']}

        dataset_files = set(os.listdir(images_folder))

        self.annotations = [
            ann for ann in data['annotations']
            if self.image_data[ann['image_id']] in dataset_files
        ]


        self.image_ids = list(set(ann['image_id'] for ann in self.annotations))
        self.transform = transform

        print(f"Loaded {len(self.annotations)} annotations for {images_folder}")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_filename = self.image_data[image_id]
        image_path = os.path.join(self.images_folder, image_filename)

        if not os.path.exists(image_path):
            return torch.zeros((3, 256, 256)), 0

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            try:
                for i, t in enumerate(self.transform.transforms):
                    image = t(image)
            except Exception as e:
                print(f"ERROR applying transform {i + 1}: {t}")
                print(f"Exception: {e}")
                print(f"Image type before transform: {type(image)}, Size: {image.size}")

                image = transforms.Resize((256, 256))(image)
                image = transforms.ToTensor()(image)

        else:
            image = transforms.Resize((256, 256))(image)
            image = transforms.ToTensor()(image)

        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Expected image to be a Tensor, but got {type(image)}")

        label = next((ann["category_id"] for ann in self.annotations if ann["image_id"] == image_id), 0)
        return image, label
