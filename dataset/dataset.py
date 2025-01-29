import torch
from PIL import Image
import os
import json
from torchvision import transforms
from torch.utils.data import Dataset


class HazardDataset(Dataset):
    def __init__(self, images_folder, annotations_file, transform=None):
        self.images_folder = images_folder
        with open(annotations_file, 'r') as f:
            data = json.load(f)
        self.image_data = {img['id']: img['file_name'] for img in data['images']}
        self.annotations = {ann['image_id']: 1 if ann['category_id'] == 1 else 0 for ann in data['annotations']}
        self.image_ids = list(self.image_data.keys())
        self.transform = transform

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

        label = self.annotations.get(image_id, 0)
        return image, label
