import json
import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class HazardDataset(Dataset):
    def __init__(self, images_folder, annotations_file, transform=None):
        self.images_folder = images_folder
        with open(annotations_file, 'r') as f:
            data = json.load(f)

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
            raise FileNotFoundError(f"Image found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                print("Error during applying the transform:", e)
                image = transforms.Resize((256, 256))(image)
                image = transforms.ToTensor()(image)
        else:
            image = transforms.Resize((256, 256))(image)
            image = transforms.ToTensor()(image)

        label = next((ann["category_id"] for ann in self.annotations if ann["image_id"] == image_id), None)
        if label is None:
            raise ValueError(f"No annotation detected for  image_id {image_id}")
        label = label - 1
        return image, label
