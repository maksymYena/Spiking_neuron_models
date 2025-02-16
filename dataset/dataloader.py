import json
import os
from collections import Counter
import random

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset.dataset import HazardDataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
images_folder = os.path.join(BASE_DIR, "../hazard_detection_data/drone/images/")
annotations_file = os.path.join(BASE_DIR, "../hazard_detection_data/drone/annotations/all_annotations_modified.json")

transforms_pipeline = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.clamp(x + torch.randn_like(x) * random.uniform(0, 0.1), 0, 1)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

with open(annotations_file, "r") as f:
    data = json.load(f)

image_id_to_category = {ann["image_id"]: ann["category_id"] for ann in data["annotations"]}

hazardous_ids = [img["id"] for img in data["images"] if image_id_to_category.get(img["id"], 0) == 1]
non_hazardous_ids = [img["id"] for img in data["images"] if image_id_to_category.get(img["id"], 0) == 2]

hazardous_train, hazardous_test = train_test_split(hazardous_ids, test_size=1, random_state=42)
non_hazardous_train, non_hazardous_test = train_test_split(non_hazardous_ids, test_size=1, random_state=42)

train_ids = set(hazardous_train + non_hazardous_train)
test_ids = set(hazardous_test + non_hazardous_test)

dataset = HazardDataset(images_folder, annotations_file, transform=transforms_pipeline)

train_dataset = [dataset[i] for i in range(len(dataset)) if dataset.image_ids[i] in train_ids]
test_dataset = [dataset[i] for i in range(len(dataset)) if dataset.image_ids[i] in test_ids]

from torch.utils.data import WeightedRandomSampler

train_labels = [label for _, label in train_dataset]
class_sample_counts = [train_labels.count(0), train_labels.count(1)]
weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)
samples_weight = torch.tensor([weights[label] for label in train_labels])
sampler = WeightedRandomSampler(samples_weight, num_samples=len(samples_weight), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

train_labels = [label for _, label in train_dataset]
test_labels = [label for _, label in test_dataset]

class_counts_train = Counter(train_labels)
class_counts_test = Counter(test_labels)

print(f"✅ Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
print(f"📊 Class distribution in train dataset: {class_counts_train}")
print(f"📊 Class distribution in test dataset: {class_counts_test}")
