import os

from torch.utils.data import DataLoader
from torchvision import transforms

from dataset.dataset import HazardDataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
train_images_folder = os.path.join(BASE_DIR, "../hazard_detection_data/drone/images/train")
test_images_folder = os.path.join(BASE_DIR, "../hazard_detection_data/drone/images/test")
annotations_file = os.path.join(BASE_DIR, "../hazard_detection_data/drone/annotations/all_annotations.json")

train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
])

test_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

train_dataset = HazardDataset(train_images_folder, annotations_file, transform=train_transforms)
test_dataset = HazardDataset(test_images_folder, annotations_file, transform=test_transforms)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
