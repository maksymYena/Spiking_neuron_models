import json
import os
from datetime import datetime
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset.dataset import HazardDataset
from models.model import DomainAdaptationModel
from torchvision import transforms

# Paths and Parameters
BASE_PATH = "./hazard_detection_data"
DRONE_IMAGES = os.path.join(BASE_PATH, "drone/images")
DRONE_ANNOTATIONS = os.path.join(BASE_PATH, "drone/annotations/all_annotations.json")
CONFIDENCE_THRESHOLD = 0.5
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 0.0005
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sync_annotation_before_running(annotations_file, images_folder):
    with open(annotations_file, 'r') as f:
        data = json.load(f)

    existing_files = {img["file_name"] for img in data["images"]}
    max_id = max([img["id"] for img in data["images"]], default=0)

    for img_file in os.listdir(images_folder):
        if img_file.endswith(('.jpeg', '.jpg', '.png')) and img_file not in existing_files:
            max_id += 1
            image_path = os.path.join(images_folder, img_file)
            data['images'].append({
                "id": max_id,
                "file_name": img_file,
                "date_captured": datetime.today().strftime('%Y-%m-%d %H:%M:%S')
            })
            data['annotations'].append({
                "id": len(data['annotations']) + 1,
                "image_id": max_id,
                "category_id": 1
            })

    with open(annotations_file, 'w') as f:
        json.dump(data, f, indent=4)


sync_annotation_before_running(DRONE_ANNOTATIONS, DRONE_IMAGES)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0, inplace=False),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


dataset = HazardDataset(DRONE_IMAGES, DRONE_ANNOTATIONS, transform=transform)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = DomainAdaptationModel().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_model(model, data_loader, torch.nn.CrossEntropyLoss(), optimizer, EPOCHS)

torch.save(model.state_dict(), "model.pth")
