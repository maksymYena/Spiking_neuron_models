import json
import os
from datetime import datetime

import norse.torch as snn  # For spiking neuron layers
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import efficientnet_b0

# Paths and Parameters
BASE_PATH = "./hazard_detection_data"
DRONE_IMAGES = os.path.join(BASE_PATH, "drone/images")
DRONE_ANNOTATIONS = os.path.join(BASE_PATH, "drone/annotations/all_annotations.json")
CONFIDENCE_THRESHOLD = 0.8
BATCH_SIZE = 16
EPOCHS = 2  # Number of epochs
LEARNING_RATE = 0.005  # Adjusted learning rate
ALPHA = 0.5  # Alpha value for domain adaptation


def sync_annotation_before_running(annotations_file, images_folder):
    # Load existing annotations
    with open(annotations_file, 'r') as f:
        data = json.load(f)

    # Get current max image id
    current_max_id = max(image['id'] for image in data['images'])

    for img_file in os.listdir(images_folder):
        if img_file.endswith(('.jpeg', '.jpg', '.png')):  # Include the desired formats
            image_id = current_max_id + 1
            image_path = os.path.join(images_folder, img_file)

            # Get image dimensions
            with Image.open(image_path) as img:
                width, height = img.size

            # Create new image entry
            new_image_entry = {
                "id": image_id,
                "file_name": img_file,
                "width": width,
                "height": height,
                "date_captured": datetime.today().strftime('%Y-%m-%d %H:%M:%S')

            }

            # Append the new entry to the images list
            data['images'].append(new_image_entry)

            # Create new annotation entry (assuming all are hazardous objects, you can adjust this logic)
            new_annotation_entry = {
                "id": len(data['annotations']) + 1,
                "image_id": image_id,
                "bbox": [50, 30, 100, 150],
                "category_id": 1,
                "iscrowd": 0
            }
            data['annotations'].append(new_annotation_entry)

            # Update the max id
            current_max_id += 1

    # Save the updated annotations back to the file
    with open(annotations_file, 'w') as f:
        json.dump(data, f, indent=4)


# Run the update function
sync_annotation_before_running(DRONE_ANNOTATIONS, DRONE_IMAGES)


# Custom Dataset to Load Images and Labels
class HazardDataset(Dataset):
    def __init__(self, images_folder, annotations_file, transform=None):
        self.images_folder = images_folder
        self.annotations = self.load_annotations(annotations_file)
        self.transform = transform
        self.image_files = [f for f in os.listdir(self.images_folder) if f.endswith((".jpeg", ".jpg"))]

    def load_annotations(self, annotations_file):
        with open(annotations_file, 'r') as f:
            data = json.load(f)
        return {ann['image_id']: 1 if ann['category_id'] == 1 else 0 for ann in data['annotations']}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_filename = self.image_files[idx]
        image_path = os.path.join(self.images_folder, image_filename)
        image = Image.open(image_path).convert("RGB")
        label = self.annotations.get(int(image_filename.split('_')[1].split('.')[0]), 0)

        if self.transform:
            image = self.transform(image)

        return image, label


# Data Preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Load Dataset and Initialize DataLoader
dataset = HazardDataset(DRONE_IMAGES, DRONE_ANNOTATIONS, transform=transform)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


# Gradient Reversal Layer for Domain Adaptation
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(x, alpha=1.0):
    return GradReverse.apply(x, alpha)


# Model with Domain-Adaptation and Spiking Neurons
class DomainAdaptationSpikingModel(nn.Module):
    def __init__(self):
        super(DomainAdaptationSpikingModel, self).__init__()
        self.feature_extractor = efficientnet_b0(weights="IMAGENET1K_V1")
        feature_size = self.feature_extractor.classifier[1].in_features
        self.feature_extractor.classifier[1] = nn.Identity()

        # Label Classifier (Spiking neuron layer)
        self.spiking_layer = snn.LIFCell()
        self.label_classifier = nn.Linear(feature_size, 2)

        # Domain Classifier
        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_size, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, x, alpha=1.0, domain=False):
        features = self.feature_extractor(x)

        if domain:
            reversed_features = grad_reverse(features, alpha)
            return self.domain_classifier(reversed_features)

        # Spiking neuron layer for label classification
        spiked, _ = self.spiking_layer(features)
        return self.label_classifier(spiked)


# Instantiate Model, Loss Functions, and Optimizer
model = DomainAdaptationSpikingModel()
class_weights = torch.tensor([0.5, 2.0])  # Adjust weights based on class distribution
label_criterion = nn.CrossEntropyLoss(weight=class_weights)
domain_criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# Training Function with Domain-Adversarial Loss and Spiking Neurons
def train_domain_adaptation(model, source_loader, target_loader, label_criterion, domain_criterion, optimizer,
                            epochs=EPOCHS, alpha=ALPHA):
    model.train()
    for epoch in range(epochs):
        total_label_loss = 0.0
        total_domain_loss = 0.0

        source_iter = iter(source_loader)
        target_iter = iter(target_loader)

        for _ in range(len(source_loader)):
            # Source data
            source_data, source_labels = next(source_iter)
            source_outputs = model(source_data)
            label_loss = label_criterion(source_outputs, source_labels)

            # Target data (domain adaptation)
            target_data, _ = next(target_iter)
            domain_source = model(source_data, alpha=alpha, domain=True)
            domain_target = model(target_data, alpha=alpha, domain=True)
            domain_labels = torch.cat((torch.ones(len(source_data)), torch.zeros(len(target_data)))).long()

            domain_loss = domain_criterion(torch.cat((domain_source, domain_target)), domain_labels)

            # Backpropagation and optimization
            loss = label_loss + domain_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_label_loss += label_loss.item()
            total_domain_loss += domain_loss.item()

        print(
            f"Epoch {epoch + 1}/{epochs}, Label Loss: {total_label_loss / len(source_loader)}, Domain Loss: {total_domain_loss / len(source_loader)}")


train_domain_adaptation(model, data_loader, data_loader, label_criterion, domain_criterion, optimizer)


def classify_image(model, image_path, transform, threshold=CONFIDENCE_THRESHOLD):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        confidence, pred_class = torch.max(probabilities, 1)

        # Checking confidence threshold for classification
        if confidence.item() >= threshold:
            print(f"pred_class.item() = {pred_class.item()}")
            is_hazardous = pred_class.item() == 1  # Hazardous if class is 1 and confidence is above threshold
            label = "hazardous_object" if is_hazardous else "non_hazardous_object"
        else:
            # Default to "non_hazardous_object" if confidence is below threshold
            label = "non_hazardous_object"

    print(f"Image {image_path} classified as {label} with confidence {confidence.item():.2f}")
    return label, confidence.item()


# Folder Classification and Results Export
def classify_images_in_folder(model, folder_path, transform, threshold=CONFIDENCE_THRESHOLD):
    results = []
    for image_file in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_file)
        if image_file.endswith((".jpeg", ".jpg")):
            label, confidence = classify_image(model, image_path, transform, threshold)
            results.append({"File Name": image_file, "Confidence": confidence, "Label": label})
    return pd.DataFrame(results)


def update_annotations_after_running(results_df, annotations_file):
    with open(annotations_file, 'r') as f:
        data = json.load(f)

    for index, row in results_df.iterrows():
        file_name = row["File Name"]
        confidence = row["Confidence"]
        label = row["Label"]

        image_id = next((img["id"] for img in data["images"] if img["file_name"] == file_name), None)
        if image_id is not None:
            annotation = next((ann for ann in data["annotations"] if ann["image_id"] == image_id), None)
            if annotation:
                annotation["category_id"] = 1 if label == "hazardous_object" else 0
            else:
                data["annotations"].append({
                    "id": len(data["annotations"]) + 1,
                    "image_id": image_id,
                    "bbox": [0, 0, 0, 0],  # Можно настроить, если нужны координаты
                    "category_id": 1 if label == "hazardous_object" else 0,
                    "iscrowd": 0
                })

    with open(annotations_file, 'w') as f:
        json.dump(data, f, indent=4)


# Run Classification on the Folder and Print Results
results_df = classify_images_in_folder(model, DRONE_IMAGES, transform)
print(results_df)

update_annotations_after_running(results_df, DRONE_ANNOTATIONS)
