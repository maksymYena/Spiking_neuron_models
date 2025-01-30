import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from dataset.dataloader import test_loader
from models.model import DomainAdaptationModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DomainAdaptationModel().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device, weights_only=True))
model.eval()

with open("models/losses.txt", "r") as f:
    losses = [float(line.strip()) for line in f.readlines()]

epochs = list(range(1, len(losses) + 1))

plt.figure(figsize=(8, 5))
plt.plot(epochs, losses, marker='o', linestyle='-', color='b', label='Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss per Epoch")
plt.legend()
plt.grid()
plt.show()

y_true = []
y_pred = []

for images, labels in test_loader:
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    predictions = torch.argmax(outputs, dim=1)
    y_true.extend(labels.cpu().numpy())
    y_pred.extend(predictions.cpu().numpy())

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Hazardous", "Hazardous"])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

noise_levels = np.linspace(0, 0.5, 10)
accuracies = []

for noise in noise_levels:
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        noisy_images = images + torch.randn_like(images) * noise
        noisy_images = torch.clamp(noisy_images, 0, 1)
        outputs = model(noisy_images)
        predictions = torch.argmax(outputs, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    accuracies.append(correct / total)

plt.figure(figsize=(8, 5))
plt.plot(noise_levels, accuracies, marker='o', linestyle='-', color='r', label='Accuracy')
plt.xlabel("Noise Level")
plt.ylabel("Accuracy")
plt.title("Model Accuracy vs Noise Level")
plt.legend()
plt.grid()
plt.show()
