import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from dataset.dataloader import test_loader
from models.model import DomainAdaptationModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DomainAdaptationModel().to(device)

# Загрузка обученных весов
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# Читаем историю потерь из файла
with open("models/losses.txt", "r") as f:
    losses = [float(line.strip()) for line in f.readlines()]

epochs = list(range(1, len(losses) + 1))

# График изменения Loss по эпохам
plt.figure(figsize=(8, 5))
plt.plot(epochs, losses, marker='o', linestyle='-', color='b', label='Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss per Epoch")
plt.legend()
plt.grid()
plt.show()

# Подсчёт confusion matrix на "чистых" данных (без шума)
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
plt.title("Confusion Matrix (No Noise)")
plt.show()

# Разные уровни шума: от 0 до 0.5 (10 точек)
noise_levels = np.linspace(0, 0.5, 10)
accuracies = []

for noise in noise_levels:
    correct = 0
    total = 0
    # Прогоняем весь test_loader при данном noise
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        # Генерируем *случайную* интенсивность шума для каждого пикселя
        # (можно сгенерировать для каждого изображения, см. комментарий ниже)
        random_scales = torch.rand_like(images) * noise
        # Добавляем шум: гауссовский шум * random_scales
        noisy_images = images + torch.randn_like(images) * random_scales

        # Зажимаем диапазон [0, 1]
        noisy_images = torch.clamp(noisy_images, 0, 1)

        outputs = model(noisy_images)
        predictions = torch.argmax(outputs, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    accuracy = correct / total
    accuracies.append(accuracy)

# График "Accuracy vs Noise Level"
plt.figure(figsize=(8, 5))
plt.plot(noise_levels, accuracies, marker='o', linestyle='-', color='r', label='Accuracy')
plt.xlabel("Max Noise Level")
plt.ylabel("Accuracy")
plt.title("Model Accuracy vs Noise Level (Random per Pixel)")
plt.legend()
plt.grid()
plt.show()
