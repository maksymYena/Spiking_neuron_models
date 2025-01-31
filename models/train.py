from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim

from dataset.dataloader import train_loader
from models.model import DomainAdaptationModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DomainAdaptationModel().to(device)

train_labels = []
for _, labels in train_loader:
    train_labels.extend(labels.cpu().numpy())

class_counts = Counter(train_labels)
print("Class distribution in dataset:", Counter([label for _, label in train_loader.dataset]))
total_samples = sum(class_counts.values())
num_classes = 2
class_weights = [total_samples / class_counts.get(i + 1, 1) for i in range(num_classes)]
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

print(f"Class weights: {class_weights}")

# Создаём `weight`
class_weights = torch.tensor(class_weights, dtype=torch.float, device=device)
print(f"Class weights: {class_weights}")

criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

num_epochs = 10
losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        labels = labels - 1

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    losses.append(avg_loss)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), "model.pth")

with open("losses.txt", "w") as f:
    for loss in losses:
        f.write(f"{loss}\n")

print("Training complete! Model saved as model.pth.")
print("Losses saved to losses.txt.")
