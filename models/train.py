from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from dataset.dataloader import train_loader, test_loader
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
class_weights = [total_samples / class_counts.get(i, 1) for i in range(num_classes)]
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

model.eval()
test_loss = 0.0
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_loss /= len(test_loader)
print(f"\nTest Loss: {test_loss:.4f}")

accuracy = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {accuracy:.4f}")

cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:")
print(cm)

target_names = ["hazardous", "non_hazardous"]
report = classification_report(all_labels, all_preds, target_names=target_names)
print("Classification Report:")
print(report)
