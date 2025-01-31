import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFilter, ImageEnhance

from models.model import DomainAdaptationModel


class NoiseRobustnessTester:
    def __init__(self, model_path, image_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DomainAdaptationModel().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()

        self.image_path = image_path
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def add_noise(self, image, noise_level):
        """ Добавляет различные искажения """
        image = image.copy()

        # Гауссовский шум
        if np.random.rand() > 0.5:
            noise = np.random.normal(0, noise_level * 50, np.array(image).shape)
            image = np.clip(np.array(image) + noise, 0, 255).astype(np.uint8)
            image = Image.fromarray(image)

        # Размытие
        if np.random.rand() > 0.5:
            image = image.filter(ImageFilter.GaussianBlur(radius=noise_level * 5))

        # JPEG-артефакты
        if np.random.rand() > 0.5:
            image.save("temp.jpg", "JPEG", quality=int(100 - noise_level * 90))
            image = Image.open("temp.jpg")

        # Контрастность
        if np.random.rand() > 0.5:
            contrast = 1.0 - noise_level
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(contrast)

        return image

    def test_noise_robustness(self):
        original_image = Image.open(self.image_path).convert("RGB")
        noise_levels = np.arange(0, 1.05, 0.1)  # От 0% до 100% с шагом 10%
        confidences = []

        plt.figure(figsize=(12, 6))
        for noise_level in noise_levels:
            noisy_image = self.add_noise(original_image, noise_level)
            transformed_image = self.transform(noisy_image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(transformed_image)
                probabilities = torch.softmax(output, dim=1).cpu().numpy().flatten()
                predicted_class = np.argmax(probabilities)
                confidence = np.max(probabilities)

            # Логирование предсказаний
            print(f"Noise: {int(noise_level * 100)}% | Class: {predicted_class} | Confidence: {confidence:.4f} | Probabilities: {probabilities}")

            confidences.append(confidence)

            plt.clf()
            plt.subplot(1, 2, 1)
            plt.imshow(noisy_image)
            plt.title(f"Noise: {int(noise_level * 100)}% | Confidence: {confidence:.2f}")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.plot(noise_levels[:len(confidences)] * 100, confidences, marker="o", linestyle="-", color="b")
            plt.xlabel("Noise Level (%)")
            plt.ylabel("Model Confidence")
            plt.title("Model Confidence vs. Noise Level")
            plt.grid(True)

            plt.pause(0.5)

        plt.show()


# Запуск теста
tester = NoiseRobustnessTester("model.pth", "./hazard_detection_data/drone/images/IMG_1.jpeg")
tester.test_noise_robustness()
