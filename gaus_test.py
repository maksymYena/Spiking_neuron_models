import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

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
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def add_noise(self, image, noise_level):
        image = np.array(image).astype(np.float32) / 255.0
        noise = np.random.normal(0, noise_level, image.shape)
        noisy_image = np.clip(image + noise, 0, 1)
        return (noisy_image * 255).astype(np.uint8)

    def test_noise_robustness(self):
        original_image = Image.open(self.image_path).convert("RGB")
        noise_levels = np.arange(0, 1.05, 0.05)
        confidences = []

        plt.figure(figsize=(12, 6))
        for noise_level in noise_levels:
            noisy_image = self.add_noise(original_image, noise_level)
            transformed_image = self.transform(Image.fromarray(noisy_image)).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(transformed_image)
                confidence = torch.softmax(output, dim=1).max().item()

            confidences.append(confidence)

            plt.clf()
            plt.subplot(1, 2, 1)
            plt.imshow(noisy_image)
            plt.title(f"Noise Level: {int(noise_level * 100)}% | Model Confidence: {confidence:.2f}")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.plot(noise_levels[:len(confidences)] * 100, confidences, marker="o", linestyle="-", color="b")
            plt.xlabel("Noise Level (%)")
            plt.ylabel("Model Confidence")
            plt.title("Model Confidence vs. Noise Level")
            plt.grid(True)

            plt.pause(0.5)

        plt.show()


tester = NoiseRobustnessTester("model.pth", "./hazard_detection_data/drone/images/IMG_2132.jpeg")
tester.test_noise_robustness()
