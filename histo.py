import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from models.model import DomainAdaptationModel

class ImageAndHistogramViewer:
    def __init__(self, model_path, image_folder):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DomainAdaptationModel().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()
        self.image_folder = image_folder
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def get_confidences(self):
        """
        Processes all images in the folder, computes the model output,
        applies softmax, and collects the maximum probability (confidence) for each image.
        """
        confidences = []
        image_files = [f for f in os.listdir(self.image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for img_file in image_files:
            img_path = os.path.join(self.image_folder, img_file)
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Error opening image {img_path}: {e}")
                continue
            transformed_image = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.model(transformed_image)
                probabilities = torch.softmax(output, dim=1)
                confidence = probabilities.max().item()
                confidences.append(confidence)
        return confidences, image_files

    def view_sample_and_histogram(self):
        """
        Displays one sample image (first in the folder) on the left,
        and a histogram of classification confidences (across the folder) on the right.
        """
        confidences, image_files = self.get_confidences()
        if not confidences:
            print("No images found or no confidences computed.")
            return

        # Select the first image as the sample image
        sample_image_path = os.path.join(self.image_folder, image_files[0])
        sample_image = Image.open(sample_image_path).convert("RGB")

        # Create a figure with two subplots: left for the sample image, right for the histogram
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Left subplot: display the sample image
        axs[0].imshow(sample_image)
        axs[0].axis("off")
        axs[0].set_title("Sample Image")

        # Right subplot: plot the histogram of confidences
        axs[1].hist(confidences, bins=10, color='skyblue', alpha=0.8, edgecolor='black')
        axs[1].set_xlabel("Model Confidence")
        axs[1].set_ylabel("Frequency")
        axs[1].set_title("Histogram of Classification Confidence")
        axs[1].grid(True)

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    viewer = ImageAndHistogramViewer("model.pth", "./hazard_detection_data/drone/images")
    viewer.view_sample_and_histogram()
