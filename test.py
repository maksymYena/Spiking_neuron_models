import json
import os
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score
from classifier.classify import classify_images_in_folder, classify_image
from torchvision import models
from torchvision import transforms


class ModelTester:
    def __init__(self, model_path, test_folder, threshold=0.8):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet50(pretrained=True)


        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()

        self.test_folder = test_folder
        self.threshold = threshold
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def test_model(self):
        results_df = classify_images_in_folder(self.model, self.test_folder, self.transform, self.threshold)
        pd.set_option("display.max_rows", None)
        print(results_df)
        return results_df

    def evaluate_metrics(self, results_df, annotations_file):
        if not os.path.exists(annotations_file):
            print(f"Error: Annotation file {annotations_file} not found.")
            return None

        with open(annotations_file, 'r') as f:
            data = json.load(f)

        if "annotations" not in data or "images" not in data:
            print("Error: Incorrect annotation format.")
            return None


        hazardous_count = sum(1 for ann in data["annotations"] if ann["category_id"] == 1)
        non_hazardous_count = len(data["images"]) - hazardous_count
        print(f"Hazardous Objects: {hazardous_count}, Non-Hazardous Objects: {non_hazardous_count}")


        true_labels = {img["file_name"]: 1 if any(
            ann["image_id"] == img["id"] and ann["category_id"] == 1 for ann in data["annotations"]
        ) else 0 for img in data["images"]}

        y_true = [true_labels.get(row["File Name"], 0) for _, row in results_df.iterrows()]
        y_pred = [1 if row["Confidence"] >= self.threshold else 0 for _, row in results_df.iterrows()]

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=1)
        recall = recall_score(y_true, y_pred, zero_division=1)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")

        return {"accuracy": accuracy, "precision": precision, "recall": recall}

    def test_single_image(self, image_path):
        if not os.path.exists(image_path):
            print(f"Error: File {image_path} not found.")
            return None, None

        label, confidence = classify_image(self.model, image_path, self.transform, self.threshold)
        print(f"Image {image_path} classified as {label} with confidence {confidence:.2f}")
        return label, confidence



tester = ModelTester("./model.pth", "./hazard_detection_data/drone/images", threshold=0.5)
results_df = tester.test_model()
tester.evaluate_metrics(results_df, "./hazard_detection_data/drone/annotations/all_annotations.json")
tester.test_single_image("./hazard_detection_data/drone/images/IMG_2115.jpeg")