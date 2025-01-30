import os
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms


def classify_image(model, image_path, transform, threshold=0.8):

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        confidence, pred_class = torch.max(probabilities, 1)

        if confidence.item() >= threshold:
            label = "hazardous_object" if pred_class.item() == 1 else "non_hazardous_object"
        else:
            label = "non_hazardous_object"

    print(f"Image {image_path} classified as {label} with confidence {confidence.item():.2f}")
    return label, confidence.item()


def classify_images_in_folder(model, folder_path, transform, threshold=0.8):

    results = []
    for image_file in os.listdir(folder_path):
        if image_file.endswith(('.jpeg', '.jpg', '.png')):
            image_path = os.path.join(folder_path, image_file)
            label, confidence = classify_image(model, image_path, transform, threshold)
            results.append({"File Name": image_file, "Confidence": confidence, "Label": label})

    results_df = pd.DataFrame(results)
    return results_df
