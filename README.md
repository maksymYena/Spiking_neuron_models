# Hazard Object Detection Model

This repository provides an implementation of a Hazard Object Detection Model that utilizes domain adaptation techniques
and spiking neural networks for robust classification of hazardous objects in diverse environmental conditions. The
model is designed to accurately classify objects even under noise and varying conditions, making it ideal for security
and monitoring applications.

## Key Features

- **Domain Adaptation:**  
  Implements gradient reversal layers to adapt the model to different data distributions (e.g., varying background
  conditions, lighting, sensor types).

- **Spiking Neural Networks:**  
  Incorporates a Leaky Integrate-and-Fire (LIF) cell in the classification layer for event-driven processing, improving
  the model’s noise robustness and energy efficiency.

- **Efficient Feature Extraction:**  
  Utilizes the pre-trained EfficientNet-B0 architecture to extract high-quality features from images.

- **Robust Evaluation Metrics:**  
  Includes training and testing pipelines that compute loss curves, confusion matrices, accuracy, precision, recall, and
  F1-scores.

- **Noise Robustness Testing:**  
  Provides scripts to add synthetic noise to images and analyze how the model’s confidence changes with increasing noise
  levels.

## Repository Structure

- `models/`  
  Contains model definitions, including the Domain Adaptation Model.

- `dataset/`  
  Contains the data loader and dataset definitions (e.g., `HazardDataset`).

- `train.py`  
  Script for training the Hazard Object Detection Model.

- `evaluate.py`  
  Script for evaluating the model on the test set and generating performance metrics.

- `noise_robustness.py`  
  Script to test the model’s noise robustness by adding Gaussian noise to images and plotting the confidence vs. noise
  level.

- `requirements.txt`  
  Contains the list of required packages:

- numpy~=2.1.3
- pandas~=2.2.3
- scikit-learn~=1.6.1
- opencv-python==4.5.3.20210929
- tensorflow==2.6.0
- requests==2.26.0
- torch~=2.5.1
- torchvision~=0.20.1
- pillow~=11.0.0
- classify~=0.5.0
- matplotlib~=3.10.0

## Installation

Make sure you have Python 3.7 or later installed. Then, install the required packages using pip:

```bash
pip install -r requirements.txt
```

## Usage

# Training the Model

- Run the training script to train the model:

```bash
python models/train.py 
```

This script:

- Loads and preprocesses the dataset.
- Splits the data into training and test sets.
- Computes class weights using the Counter from the collections module.
- Trains the model for a specified number of epochs.
- Saves the trained model as model.pth and logs the loss values in losses.txt.

## Evaluating the Model

After training, evaluate the model using the evaluation script (or within train.py if evaluation is integrated):

```bash
python models/evaluate.py
```