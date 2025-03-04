import kagglehub
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

""" 
TODO:

Stratified split (keep class distribution):
    Train: 70% (e.g. 70% from each class).
    Validation: 10% (for early stopping and hyperparameter tuning).
    Test: 20% (only for final evaluation).


output size: 256 * 256 * 3

add requirements to requirements.txt
"""

def data_preprocessing_tumor():
    # 1. Download the dataset from KaggleHub
    dataset_path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")

    # 2. Define paths to training and testing directories
    train_path = os.path.join(dataset_path, "Training")
    test_path = os.path.join(dataset_path, "Testing")

    # 3. Define image transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize images to 256x256 pixels
        transforms.ToTensor(),          # Convert images to PyTorch tensors
    ])

    # 4. Use ImageFolder to automatically load images and assign labels
    train_dataset = ImageFolder(root=train_path, transform=transform)
    test_dataset = ImageFolder(root=test_path, transform=transform)

    # 5. Create DataLoader to load batches of images efficiently
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # Shuffle training data
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # No shuffle for testing

    return train_loader, test_loader

    # 6. Print dataset details
    # print("Number of training samples:", len(train_dataset))
    # print("Number of testing samples:", len(test_dataset))
    # print("Class dictionary:", train_dataset.class_to_idx)  # Mapping of class names to indices

    # 7. Display a sample image with its class label
    # import matplotlib.pyplot as plt

    # def show_image(tensor, label):
    #     img = tensor.permute(1, 2, 0)  # Convert (C, H, W) to (H, W, C) for visualization
    #     plt.imshow(img)  # Display the image
    #     plt.title(f"Class: {label}")  # Show class label
    #     plt.axis("off")  # Hide axis for cleaner visualization
    #     plt.show()

    # #  8. Select the first image from the dataset and display it
    # image, label = train_dataset[0]  # Get the first image and its label
    # show_image(image, label)

