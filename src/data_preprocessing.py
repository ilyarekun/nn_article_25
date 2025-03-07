import kagglehub
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
    

def data_preprocessing_tumor():
    # 1. Download the dataset from KaggleHub
    dataset_path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")

    # 2. Define paths to training and testing directories
    train_path = os.path.join(dataset_path, "Training")
    test_path = os.path.join(dataset_path, "Testing")

    # 3. Define image transformations
    transform = transforms.Compose([
        transforms.CenterCrop((400,400)), #from input 512x512 to 400x400
        transforms.Resize((200, 200)),  # Resize images to 200х200 pixels
        transforms.ToTensor(),          # Convert images to PyTorch tensors
    ])

    # 4. Use ImageFolder to automatically load images and assign labels
    train_dataset = ImageFolder(root=train_path, transform=transform)
    test_dataset = ImageFolder(root=test_path, transform=transform)

    # 5. Create DataLoader to load batches of images efficiently
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # Shuffle training data
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # No shuffle for testing

    return train_loader, test_loader


