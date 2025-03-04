import kagglehub
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split

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

    # 4. Load datasets with ImageFolder
    train_dataset = ImageFolder(root=train_path, transform=transform)
    test_dataset = ImageFolder(root=test_path, transform=transform)

    # 5. Perform stratified split on the training dataset
    indices = list(range(len(train_dataset)))
    # 70% training, 10% validation from the training set (например, если training set уже 80% от общего количества, 
    # то можно скорректировать проценты)
    train_idx, valid_idx = train_test_split(
        indices,
        test_size=0.125,  # 0.125 от train_dataset = примерно 10% от общего набора, если исходный split 70/20/10
        stratify=train_dataset.targets,
        random_state=42
    )
    train_subset = Subset(train_dataset, train_idx)
    valid_subset = Subset(train_dataset, valid_idx)

    # 6. Create DataLoaders for training, validation, and testing
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_subset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, valid_loader, test_loader
