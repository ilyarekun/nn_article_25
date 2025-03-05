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
    full_train_dataset = ImageFolder(root=train_path, transform=transform)
    full_test_dataset = ImageFolder(root=test_path, transform=transform)

    # 5. Adjust test set to 10% of total dataset
    test_indices = list(range(len(full_test_dataset)))
    new_test_size = len(test_indices) // 2  # 10% of total dataset (50% of current test set)

    test_idx, _ = train_test_split(
        test_indices,
        test_size=0.5,  # Keep only half of the original test set
        stratify=full_test_dataset.targets,
        random_state=42
    )
    new_test_subset = Subset(full_test_dataset, test_idx)

    # 6. Split train into 70% train and 20% validation
    train_indices = list(range(len(full_train_dataset)))
    train_idx, valid_idx = train_test_split(
        train_indices,
        test_size=0.25,  # 20% of total dataset (since train was originally 80%)
        stratify=full_train_dataset.targets,
        random_state=42
    )

    train_subset = Subset(full_train_dataset, train_idx)
    valid_subset = Subset(full_train_dataset, valid_idx)

    # 7. Create DataLoaders
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_subset, batch_size=32, shuffle=False)
    test_loader = DataLoader(new_test_subset, batch_size=32, shuffle=False)

    return train_loader, valid_loader, test_loader
