import kagglehub
import os
import shutil
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split as randspl

from torchvision.datasets import ImageFolder


def data_preprocessing_tumor():
    # Download the dataset from KaggleHub
    dataset_path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")

    # Define paths to training and testing directories
    train_path = os.path.join(dataset_path, "Training")
    test_path = os.path.join(dataset_path, "Testing")
    
    # Making new data split by aggregating all the data together and splitting again
    general_dataset_path = os.path.join(dataset_path, "General_Dataset")
    os.makedirs(general_dataset_path, exist_ok=True)
    
    
    for source_path in [train_path, test_path]:
        
        for class_name in os.listdir(source_path):
            class_path = os.path.join(source_path, class_name)
            general_class_path = os.path.join(general_dataset_path, class_name)
            os.makedirs(general_class_path, exist_ok=True)
            
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                shutil.move(img_path, os.path.join(general_class_path, img_name))
    

    # Define image transformations
    transform = transforms.Compose([
        transforms.CenterCrop((400,400)), #from input 512x512 to 400x400
        transforms.Resize((200, 200)),  # Resize images to 200х200 pixels
        transforms.ToTensor(),          # Convert images to PyTorch tensors
    ])

    # Load datasets with ImageFolder
    general_dataset = ImageFolder(root=general_dataset_path, transform=transform)
    
    total_size = len(general_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size
    
    train_set, val_set, test_set = randspl(
        general_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
        
    )
        
    # Create DataLoaders
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    return train_loader, val_loader, test_loader
