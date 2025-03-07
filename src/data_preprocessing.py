import kagglehub
import os
import shutil
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split as randspl

from torchvision.datasets import ImageFolder



def data_norm(train_path):
    transform = transforms.Compose([
        transforms.CenterCrop((400,400)), #from input 512x512 to 400x400
        transforms.Resize((200, 200)),  # Resize images to 200х200 pixels
        transforms.ToTensor()        
    ])
    
    dataset = ImageFolder(root=train_path, transform=transform)

    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_samples = 0    
    for data,_ in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        total_samples += batch_samples
        
    mean /= total_samples
    std /= total_samples
    print(f"Mean: {mean}")
    print(f"Std: {std}")
    
    return mean, std




def data_preprocessing_tumor():
    # 1. Download the dataset from KaggleHub
    dataset_path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")

    # 2. Define paths to training and testing directories
    train_path = os.path.join(dataset_path, "Training")
    test_path = os.path.join(dataset_path, "Testing")
    
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
                
    

    # 3. Define image transformations
    transform = transforms.Compose([
        transforms.CenterCrop((400,400)), #from input 512x512 to 400x400
        transforms.Resize((200, 200)),  # Resize images to 200х200 pixels
        transforms.ToTensor(),          # Convert images to PyTorch tensors
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])

    # 4. Load datasets with ImageFolder
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

    

    # 7. Create DataLoaders
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    return train_loader, val_loader, test_loader


    

    