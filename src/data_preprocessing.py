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
        transforms.CenterCrop((400,400)),
        transforms.Resize((200, 200)),  # Resize images to 256x256 pixels
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


import os
import kagglehub
import cv2
import matplotlib.pyplot as plt
import numpy as np
def cropping_images():
    # Скачивание датасета
    dataset_path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")

    # Пути к папкам с изображениями
    train_path = os.path.join(dataset_path, "Training")
    test_path = os.path.join(dataset_path, "Testing")

    # Получаем список файлов изображений из тренировочной выборки
    class_dirs = os.listdir(train_path)  # Папки с разными классами
    image_paths = []
    
    for class_dir in class_dirs:
        class_path = os.path.join(train_path, class_dir)
        if os.path.isdir(class_path):
            images = [os.path.join(class_path, img) for img in os.listdir(class_path) if img.endswith(('.png', '.jpg', '.jpeg'))]
            image_paths.extend(images)
    
    # Выбираем первые два изображения
    sample_images = image_paths[:2]

    # Обрезка и отображение изображений
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    for i, img_path in enumerate(sample_images):
        img = cv2.imread(img_path)  # Загружаем изображение
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Преобразуем в RGB

        # Обрезаем 28 пикселей с каждой стороны
        cropped_img = img[56:-56, 56:-56]
        print(cropped_img.shape)
        # Отображаем исходное и обрезанное изображения
        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Original Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(cropped_img)
        axes[i, 1].set_title("Cropped Image")
        axes[i, 1].axis("off")

    plt.show()
    

    