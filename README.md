# Brain Tumor Classification with CNN (Replication Study)

This repository contains an implementation of a convolutional neural network (CNN) for brain tumor classification, replicating the methodology described in Comprehensive CNN Model for [Brain Tumour Identification and Classification using MRI Images](https://ieeexplore.ieee.org/document/10502486). The model achieves multiclass classification on MRI scans from the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).

## Dataset Overview

**Source:** Brain Tumor MRI Dataset (Kaggle)  
**Original Size:** 7,023 images (512×512×3)  
**Classes:** 
- Pituitary (1,757 total)
- Meningioma (1,645 total)
- Glioma (1,621 total)
- No Tumor (2,000 total)

**Preprocessing:**
1. Resized to 256×356×3
2. Center-cropped to 200×200×3
3. Training/Test Split (80/20):
   | Class          | Training | Testing |
   |----------------|----------|---------|
   | Pituitary      | 1,457    | 300     |
   | Meningioma     | 1,339    | 306     |
   | Glioma         | 1,321    | 300     |
   | No Tumor       | 1,595    | 405     |

*Note: The test set was used for both validation and testing during training, which may lead to optimistic performance estimates.*

## Model Architecture

### Convolutional Base (6 Layers)

| Layer | Channels In/Out | Kernel | Padding | Stride | Activation | Pooling | Output Shape |
|-------|-----------------|--------|---------|--------|------------|---------|--------------|
| 1     | 3 → 64          | 7×7    | 3       | 1      | ReLU       | 2×2     | 100×100×64   |
| 2     | 64 → 128        | 7×7    | 3       | 1      | ReLU       | 2×2     | 50×50×128    |
| 3     | 128 → 128       | 7×7    | 3       | 1      | ReLU       | 2×2     | 25×25×128    |
| 4     | 128 → 256       | 7×7    | 3       | 1      | ReLU       | 2×2     | 12×12×256    |
| 5     | 256 → 256       | 7×7    | 3       | 1      | ReLU       | 2×2     | 6×6×256      |
| 6     | 256 → 512       | 7×7    | 3       | 1      | ReLU       | 2×2     | 3×3×512      |

### Fully Connected Classifier

| Layer | Units In/Out | Activation | Dropout | Output Shape |
|-------|--------------|------------|---------|--------------|
| 1     | 4608 → 1024  | ReLU       | 0.25    | 1024         |
| 2     | 1024 → 512   | ReLU       | 0.25    | 512          |
| 3     | 512 → 4      | Linear     | -       | 4            |

**Final Layer Note:** Explicit softmax activation is omitted as `nn.CrossEntropyLoss` internally combines log-softmax and negative log-likelihood loss.

## Training Protocol

- **Optimizer:** Stochastic Gradient Descent (SGD)
- **Learning Rate:** 0.001
- **Loss Function:** Cross Entropy Loss
- **Regularization:**
  - Batch Normalization after each conv layer
  - Dropout (p=0.25) in FC layers
  - Early Stopping (patience=3, monitoring validation loss)
- **Batch Size:** 32 (standard configuration)
- **Epochs:** 100 (with early stopping)

