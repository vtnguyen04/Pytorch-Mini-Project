import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from typing import List

import config
from model import DogCatClassifier

def evaluate_test(model: DogCatClassifier, test_data_loader: DataLoader) -> None:
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_data_loader:
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_acc = 100. * correct / total
    print(f'Test Accuracy: {test_acc:.2f}%')

def display_misclassified(model: DogCatClassifier, test_data_loader: DataLoader, num_images: int = 10) -> None:
    model.eval()
    misclassified_images: List[torch.Tensor] = []
    misclassified_labels: List[int] = []
    predicted_labels: List[int] = []
    
    with torch.no_grad():
        for images, labels in test_data_loader:
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            mask = (predicted != labels)
            misclassified_images.extend(images[mask].cpu())
            misclassified_labels.extend(labels[mask].cpu())
            predicted_labels.extend(predicted[mask].cpu())
            
            if len(misclassified_images) >= num_images:
                break
    
    misclassified_images = misclassified_images[:num_images]
    misclassified_labels = misclassified_labels[:num_images]
    predicted_labels = predicted_labels[:num_images]
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle("Misclassified Images", fontsize=16)
    
    for i, ax in enumerate(axes.flat):
        if i < len(misclassified_images):
            img = misclassified_images[i]
            true_label = "Dog" if misclassified_labels[i] == 0 else "Cat"
            pred_label = "Dog" if predicted_labels[i] == 0 else "Cat"
            
            img = img * torch.tensor(config.STD).view(3, 1, 1) + torch.tensor(config.MEAN).view(3, 1, 1)
            img = torch.clamp(img, 0, 1)
            
            ax.imshow(img.permute(1, 2, 0))
            ax.set_title(f"True: {true_label}\nPred: {pred_label}")
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()