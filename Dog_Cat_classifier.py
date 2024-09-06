#%%

"""import dependencies """

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torchvision.models import resnet18
from torch.utils.tensorboard import SummaryWriter

import seaborn as sns
sns.set_style("darkgrid")

BATCH_SIZE = 512
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
PATIENCE = 10

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
#%%

""" build dataset"""

class DogCat_dataset(Dataset):
    def __init__(self, root_dir: str, train_transform = None, val_transform = None):
        self.images_path_label = []

        category = {'dogs': 0, 'cats': 1}

        for sub_dir in os.listdir(root_dir):
            sub_dir_path = os.path.join(root_dir, sub_dir)
            for image_name in os.listdir(sub_dir_path):
                image_path = os.path.join(sub_dir_path, image_name)
                self.images_path_label.append((image_path, category[sub_dir]))

        self.train_transform = train_transform
        self.val_transform = val_transform

    def __len__(self):
        return len(self.images_path_label)

    def __getitem__(self, idx):
        image_path, label = self.images_path_label[idx]
        image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = image.astype('float32')
        label = np.array(label, dtype=np.int64)

        if isinstance(idx, int):  # Single index, use train transform
            if self.train_transform:
                image = self.train_transform(image)
        else:  # Likely a boolean mask or list, use val transform
            if self.val_transform:
                image = self.val_transform(image)

        return (image, label)

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness = 0.2, contrast = 0.2, saturation = 0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

val_test_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

dataset_path = '/home/nguyen/Documents/deeplearning/basic_pytorch/Dog_Cat_classifier/Dataset'
train_path = os.path.join(dataset_path, 'train')
full_dataset = DogCat_dataset(train_path, train_transform = train_transforms, val_transform = val_test_transforms)

indices = list(range(len(full_dataset)))
split = int(np.floor(0.2 * len(full_dataset)))
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Tạo samplers
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

# Tạo dataloaders
train_loader = DataLoader(full_dataset, batch_size = BATCH_SIZE, sampler=train_sampler)
val_loader = DataLoader(full_dataset, batch_size = BATCH_SIZE, sampler=valid_sampler)

# Test dataset vẫn giữ nguyên
test_path = os.path.join(dataset_path, 'test')
test_dataset = DogCat_dataset(test_path, val_transform=val_test_transforms)
test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle=False)


print(f"Number of training samples: {len(train_idx)}")
print(f"Number of validation samples: {len(val_idx)}")
print(f"Number of test samples: {len(test_dataset)}")

# %%
""" build model """
class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet18(pretrained = True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2) 

    def forward(self, x):
        return self.model(x)

model = NN().to(device)

#%%
""" test output shape"""
input_tensor = torch.rand(4, 3, 224, 224).to(device)

output = model(input_tensor)
print(output.shape)  

# %%

""" training model """

optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

# Early Stopping
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def train_fn(num_epochs, train_data_loader, val_data_loader):

    train_losses = []
    valid_losses = []
    train_accs = []
    val_accs = []

    # Initialize TensorBoard
    writer = SummaryWriter()

    # Initialize Early Stopping
    early_stopping = EarlyStopping(patience=10, verbose=True)

    def train(epoch):
        batch_bar = tqdm(train_data_loader, desc = f"epoch {epoch + 1}/{num_epochs}", 
                         colour = 'green')

        model.train()
        running_loss = 0
        correct = 0
        total = 0
        for images, labels in batch_bar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(train_data_loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc

    def validate(data_loader):
        model.eval()
        running_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(data_loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc

    epoch_bar = tqdm(range(num_epochs), desc = 'training', colour = 'cyan')

    for epoch in epoch_bar:
        
        train_loss, train_acc = train(epoch)
        valid_loss, valid_acc = validate(val_data_loader)

        tqdm.write(f'Epochs: {epoch}, Train_loss: {train_loss:.4f}, Valid_loss: {valid_loss:.4f}, '
                   f'Train_accuracy: {train_acc:.2f}, Valid_accuracy: {valid_acc:.2f}')

        train_accs.append(train_acc)
        val_accs.append(valid_acc)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        
        # Step the scheduler
        scheduler.step(valid_loss)

        # TensorBoard logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/validation', valid_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/validation', valid_acc, epoch)
        writer.add_scalar('Learning rate', optimizer.param_groups[0]['lr'], epoch)

        # Early stopping
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    writer.close()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))

    ax1.plot(train_accs, color = "blue", label = "Train_acc")
    ax1.plot(val_accs, color = "red", label = "Validation_acc")
    ax1.set(title = "Acc over epochs", xlabel = "Epoch", ylabel = "Acc")
    ax1.legend()

    ax2.plot(train_losses, color = "blue", label = "Train_loss")
    ax2.plot(valid_losses, color = "red", label = "Validation_loss")
    ax2.set(title = "loss over epochs", xlabel = "Epoch", ylabel = "Loss")
    ax2.legend()

    plt.show()

train_fn(20, train_data_loader, val_data_loader)

# %%
""" Final evaluation on test set """
def evaluate_test(model, test_data_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_acc = 100. * correct / total
    print(f'Test Accuracy: {test_acc:.2f}%')

# Load the best model
model.load_state_dict(torch.load('checkpoint.pt'))
evaluate_test(model, test_data_loader)
# %%

""" show result """

def display_misclassified(model, test_data_loader, num_images=10):
    model.eval()
    misclassified_images = []
    misclassified_labels = []
    predicted_labels = []
    
    with torch.no_grad():
        for images, labels in test_data_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            mask = (predicted != labels)
            misclassified_images.extend(images[mask].cpu())
            misclassified_labels.extend(labels[mask].cpu())
            predicted_labels.extend(predicted[mask].cpu())
            
            if len(misclassified_images) >= num_images:
                break
    
    # Limit to the requested number of images
    misclassified_images = misclassified_images[:num_images]
    misclassified_labels = misclassified_labels[:num_images]
    predicted_labels = predicted_labels[:num_images]
    
    # Set up the plot
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle("Misclassified Images", fontsize=16)
    
    for i, ax in enumerate(axes.flat):
        if i < len(misclassified_images):
            img = misclassified_images[i]
            true_label = "Dog" if misclassified_labels[i] == 0 else "Cat"
            pred_label = "Dog" if predicted_labels[i] == 0 else "Cat"
            
            # Denormalize the image
            img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            img = torch.clamp(img, 0, 1)
            
            ax.imshow(img.permute(1, 2, 0))
            ax.set_title(f"True: {true_label}\nPred: {pred_label}")
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# After training and evaluation, call this function
model.load_state_dict(torch.load('checkpoint.pt'))
display_misclassified(model, test_data_loader)