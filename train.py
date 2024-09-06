import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt

import config
from model import DogCatClassifier

class EarlyStopping:
    def __init__(self, patience: int = 7, verbose: bool = False, delta: float = 0, path: str = 'checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss: float, model: nn.Module) -> None:
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

    def save_checkpoint(self, val_loss: float, model: nn.Module) -> None:
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def plot_results(train_accs: List[float], val_accs: List[float], train_losses: List[float], valid_losses: List[float]) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15,5))

    ax1.plot(train_accs, color="blue", label="Train_acc")
    ax1.plot(val_accs, color="red", label="Validation_acc")
    ax1.set(title="Accuracy over epochs", xlabel="Epoch", ylabel="Accuracy")
    ax1.legend()

    ax2.plot(train_losses, color="blue", label="Train_loss")
    ax2.plot(valid_losses, color="red", label="Validation_loss")
    ax2.set(title="Loss over epochs", xlabel="Epoch", ylabel="Loss")
    ax2.legend()

    plt.show()

def train_fn(num_epochs: int, train_data_loader: DataLoader, val_data_loader: DataLoader, model: DogCatClassifier) -> None:
    train_losses: List[float] = []
    valid_losses: List[float] = []
    train_accs: List[float] = []
    val_accs: List[float] = []

    writer = SummaryWriter()
    early_stopping = EarlyStopping(patience = config.EARLY_STOPPING_PATIENCE, verbose = True, path = config.CHECKPOINT_PATH)

    optimizer = torch.optim.Adam(model.parameters(), lr = config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.1, patience = 5, verbose = True)

    def train(epoch: int) -> Tuple[float, float]:
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        batch_bar = tqdm(train_data_loader, desc=f"epoch {epoch + 1}/{num_epochs}", colour='green')

        for images, labels in batch_bar:
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

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

    def validate(data_loader: DataLoader) -> Tuple[float, float]:
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(config.DEVICE)
                labels = labels.to(config.DEVICE)

                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(data_loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc

    epoch_bar = tqdm(range(num_epochs), desc='training', colour='cyan')

    for epoch in epoch_bar:
        train_loss, train_acc = train(epoch)
        valid_loss, valid_acc = validate(val_data_loader)

        tqdm.write(f'Epochs: {epoch}, Train_loss: {train_loss:.4f}, Valid_loss: {valid_loss:.4f}, '
                   f'Train_accuracy: {train_acc:.2f}, Valid_accuracy: {valid_acc:.2f}')

        train_accs.append(train_acc)
        val_accs.append(valid_acc)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        
        scheduler.step(valid_loss)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/validation', valid_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/validation', valid_acc, epoch)
        writer.add_scalar('Learning rate', optimizer.param_groups[0]['lr'], epoch)

        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            tqdm.write("Early stopping")
            break

    writer.close()

    plot_results(train_accs, val_accs, train_losses, valid_losses)
