import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import models
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm

class TestDataset(Dataset):
    def __init__(self, test_dir, transform=None):
        if not os.path.isdir(test_dir):
            raise ValueError(f"Provided test directory '{test_dir}' does not exist.")

        self.test_dir = test_dir
        self.transform = transform
        self.image_paths = [
            os.path.join(test_dir, f)
            for f in os.listdir(test_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, os.path.basename(img_path)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

    
def train_transform():
    size = 400
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.3, saturation=0.2, hue=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    ])

def test_transform():
    return transforms.Compose([
        transforms.Resize((400, 400)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

def create_dataloaders(train_dir, val_dir, test_dir, batch_size=128):
    # train 和 val 是兩層資料夾，使用 ImageFolder 自動處理
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform())
    val_dataset = datasets.ImageFolder(val_dir, transform=test_transform())
    test_dataset = TestDataset(test_dir, transform=test_transform())  # Test 目錄是單層圖片

    class_names = train_dataset.classes
    idx_to_class = {cls: i for cls, i in enumerate(class_names)}
    # print("Index to class:", idx_to_class)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=32
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=32
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, 
        num_workers=32
    )

    return train_loader, val_loader, test_loader, idx_to_class

def train_model(
    train_loader, val_loader, num_epochs=100, learning_rate=0.0007, patience=10
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnext50_32x4d(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 100)
    model = model.to(device)
    print('#Params:', sum(p.numel() for p in model.parameters()))

    criterion = FocalLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.6, patience=5, verbose=True
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses, val_accuracies = [], [], []


    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        if epoch > 3:
            for param in model.parameters():
                param.requires_grad = True

        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                pbar.set_postfix_str(f"Loss: {loss.item():.4f}")
            
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        avg_train_loss = running_loss / len(train_loader)
        val_accuracy = correct / total

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, Val Acc: {correct/total:.4f}"
        )
        
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
            print("Best model saved.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
    
    # Loss plot
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label='Train Loss', color='green', marker='s', markersize=1)
    plt.plot(val_losses, label='Val Loss', color='blue', marker='s', markersize=1)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_curve_loss.png')
    plt.close()
    
    # Accuracy plot
    plt.figure(figsize=(8, 6))
    plt.plot(val_accuracies, label='Validation Accuracy', color='green', marker='s', markersize=1)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_curve_acc')
    plt.close()

    return model

def predict(model, test_loader, idx_to_class):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    predictions = []
    image_names = []

    with torch.no_grad():
        for images, filenames in test_loader:
            filenames = [os.path.splitext(f)[0] for f in filenames]
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            predictions.extend([idx_to_class[p.item()] for p in preds])
            image_names.extend(filenames)

    return image_names, predictions

def save_predictions_to_csv(image_names, predictions, output_file="prediction.csv"):
    df = pd.DataFrame({"image_name": image_names, "pred_label": predictions})
    df.to_csv(output_file, index=False)

def main():
    train_dir = "./data/train"
    val_dir = "./data/val"
    test_dir = "./data/test"
    train_loader, val_loader, test_loader, idx_to_class = create_dataloaders(
        train_dir, val_dir, test_dir
    )
    model = train_model(train_loader, val_loader)
    model = models.resnext50_32x4d(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 100)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.load_state_dict(torch.load("best_model.pth"))

    model_size = sum(p.numel() for p in model.parameters()) * 4 / (1024 ** 2)
    print(f"Model size: {model_size:.2f} MB")
    
    image_names, predictions = predict(model, test_loader, idx_to_class)
    save_predictions_to_csv(image_names, predictions)
    print("Predictions saved to prediction.csv")

if __name__ == "__main__":
    main()
