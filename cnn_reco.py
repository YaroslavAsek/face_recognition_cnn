import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from sklearn.metrics import classification_report


#Константы с классами и характеристиками
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
NUM_CLASSES = len(EMOTIONS)
IMG_SIZE = 48
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = r"dataset\1"           # путь до папки с датасетом
CSV_PATH = "fer2013.csv"       # название файла с метками классов эмоций
CHECKPOINT = "emotion_cnn_best.pth" # название файла для сохранения пути


class FER2013CSV(Dataset):

    def __init__(self, csv_path: str, split: str = "Training", transform=None):
        df = pd.read_csv(csv_path)
        df = df[df["Usage"] == split].reset_index(drop=True)
        self.pixels = df["pixels"].values
        self.labels = df["emotion"].values.astype(np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        pixels = np.array(self.pixels[idx].split(), dtype=np.uint8).reshape(IMG_SIZE, IMG_SIZE)
        image = Image.fromarray(pixels, mode="L")          
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]


def get_transforms(augment: bool):
    base = [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
    ]
    if augment:
        aug = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
        base.extend(aug)
    base.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    return transforms.Compose(base)


def build_dataloaders():
    if os.path.isdir(DATA_DIR):
        print("Загружаем папку с датасетом")
        train_ds = datasets.ImageFolder(
            os.path.join(DATA_DIR, "train"), transform=get_transforms(augment=True)
        )
        test_ds = datasets.ImageFolder(
            os.path.join(DATA_DIR, "test"), transform=get_transforms(augment=False)
        )
    elif os.path.isfile(CSV_PATH):
        print("Загружаем папку с метками")
        train_ds = FER2013CSV(CSV_PATH, "Training",  transform=get_transforms(augment=True))
        test_ds = FER2013CSV(CSV_PATH, "PublicTest", transform=get_transforms(augment=False))
    else:
        raise FileNotFoundError(
            "Датасет не был найден"
        )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)
    print(f"Train samples: {len(train_ds):,}  |  Test samples: {len(test_ds):,}")
    return train_loader, test_loader



class ConvBlock(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.skip = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + self.skip(x))
        return out


class EmotionCNN(nn.Module):

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        self.stage1 = nn.Sequential(
            ConvBlock(32,  64,  stride=2),   
            ConvBlock(64,  64),
        )
        self.stage2 = nn.Sequential(
            ConvBlock(64,  128, stride=2),   
            ConvBlock(128, 128),
        )
        self.stage3 = nn.Sequential(
            ConvBlock(128, 256, stride=2),   
            ConvBlock(256, 256),
        )
        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc      = nn.Linear(256, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)

#Обучение нейросети
def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="  train", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for images, labels in tqdm(loader, desc="  eval ", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)
        preds = outputs.argmax(1)
        total_loss += loss.item() * images.size(0)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels


def plot_history(history: dict):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"],   label="Val")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"],   label="Val")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=120)
    plt.show()
    print("Saved: training_history.png")


# Функция для проверки нейросети
def predict(image_path: str, model: nn.Module):
    """Run inference on a single image file."""
    transform = get_transforms(augment=False)
    image = Image.open(image_path).convert("L")
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    for emotion, prob in sorted(zip(EMOTIONS, probs), key=lambda x: -x[1]):
        bar = "█" * int(prob * 30)
        print(f"  {emotion:<10} {prob*100:5.1f}%  {bar}")
    print(f"\n→ Predicted: {EMOTIONS[probs.argmax()]}")


# Основной файл
def main():
    train_loader, test_loader = build_dataloaders()

    model     = EmotionCNN(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler    = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss,   val_acc, preds, labels = evaluate(model, test_loader, criterion)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"Loss {train_loss:.4f}/{val_loss:.4f} | "
              f"Acc {train_acc*100:.2f}%/{val_acc*100:.2f}% | "
              f"{elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT)
            print(f"  ✓ Checkpoint saved (val_acc={val_acc*100:.2f}%)")

    print(f"\nBest validation accuracy: {best_val_acc*100:.2f}%")

    # Загружаем лучшую модель
    model.load_state_dict(torch.load(CHECKPOINT))
    _, _, preds, labels = evaluate(model, test_loader, criterion)

    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=EMOTIONS))

    plot_history(history)


if __name__ == "__main__":
    main()