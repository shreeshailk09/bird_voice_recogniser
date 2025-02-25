import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader

DATASET_FILE = "data/processed/features.npy"

class BirdDataset(Dataset):
    def __init__(self):
        data = np.load(DATASET_FILE, allow_pickle=True).item()
        self.features = np.array(data["features"], dtype=np.float32)
        self.labels = np.array(data["labels"], dtype=np.int64)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class BirdClassifier(nn.Module):
    def __init__(self, num_classes):
        super(BirdClassifier, self).__init__()
        self.fc1 = nn.Linear(141, 1024)  # 🔹 Match feature size
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.batchnorm1 = nn.BatchNorm1d(1024)
        self.batchnorm2 = nn.BatchNorm1d(512)
        self.batchnorm3 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = self.relu(self.batchnorm1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.batchnorm2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.batchnorm3(self.fc3(x)))
        x = self.fc4(x)
        return x


if __name__ == "__main__":
    dataset = BirdDataset()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    num_classes = len(set(dataset.labels))
    model = BirdClassifier(num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    epochs = 50

    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        correct, total, epoch_loss = 0, 0, 0.0

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_accuracy = 100 * correct / total
        scheduler.step(epoch_loss)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}, Train Acc: {train_accuracy:.2f}%")

        if train_accuracy > best_val_acc:
            best_val_acc = train_accuracy
            torch.save(model.state_dict(), "models/bird_classifier_best.pth")
            print(f"✅ Best model saved with Train Acc: {train_accuracy:.2f}%")
