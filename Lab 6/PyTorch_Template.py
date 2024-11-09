import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import json
import torchvision.transforms as transforms

# Custom Dataset Class
class MyDataset(Dataset):
    def __init__(self, json_filename, image_dir, transform=None):
        super(MyDataset, self).__init__()
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Load JSON data
        with open(json_filename, 'r') as f:
            data = json.load(f)
            for filename, label in data.items():
                label = int(label) - 1  # Adjust labels to be 0-indexed
                if label < 0 or label >= 196:  # Replace 196 with your actual number of classes
                    raise ValueError(f"Label {label} is out of bounds for 196 classes.")
                self.image_paths.append(filename)
                self.labels.append(label)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_paths[idx])
        label = self.labels[idx]

        # Load the image
        image = Image.open(image_path).convert("RGB")

        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)

        # Convert label to tensor
        label = torch.tensor(label, dtype=torch.long)

        return image, label

    def __len__(self):
        return len(self.image_paths)

# Define a simple CNN architecture
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 16 * 16, 256)  # Adjusted for 64x64 input images
        self.fc2 = nn.Linear(256, 196)  # Replace 196 with the number of classes you have
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate the dataset and dataloaders
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize images to a fixed size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = MyDataset("train_annos.json", "cars_train", transform=transform)
test_dataset = MyDataset("test_annos.json", "cars_test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Instantiate the model, loss function, and optimizer
model = MyNetwork()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss_func = nn.CrossEntropyLoss()

# Training Loop
num_epochs = 5
model.train()

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()  # Clear gradients
        outputs = model(images)  # Forward pass
        loss = loss_func(outputs, labels)  # Compute loss
        loss.backward()  # Backpropagate
        optimizer.step()  # Update weights

        running_loss += loss.item()
    
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

print("Training complete")

# Testing Loop
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

# Additional code for plotting results (optional)
