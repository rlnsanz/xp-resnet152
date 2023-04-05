import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from transformers import AutoFeatureExtractor, ResNetForImageClassification
from datasets import load_dataset

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper-parameters
num_epochs = 5
batch_size = 128
learning_rate = 0.001

# Data loader
data = load_dataset("imagenet-1k")

import sys

sys.exit(0)

feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-152")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-152").to(device)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(
                "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                    epoch + 1, num_epochs, i + 1, total_step, loss.item()
                )
            )

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(
        "Accuracy of the network on the 10000 test images: {} %".format(
            100 * correct / total
        )
    )
