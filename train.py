import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data as torchdata
import torchvision
import torchvision.transforms as transforms

from transformers import AutoFeatureExtractor, ResNetForImageClassification
from datasets import load_dataset, DatasetDict

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper-parameters
num_epochs = 5
batch_size = 64
learning_rate = 0.001

# Data loader
data = load_dataset("imagenet-1k")
assert isinstance(data, DatasetDict)
assert set(data.keys()) == {"train", "validation", "test"}  # type: ignore


randcrop = transforms.RandomResizedCrop((256, 256))
randcrop.to(device)


def my_collate(batch):
    new_batch = []
    for i, record in enumerate(batch):
        im = record["image"]
        print(i, len(im.shape))
        if len(im.shape) == 3:
            h, w, c = im.shape
            im = im.to(device).reshape((3, h, w))
        elif len(im.shape) == 2:
            h, w = im.shape
            im = im.to(device).reshape((1, h, w))
            im = im.repeat(3, 1, 1)
        else:
            raise
        new_batch.append(
            {
                "label": record["label"],
                "image": randcrop(im),
            }
        )

    out = torchdata.default_collate(new_batch)
    print("called collate")
    return out


train_loader = torchdata.DataLoader(dataset=data["train"].with_format("torch"), batch_size=batch_size, shuffle=True, collate_fn=my_collate)  # type: ignore
val_loader = torchdata.DataLoader(dataset=data["validation"].with_format("torch"), batch_size=batch_size, shuffle=False, collate_fn=my_collate)  # type: ignore
test_loader = torchdata.DataLoader(dataset=data["test"].with_format("torch"), batch_size=batch_size, shuffle=False, collate_fn=my_collate)  # type: ignore

feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-152")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-152").to(device)  # type: ignore


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, batch in enumerate(train_loader):
        # Move tensors to the configured device
        print(batch)
        import sys

        sys.exit(0)
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
