import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data as torchdata

from transformers import AutoFeatureExtractor, ResNetForImageClassification
from datasets import load_dataset, DatasetDict

import flor

# Device configuration
device = torch.device(
    flor.arg(
        "device",
        (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        ),
    )
)

# Hyper-parameters
num_epochs = flor.arg("epochs", 2)
batch_size = flor.arg("batch_size", 8)
learning_rate = flor.arg("lr", 0.001)

# Data loader
data = load_dataset("imagenet-1k")
assert isinstance(data, DatasetDict)
assert set(data.keys()) == {"train", "validation", "test"}  # type: ignore

feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-152")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-152").to(device)  # type: ignore


def my_collate(batch):
    new_batch = []
    for i, record in enumerate(batch):
        if len(record["image"].shape) == 3:  # Already in (H, W, C)
            h, w, c = record["image"].shape
            assert c == 3, f"Unexpected channel count: {c}"
            im = record["image"].permute(2, 0, 1)  # Convert to (C, H, W)
        elif len(record["image"].shape) == 2:  # Grayscale image (H, W)
            h, w = record["image"].shape
            im = record["image"].unsqueeze(0).repeat(3, 1, 1)  # Add channels
        else:  # Flat tensor or invalid shape
            flat_size = record["image"].numel()
            edge = int(flat_size**0.5)
            if edge * edge == flat_size:  # Is it a square image?
                im = record["image"].reshape(1, edge, edge).repeat(3, 1, 1)
            else:
                raise ValueError(f"Cannot reshape image of size {flat_size}")

        im = feature_extractor(im, return_tensors="pt")["pixel_values"]
        new_batch.append(
            {
                "label": record["label"],
                "image": im,
            }
        )

    out = torchdata.default_collate(new_batch)
    out["image"] = out["image"].reshape(batch_size, 3, 224, 224)
    return out


train_loader = torchdata.DataLoader(dataset=data["train"].with_format("torch"), batch_size=batch_size, shuffle=True, collate_fn=my_collate)  # type: ignore
val_loader = torchdata.DataLoader(dataset=data["validation"].with_format("torch"), batch_size=batch_size, shuffle=False, collate_fn=my_collate)  # type: ignore
test_loader = torchdata.DataLoader(dataset=data["test"].with_format("torch"), batch_size=batch_size, shuffle=False, collate_fn=my_collate)  # type: ignore

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
print(total_step)

with flor.checkpointing(model=model, optimizer=optimizer):
    for epoch in flor.loop("epoch", range(num_epochs)):
        model.train()
        for i, batch in flor.loop("step", enumerate(train_loader)):
            # Move tensors to the configured device
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs.logits, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                        epoch + 1,
                        num_epochs,
                        i,
                        total_step,
                        flor.log("loss", loss.item()),
                    )
                )
            if i >= flor.arg("step_cap", 20000):
                break

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    step_cap = flor.arg("eval_step_cap", 1000)
    print(f"evaluating for {step_cap} of {len(val_loader)} rounds")
    for i, batch in enumerate(val_loader):
        # Move tensors to the configured device
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        # Forward pass
        outputs = model(images)
        left, predicted = torch.max(outputs.logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if i % 100 == 0:
            print(i, correct, total)

        if i >= step_cap:
            break

    print(
        f"Accuracy of the network on the {len(val_loader) * batch_size} test images: {flor.log('test_acc', 100 * correct / total)}"
    )
