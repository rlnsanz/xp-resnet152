import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor, ResNetForImageClassification
from datasets import load_dataset, DatasetDict
import flor
from PIL import Image

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

# Load dataset
data = load_dataset("imagenet-1k")
assert isinstance(data, DatasetDict)
assert set(data.keys()) == {"train", "validation", "test"}

# Initialize feature extractor and model
feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-152")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-152").to(device)


def preprocess_function(examples):
    images = []
    for image in examples["image"]:
        # Convert images to RGB
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        if image.mode != "RGB":
            image = image.convert("RGB")
        images.append(image)
    # Process images without return_tensors="pt"
    inputs = feature_extractor(images)
    examples["pixel_values"] = inputs["pixel_values"]
    return examples


# Apply preprocessing with a smaller batch size
data["train"] = data["train"].map(
    preprocess_function, batched=True, batch_size=8, remove_columns=["image"]
)
data["validation"] = data["validation"].map(
    preprocess_function, batched=True, batch_size=8, remove_columns=["image"]
)
data["test"] = data["test"].map(
    preprocess_function, batched=True, batch_size=8, remove_columns=["image"]
)

# Set the format of the datasets to return PyTorch tensors
data["train"].set_format(type="torch", columns=["pixel_values", "label"])
data["validation"].set_format(type="torch", columns=["pixel_values", "label"])
data["test"].set_format(type="torch", columns=["pixel_values", "label"])

# Create DataLoaders without a custom collate function
train_loader = DataLoader(dataset=data["train"], batch_size=batch_size, shuffle=True)
val_loader = DataLoader(
    dataset=data["validation"], batch_size=batch_size, shuffle=False
)
test_loader = DataLoader(dataset=data["test"], batch_size=batch_size, shuffle=False)

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
            images = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images)
            loss = criterion(outputs.logits, labels)

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
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    step_cap = flor.arg("eval_step_cap", 1000)
    print(f"evaluating for {step_cap} of {len(test_loader)} rounds")
    for i, batch in enumerate(test_loader):
        images = batch["pixel_values"].to(device)
        labels = batch["label"].to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if i % 100 == 0:
            print(i, correct, total)

        if i >= step_cap:
            break

    print(
        f"Accuracy of the network on the {len(test_loader) * batch_size} test images: {flor.log('test_acc', 100 * correct / total)}"
    )
