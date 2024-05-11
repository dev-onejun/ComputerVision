from train import create_vit_classifier, get_parse_arguments

from transformers import ViTImageProcessor, ViTPreTrainedModel
import torch
from torch import nn
from torch.utils import data

from datasets.RecaptchaDataset import RecaptchaDataset

from tqdm import tqdm

from transformers import ViTModel, ViTPreTrainedModel


class ViTClassifier(ViTPreTrainedModel):
    def __init__(self, config):
        super(ViTClassifier, self).__init__(config)

        self.vit = ViTModel(config)

        self.classifier = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10),
        )

        for name, param in self.vit.named_parameters():
            if "classifier" not in name:  # Freeze layers that are not the classifier
                param.requires_grad = False

    def forward(self, pixel_values):
        vit_outputs = self.vit(pixel_values=pixel_values)
        vit_output = vit_outputs.pooler_output

        logits = self.classifier(vit_output)

        return logits


args = get_parse_arguments()

processor = ViTImageProcessor.from_pretrained(args.vit_pretrained_model)
# model = create_vit_classifier(args)
model = ViTClassifier.from_pretrained(args.vit_pretrained_model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# processor.to(device)
model.to(device)

# Data transformation and loader
train_dataset = RecaptchaDataset(args.dataset_path, "train")
train_loader = data.DataLoader(train_dataset, batch_size=32, shuffle=True)

validate_dataset = RecaptchaDataset(args.dataset_path, "validate")
validate_loader = data.DataLoader(
    validate_dataset, batch_size=args.batch_size, shuffle=False
)


def process_large_dataset(dataloader):
    for data in tqdm(dataloader):
        yield data


# Optimizer (only optimize parameters that require gradients)
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4
)

num_epochs = 50

for epoch in range(num_epochs):
    model.train()

    epoch_loss = 0.0
    correct = 0
    # for images, targets in tqdm(train_loader):
    for images, targets in process_large_dataset(train_loader):
        images, targets = images.to(device), targets.to(device)

        inputs = processor(images, do_rescale=False)
        inputs = torch.Tensor(inputs["pixel_values"]).to(device)
        outputs = model(inputs)

        loss = nn.CrossEntropyLoss()(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        predicts = torch.argmax(outputs, dim=1)
        correct += (targets == predicts).sum().float()

        """
        batch_correct = (targets == predicts).sum().float()
        batch_accuracy = batch_correct / args.batch_size * 100
        print(f"Step Accuracy {batch_accuracy:.4f}")
        """

    model.eval()
    validate_epoch_loss = 0.0
    validate_correct = 0
    with torch.no_grad():
        # for images, targets in tqdm(validate_loader):
        for images, targets in process_large_dataset(validate_loader):
            images, targets = images.to(device), targets.to(device)

            inputs = processor(images, do_rescale=False)
            inputs = torch.Tensor(inputs["pixel_values"]).to(device)
            outputs = model(inputs)

            loss = nn.CrossEntropyLoss()(outputs, targets)

            validate_epoch_loss += loss.item()
            predicts = torch.argmax(outputs, dim=1)
            validate_correct += (targets == predicts).sum().float()

    epoch_loss = epoch_loss / len(train_loader)
    accuracy = correct / (len(train_loader) * args.batch_size) * 100

    validate_epoch_loss = validate_epoch_loss / len(validate_loader)
    validate_accuracy = (
        validate_correct / (len(validate_loader) * args.batch_size) * 100
    )

    print(
        f"Epoch {epoch + 1}\nTrain Accuracy: {accuracy:.4f}\tTrain Loss: {epoch_loss:.4f}\tValidate Accuracy: {validate_accuracy:.4f}\tValidate Loss: {validate_epoch_loss:.4f}\n"
    )

from datetime import datetime

torch.save(model.state_dict(), f"./{datetime.now()}.pt")
