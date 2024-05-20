import os
from utils.args_parser import get_train_parsed_arguments

import torch

from torch.utils.tensorboard.writer import SummaryWriter

writer = SummaryWriter()

args = get_train_parsed_arguments()

if args.cuda:
    assert (
        args.cuda == torch.cuda.is_available()
    ), "A CUDA Device is required to use cuda"
device = torch.device("cuda" if args.cuda else "cpu")

torch.manual_seed(args.seed)

saved_model_dir = "./saved_models"
os.makedirs(saved_model_dir, exist_ok=True)

import warnings

warnings.filterwarnings("ignore")

from torchvision.models import resnet152, ResNet152_Weights
from torch import nn

weights = ResNet152_Weights.IMAGENET1K_V2
preprocess = weights.transforms()
model = resnet152(weights=weights)
model.fc = nn.Sequential(
    nn.Linear(2048, 2048),
    nn.ReLU(),
    nn.Linear(2048, 10),
)
model.to(device)
print(model)

from datasets import RecaptchaDataset as Dataset
from torch.utils import data

train_dataset = Dataset(args.dataset_path, "train")
validate_dataset = Dataset(args.dataset_path, "validate")

train_loader = data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.number_workers,
)
validate_loader = data.DataLoader(
    validate_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.number_workers,
)

optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
loss_fn = nn.CrossEntropyLoss()

from utils.train_helpers import process_large_dataset

for epoch in range(1, args.epochs):
    model.train()

    epoch_loss = 0.0
    correct = 0
    for images, targets in process_large_dataset(train_loader):
        images, targets = images.to(device), targets.to(device)

        inputs = preprocess(images).to(device)
        # outputs = model(inputs).squeeze(0).softmax(0)
        outputs = model(inputs)

        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        predicts = torch.argmax(outputs, dim=1)
        correct += (targets == predicts).sum().float()

    validate_loss, validate_correct = 0.0, 0
    model.eval()
    with torch.no_grad():
        for images, targets in process_large_dataset(validate_loader):
            images, targets = images.to(device), targets.to(device)

            inputs = preprocess(images).to(device)
            outputs = model(inputs).squeeze(0).softmax(0)

            loss = loss_fn(outputs, targets)

            validate_loss += loss.item()
            predicts = torch.argmax(outputs, dim=1)
            validate_correct += (targets == predicts).sum().float()

    train_loss = epoch_loss / len(train_loader)
    train_accuracy = correct / (len(train_loader) * train_loader.batch_size) * 100
    validate_loss = validate_loss / len(validate_loader)
    validate_accuracy = (
        validate_correct / (len(validate_loader) * validate_loader.batch_size) * 100
    )

    print(
        f"Epoch {epoch}\t|\tTrain Loss {train_loss:.4f}\tTrain Accuracy {train_accuracy:.4f}\tValidate Loss {validate_loss:.4f}\tValidate Accuracy {validate_accuracy:.4f}"
    )

    writer.add_scalar("loss/train", train_loss, epoch)
    writer.add_scalar("accuracy/train", train_accuracy, epoch)
    writer.add_scalar("loss/validate", validate_loss, epoch)
    writer.add_scalar("accuracy/validate", validate_accuracy, epoch)

    if epoch % 10 == 0:
        torch.save(
            model.state_dict(),
            os.path.join(saved_model_dir, "epoch_{}.pt".format(epoch)),
        )

writer.flush()

from datetime import datetime

torch.save(
    model.state_dict(),
    os.path.join(saved_model_dir, "final_{}.pt".format(datetime.now())),
)
