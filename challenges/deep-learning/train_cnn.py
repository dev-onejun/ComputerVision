import os
from utils.args_parser import get_train_parsed_arguments

from models import EfficientNet, ResNet

from datasets import RecaptchaDataset as Dataset
from torch.utils import data
import torch

from utils.train_helpers import fit_cnn, evaluate_cnn

from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime


def train(train_loader, validate_loader, model, n_epochs, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(1, n_epochs + 1):
        train_loss, train_accuracy = fit_cnn(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
        )
        validate_loss, validate_accuracy = evaluate_cnn(
            model,
            validate_loader,
            optimizer,
            loss_fn,
            device,
        )

        print(
            f"Epoch {epoch}\t|\tTrain Loss {train_loss:.4f}\tTrain Accuracy {train_accuracy:.4f}\tValidate Loss {validate_loss:.4f}\tValidate Accuracy {validate_accuracy:.4f}"
        )

        args.writer.add_scalar("loss/train", train_loss, epoch)
        args.writer.add_scalar("accuracy/train", train_accuracy, epoch)
        args.writer.add_scalar("loss/validate", validate_loss, epoch)
        args.writer.add_scalar("accuracy/validate", validate_accuracy, epoch)

        if epoch % 10 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(saved_model_dir, "epoch_{}.pt".format(epoch)),
            )

    args.writer.flush()

    torch.save(
        model.state_dict(),
        os.path.join(saved_model_dir, "final_{}.pt".format(datetime.now())),
    )


def main():
    model = EfficientNet() if args.model == "efficientnet" else ResNet()
    model.to(device)

    train_dataset = Dataset(args.dataset_path, "train", transform=model.transform)
    validate_dataset = Dataset(args.dataset_path, "validate", transform=model.transform)

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
    train(
        train_loader,
        validate_loader,
        model,
        n_epochs=args.epochs,
        lr=args.learning_rate,
    )


args = get_train_parsed_arguments()

if args.cuda:
    assert (
        args.cuda == torch.cuda.is_available()
    ), "A CUDA Device is required to use cuda"
device = torch.device("cuda" if args.cuda else "cpu")


torch.manual_seed(args.seed)

saved_model_dir = "./saved_models"
os.makedirs(saved_model_dir, exist_ok=True)

if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    writer = SummaryWriter()
    args.writer = writer

    main()
