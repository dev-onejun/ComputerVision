from torch.utils.data import Dataset
from torchvision import transforms
from pandas.core.common import flatten

import glob
from PIL import Image
from torchvision.models import (
    EfficientNet_V2_L_Weights,
    ResNet152_Weights,
    ResNet50_Weights,
)


"""
class TestDataset(Dataset):
    def __init__(self, dataset_path, transform):
        super(TestDataset, self).__init__()

        image_paths = []
        for image_path in glob.glob(dataset_path + "/*"):
            image_paths.append(image_path)

        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        image = image.convert("RGB")
        image = self.transform(image)

        return image
        """


class TestDataset(Dataset):
    def __init__(self, dataset_path, transform):
        super(TestDataset, self).__init__()

        labels, image_paths = [], []
        for dataset in glob.glob(dataset_path + "/*"):
            label = dataset.split("/")[-1]
            labels.append(label)

            image_paths_for_label = glob.glob(dataset + "/*")
            image_paths.append(image_paths_for_label)

        self.image_paths = list(flatten(image_paths))
        self.idx_to_label = {i: j for i, j in enumerate(labels)}
        self.label_to_idx = {value: key for key, value in self.idx_to_label.items()}
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        image = image.convert("RGB")
        image = self.transform(image)

        label = self.image_paths[idx].split("/")[-2]
        label = self.label_to_idx[label]

        return image, label


from torch import nn
import torch
from CNNs import ResNet50, ResNet152, EfficientNet


class VotingEnsemble(nn.Module):
    def __init__(self, **kwargs):
        super(VotingEnsemble, self).__init__()

        self.resnet50 = ResNet50()
        self.resnet50.load_state_dict(torch.load(kwargs["resnet50"]))
        for name, param in self.resnet50.named_parameters():
            param.requires_grad = False

        self.resnet152 = ResNet152()
        self.resnet152.load_state_dict(torch.load(kwargs["resnet152"]))
        for name, param in self.resnet152.named_parameters():
            param.requires_grad = False

        self.efficientnetv2 = EfficientNet()
        self.efficientnetv2.load_state_dict(torch.load(kwargs["efficientnetv2"]))
        for name, param in self.efficientnetv2.named_parameters():
            param.requires_grad = False

    def forward(self, resnet50_input, resnet152_input, efficientnetv2_input):
        outputs = [
            self.resnet50(resnet50_input),
            self.resnet152(resnet152_input),
            self.efficientnetv2(efficientnetv2_input),
        ]
        outputs = [torch.argmax(output, dim=1) for output in outputs]

        temp = outputs.copy()

        # Stack outputs to shape (num_models, batch_size, num_classes)
        outputs = torch.stack(outputs)

        mode_outputs, mode_indicies = torch.mode(outputs, dim=0)

        for i, mode_index in enumerate(mode_indicies):
            # Refers to the frequency of the classes is same
            if mode_index == 0:
                # Follows the highest accuracy model while training
                mode_outputs[i] = temp[0]

        return mode_outputs


def process_large_dataset(dataloader):
    for (resnet50, _), (resnet152, _), (efficientnetv2, _targets) in zip(
        dataloader["resnet50"],
        dataloader["resnet152"],
        dataloader["efficientnetv2"],
    ):
        yield resnet50, resnet152, efficientnetv2, _targets


if __name__ == "__main__":
    DATASET_PATH = "./test-dataset"

    import warnings

    warnings.filterwarnings("ignore")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from torchvision.models import (
        ResNet152_Weights,
        ResNet50_Weights,
        EfficientNet_V2_L_Weights,
    )

    transforms = {
        "efficientnetv2": EfficientNet_V2_L_Weights.IMAGENET1K_V1.transforms(),
        "resnet50": ResNet50_Weights.IMAGENET1K_V2.transforms(),
        "resnet152": ResNet152_Weights.IMAGENET1K_V2.transforms(),
    }

    from torch.utils import data

    dataset = {}
    dataloader = {}
    for key, transform in transforms.items():
        dataset[key] = TestDataset(DATASET_PATH, transform=transform)
        dataloader[key] = data.DataLoader(
            dataset[key],
            batch_size=32,
            num_workers=8,
        )

    model = VotingEnsemble(
        resnet50="./saved_models/resnet50_acc61.pt",
        resnet152="./saved_models/resnet152_acc60.pt",
        efficientnetv2="./saved_models/efficientnetv2.pt",
    )
    model.to(device)

    """ No required
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=0
    )
    """

    correct = 0

    model.eval()
    with torch.no_grad():
        for (
            resnet50_input,
            resnet152_input,
            efficientnetv2_input,
            _targets,
        ) in process_large_dataset(dataloader):
            resnet50_input = resnet50_input.to(device)
            resnet152_input = resnet152_input.to(device)
            efficientnetv2_input = efficientnetv2_input.to(device)

            _targets = _targets.to(device)

            outputs = model(resnet50_input, resnet152_input, efficientnetv2_input)

            correct += (_targets == outputs).sum().float()

        accuracy = correct / len(dataloader["resnet50"].dataset) * 100

    print(accuracy.item())
