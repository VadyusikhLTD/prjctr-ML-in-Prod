import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import pytest


@pytest.fixture()
def train_dataset() -> DataLoader:
    return datasets.FashionMNIST(root="../dataset", train=True, transform=transforms.ToTensor(), download=True)


@pytest.fixture()
def val_dataset() -> DataLoader:
    return datasets.FashionMNIST(root="../dataset", train=False, transform=transforms.ToTensor(), download=True)

    # val_dataset = datasets.FashionMNIST(root="../dataset", train=False, transform=transforms.ToTensor(), download=True)
    # return DataLoader(dataset=val_dataset, batch_size=64, shuffle=True)


def test_data_shape(train_dataset, val_dataset):
    assert train_dataset.data.shape == torch.Size([60000, 28, 28])
    assert train_dataset.targets.shape == torch.Size([60000])
    assert val_dataset.data.shape == torch.Size([10000, 28, 28])
    assert val_dataset.targets.shape == torch.Size([10000])


def test_data_boundaries(train_dataset, val_dataset):
    assert train_dataset.data.min() == 0 and train_dataset.data.max() == 255
    assert train_dataset.targets.min() == 0 and train_dataset.targets.max() == 9

    assert val_dataset.data.min() == 0 and val_dataset.data.max() == 255
    assert val_dataset.targets.min() == 0 and val_dataset.targets.max() == 9
