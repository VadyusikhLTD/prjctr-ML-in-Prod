import torch
import torch.nn as nn
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from models import SimpleCNN, calc_accuracy
from fashion_mnist import train_loop
from utils.DataTrainingArguments import DataTrainingArguments
from utils.ModelArguments import ModelArguments
from transformers import TrainingArguments, set_seed


import pytest


@pytest.fixture
def model() -> SimpleCNN:
    model = SimpleCNN()
    return model


@pytest.fixture
def simple_data():
    return torch.rand(16, 1, 28, 28)


@pytest.fixture()
def model_args() -> ModelArguments:
    return ModelArguments(
        model_name_or_path="simpleCNN",
        conv1channels_num=40,
        conv2channels_num=20,
        final_activation="relu"
        )


@pytest.fixture()
def data_args() -> DataTrainingArguments:
    return DataTrainingArguments(
        dataset_name='fashion_mnist',
    )


@pytest.fixture()
def one_batch_training_args() -> TrainingArguments:
    return TrainingArguments(
        do_train=True,
        do_eval=False,
        learning_rate=1e-4,
        num_train_epochs=300,
        output_dir="/tmp/test",
        per_device_train_batch_size=16
    )


@pytest.fixture()
def training_args() -> TrainingArguments:
    return TrainingArguments(
        do_train=True,
        do_eval=False,
        learning_rate=1e-4,
        num_train_epochs=3,
        output_dir="/tmp/test",
        per_device_train_batch_size=16
    )


@pytest.fixture()
def train_data_loader() -> DataLoader:
    train_dataset = datasets.FashionMNIST(root="../dataset", train=True, transform=transforms.ToTensor(), download=True)
    return DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)


@pytest.fixture()
def val_data_loader() -> DataLoader:
    val_dataset = datasets.FashionMNIST(root="../dataset", train=False, transform=transforms.ToTensor(), download=True)
    return DataLoader(dataset=val_dataset, batch_size=64, shuffle=True)


def test_simple_predict(model, simple_data):
    result = model(simple_data)
    assert result.shape == torch.Size([16, 10])


def test_overfit_batch(model: nn.Module, train_data_loader: DataLoader, one_batch_training_args: TrainingArguments):
    batch = next(iter(train_data_loader))
    dt_set = data_utils.TensorDataset(batch[0], batch[1])
    train_loader = data_utils.DataLoader(dt_set, batch_size=64, shuffle=True)

    train_loop(train_loader, model, one_batch_training_args, device='cpu', is_use_wandb=False)
    train_acc = calc_accuracy(train_loader, model, 'cpu')

    assert train_acc > 0.95


def test_train_to_completion(
        model: nn.Module,
        train_data_loader: DataLoader,
        val_data_loader: DataLoader,
        training_args: TrainingArguments
):
    train_loop(train_data_loader, model, training_args, device='cpu', is_use_wandb=False)

    val_acc = calc_accuracy(val_data_loader, model, 'cpu')

    assert val_acc > 0.80
