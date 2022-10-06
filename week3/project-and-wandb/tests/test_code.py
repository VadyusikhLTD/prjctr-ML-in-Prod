import torch
import torch.utils.data as data_utils
from models import calc_accuracy, IdentityModel

import pytest


@pytest.fixture
def my_identity_model() -> IdentityModel:
    model = IdentityModel()
    return model


@pytest.fixture
def my_test_data():
    train = data_utils.TensorDataset(torch.tensor([[.1, 0.9], [-0.5, 22], [0, 1]]), torch.tensor([1, 1, 0]) )
    train_loader = data_utils.DataLoader(train, batch_size=50, shuffle=True)
    return train_loader


def test_calc_accuracy(my_test_data, my_identity_model):
    metrics = calc_accuracy(my_test_data, my_identity_model, "cpu")
    assert abs(metrics - 2/3) < 1e-6