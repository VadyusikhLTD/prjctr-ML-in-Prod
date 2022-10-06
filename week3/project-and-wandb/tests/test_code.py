import torch
import torch.utils.data as data_utils
from models import SimpleCNN

import pytest


@pytest.fixture
def my_model() -> SimpleCNN:
    model = SimpleCNN()
    return model


@pytest.fixture
def my_test_data():
    return torch.rand(16, 1, 28, 28)


def test_simple_predict(my_model, my_test_data):
    result = my_model(my_test_data)
    assert result.shape == torch.Size([16, 10])
