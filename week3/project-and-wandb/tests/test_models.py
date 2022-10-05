import torch
# from torch import nn
# import torch.nn.functional as F
# from tqdm import tqdm

import pytest

import image_classification.models.SimpleCNN as SimpleCNN

@pytest.fixture
def my_model():
    return SimpleCNN()



# class SimpleCNN(nn.Module):
#     def __init__(self,
#                  conv1channels_num=20,
#                  conv2channels_num=20,
#                  class_num=10,
#                  use_bn_for_input=False,
#                  use_bn_inside=True,
#                  final_activation=None):



# def calc_accuracy(test_data, model_, device):
#     num_correct, num_samples = 0, 0
#     model_.eval()
#
#     with torch.no_grad():
#         data_tqdm = tqdm(test_data, desc="Accuracy calculation", ascii=True, miniters=len(test_data)//5)
#         for x, y in data_tqdm:
#             x = x.to(device)
#             y = y.to(device)
#
#             pred_ = model_(x)
#             _, pred_ = pred_.max(1)
#             num_correct += (pred_ == y).sum()
#             num_samples += pred_.size(0)
#
#     model_.train()
#     return num_correct/num_samples


if __name__ == "__main__":
    model = SimpleCNN(
        use_bn_for_input=True,
                 use_bn_inside=True,
                 final_activation='relu')
    img = torch.randn(16, 1, 28, 28)
    res = model(img)
    print(model)
    print(f"{img.shape} -> {res.shape}")