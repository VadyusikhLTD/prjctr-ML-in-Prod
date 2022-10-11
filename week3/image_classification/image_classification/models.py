import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm


class SimpleCNN(nn.Module):
    def __init__(self,
                 conv1channels_num=20,
                 conv2channels_num=20,
                 class_num=10,
                 use_bn_for_input=False,
                 use_bn_inside=True,
                 final_activation=None):

        super(SimpleCNN, self).__init__()
        self.conv1channels_num = conv1channels_num
        self.conv2channels_num = conv2channels_num
        self.class_num = class_num
        self.use_bn_for_input = use_bn_for_input
        self.use_bn_inside = use_bn_inside
        self.final_activation = final_activation

        if use_bn_for_input:
            self.bn_input = nn.BatchNorm2d(1)
        else:
            self.bn_input = nn.Identity()

        if use_bn_inside:
            self.bn1 = nn.BatchNorm2d(conv1channels_num)
            self.bn2 = nn.BatchNorm2d(conv2channels_num)
        else:
            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()

        if final_activation == 'softmax':
            self.final_activation = nn.Softmax()
        elif final_activation == 'relu':
            self.final_activation = nn.ReLU()
        elif (final_activation is None) or (final_activation == 'None'):
            self.final_activation = nn.Identity()
        else:
            raise ValueError(f"Wrong final_activation - '{final_activation}'.")

        self.pool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=conv1channels_num, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=conv1channels_num, out_channels=conv2channels_num, kernel_size=3, padding=1)
        self.fc = nn.Linear(conv2channels_num * 7 * 7, class_num)


    def forward(self, x):
        if self.use_bn_for_input:
            x = self.bn_input(x)

        x = self.conv1(x)
        if self.use_bn_inside:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        if self.use_bn_inside:
            x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return self.final_activation(x)


class IdentityModel(nn.Module):
    def __init__(self):
        super(IdentityModel, self).__init__()

    def forward(self, x):
        return x


def calc_accuracy(test_data, model_, device):
    num_correct, num_samples = 0, 0
    model_.eval()

    with torch.no_grad():
        data_tqdm = tqdm(test_data, desc="Accuracy calculation", ascii=True, miniters=len(test_data)//5)
        for x, y in data_tqdm:
            x = x.to(device)
            y = y.to(device)

            pred_ = model_(x)
            _, pred_ = pred_.max(1)
            num_correct += (pred_ == y).sum()
            num_samples += pred_.size(0)

        data_tqdm.set_postfix_str(f"Accuracy is {num_correct/num_samples:.4f}")

    model_.train()
    return num_correct/num_samples


if __name__ == "__main__":
    model = SimpleCNN(
        use_bn_for_input=True,
                 use_bn_inside=True,
                 final_activation='relu')
    img = torch.randn(16, 1, 28, 28)
    res = model(img)
    print(model)
    print(f"{img.shape} -> {res.shape}")
