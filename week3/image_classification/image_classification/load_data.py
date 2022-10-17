from pathlib import Path

import torchvision.transforms as transforms
import torchvision.datasets as datasets


def load_fashion_mnist(path_to_save: Path):
    path_to_save.mkdir(parents=True, exist_ok=True)
    datasets.FashionMNIST(root=path_to_save/Path('train'), train=True, download=True)
    datasets.FashionMNIST(root=path_to_save/Path('valid'), train=False, download=True)

import os
print(os.path.dirname(__file__) )