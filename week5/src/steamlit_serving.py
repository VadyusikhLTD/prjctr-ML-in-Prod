import streamlit as st
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from pathlib import Path

from predictor import Predictor


@st.cache(hash_funcs={Predictor: lambda _: None})
def get_model() -> Predictor:
    return Predictor.load_from_model_registry()

predictor = get_model()


@st.cache(hash_funcs={DataLoader: lambda _: None})
def load_fashion_mnist(dataset_path: Path = "data/dataset") -> DataLoader:
    test_dataset = datasets.FashionMNIST(
        root=dataset_path,
        train=False,
        transform=transforms.ToTensor(),
        download=True)
    return DataLoader(dataset=test_dataset, batch_size=10, shuffle=True)


dataloader = load_fashion_mnist()


def get_message(pred: torch.Tensor, label: int) -> str:
    pred_label = pred.argmax()
    prob = torch.nn.functional.softmax(pred)

    if pred_label != label:
        return f"Class {pred_label} predicted, it's probability is {prob[0][pred_label]:.4f}" \
               f"\n\nReal label {label}, it's probability is {prob[0][label]:.4f}"
    else:
        return f"Right class {pred_label} predicted, it's probability is {prob[0][pred_label]:.4f}"


def single_pred():
    imgs, labels = next(iter(dataloader))
    st.image(imgs[0][0].numpy(), caption=f"Class {labels[0].item()}")
    if st.button("Run inference"):
        pred = predictor.predict(imgs[0].unsqueeze(0))
        st.write(get_message(pred, labels[0].item()))


def batch_pred():
    imgs, labels = next(iter(dataloader))

    if st.button("Batch inference"):
        pred = predictor.predict(imgs)
        for i in range(0, len(imgs) // 2):
            cols = st.columns(2)

            cols[0].image(
                imgs[i * 2][0].numpy(),
                caption=get_message(pred[i * 2].unsqueeze(0), labels[i*2].item()),
                use_column_width=True
            )
            cols[1].image(
                imgs[i * 2 + 1][0].numpy(),
                caption=get_message(pred[i * 2+1].unsqueeze(0), labels[i*2+1].item()),
                use_column_width=True
            )


def main():
    st.header("UI serving demo")

    tab1, tab2 = st.tabs(["Single prediction", "Batch prediction"])

    with tab1:
        single_pred()

    with tab2:
        batch_pred()


if __name__ == "__main__":
    main()
