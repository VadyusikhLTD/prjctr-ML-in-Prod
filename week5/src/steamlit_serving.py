import streamlit as st
import torch
from predictor import Predictor


@st.cache(hash_funcs={Predictor: lambda _: None})
def get_model() -> Predictor:
    return Predictor.load_from_model_registry()


predictor = get_model()


def single_pred():
    if st.button("Run inference"):
        img = torch.rand(1, 1, 28, 28)
        pred = predictor.predict(img)
        prob = torch.nn.functional.softmax(pred)
        st.write(f"Class {pred.argmax()} predicted, it's probability is {prob[0][pred.argmax()]:.4f}")


def batch_pred():
    if st.button("Batch inference"):
        img = torch.rand(4, 1, 28, 28)
        pred = predictor.predict(img)
        prob = torch.nn.functional.softmax(pred)
        for i, c_num in enumerate(pred.argmax(axis=1)):
            st.write(f"Class {c_num} predicted, it's probability is {prob[i][c_num]:.4f}")


def main():
    st.header("UI serving demo")

    tab1, tab2 = st.tabs(["Single prediction", "Batch prediction"])

    with tab1:
        single_pred()

    with tab2:
        batch_pred()


if __name__ == "__main__":
    main()
