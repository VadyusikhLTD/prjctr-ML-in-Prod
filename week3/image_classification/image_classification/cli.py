import typer

from load_data import load_fashion_mnist
from fashion_mnist import full_train
# from nlp_sample.utils import load_from_registry, upload_to_registry
# from nlp_sample.predictor import run_inference_on_dataframe

app = typer.Typer()
app.command()(full_train)
app.command()(load_fashion_mnist)
# app.command()(upload_to_registry)
# app.command()(load_from_registry)
# app.command()(run_inference_on_dataframe)


if __name__ == "__main__":
    app()