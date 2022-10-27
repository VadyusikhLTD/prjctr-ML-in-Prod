import wandb
from pathlib import Path


def add_wandb_artifact(artifact_path: Path, artifact_name: str, type: str) -> None:
    with wandb.init(project="model-registry", entity="vadyusikh") as run:
        artifact = wandb.Artifact(artifact_name, type=type)
        artifact.add_file(artifact_path)
        run.log_artifact(artifact)
        run.join()


def load_wandb_artifact(model_path: Path, model_name: str, type: str):
    with wandb.init() as run:
        model_path.mkdir(parents=True, exist_ok=True)
        artifact = run.use_artifact(model_name, type=type)
        artifact_dir = artifact.download(root=model_path)
        print(f"{type} '{model_name}' loaded to f'{artifact_dir}'")


if __name__ == "__main__":
    add_wandb_artifact(
        artifact_path=Path("data/model/model_scripted.jit"),
        artifact_name='model_scripted',
        type='model'
    )


