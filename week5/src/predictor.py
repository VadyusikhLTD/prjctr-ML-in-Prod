import logging
from pathlib import Path

import numpy as np
import torch
import wandb
from filelock import FileLock

logger = logging.getLogger()

MODEL_ID = 'vadyusikh/model-registry/model_scripted:v0'
MODEL_PATH = Path("data/model")
MODEL_NAME = 'model_scripted.jit'
MODEL_LOCK = ".lock-file"


def load_from_registry(model_name: str, model_path: Path):
    with wandb.init() as run:
        model_path.mkdir(parents=True, exist_ok=True)
        artifact = run.use_artifact(model_name, type="model")
        artifact_dir = artifact.download(root=model_path)
        print(f"Model '{model_name}' loaded to f'{artifact_dir}'")


class Predictor:
    def __init__(self, model_load_path: Path):
        self.model = torch.jit.load(model_load_path)
        self.model.eval()

    @torch.no_grad()
    def predict(self, in_img: np.array):
        return self.model(in_img)

    @classmethod
    def load_from_model_registry(cls) -> 'Predictor':
        with FileLock(MODEL_LOCK):
            MODEL_PATH.mkdir(parents=True, exist_ok=True)
            if not (MODEL_PATH / MODEL_NAME).exists():
                load_from_registry(model_name=MODEL_ID, model_path=MODEL_PATH )

        return cls(model_load_path=MODEL_PATH / MODEL_NAME)
