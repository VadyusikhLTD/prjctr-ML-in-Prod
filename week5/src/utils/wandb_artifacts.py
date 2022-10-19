import wandb

with wandb.init(project="model-registry", entity="vadyusikh") as run:
    artifact = wandb.Artifact('model_scripted', type='model')
    artifact.add_file("data/model/model_scripted.jit")
    run.log_artifact(artifact)
    run.join()


from pathlib import Path

p = Path("data/model/model_scripted.jit")
print(p.is_file(), p)

