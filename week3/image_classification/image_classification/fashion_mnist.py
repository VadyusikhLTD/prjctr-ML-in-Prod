import logging
import os
import sys

from models import SimpleCNN, calc_accuracy

import torch
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from tqdm import tqdm

from utils.DataTrainingArguments import DataTrainingArguments
from utils.ModelArguments import ModelArguments

USE_WANDB = True

if USE_WANDB:
    import wandb
    wandb.init(project="prjctr-ML-in-Prod", entity="vadyusikh")

logger = logging.getLogger(__name__)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if data_args.dataset_name == 'fashion_mnist':
        # LOAD DATA
        train_dataset = datasets.FashionMNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
        train_loader = DataLoader(dataset=train_dataset, batch_size=training_args.train_batch_size, shuffle=True)

        test_dataset = datasets.FashionMNIST(root="dataset/", train=False, transform=transforms.ToTensor(), download=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=training_args.train_batch_size, shuffle=True)
    else:
        raise ValueError(f"Proper dataset not provided! Provided '{data_args.dataset_name}'")

    # INIT NETWORK
    model = None
    if model_args.model_name_or_path.lower() == 'SimpleCNN'.lower():
        model = SimpleCNN().to(device)
    else:
        raise ValueError(f"Proper model not provided! Provided '{model_args.model_name_or_path}'")

    if training_args.do_train:
        logger.info("*** Training ***")

        if USE_WANDB:
            wandb.config = {
                "learning_rate": training_args.learning_rate,
                "epochs": training_args.num_train_epochs,
                "batch_size": training_args.train_batch_size,
                "seed": training_args.seed
            }

        # LOSS AND OPTIMIZER
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=training_args.learning_rate)

        # TRAIN
        update_rate = len(train_loader)//20
        epoch_tqdm = tqdm(range(int(training_args.num_train_epochs)), desc="Epochs", ascii=True)
        for epoch in epoch_tqdm:
            if epoch > 0:
                checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}

            train_tqdm = tqdm(enumerate(train_loader), desc="Training batches", ascii=True, total=len(train_loader))
            loss_vals = list()
            for step, (data, target) in train_tqdm:
                data = data.to(device)
                target = target.to(device)

                #forward
                pred = model(data)
                loss = loss_fn(pred, target)

                #backward
                optimizer.zero_grad()
                loss.backward()

                #optimizer step
                optimizer.step()

                if step % update_rate == 0 and step > 0:
                    mean_loss = torch.tensor(loss_vals).mean()
                    train_tqdm.set_postfix_str(f"Train mean loss is {mean_loss:.4f} (step no. {step})")
                    if USE_WANDB:
                        wandb.log({"train_loss": mean_loss}, step=step + epoch*len(train_loader))
                else:
                    loss_vals += [loss.item()]

            train_acc = calc_accuracy(train_loader, model, device)
            epoch_tqdm.set_postfix_str(f"Train acc is {train_acc:.4f}")
            if USE_WANDB:
                wandb.log({"train_accuracy": train_acc, 'epoch': epoch})

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        test_acc = calc_accuracy(test_loader, model, device)
        logger.info(f"Test acc is {test_acc:.4f}")
        if USE_WANDB:
            wandb.log({"test_accuracy": test_acc})


if __name__ == "__main__":
    main()
