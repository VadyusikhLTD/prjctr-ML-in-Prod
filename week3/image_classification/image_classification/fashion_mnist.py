import logging
import os
import sys
from pathlib import Path

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

import wandb

logger = logging.getLogger(__name__)
MODELS_DIR_PARH = Path("../data/models")


def full_train():
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

    if data_args.is_use_wandb:
        wandb.init(project="prjctr-ML-in-Prod", entity="vadyusikh")

    train(model_args, data_args, training_args)


def train(model_args, data_args, training_args):
    # Set seed before initializing model.
    set_seed(training_args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if data_args.dataset_name == 'fashion_mnist':
        # LOAD DATA
        train_dataset = datasets.FashionMNIST(root=data_args.dataset_path, train=True, transform=transforms.ToTensor(), download=True)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=training_args.train_batch_size, shuffle=True)

        test_dataset = datasets.FashionMNIST(root=data_args.dataset_path, train=False, transform=transforms.ToTensor(), download=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=training_args.train_batch_size, shuffle=True)
    else:
        raise ValueError(f"Proper dataset not provided! Provided '{data_args.dataset_name}'")

    # INIT NETWORK
    model = None
    if model_args.model_name_or_path.lower() == 'SimpleCNN'.lower():
        model = SimpleCNN(
            conv1channels_num=model_args.conv1channels_num,
            conv2channels_num=model_args.conv2channels_num,
            final_activation=model_args.final_activation
        ).to(device)
    else:
        raise ValueError(f"Proper model not provided! Provided '{model_args.model_name_or_path}'")

    if training_args.do_train:
        logger.info("*** Training ***")

        if data_args.is_use_wandb:
            wandb.config = {
                "learning_rate": training_args.learning_rate,
                "epochs": training_args.num_train_epochs,
                "batch_size": training_args.train_batch_size,
                "seed": training_args.seed,
                'conv1channels_num': model_args.conv1channels_num,
                'conv2channels_num': model_args.conv2channels_num,
                'final_activation': model_args.final_activation
            }

        train_loss = train_loop(train_dataloader, model, training_args, device, is_use_wandb=data_args.is_use_wandb)

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        test_acc = calc_accuracy(test_loader, model, device)
        logger.info(f"Test acc is {test_acc:.4f}")
        if data_args.is_use_wandb:
            wandb.log({"val_acc": test_acc})

    if model_args.save_model:
        if training_args.do_eval:
            torch.save(model, MODELS_DIR_PARH/f"simple_model_valid_acc-{train_loss:.4f}.pt")
        elif training_args.do_train:
            torch.save(model, MODELS_DIR_PARH/f"simple_model_train_loss-{test_acc:.4f}.pt")
        else:
            torch.save(model, MODELS_DIR_PARH/f"simple_model.pt")


def train_loop(train_dataloader, model, training_args, device, is_use_wandb, optimizer=None, loss_fn=None):
    # LOSS AND OPTIMIZER
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=training_args.learning_rate)

    # TRAIN
    update_rate = max(1, len(train_dataloader)//20)
    epoch_tqdm = tqdm(range(int(training_args.num_train_epochs)), desc="Epochs", ascii=True)
    for epoch in epoch_tqdm:
        if epoch > 0:
            checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}

        train_tqdm = tqdm(enumerate(train_dataloader), desc="Training batches", ascii=True, total=len(train_dataloader),
                                  leave=True, miniters=len(train_dataloader)//10)
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

            loss_vals += [loss.item()]
            if step % update_rate == 0:
                mean_loss = torch.tensor(loss_vals).mean()
                loss_vals.clear()
                train_tqdm.set_postfix_str(f"Train mean loss is {mean_loss:.4f} (step no. {step})")
                if is_use_wandb:
                    wandb.log({"train_loss": mean_loss}, step=step + epoch*len(train_dataloader))


        train_acc = calc_accuracy(train_dataloader, model, device)
        epoch_tqdm.set_postfix_str(f"Train acc is {train_acc:.4f}")
        if is_use_wandb:
            wandb.log({"train_accuracy": train_acc, 'epoch': epoch})

    return loss.item()


if __name__ == "__main__":
    full_train()
