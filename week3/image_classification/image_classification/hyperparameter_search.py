import wandb
import os


sweep_configuration = {
    'method': 'bayes',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'val_acc'},
    'parameters':
    {
        'batch_size': {'values': [8, 64]},
        'conv1channels_num': {'values': [20, 40, 70]},
        'conv2channels_num': {'values': [10, 30, 50]},
        'final_activation': {'values': [None, 'softmax', 'relu']}
     }
}


def train_func():
    run = wandb.init(project="prjctr-ML-in-Prod", entity="vadyusikh")
    command = f"python3 fashion_mnist.py --dataset_name fashion_mnist " \
              f"--do_train " \
              f"--do_eval " \
              f"--per_device_train_batch_size {wandb.config.batch_size} " \
              f"--conv1channels_num {wandb.config.conv1channels_num} " \
              f"--conv2channels_num {wandb.config.conv2channels_num} " \
              f"--final_activation {wandb.config.final_activation} " \
              f"--learning_rate 5e-5 " \
              f"--num_train_epochs 3 " \
              f"--model_name_or_path simpleCNN " \
              f"--output_dir tmp/"
    # --model_name_or_path, --output_dir
    print(f"Command to run : \n{command}\n")
    # raise Exception("sadfg")
    os.system(command)


if __name__ == "__main__":
    # 🐝 Step 3: Initialize sweep by passing in config
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="prjctr-ML-in-Prod", entity="vadyusikh")

    # 🐝 Step 4: Call to `wandb.agent` to start a sweep
    wandb.agent(sweep_id, function=train_func, count=8)
