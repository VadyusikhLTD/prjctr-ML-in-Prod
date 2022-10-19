# WEEK 5


## Streamlit serving for model
To install streamlit run

```python3 -m pip install streamlit```



## Fast API for model + tests + CI



## Seldon API for model + tests


f"python3 fashion_mnist.py --dataset_name fashion_mnist " \
              f"--do_train " \
              f"--do_eval " \
              f"--per_device_train_batch_size 32 " \
              f"--conv1channels_num 40 " \
              f"--conv2channels_num 20 " \
              f"--final_activation relu " \
              f"--learning_rate 5e-5 " \
              f"--num_train_epochs 3 " \
              f"--model_name_or_path simpleCNN " \
              f"--output_dir tmp/"