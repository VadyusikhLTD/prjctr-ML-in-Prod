# import concurency
import time
from concurrent.futures.process import ProcessPoolExecutor

import numpy as np
from tqdm import tqdm

def predict(input):
    time.sleep(.05)
    return np.sum(input*input.T)


def single_process(in_dataset):
    result = list()

    for arr in tqdm(in_dataset):
        result.append(predict(arr) )

    return result


def multi_process(in_dataset, num_workers=5):
    result = list()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for pred in executor.map(predict, in_dataset):
            result.append(pred)

    return result

def main():
    dataset = [np.random.random((228, 228)) for arr in range(100)]

    tm = time.time()
    single_process(dataset)
    tm = time.time() - tm
    print(f"Single process evaluation takes {tm*1_000:.2f} ms")

    for num_w in range(1, 15):
        tm = time.time()
        multi_process(dataset, num_w )
        tm = time.time() - tm
        print(f"Multiple process ({num_w} workers) evaluation takes {tm*1_000:.2f} ms")


if __name__=="__main__":
    main()

