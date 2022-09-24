import time

import pandas as pd
import pyarrow as pa

import numpy as np
import random
import string
import timeit
import matplotlib.pyplot as plt


def generate_str(string_len=10):
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(string_len))

def generate_row(types_list):
    result = list()
    for t, param in types_list:
        if t == "str":
            result += [generate_str(param)]
        elif t == "int":
            result += [random.randint(param[0], param[1])]
        elif t == "float":
            result += [random.random()*param]
        else:
            raise ValueError('Wrong type')
    return result


def main():
    table_arc = [('str', 10), ('int', (-10, 200)), ('float', 5), ]
    row_nums = [100, 1_000, 10_000, 100_000, 1_000_000]
    formats = ["CSV", "feather", "h5"]

    for f_format in formats:
        print(f"{f_format}:")
        for row_num in row_nums:
            generated_data = [generate_row(table_arc + [('str', random.randint(10, 100))])
                              for _ in range(row_num)]
            df = pd.DataFrame(generated_data, columns=['a', 'b', 'c', 'd'])
            rep_num = (200 if row_num < 100_000 else 15)
            if f_format == "CSV":
                tm_w = timeit.timeit(lambda: df.to_csv("data/table.csv"), number=rep_num)/rep_num*1_000
                time.sleep(1)
                tm_r = timeit.timeit(lambda: pd.read_csv("data/table.csv"), number=rep_num)/rep_num*1_000
            elif f_format == "feather":
                tm_w = timeit.timeit(lambda: df.to_feather("data/data.feather"), number=rep_num)/rep_num*1_000
                time.sleep(1)
                tm_r = timeit.timeit(lambda: pd.read_feather("data/data.feather"), number=rep_num)/rep_num*1_000
            elif f_format == "h5":
                tm_w = timeit.timeit(lambda: df.to_hdf("data/data.h5", key="dataset", mode="w"), number=rep_num)/rep_num*1_000
                time.sleep(1)
                tm_r = timeit.timeit(lambda: pd.read_hdf("data/data.h5"), number=rep_num)/rep_num*1_000
            else:
                raise ValueError("Wrong format")

            print(f"\t{row_num}: write - {tm_w:.3f} ms; read - {tm_r:.3f} ms")



if __name__=="__main__":
    main()