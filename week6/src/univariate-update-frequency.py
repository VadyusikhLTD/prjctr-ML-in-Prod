# import json
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
# from alibi_detect.cd import KSDrift
from load_data import get_tables_from_folder, get_tables_from_path
from datatypes import TableInfo, Distribution, TableDistribution, ReferenceDistribution
from typing import List, Tuple


def get_time_distribution(
        table_info: TableInfo,
        hist_bins: int = 24 * 6,
        hist_range: Tuple[float, float] = (-1, 23)
) -> Distribution:
    df = pd.read_csv(table_info.path)
    updates_time = pd.to_datetime(df['gpstime']).values - np.datetime64(table_info.date)
    updates_time = updates_time.astype(float) / 1e9 / 3600
    dist = np.histogram(updates_time, bins=hist_bins, range=hist_range)
    dist[0][0] += np.sum(updates_time < dist[1][0])
    dist[0][-1] += np.sum(updates_time > dist[1][-1])

    return TableDistribution(
        table_info=table_info,
        distribution_pdf=dist[0],
        distribution_space=dist[1][:-1]
    )


def grab_time_distributions(table_info_list: List[TableInfo]) -> List[Distribution]:
    return [get_time_distribution(table_info) for table_info in table_info_list]


def get_update_references(reference_distribution_path: Path = Path("../data/gps_data_update_frequency.json")):
    with open(reference_distribution_path, 'r') as f:
        ref_dists = json.load(f)
        return (
            ReferenceDistribution(
                distribution_space=ref_dists["workday"][0],
                distribution_pdf=ref_dists["workday"][1],
                info={"name": 'workday'}
            ),
            ReferenceDistribution(
                distribution_space=ref_dists["weekend"][0],
                distribution_pdf=ref_dists["weekend"][1],
                info={"name": 'weekend'}
            ),
        )


DATA_PATH = Path('../../../pet_project/tables')
DATA2_PATH = Path('../../../PTETA/PTETA/data/local/tables')


def show_distributions(dists: List[Distribution]):
    plt.figure(figsize=(15, 5) )
    for dist in dists:
        label, linestyle = None, None
        if isinstance(dist, ReferenceDistribution):
            label = f"Reference {dist.info['name']}"
            linestyle = '-'
        elif isinstance(dist, TableDistribution):
            label = f"Reference {dist.table_info.date}"
            linestyle = '--'

        plt.plot(dist.distribution_space,
                 dist.distribution_pdf,
                 # label=f"is_drift: {skd['is_drift']}, dist: {skd['distance'][0]:.3f}, p_val: {skd['p_val'][0]:.3f}",
                 label=label,
                 linestyle=linestyle)

    plt.legend()
    plt.grid()
    # f.suptitle(wd_name)
    plt.show()


if __name__ == "__main__":
    # table_info_list = get_tables_from_folder(DATA2_PATH)[:5]
    # print(grab_time_distributions(table_info_list))
    # print(get_update_references())

    show_distributions(list(get_update_references()))
