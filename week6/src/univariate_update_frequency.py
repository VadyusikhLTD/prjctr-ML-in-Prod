import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from alibi_detect.cd import KSDrift
from load_data import get_tables_from_folder, get_tables_from_path
from datatypes import TableInfo, Distribution, TableDistribution
from typing import List, Tuple, Callable

from config import DATA_PATH, DATA2_PATH


def get_time_distribution(
        table_info: TableInfo,
        hist_bins: int = 24 * 6,
        hist_range: Tuple[float, float] = (-1, 23)
) -> TableDistribution:
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


def grab_time_distributions(table_info_list: List[TableInfo]) -> List[TableDistribution]:
    return [get_time_distribution(table_info) for table_info in table_info_list]


def calculate_reference_distribution(
        table_distribution_list: List[TableDistribution],
        solving_func: Callable = lambda x: np.median(x, axis=-1)
) -> Distribution:
    dist_space = table_distribution_list[0].distribution_space
    is_same_space = all([np.allclose(dist.distribution_space, dist_space)
                         for dist in table_distribution_list])
    if not is_same_space:
        raise ValueError("Tables have different distribution space")

    dist_pdf_list = [dist.distribution_pdf for dist in table_distribution_list]

    ref_dist = Distribution(
        distribution_space=dist_space,
        distribution_pdf=solving_func(np.dstack(dist_pdf_list)[0])
    )

    return ref_dist


def get_update_references(reference_distribution_path: Path = Path("../data/gps_data_update_frequency.json")):
    with open(reference_distribution_path, 'r') as f:
        ref_dists = json.load(f)
        return (
            Distribution(
                distribution_space=np.array(ref_dists["workday"][0]),
                distribution_pdf=np.array(ref_dists["workday"][1]),
                info={"name": 'workday', "label": "Workday ref dist"}
            ),
            Distribution(
                distribution_space=np.array(ref_dists["weekend"][0]),
                distribution_pdf=np.array(ref_dists["weekend"][1]),
                info={"name": 'weekend', "label": "Weekend ref dist"}
            ),
        )


def detect_ks_drift(
        ref: Distribution,
        dist_to_check_list: List[Distribution],
        p_val: float = 0.05
) -> List[Distribution]:
    drift_detector = KSDrift(ref.distribution_pdf, p_val=p_val)
    for i, dist in enumerate(dist_to_check_list):
        skd = drift_detector.predict(dist.distribution_pdf)['data']
        if dist.info:
            dist.info["KSDrift result"] = skd
        else:
            dist.info = {"KSDrift result": skd}

    return dist_to_check_list


def detect_bounded_drift(
        ref: Distribution,
        dist_to_check_list: List[Distribution],
        bounds: Tuple[float, float] = (0.5, 1.35),
        abs_tol: float = 300
) -> List[Distribution]:

    l_bound = ref.distribution_pdf * bounds[0]
    u_bound = ref.distribution_pdf * bounds[1]

    mask = ref.distribution_pdf - l_bound < abs_tol
    l_bound[mask] = ref.distribution_pdf[mask] - abs_tol

    mask = u_bound - ref.distribution_pdf < abs_tol
    u_bound[mask] = ref.distribution_pdf[mask] + abs_tol

    for i, dist in enumerate(dist_to_check_list):
        is_drifted = (l_bound < dist.distribution_pdf) & (dist.distribution_pdf < u_bound)
        b_drift_value = (1 - sum(is_drifted)/len(is_drifted))*100
        if dist.info:
            dist.info["bound drift"] = is_drifted
            dist.info["bound drift value"] = b_drift_value
        else:
            dist.info = {"bound drift": is_drifted, "bound drift value": b_drift_value}

    return dist_to_check_list


def detect_bounded_drift_on_tables(
        tables_to_check_list: List[TableInfo],
        bounds: Tuple[float, float] = (0.5, 1.35),
        abs_tol: float = 300
) -> List[Distribution]:

    table_distribution_list = grab_time_distributions(tables_to_check_list)
    workday_ref_dist, weekend_ref_dist = get_update_references()

    weekends_table_dist_list = [dist for dist in table_distribution_list
                                if dist.table_info.weekday in ['Saturday', 'Sunday']]
    weekends_table_dist_list = detect_bounded_drift(weekend_ref_dist, weekends_table_dist_list, bounds, abs_tol)

    workdays_table_dist_list = [dist for dist in table_distribution_list
                                if dist.table_info.weekday not in ['Saturday', 'Sunday']]
    workdays_table_dist_list = detect_bounded_drift(workday_ref_dist, workdays_table_dist_list, bounds, abs_tol)

    return weekends_table_dist_list + workdays_table_dist_list


def show_distributions(dists: List[Distribution]):
    plt.figure(figsize=(15, 5))
    for dist in dists:
        plt.plot(dist.distribution_space, dist.distribution_pdf,
                 label=dist.info.get('label', None),
                 linestyle=dist.info.get('linestyle', None)
                 )

    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":

    workday_ref_dist, weekend_ref_dist = get_update_references()
    show_distributions([workday_ref_dist, weekend_ref_dist])
    # table_info_list = get_tables_from_folder(DATA_PATH)[-10:]
    #
    # table_distribution_list = grab_time_distributions(table_info_list)
    # table_distribution_list = detect_KS_drift(workday_ref_dist, table_distribution_list)
    # for dist in table_distribution_list:
    #     ksd_res = dist.info['KSDrift result']
    #     dist.info['label'] = f"{dist.table_info.date.strftime('%d_%B')} ({dist.table_info.weekday}) |" \
    #                          f"is_drift - {ksd_res['is_drift']}, dist={ksd_res['distance'][0]:.4f}, p_val={ksd_res['p_val'][0]:.4f}"
    #     dist.info['linestyle'] = '-.'

    # table_distribution_list = detect_bounded_drift_on_tabels(table_info_list)
    # for dist in table_distribution_list:
    #     b_drift = dist.info['bound drift']
    #     dist.info['label'] = f"{dist.table_info.date.strftime('%d_%B')} ({dist.table_info.weekday}) |" \
    #                          f"{dist.info['bound drift value']:.2f}"
    #     dist.info['linestyle'] = '-.'

    # show_distributions([workday_ref_dist, weekend_ref_dist] + table_distribution_list)
