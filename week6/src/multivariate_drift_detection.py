import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from alibi_detect.cd import KSDrift
import folium
import pymap3d

from load_data import get_tables_from_folder, get_tables_from_path
from datatypes import TableInfo, Bound2D, Distribution2D, TableDistribution2D
from typing import List, Tuple, Callable
import webbrowser


from config import \
    DATA_PATH, \
    DATA2_PATH, \
    CHERNIVTSI_CENTER, \
    CHERNIVTSI_BOUND_2D


def convert_step_from_meters_to_geod(
        base_point_geod: np.array,
        step: Tuple[float, float] = (100, 100)
) -> Tuple[float, float]:
    lat1, lon1, _ = pymap3d.enu2geodetic(
        e=step[0], n=step[1], u=0,
        lat0=base_point_geod[0], lon0=base_point_geod[1], h0=0,
        ell=None, deg=True)
    return lat1 - base_point_geod[0], lon1 - base_point_geod[1]


def get_coordinate_distribution(
        table_info: TableInfo,
        bounds: Bound2D,
        step: Tuple[float, float] = (100, 100)
) -> TableDistribution2D:
    df = pd.read_csv(table_info.path)

    df.dropna(subset=['lat', 'lng'], inplace=True)
    mask = (bounds.x.lower < df["lat"]) & (df["lat"] < bounds.x.upper)  # 48.0 < 48.5
    mask &= (bounds.y.lower < df["lng"]) & (df["lng"] < bounds.y.upper) # 25.8 < 26.1
    df = df[mask]

    dist = np.histogram2d(
        df["lat"].values,
        df["lng"].values,
        range=[[bounds.x.lower, bounds.x.upper],
               [bounds.y.lower, bounds.y.upper]],
        bins=(int((bounds.x.upper - bounds.x.lower)//step[0]),
              int((bounds.y.upper - bounds.y.lower)//step[1]))
    )

    return TableDistribution2D(
        table_info=table_info,
        distribution_pdf=dist[0],
        distribution_x_space=dist[1][:-1],
        distribution_y_space=dist[2][:-1],
        bounds=bounds
    )


def grab_coordinate_distribution(
        table_info_list: List[TableInfo],
        bounds: Bound2D,
        step: Tuple[float, float] = (100, 100)
) -> List[TableDistribution2D]:
    result = list()

    for table_info in tqdm(table_info_list,
                           desc="Grabbing coordinate distribution from table info list"):
        result += [get_coordinate_distribution(table_info, bounds, step)]
    return result


def calculate_reference_distribution2d(
        table_distribution2d_list: List[TableDistribution2D],
        solving_func: Callable = lambda x: np.median(x, axis=-1)
) -> Distribution2D:
    dist_x_space = table_distribution2d_list[0].distribution_x_space
    dist_y_space = table_distribution2d_list[0].distribution_y_space
    is_same_space = all(
        [np.allclose(dist.distribution_x_space, dist_x_space) & np.allclose(dist.distribution_y_space, dist_y_space)
         for dist in table_distribution2d_list])
    if not is_same_space:
        raise ValueError("Tables have different distribution space")

    dist_pdf_list = [dist.distribution_pdf for dist in table_distribution2d_list]

    ref_dist = Distribution2D(
        distribution_x_space=dist_x_space,
        distribution_y_space=dist_y_space,
        distribution_pdf=solving_func(np.dstack(dist_pdf_list)),
        bounds=table_distribution2d_list[0].bounds
    )

    return ref_dist


def get_coordinate_reference_distribution(
        reference_distribution_path: Path = Path("../data/coordinate_distribution2d.json")):
    with open(reference_distribution_path, 'r') as f:
        ref_dists = json.load(f)["coordinate_distribution2d"]

        return Distribution2D(
            distribution_x_space=np.array(ref_dists["distribution_x_space"]),
            distribution_y_space=np.array(ref_dists["distribution_y_space"]),
            distribution_pdf=np.array(ref_dists["distribution_pdf"]),
            bounds=None,
            info={"name": 'allday', "label": "Allday ref dist"}
        )


def detect2d_ks_drift(
        ref: Distribution2D,
        dist_to_check_list: List[Distribution2D],
        p_val: float = 0.05
) -> List[Distribution2D]:

    drift_detector = KSDrift(ref.distribution_pdf, p_val=p_val)
    for i, dist in enumerate(dist_to_check_list):
        skd = drift_detector.predict(dist.distribution_pdf)['data']
        if dist.info:
            dist.info["KSDrift result"] = skd
        else:
            dist.info = {"KSDrift result": skd}

    return dist_to_check_list


def show_distribution_heatmap(dists: Distribution2D, map_file_save_path: Path = Path("../data/map.html")) -> None:
    map_hooray = folium.Map(location=CHERNIVTSI_CENTER, zoom_start=12)

    folium.PolyLine([[CHERNIVTSI_BOUND_2D.x.lower, CHERNIVTSI_BOUND_2D.y.lower],
                     [CHERNIVTSI_BOUND_2D.x.lower, CHERNIVTSI_BOUND_2D.y.upper],
                     [CHERNIVTSI_BOUND_2D.x.upper, CHERNIVTSI_BOUND_2D.y.upper],
                     [CHERNIVTSI_BOUND_2D.x.upper, CHERNIVTSI_BOUND_2D.y.lower],
                     [CHERNIVTSI_BOUND_2D.x.lower, CHERNIVTSI_BOUND_2D.y.lower]
                     ]).add_to(map_hooray)

    points = list()
    for i, x in enumerate(dists.distribution_x_space):
        for j, y in enumerate(dists.distribution_y_space):
            points += [[x, y]] * int(dists.distribution_pdf[i, j])

    # Plot it on the map
    folium.plugins.HeatMap(points, radius=5, blur=3).add_to(map_hooray)

    # Display the map
    map_hooray.save(map_file_save_path)
    webbrowser.open(f"file://{str(map_file_save_path.resolve())}", new=2)


if __name__ == "__main__":
    step_unit = convert_step_from_meters_to_geod(CHERNIVTSI_CENTER, (100, 100))
    table_info_list = get_tables_from_folder(DATA_PATH) + get_tables_from_folder(DATA2_PATH)
    table_distribution_list = grab_coordinate_distribution(table_info_list[-5:], bounds=CHERNIVTSI_BOUND_2D, step=step_unit)

    ref_dist = get_coordinate_reference_distribution()
    show_distribution_heatmap(ref_dist)
    table_distribution_list = detect2d_ks_drift(ref_dist, table_distribution_list)
    for dist in table_distribution_list:
        ksd_res = dist.info['KSDrift result']
        fpath = Path(f"../data/map-dist={ksd_res['distance'][0]:.4f}-p_val={ksd_res['p_val'][0]:.4f}.html")
        dist.distribution_pdf -= ref_dist.distribution_pdf
        dist.distribution_pdf = np.abs(dist.distribution_pdf)
        show_distribution_heatmap(dist, fpath)
