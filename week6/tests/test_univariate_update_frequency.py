import pytest
from univariate_update_frequency import \
    get_update_references, \
    get_tables_from_folder, \
    grab_time_distributions, \
    calculate_reference_distribution


def test_all():
    table_info_list = get_tables_from_folder(DATA_PATH)
    table_distribution_list = grab_time_distributions(table_info_list)

    ref_dist = calculate_reference_distribution(table_distribution_list)
    if ref_dist.info:
        ref_dist.info["label"] = "Total ref dist"
    else:
        ref_dist.info = {"label": "Total ref dist"}

    workday_ref_dist, weekend_ref_dist = get_update_references()
    if ref_dist.info:
        workday_ref_dist.info["label"] = "Workday ref dist"
    else:
        workday_ref_dist.info = {"label": "Workday ref dist"}

    if ref_dist.info:
        weekend_ref_dist.info["label"] = "Weekend ref dist"
    else:
        weekend_ref_dist.info = {"label": "Weekend ref dist"}

    # show_distributions([ref_dist, workday_ref_dist, weekend_ref_dist])