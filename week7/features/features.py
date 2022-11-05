from datetime import timedelta

from feast import Entity, \
    FeatureView, \
    Field, \
    FileSource

from feast.types import Float32, String

bus = Entity(name="bus", join_keys=["id"])

bus_stats_source = FileSource(
    name="bus_decimal_stats_source",
    path="../data/data_for_21_SEP_to_30_SEP_2022.parquet",
    timestamp_field="gpstime",
)
# driver_stats_source = FileSource(
#     name="driver_hourly_stats_source",
#     path="/app/data/data_for_21_SEP_to_30_SEP_2022.parquet",
#     timestamp_field="event_timestamp",
#     created_timestamp_column="created",
# )

bus_stats_fv = FeatureView(
    name="bus_decimal_stats_source",
    entities=[bus],
    ttl=timedelta(minutes=10),
    schema=[
        Field(name="lat", dtype=Float32),
        Field(name="lng", dtype=Float32),
        Field(name="routeName", dtype=String),
    ],
    online=True,
    source=bus_stats_source,
    tags={"team": "bus_performance"},
)

# driver_stats_fv = FeatureView(
#     name="driver_hourly_stats",
#     entities=[driver],
#     ttl=timedelta(days=1),
#     schema=[
#         Field(name="conv_rate", dtype=Float32),
#         Field(name="acc_rate", dtype=Float32),
#         Field(name="avg_daily_trips", dtype=Int64),
#     ],
#     online=True,
#     source=driver_stats_source,
#     tags={"team": "driver_performance"},
# )
