from datetime import timedelta

from feast import Entity, \
    FeatureView, \
    Field, \
    FileSource

from feast.types import Float32, String

bus = Entity(name="bus", join_keys=["id"])

bus_decimal_stats_source = FileSource(
    name="bus_decimal_stats_source",
    path="data/data_for_13_SEP_2022.parquet",
    timestamp_field="gpstime",
)

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
    source=bus_decimal_stats_source,
    tags={"team": "bus_performance"},
)
