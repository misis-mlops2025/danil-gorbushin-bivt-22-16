from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32

driver_daily_metrics_source = FileSource(
    path="data/driver_daily_metrics.parquet",
    event_timestamp_column="event_timestamp",
)

driver_entity = Entity(
    name="driver",
    join_keys=["driver_id"],
    value_type=ValueType.INT64,
    description="error identifier",
)

driver_daily_metrics_view = FeatureView(
    name="driver_daily_metrics",
    entities=[driver_entity],
    ttl=timedelta(days=730),
    schema=[
        Field(name="conv_rate", dtype=Float32),
        Field(name="acc_rate", dtype=Float32),
        Field(name="avg_daily_trips", dtype=Float32),
    ],
    online=True,
    source=driver_daily_metrics_source,
    tags={"owner": "hw6"},
)
