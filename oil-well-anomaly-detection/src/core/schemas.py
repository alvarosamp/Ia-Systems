"""
Pandera schemas for 3w dataframes

Two stages are validated:

1 - '''raw_schema''' - what we expect to see *immediately after* loading a single 3w parquet file. Sensors are nullable floats( the dataset has real instances with frozen/missing channels by design), the 'class'
column is 'int64' and may also be NaN and the index is a monotonically incresing ''DatetimeIndex'

2 - 'processed_schema' - what we guaranteee at the output of data/preapre. After preparation we kepp the same columns, but:
    1 - class is restricted to the set of target labels ( no events sneak in);
    2 - an extra instance_id colum tags the source file so that later splits never leak across instances
    3 - an extra ''source_event_dir' column records the event-type folder thefile came from ('0' or '2')
    
"""

from __future__ import annotations
from distro import name
import pandas as pd
import pandera.pandas as pa
from pandera.engines.pandas_engine import Datetime
from core.settings import TARGET_LABELS, EVENT_DIRS_FOR_DHSV, SENSORS
from pathlib import Path

#Helpers
def _sensor_columns(*, nullable:bool) -> dict[str, pa.Column]:
    """
    Helper to generate the dict of sensor columns for the schemas 
    
    Build the eight sensor columns. Centralised so raw/processed agree
    """
    return {
        name: pa.Column(
            float,
            nullable = nullable,
            required = True,
            description = f'3W sensor channel {name[-1]}'
        )
        for name in SENSORS
    }
    
#Raw - one parquet file as it lives on disk

raw_schema = pa.DataFrameSchema(
    columns = {
        **_sensor_columns(nullable = True),
        LABEL_COL: pa.Column(
            pd.Int64Dtype(),
            nullable = True,
            required = True,
            description = '3W class label. May be NaN for unlabeled data'
        )
    },
    index = pa.Index(
        Datetime(tz = None),
        name = None,
        unique = False,
        description = 'Observation timestamp (file index)',
    ),
    strict = False, # Raw files may have extra columns we ignore
    coerce = False, 
    ordered = False,
)

#Processed - after data/preapre, before splitting
processed_schema = pa.DataFrameSchema(
    columns = {
        **_sensor_columns(nullable = True),
        LABEL_COL: pa.Column(
            pd.Int64Dtype(),
            nullable = False,
            required = True,
            checks = pa.Check.isin(TARGET_LABELS),
            description = '3W class label. Restricted to target labels for DHSV problem'
        ),
        'instance_id': pa.Column(
            str,
            nullable = False,
            required = True,
            description = 'Unique identifier for the source file, to prevent data leakage across splits'
        ),
        'source_event_dir': pa.Column(
            str,
            nullable = False,
            required = True,
            checks = pa.Check.isin(EVENT_DIRS_FOR_DHSV),
            description = 'Event-type folder the file came from (\'0\' or \'2\')'
        ),
    },
    index = pa.Index(
        Datetime(tz = None),
        name = None,
        unique = False,
        description = 'Observation timestamp (file index)',
    ),
    strict = True, # No extra columns allowed after processing
    coerce = True,
    ordered = False,
)

__all__ = ['raw_schema', 'processed_schema']