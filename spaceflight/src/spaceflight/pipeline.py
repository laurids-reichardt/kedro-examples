"""Construction of the master pipeline.
"""

from typing import Dict
from kedro.pipeline import Pipeline

from .pipelines.data_science import pipeline as ds
from .pipelines.data_engineering import pipeline as de
from .pipelines.data_engineering.nodes import log_running_time


def create_pipelines(**kwargs):
    """Create the project's pipeline.

    Args:
        kwargs: Ignore any additional arguments added in the future.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.

    """

    data_engineering_pipeline = de.create_pipeline().decorate(log_running_time)
    data_science_pipeline = ds.create_pipeline().decorate(log_running_time)

    return {
        "de": data_engineering_pipeline,
        "ds": data_science_pipeline,
        "__default__": data_engineering_pipeline + data_science_pipeline,
    }
