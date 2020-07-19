"""Construction of the master pipeline.
"""

from typing import Dict
from kedro.pipeline import Pipeline
from kedro_mlflow.pipeline import pipeline_ml

from .pipelines import pipeline

def create_pipelines(**kwargs) -> Dict[str, Pipeline]:
    """Create the project's pipeline.

    Args:
        kwargs: Ignore any additional arguments added in the future.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.

    """
    kedro_mlflow_pipeline = pipeline_ml(
        training=pipeline.create_pipeline().only_nodes_with_tags("training"),
        inference=pipeline.create_pipeline().only_nodes_with_tags("inference"),
        input_name="features"
    )

    return {
        "kedro_mlflow": kedro_mlflow_pipeline,
        "__default__": pipeline.create_pipeline().only_nodes_with_tags("training"),
    }
