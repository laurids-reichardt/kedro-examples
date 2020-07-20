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

    preprocessing_pipeline = pipeline.create_pipeline().only_nodes_with_tags("preprocessing")
    training_pipeline = pipeline.create_pipeline().only_nodes_with_tags("training")
    evaluation_pipeline = pipeline.create_pipeline().only_nodes_with_tags("evaluation")
    inference_pipeline = pipeline.create_pipeline().only_nodes_with_tags("inference")

    kedro_mlflow_pipeline = pipeline_ml(
        training=preprocessing_pipeline + training_pipeline + evaluation_pipeline,
        inference=inference_pipeline,
        input_name="features"
    )

    return {
        "preprocessing": preprocessing_pipeline,
        "training": training_pipeline,
        "evaluation": evaluation_pipeline,
        "kedro_mlflow": kedro_mlflow_pipeline,
        "__default__": preprocessing_pipeline + training_pipeline + evaluation_pipeline,
    }
