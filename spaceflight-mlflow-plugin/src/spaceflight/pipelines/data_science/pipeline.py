from kedro.pipeline import Pipeline, node

from .nodes import evaluate_model, split_data, train_model, make_prediction


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=split_data,
                inputs=["master_table", "parameters"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                tags=["training"]
            ),
            node(
                func=train_model,
                inputs=["X_train", "y_train"],
                outputs="regressor",
                tags=["training"]
            ),
            node(
                func=evaluate_model,
                inputs=["regressor", "X_test", "y_test"],
                outputs="regressor",
                tags=["training"]
            ),
            node(
                func=make_prediction,
                inputs=["regressor", "features"],
                outputs=None,
                tags=["inference"]
            ),
        ]
    )
