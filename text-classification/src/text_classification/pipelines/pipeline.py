from kedro.pipeline import Pipeline, node

from .nodes import (
    split_data,
    fit_label_binarizer,
    transform_labels,
    train_model,
    evaluate_model,
    make_prediction,
    tfid_vectorize_fit,
    tfid_vectorize_transform,
    transform_df_to_ndarray,
)


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                name="Split data",
                func=split_data,
                inputs=["text_samples", "parameters"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                tags=["preprocessing"],
            ),
            node(
                name="Fit MultiLabelBinarizer",
                func=fit_label_binarizer,
                inputs="y_train",
                outputs="mlb",
                tags=["preprocessing"],
            ),
            node(
                name="Transform labels",
                func=transform_labels,
                inputs=["mlb", "y_train", "y_test"],
                outputs=["Y_train", "Y_test"],
                tags=["preprocessing"],
            ),
            node(
                name="Fit TfidfVectorizer",
                func=tfid_vectorize_fit,
                inputs=["X_train"],
                outputs="vectorizer",
                tags=["preprocessing"],
            ),
            node(
                name="Transform X_train features",
                func=tfid_vectorize_transform,
                inputs=["vectorizer", "X_train"],
                outputs="X_train_transformed",
                tags=["preprocessing"],
            ),
            node(
                name="Transform X_test features",
                func=tfid_vectorize_transform,
                inputs=["vectorizer", "X_test"],
                outputs="X_test_transformed",
                tags=["preprocessing"],
            ),
            node(
                name="Train model",
                func=train_model,
                inputs=["X_train_transformed", "Y_train"],
                outputs="classifier",
                tags=["training"],
            ),
            node(
                name="Evaluate model",
                func=evaluate_model,
                inputs=["classifier", "X_test_transformed", "Y_test"],
                outputs=None,
                tags=["evaluation"],
            ),
            node(
                name="Transform inference input to ndarray",
                func=transform_df_to_ndarray,
                inputs=["features"],
                outputs="features_ndarray",
                tags=["inference"],
            ),
            node(
                name="Vectorize inference input",
                func=tfid_vectorize_transform,
                inputs=["vectorizer", "features_ndarray"],
                outputs="features_transformed",
                tags=["inference"],
            ),
            node(
                name="Make prediction from inference input",
                func=make_prediction,
                inputs=["classifier", "mlb", "features", "features_transformed"],
                outputs="predictions",
                tags=["inference"],
            ),
        ]
    )
