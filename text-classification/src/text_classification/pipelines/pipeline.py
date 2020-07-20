from kedro.pipeline import Pipeline, node

from .nodes import split_data, fit_label_binarizer, transform_labels, train_model, evaluate_model, make_prediction, tfid_vectorize_fit, tfid_vectorize_transform


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                name="Split Data",
                func=split_data,
                inputs=["text_samples", "parameters"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                tags=["training"],
            ),
            node(
                name="Fit MultiLabelBinarizer",
                func=fit_label_binarizer,
                inputs="y_train",
                outputs="mlb",
                tags=["training"],
            ),
            node(
                name="Transform Labels",
                func=transform_labels,
                inputs=["mlb", "y_train", "y_test"],
                outputs=["Y_train", "Y_test"],
                tags=["training"],
            ),
            node(
                name="Fit TfidfVectorizer",
                func=tfid_vectorize_fit,
                inputs=["X_train"],
                outputs="vectorizer",
                tags=["training"],
            ),
            node(
                name="Transform X_train features",
                func=tfid_vectorize_transform,
                inputs=["vectorizer", "X_train"],
                outputs="X_train_transformed",
                tags=["training"],
            ),
            node(
                name="Transform X_test features",
                func=tfid_vectorize_transform,
                inputs=["vectorizer", "X_test"],
                outputs="X_test_transformed",
                tags=["training"],
            ),
            node(
                name="Train Model",
                func=train_model,
                inputs=["X_train_transformed", "Y_train"],
                outputs="classifier",
                tags=["training"],
            ),
            node(
                name="Evaluate Model",
                func=evaluate_model,
                inputs=["classifier", "X_test_transformed", "Y_test"],
                outputs=None,
                tags=["training"],
            ),
            node(
                name="Make Prediction",
                func=make_prediction,
                inputs=["classifier", "mlb", "features"],
                outputs="predictions",
                tags=["inference"],
            ),
        ]
    )
