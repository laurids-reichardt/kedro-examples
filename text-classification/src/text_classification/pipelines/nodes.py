import pandas as pd
import numpy as np
from typing import Dict, List
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import mlflow


def split_data(text_samples: pd.DataFrame, parameters: Dict) -> List:
    # extract features
    X = text_samples["features"].values

    # extract labels
    y = text_samples["labels"].values

    # split dataset into train and test data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )

    return [X_train, X_test, y_train, y_test]

def fit_label_binarizer(y_train: np.ndarray) -> MultiLabelBinarizer:
    # multi label binarizer to transform data labels
    mlb = MultiLabelBinarizer()

    # fit the mlb on the train label data
    mlb.fit(y_train)

    return mlb

def transform_labels(mlb: MultiLabelBinarizer, y_train: np.ndarray, y_test: np.ndarray) -> List:
    # transform train label data
    Y_train = mlb.transform(y_train)

    # transform test label data
    Y_test = mlb.transform(y_test)

    return [Y_train, Y_test]


def tfid_vectorize_fit(X_train: np.ndarray) -> TfidfVectorizer:
    # equivalent to CountVectorizer followed by TfidfTransformer
    vectorizer = TfidfVectorizer()

    # fit on train feature data
    vectorizer.fit(X_train)

    return vectorizer


def tfid_vectorize_transform(vectorizer: TfidfVectorizer, features: np.ndarray) -> np.ndarray:
    # transform feature data
    features_transformed = vectorizer.transform(features)

    return features_transformed


def train_model(X_train_transformed: np.ndarray, Y_train: np.ndarray) -> OneVsRestClassifier:
    # scikit-learn classifier
    classifier = OneVsRestClassifier(LinearSVC())

    # fit the classifier on the train data
    classifier.fit(X_train_transformed, Y_train)

    return classifier
    

def evaluate_model(classifier: OneVsRestClassifier, X_test_transformed: np.ndarray, Y_test: np.ndarray):
    # make prediction with test data
    predicted = classifier.predict(X_test_transformed)

    # accuracy score of the trained classifier
    accu = accuracy_score(Y_test, predicted)

    # log accuracy
    logger = logging.getLogger(__name__)
    logger.info("Model has an accuracy of %.3f", accu)

    # log metric to MLflow
    mlflow.log_metric("accuracy", accu)


def make_prediction(classifier: OneVsRestClassifier, mlb: MultiLabelBinarizer, features: pd.DataFrame) -> List:
    # convert dataframe of string values to numpy ndarray
    values = features.to_numpy().flatten()

    # model inference on values
    predicted = classifier.predict(values)

    # inverse transform prediction matrix back to string labels
    all_labels = mlb.inverse_transform(predicted)

    # map input values to predicted label
    predictions = []
    for item, labels in zip(values, all_labels):
        predictions.append({"value": item, "label": labels})

    # return predictions as list of dicts
    return predictions

