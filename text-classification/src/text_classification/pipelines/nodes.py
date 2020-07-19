import pandas as pd
import numpy as np
from functools import wraps
from typing import Callable, Dict, List
import time
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


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


def train_model(X_train: np.ndarray, Y_train: np.ndarray) -> Pipeline:
    # scikit-learn classifier pipeline
    classifier = Pipeline(
        [
            ("vectorizer", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("clf", OneVsRestClassifier(LinearSVC())),
        ]
    )

    # fit the classifier on the train data
    classifier.fit(X_train, Y_train)

    return classifier


def evaluate_model(classifier: Pipeline, X_test: np.ndarray, Y_test: np.ndarray):
    # make prediction with test data
    predicted = classifier.predict(X_test)

    # accuracy score of the trained classifier
    accu = accuracy_score(Y_test, predicted)

    # log accuracy
    logger = logging.getLogger(__name__)
    logger.info("Model has an accuracy of %.3f", accu)

def synthetic_node(classifier: Pipeline, mlb: MultiLabelBinarizer) -> List:
    return [classifier, mlb]


def make_prediction(classifier: Pipeline, mlb: MultiLabelBinarizer, features: np.ndarray) -> List:
    # model inference on features
    predicted = classifier.predict(features)

    # inverse transform prediction matrix back to string labels
    all_labels = mlb.inverse_transform(predicted)

    # map input values to predicted label
    predictions = []
    for item, labels in zip(values, all_labels):
        predictions.append({"value": item, "label": labels})

    # return predictions as list of dicts
    return predictions

