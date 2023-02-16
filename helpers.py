"""Module containing helper functions for reading and writing data, as well as making predictions."""

import csv
import pickle

import numpy as np


def load_csv_data(data_path, sub_sample=False, logistic=False):
    """Loads data and returns `y` (class labels), `tX` (features) and `ids` (event ids).

    Args:
        data_path: Data path of data to load.
        sub_sample: Return subsample of the data.
        logistic: Load labels to be used for logistic regression.

    Returns:
        yb: Class labels.
        input_data: Feature values.
        ids: IDs of each data sample.
    """

    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # Convert class labels from strings to binary (-1, 1)
    yb = np.ones(len(y))
    yb[np.where(y == "b")] = 0 if logistic else -1

    # Sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def create_csv_submission(ids, y_pred, name):
    """Creates an output file in .csv format for submission to Kaggle or AIcrowd.

    Args:
        ids: Event IDs associated with each prediction.
        y_pred: Predicted class labels.
        name: String name of .csv output file to be created.
    """

    with open(name, "w") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})


def predict_labels(weights, features, logistic=False):
    """Create predictions given weights and features.

    Args:
        weights: Model weight vector.
        features: Sample feature vectors.
        logistic: Quantise prediction for logistic regression.

    Returns:
        y: Predictions for each data sample.
    """

    y = np.dot(features, weights)

    if logistic:
        y[y > 0.5] = 1
        y[y <= 0.5] = -1
    else:
        y[y > 0.0] = 1
        y[y <= 0.0] = -1

    return y


def save_model_weights(w, file_name="model_weights.pickle"):
    """Save model weights into a pickle file.

    Args:
        w: Model weights.
        file_name: String name of .pickle file to be created.
    """

    with open(file_name, "wb") as file:
        pickle.dump(w, file)


def load_model_weights(file_name="model_weights.pickle"):
    """Load model weights from a pickle file.

    Args:
        file_name: String name of .pickle file to load weights from.

    Returns:
        weights: Model weights loaded from pickle file.
    """

    with open(file_name, "rb") as file:
        weights = pickle.load(file)

    return weights
