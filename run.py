"""Script loading pre-trained model and creating submission file of predictions."""

import numpy as np

from helpers import *
from implementations import *
from processing import *


TRAIN_DATA_PATH = "./data/train.csv"
TEST_DATA_PATH = "./data/test.csv"

# Model parameters
degrees = [12, 12, 12]


# Get columns to drop per group
_, data, _ = load_csv_data(TRAIN_DATA_PATH)
_, _, drop_cols = preprocess_data(data)

# Load pre-trained model weights
weights = load_model_weights()

# Load test data
test_labels, test_data, test_ids = load_csv_data(TEST_DATA_PATH)
# Split data
test_data, group_rows, _ = preprocess_data(test_data)

pred_ids = []
pred_labels = []

# Predict labels
for idx, (rows, cols, w, deg) in enumerate(
    zip(group_rows, drop_cols, weights, degrees)
):
    sel_data = np.delete(test_data[rows], cols, axis=1)
    tx = build_poly(sel_data, deg, sqrt=True, cbrt=True, pairs=True)
    tx = standardize(tx, bias=True)

    group_ids = test_ids[rows]
    group_pred_labels = predict_labels(w, tx)

    pred_ids = np.concatenate([pred_ids, group_ids])
    pred_labels = np.concatenate([pred_labels, group_pred_labels])

create_csv_submission(pred_ids, pred_labels, "./data/submission.csv")
