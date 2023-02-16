"""Module containing functions to process and transform the data, as well as for feature engineering."""

import numpy as np


DER_mass_MMC = 0
PRI_jet_num = 22
PRI_jet_all_pt = 29


def preprocess_data(data):
    """Performs the first steps of data processing:
        - Replaces -999.0 with NaN
        - Imputes missing data
        - Performs any data transformations
        - Removes outliers
        - Splits the data into groups and identifies which columns are meaningless for each group

    Does not standardise the data.
    """

    # -999.0 used to denote NaN
    data[data == -999.0] = np.nan

    # Impute data
    data[:, DER_mass_MMC] = impute_col(data, DER_mass_MMC)
    # Treat 0 as missing value for PRI_jet_all_pt. Replace it with the median
    data[:, PRI_jet_all_pt] = impute_col(data, PRI_jet_all_pt, 0)

    # Transform features
    data = log_transform(data)
    data = pseudorapidity_transformation(data)
    data = angular_transformation(data)

    # Remove outliers
    data = remove_outliers(data, 0.05)

    # Split data into 3 groups
    group_rows, drop_cols = group_data(data)

    return data, group_rows, drop_cols


def impute_col(data, col, val=None):
    """Replace missing data in a column with the median."""

    col_data = data[:, col]

    if val is None:
        median = np.nanmedian(col_data)
        data[:, col] = np.where(np.isnan(data[:, col]), median, data[:, col])
    else:
        median = np.median(col_data[col_data != val])
        data[:, col] = np.where(data[:, col] == val, median, data[:, col])

    return data[:, col]


def standardize(x, bias=False):
    """Compute the standard scores of the data. If bias is True, ignores the first column."""

    mean, std = np.mean(x, axis=0), np.std(x, axis=0)
    x = x - mean
    x[:, std > 0] = x[:, std > 0] / std[std > 0]

    # Reset normalisation of intercept column
    if bias:
        x[:, 0] = 1

    return x


def abs_transform(data):
    """Absolute value transform on identified columns."""

    idxs = [6, 11]
    data[:, idxs] = np.abs(data[:, idxs])
    return data


def log_transform(data):
    """Log transform on identified columns."""

    idxs = [0, 1, 2, 3, 5, 8, 9, 10, 13, 16, 19, 21, 23, 26, 29]
    data[:, idxs] = np.log1p(data[:, idxs])
    return data


def pseudorapidity_transformation(data):
    """Transform of pseudorapidity features."""

    idxs = [14, 17, 24, 27]
    data[:, idxs] = np.abs(data[:, idxs]) ** (3 / 2)
    return data


def angular_transformation(data):
    """Angular trasformation on identified columns."""

    idxs = [15, 18, 20, 25, 28]
    data[:, idxs] = np.cos(data[:, idxs] + np.pi * 0.25) + np.sin(
        data[:, idxs] + np.pi * 0.25
    )
    return data


def remove_outliers(data, alpha=0.05):
    """Given a quantile, substitute data outside this treshold with the threshold itself."""

    for i in range(data.shape[1]):
        upper_lim = np.nanquantile(data[:, i], 1 - alpha)
        lower_lim = np.nanquantile(data[:, i], alpha)

        data[np.where(data[:, i] > upper_lim), i] = upper_lim
        data[np.where(data[:, i] < lower_lim), i] = lower_lim

    return data


def group_data(data):
    """Return groups of rows of data points, as well as columns to ignore per group."""

    group_rows_0 = np.where(data[:, PRI_jet_num] == 0)
    group_rows_1 = np.where(data[:, PRI_jet_num] == 1)
    group_rows_2 = np.where((data[:, PRI_jet_num] == 2) | (data[:, PRI_jet_num] == 3))

    drop_cols = (
        [4, 5, 6, 12, 22, 23, 24, 25, 26, 27, 28, 29],
        [4, 5, 6, 12, 22, 26, 27, 28],
        [22],
    )

    return (group_rows_0, group_rows_1, group_rows_2), drop_cols


### Feature expansion


def build_poly(x, degree, sqrt=False, cbrt=False, pairs=False):
    """Polynomial basis functions for input data x, for j=0 up to j=degree.
    Optionally can add square or cube roots of x as additional features,
    or the basis of products between the features.

    Args:
        x: numpy array of shape (N,). N is the number of samples
        degree: integer
        sqrt: boolean
        cbrt: boolean
        pairs: boolean
    Returns:
        poly: numpy array of shape (N,d+1)
    """

    poly = np.ones((len(x), 1))
    for deg in range(1, degree + 1):
        poly = np.c_[poly, np.power(x, deg)]

    if sqrt:
        poly = np.c_[poly, np.sqrt(np.abs(x))]

    if cbrt:
        poly = np.c_[poly, np.cbrt(x)]

    if pairs:
        for i in range(x.shape[1]):
            for j in range(i + 1, x.shape[1]):
                poly = np.c_[poly, x[:, i] * x[:, j]]

    return poly
