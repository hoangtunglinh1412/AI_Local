import os
import pandas as pd


DATA_FOLDER = "data"


def load_data(filename):

    filename = os.path.normpath(
        filename
    )

    if os.path.isabs(filename):

        file_path = filename

    elif filename.startswith(
        DATA_FOLDER + os.sep
    ):

        file_path = filename

    else:

        file_path = os.path.join(
            DATA_FOLDER,
            filename
        )

    if not os.path.exists(file_path):

        return None

    extension = os.path.splitext(
        file_path
    )[1].lower()

    if extension == ".csv":

        data = pd.read_csv(
            file_path
        )

    elif extension == ".xlsx":

        data = pd.read_excel(
            file_path
        )

    else:

        return None

    return data


def analyze_basic(data):

    result = {}

    result["rows"] = data.shape[0]

    result["columns"] = data.shape[1]

    result["column_names"] = list(
        data.columns
    )

    result["missing_values"] = (
        data.isnull().sum().to_dict()
    )

    return result

def analyze_statistics(data):

    numeric_data = data.select_dtypes(
        include="number"
    )

    if numeric_data.empty:

        return None

    statistics = numeric_data.describe()

    return statistics

def analyze_group(data, group_column, value_column):

    if group_column not in data.columns:

        return None

    if value_column not in data.columns:

        return None

    result = (
        data.groupby(group_column)[value_column]
        .agg(["count", "mean", "min", "max"])
    )

    return result

def analyze_max_by_group(
    data,
    group_column,
    value_column
):
    if group_column not in data.columns:
        return None

    if value_column not in data.columns:
        return None

    if not pd.api.types.is_numeric_dtype(
        data[value_column]
    ):
        return None

    missing_count = data[value_column].isnull().sum()

    missing_group_count = data[group_column].isnull().sum()

    valid_count = data[value_column].notnull().sum()

    if valid_count == 0:
        return {
            "groups": [],
            "max_value": None,
            "missing_count": missing_count,
            "no_valid_values": True
        }

    result = (
        data.groupby(group_column)[value_column]
        .max()
    )

    if result.empty:
        return None

    max_value = result.max()

    groups = result[
        result == max_value
    ].index.tolist()

    return {
    "groups": groups,
    "max_value": max_value,
    "missing_count": missing_count,
    "missing_group_count": missing_group_count
    }

def detect_outliers(data):

    numeric_data = data.select_dtypes(
        include="number"
    )

    result = {}

    for column in numeric_data.columns:

        q1 = numeric_data[column].quantile(0.25)

        q3 = numeric_data[column].quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr

        upper_bound = q3 + 1.5 * iqr

        outliers = numeric_data[
            (numeric_data[column] < lower_bound)
            |
            (numeric_data[column] > upper_bound)
        ][column]

        result[column] = {
            "count": len(outliers),
            "values": outliers.tolist(),
            "lower_bound": lower_bound,
            "upper_bound": upper_bound
        }

    return result

def analyze_correlation(data):

    numeric_data = data.select_dtypes(
        include="number"
    )

    if numeric_data.empty:

        return None

    correlation = numeric_data.corr()

    return correlation

def create_histogram(data, column):

    if column not in data.columns:

        return None

    if not pd.api.types.is_numeric_dtype(
        data[column]
    ):

        return None

    import matplotlib.pyplot as plt

    plt.figure()

    plt.hist(
        data[column].dropna(),
        bins=10
    )

    plt.title(
        "Distribution of " + column
    )

    plt.xlabel(
        column
    )

    plt.ylabel(
        "Frequency"
    )

    plt.tight_layout()

    output_file = (
        "data/"
        + column
        + "_histogram.png"
    )

    plt.savefig(
        output_file
    )

    plt.close()

    return output_file

def create_scatter(data, x_column, y_column):

    if x_column not in data.columns:

        return None

    if y_column not in data.columns:

        return None

    if not pd.api.types.is_numeric_dtype(
        data[x_column]
    ):

        return None

    if not pd.api.types.is_numeric_dtype(
        data[y_column]
    ):

        return None

    import matplotlib.pyplot as plt

    plt.figure()

    plt.scatter(
        data[x_column],
        data[y_column]
    )

    plt.title(
        x_column
        + " vs "
        + y_column
    )

    plt.xlabel(
        x_column
    )

    plt.ylabel(
        y_column
    )

    plt.tight_layout()

    output_file = (
        "data/"
        + x_column
        + "_vs_"
        + y_column
        + ".png"
    )

    plt.savefig(
        output_file
    )

    plt.close()

    return output_file

def generate_report(data):

    if data is None:

        return None

    report = {}

    # =========================
    # BASIC INFORMATION
    # =========================

    report["rows"] = data.shape[0]

    report["columns"] = data.shape[1]

    report["column_names"] = list(
        data.columns
    )

    # =========================
    # MISSING VALUES
    # =========================

    report["missing_values"] = (
        data.isnull().sum().to_dict()
    )

    # =========================
    # NUMERIC COLUMNS
    # =========================

    numeric_data = data.select_dtypes(
        include="number"
    )

    report["numeric_columns"] = list(
        numeric_data.columns
    )

    # =========================
    # CATEGORICAL COLUMNS
    # =========================

    categorical_data = data.select_dtypes(
        exclude="number"
    )

    report["categorical_columns"] = list(
        categorical_data.columns
    )

    # =========================
    # STATISTICS
    # =========================

    if not numeric_data.empty:

        report["statistics"] = (
            numeric_data.describe()
        )

    else:

        report["statistics"] = None

    # =========================
    # OUTLIERS
    # =========================

    report["outliers"] = detect_outliers(
        data
    )

    # =========================
    # CORRELATION
    # =========================

    report["correlation"] = (
        analyze_correlation(data)
    )

    return report