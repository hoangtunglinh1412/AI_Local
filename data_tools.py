import pandas as pd

from data_analyzer import (
    analyze_statistics,
    analyze_group,
    analyze_max_by_group,
    detect_outliers,
    analyze_correlation
)

# =========================
# TOOL: STATISTICS
# =========================

def tool_statistics(data):

    result = analyze_statistics(
        data
    )

    if result is None:

        return "Không có cột dữ liệu số."

    return result.to_string()


# =========================
# TOOL: GROUP ANALYSIS
# =========================

def tool_group_analysis(
    data,
    group_column,
    value_column
):

    result = analyze_group(
        data,
        group_column,
        value_column
    )

    if result is None:

        return (
            "Không thể phân tích. "
            "Hãy kiểm tra tên cột."
        )

    output = ""

    for group in result.index:

        output += (
            "Group: "
            + str(group)
            + "\n"
        )

        output += (
            "Count: "
            + str(result.loc[group, "count"])
            + "\n"
        )

        output += (
            "Mean: "
            + str(result.loc[group, "mean"])
            + "\n"
        )

        output += (
            "Minimum: "
            + str(result.loc[group, "min"])
            + "\n"
        )

        output += (
            "Maximum: "
            + str(result.loc[group, "max"])
            + "\n\n"
        )

    return output.strip()

# =========================
# TOOL: OUTLIER
# =========================

def tool_outliers(data):

    result = detect_outliers(
        data
    )

    if len(result) == 0:

        return "Không có cột dữ liệu số."

    output = ""

    for column, info in result.items():

        output += (
            "Column: "
            + column
            + "\n"
        )

        output += (
            "Outlier count: "
            + str(info["count"])
            + "\n"
        )

        output += (
            "Outlier values: "
            + str(info["values"])
            + "\n"
        )

        output += (
            "Lower bound: "
            + str(info["lower_bound"])
            + "\n"
        )

        output += (
            "Upper bound: "
            + str(info["upper_bound"])
            + "\n\n"
        )

    return output


# =========================
# TOOL: CORRELATION
# =========================

def tool_correlation(data):

    result = analyze_correlation(
        data
    )

    if result is None:

        return "Không có cột dữ liệu số."

    return result.to_string()

def tool_max_by_group(
    data,
    group_column,
    value_column
):
    result = analyze_max_by_group(
        data,
        group_column,
        value_column
    )

    if result is None:
        return (
            "Không thể phân tích. "
            "Hãy kiểm tra tên cột."
        )

    if result.get("no_valid_values", False):
        return result

    return result