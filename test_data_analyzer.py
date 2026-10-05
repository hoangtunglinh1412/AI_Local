from data_analyzer import (
    load_data,
    analyze_basic,
    analyze_statistics,
    analyze_group,
    detect_outliers,
    analyze_correlation,
    create_histogram,
    create_scatter,
    generate_report
)

data = load_data(
    "test.csv"
)


if data is None:

    print("Không thể đọc dữ liệu.")

else:

    result = analyze_basic(
        data
    )

    print("\n===== DATA ANALYSIS =====")

    print(
        "Rows:",
        result["rows"]
    )

    print(
        "Columns:",
        result["columns"]
    )

    print(
        "Column names:",
        result["column_names"]
    )

    print(
        "\nMissing values:"
    )

    for column, count in result[
        "missing_values"
    ].items():

        print(
            column,
            ":",
            count
        )
        statistics = analyze_statistics(
        data
    )

    print(
        "\n===== STATISTICS ====="
    )

    if statistics is None:

        print(
            "Không có cột dữ liệu số."
        )

    else:

        print(
            statistics
        )

    group_result = analyze_group(
        data,
        "city",
        "score"
    )

    print(
        "\n===== GROUP ANALYSIS ====="
    )

    if group_result is None:

        print(
            "Không thể phân tích dữ liệu theo nhóm."
        )

    else:

        print(
            group_result
        )

    outlier_result = detect_outliers(
        data
    )

    print(
        "\n===== OUTLIER DETECTION ====="
    )

    for column, result in outlier_result.items():

        print(
            "\nColumn:",
            column
        )

        print(
            "Outlier count:",
            result["count"]
        )

        print(
            "Outlier values:",
            result["values"]
        )

        print(
            "Lower bound:",
            result["lower_bound"]
        )

        print(
            "Upper bound:",
            result["upper_bound"]
        )

    correlation = analyze_correlation(
        data
    )

    print(
        "\n===== CORRELATION ====="
    )

    if correlation is None:

        print(
            "Không có dữ liệu số để phân tích."
        )

    else:

        print(
            correlation
        )

    histogram_file = create_histogram(
        data,
        "score"
    )

    print(
        "\n===== HISTOGRAM ====="
    )

    if histogram_file is None:

        print(
            "Không thể tạo histogram."
        )

    else:

        print(
            "Đã tạo:",
            histogram_file
        )


    scatter_file = create_scatter(
        data,
        "age",
        "score"
    )

    print(
        "\n===== SCATTER PLOT ====="
    )

    if scatter_file is None:

        print(
            "Không thể tạo scatter plot."
        )

    else:

        print(
            "Đã tạo:",
            scatter_file
        )

    report = generate_report(
        data
    )

    print(
        "\n===== DATASET REPORT ====="
    )

    print(
        "Rows:",
        report["rows"]
    )

    print(
        "Columns:",
        report["columns"]
    )

    print(
        "Column names:",
        report["column_names"]
    )

    print(
        "\nNumeric columns:",
        report["numeric_columns"]
    )

    print(
        "Categorical columns:",
        report["categorical_columns"]
    )

    print(
        "\nMissing values:"
    )

    for column, count in report[
        "missing_values"
    ].items():

        print(
            column,
            ":",
            count
        )

    print(
        "\nStatistics:"
    )

    print(
        report["statistics"]
    )

    print(
        "\nOutliers:"
    )

    for column, result in report[
        "outliers"
    ].items():

        print(
            column,
            ":",
            result["values"]
        )

    print(
        "\nCorrelation:"
    )

    print(
        report["correlation"]
    )