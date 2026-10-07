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


# ==========================================
# TEST LOAD_DATA GUARDRAILS (MILESTONE 1B)
# ==========================================
import os
import tempfile
import pandas as pd

print("\n==========================================")
print("TEST LOAD_DATA GUARDRAILS")
print("==========================================")

with tempfile.TemporaryDirectory() as temp_dir:

    # A. File không tồn tại
    non_existent = load_data("non_existent_file_xyz123.csv")
    assert non_existent is None, f"File không tồn tại phải trả về None, thực tế: {non_existent}"
    print("=> PASS: Case A (File không tồn tại -> None)")

    # B. CSV hợp lệ
    csv_valid = load_data("test.csv")
    assert csv_valid is not None and not csv_valid.empty and csv_valid.shape[0] > 0
    print("=> PASS: Case B (CSV hợp lệ -> DataFrame)")

    # C. XLSX hợp lệ
    xlsx_valid = load_data("test.xlsx")
    assert xlsx_valid is not None and not xlsx_valid.empty and xlsx_valid.shape[0] > 0
    print("=> PASS: Case C (XLSX hợp lệ -> DataFrame)")

    # D. CSV rỗng (0 bytes)
    empty_csv_path = os.path.join(temp_dir, "empty.csv")
    with open(empty_csv_path, "w", encoding="utf-8") as f:
        pass
    empty_csv_res = load_data(empty_csv_path)
    assert empty_csv_res is None, f"CSV rỗng phải trả về None, thực tế: {empty_csv_res}"
    print("=> PASS: Case D (CSV rỗng 0 bytes -> None)")

    # E. CSV chỉ có header (không có data rows)
    header_only_csv_path = os.path.join(temp_dir, "header_only.csv")
    with open(header_only_csv_path, "w", encoding="utf-8") as f:
        f.write("name,age,score\n")
    header_only_res = load_data(header_only_csv_path)
    assert header_only_res is None, f"CSV chỉ có header phải trả về None, thực tế: {header_only_res}"
    print("=> PASS: Case E (CSV chỉ có header -> None)")

    # F. XLSX không có data rows
    empty_xlsx_path = os.path.join(temp_dir, "header_only.xlsx")
    empty_df = pd.DataFrame(columns=["name", "age", "score"])
    empty_df.to_excel(empty_xlsx_path, index=False)
    empty_xlsx_res = load_data(empty_xlsx_path)
    assert empty_xlsx_res is None, f"XLSX không có data rows phải trả về None, thực tế: {empty_xlsx_res}"
    print("=> PASS: Case F (XLSX không có data rows -> None)")

    # G. Extension không hỗ trợ
    unsupported_path = os.path.join(temp_dir, "sample.txt")
    with open(unsupported_path, "w", encoding="utf-8") as f:
        f.write("hello world")
    unsupported_res = load_data(unsupported_path)
    assert unsupported_res is None, f"Extension không hỗ trợ phải trả về None, thực tế: {unsupported_res}"
    print("=> PASS: Case G (Extension không hỗ trợ -> None)")

    # H. Đường dẫn tuyệt đối
    abs_csv_path = os.path.abspath("data/test.csv")
    abs_res = load_data(abs_csv_path)
    assert abs_res is not None and not abs_res.empty and abs_res.shape[0] > 0
    print("=> PASS: Case H (Đường dẫn tuyệt đối -> DataFrame)")

print("\n>> TẤT CẢ TEST CASES LOAD_DATA ĐÃ PASS HOÀN TOÀN! <<")