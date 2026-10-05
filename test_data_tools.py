from data_analyzer import load_data

from data_tools import (
    tool_statistics,
    tool_group_analysis,
    tool_outliers,
    tool_correlation
)


data = load_data(
    "test.xlsx"
)


if data is None:

    print(
        "Không thể đọc dữ liệu."
    )

else:

    print(
        "\n===== TOOL: STATISTICS ====="
    )

    print(
        tool_statistics(data)
    )


    print(
        "\n===== TOOL: GROUP ANALYSIS ====="
    )

    print(
        tool_group_analysis(
            data,
            "city",
            "score"
        )
    )


    print(
        "\n===== TOOL: OUTLIERS ====="
    )

    print(
        tool_outliers(data)
    )


    print(
        "\n===== TOOL: CORRELATION ====="
    )

    print(
        tool_correlation(data)
    )