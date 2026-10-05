from data_analyzer import load_data

from tool_parser import parse_tool_response

from tool_executor import execute_tool


data = load_data(
    "test.xlsx"
)


responses = [

    '{"tool": "statistics", "parameters": {}}',

    '{"tool": "correlation", "parameters": {}}',

    '{"tool": "group_analysis", "parameters": {"group_column": "city", "value_column": "score"}}',

    '{"tool": "outliers", "parameters": {}}',

    '{"tool": "invalid", "parameters": {}}',

    '{"tool": "clarification", "parameters": {"question": "Bạn muốn nhóm dữ liệu theo cột nào?"}}'

]

for response in responses:

    print(
        "\n=============================="
    )

    print(
        "Response:",
        response
    )

    tool_result, error = parse_tool_response(
        response,
        data
    )

    if error:

        print(
            "ERROR:",
            error
        )

        continue


    result = execute_tool(
        tool_result,
        data
    )


    print(
        "\n===== TOOL RESULT ====="
    )

    print(
        result
    )