from data_analyzer import load_data

from tool_parser import parse_tool_response


data = load_data(
    "test.xlsx"
)


responses = [

    '{"tool": "statistics", "parameters": {}}',

    '{"tool": "correlation", "parameters": {}}',

    '{"tool": "invalid", "parameters": {}}',

    '{"tool": "group_analysis", "parameters": {}}',

    '{"tool": "group_analysis", "parameters": {"group_column": "city"}}',

    '{"tool": "group_analysis", "parameters": {"group_column": "city", "value_column": "score"}}',

    '{"tool": "group_analysis", "parameters": {"group_column": "city", "value_column": "name"}}',

    '{"tool": "group_analysis", "parameters": {"group_column": "unknown", "value_column": "score"}}',

    '{"tool": "outliers", "parameters": {}}',

    '{"tool": "unknown", "parameters": {}}',

    'Đây không phải JSON'

]


for response in responses:

    print(
        "\n=============================="
    )

    print(
        "Response:",
        response
    )

    result, error = parse_tool_response(
        response,
        data
    )

    if error:

        print(
            "ERROR:",
            error
        )

    else:

        print(
            "PARSED:",
            result
        )