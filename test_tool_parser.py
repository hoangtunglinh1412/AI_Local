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

    'Đây không phải JSON',

    '```json\n{"tool": "statistics", "parameters": {}}\n```',

    '```\n{"tool": "statistics", "parameters": {}}\n```',

    'Dưới đây là kết quả:\n{"tool": "statistics", "parameters": {}}\nHy vọng giúp ích.'

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

# ==========================================
# TEST ROBUST JSON FOR PARSE_TOOL_PLAN
# ==========================================
from tool_parser import parse_tool_plan

print("\n==============================")
print("TEST PARSE_TOOL_PLAN WITH MARKDOWN")
print("==============================")

plan_markdown_responses = [
    '```json\n{"steps": [{"tool": "none", "parameters": {}}]}\n```',
    'Dưới đây là kế hoạch:\n```json\n{"steps": [{"tool": "group_analysis", "parameters": {"group_column": "city", "value_column": "score"}}]}\n```\nChúc bạn thành công.'
]

for resp in plan_markdown_responses:
    print("\nInput plan:\n", resp)
    plan_res, plan_err = parse_tool_plan(resp, data)
    print("Plan Error:", plan_err)
    print("Plan Result:", plan_res)
    assert plan_err is None, f"Parse tool plan lỗi: {plan_err}"
    assert "steps" in plan_res, "Plan phải chứa steps"
    print("=> PASS: parse_tool_plan bóc tách JSON từ markdown thành công!")