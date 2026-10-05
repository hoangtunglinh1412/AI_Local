from data_analyzer import load_data

from tool_selector import select_tool

from tool_parser import parse_tool_response

from tool_executor import execute_tool


data = load_data(
    "test.xlsx"
)


question = "Dataset này có bao nhiêu dòng?"


print(
    "\n===== QUESTION ====="
)

print(
    question
)


tool_response = select_tool(
    question,
    data
)


print(
    "\n===== TOOL SELECTOR ====="
)

print(
    tool_response
)


tool_result, error = parse_tool_response(
    tool_response,
    data
)


print(
    "\n===== PARSER ====="
)

if error:

    print(
        "ERROR:",
        error
    )

else:

    print(
        tool_result
    )


if error is None:

    result = execute_tool(
        tool_result,
        data
    )

    print(
        "\n===== EXECUTOR ====="
    )

    print(
        result
    )