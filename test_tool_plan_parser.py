from data_analyzer import load_data

from tool_planner import create_plan

from tool_parser import parse_tool_plan


data = load_data(
    "test.xlsx"
)

question = (
    "Hãy so sánh tuổi trung bình "
    "và điểm trung bình của từng thành phố."
)

plan_response = create_plan(
    question,
    data
)

print(
    "\n===== RAW PLAN ====="
)

print(
    plan_response
)

plan, error = parse_tool_plan(
    plan_response,
    data
)

if error:

    print(
        "\n===== PARSER ERROR ====="
    )

    print(
        error
    )

else:

    print(
        "\n===== PARSED PLAN ====="
    )

    print(
        plan
    )