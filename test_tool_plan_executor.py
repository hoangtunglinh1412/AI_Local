from data_analyzer import load_data

from tool_planner import create_plan

from tool_parser import parse_tool_plan

from tool_executor import execute_tool_plan


data = load_data(
    "test.xlsx"
)

question = (
    "Hãy so sánh tuổi trung bình "
    "và điểm trung bình của từng thành phố."
)

# CREATE PLAN

plan_response = create_plan(
    question,
    data
)

# PARSE PLAN

plan, error = parse_tool_plan(
    plan_response,
    data
)

if error:

    print(
        "\nERROR:",
        error
    )

else:

    print(
        "\n===== VALIDATED PLAN ====="
    )

    print(
        plan
    )

    # EXECUTE PLAN

    results = execute_tool_plan(
        plan,
        data
    )

    print(
        "\n===== TOOL RESULTS ====="
    )

    for result in results:

        print(
            "\nTool:",
            result["tool"]
        )

        print(
            "Parameters:",
            result["parameters"]
        )

        print(
            "Result:"
        )

        print(
            result["result"]
        )