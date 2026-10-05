from data_analyzer import load_data

from tool_planner import create_plan

from tool_parser import parse_tool_plan

from tool_executor import execute_tool_plan

from data_agent import generate_multi_tool_explanation


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
            result["result"]
        )

    # AI EXPLANATION

    explanation = generate_multi_tool_explanation(
        question,
        data,
        results
    )

    print(
        "\n===== AI EXPLANATION ====="
    )

    print(
        explanation
    )