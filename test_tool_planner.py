from data_analyzer import load_data
from tool_planner import create_plan


data = load_data(
    "test.xlsx"
)

questions = [
    "Hãy tính điểm trung bình theo phòng ban.",
    "Điểm trung bình theo nhóm là bao nhiêu?"
]


for question in questions:

    print(
        "\n========================================"
    )

    print(
        "QUESTION:",
        question
    )

    plan = create_plan(
        question,
        data
    )

    print(
        "\n===== TOOL PLAN ====="
    )

    print(
        plan
    )