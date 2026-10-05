from data_analyzer import load_data

from tool_selector import select_tool


data = load_data(
    "test.xlsx"
)


if data is None:

    print(
        "Không thể đọc dataset."
    )

else:

    questions = [

        "Hãy phân tích thống kê của dataset.",

        "Hãy phân tích mối quan hệ giữa tuổi và điểm.",

        "Điểm trung bình của từng thành phố là bao nhiêu?",

        "Hãy tìm các giá trị ngoại lệ trong dataset."

    ]


    for question in questions:

        print(
            "\n=============================="
        )

        print(
            "Question:",
            question
        )

        result = select_tool(
            question,
            data
        )

        print(
            "Qwen:",
            result
        )