from data_agent import run_data_agent


questions = [
    "Điểm trung bình của từng thành phố là bao nhiêu?",
    "Hãy so sánh tuổi trung bình và điểm trung bình của từng thành phố.",
    "Dataset có bao nhiêu dòng và bao nhiêu cột?",
    "Hãy tính điểm trung bình theo phòng ban.",
    "Hãy tính tuổi trung bình theo thành phố.",
    "Điểm trung bình theo nhóm là bao nhiêu?"
]


for index, question in enumerate(questions):

    print("\n====================================")
    print("TEST", index + 1)
    print("====================================")

    print("\nQUESTION:")
    print(question)

    result, error = run_data_agent(
        question,
        "test.xlsx"
    )

    print("\nERROR:")
    print(error)

    print("\nEXPLANATION:")

    if result is not None:
        print(
            result["explanation"]
        )
    else:
        print(
            "Không có kết quả."
        )