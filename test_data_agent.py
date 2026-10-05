from data_agent import run_data_agent


questions = [

    "Hãy so sánh tuổi trung bình và điểm trung bình của từng thành phố."

]

for question in questions:

    print(
        "\n========================================"
    )

    print(
        "QUESTION:",
        question
    )


    result, error = run_data_agent(
        question,
        "test.xlsx"
    )


    if error:

        print(
            "\nERROR:",
            error
        )

    else:

        print(
            "\n===== PYTHON TOOL RESULT ====="
        )

        print(
            result["tool_result"]
        )


        print(
            "\n===== AI EXPLANATION ====="
        )

        print(
            result["explanation"]
        )