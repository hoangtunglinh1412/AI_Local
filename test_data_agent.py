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
        assert error is None, f"Agent trả về lỗi: {error}"

    assert result is not None, "Kết quả không được là None."

    # Kiểm tra cấu trúc multi-tool hiện tại
    if "tool_results" in result:
        print(
            "\n===== PYTHON TOOL RESULTS (MULTI-TOOL) ====="
        )
        tool_results = result["tool_results"]
        for idx, item in enumerate(tool_results):
            print(f"\nStep {idx + 1} - Tool: {item['tool']}")
            print(f"Parameters: {item['parameters']}")
            print(f"Result:\n{item['result']}")

        # Kiểm tra tính đúng đắn của multi-tool results
        assert isinstance(tool_results, list) and len(tool_results) >= 2, (
            "Kế hoạch multi-tool phải có ít nhất 2 kết quả."
        )
        tools_used = [item["tool"] for item in tool_results]
        assert all(t == "group_analysis" for t in tools_used), (
            f"Các tool sử dụng phải là group_analysis, thực tế: {tools_used}"
        )
        value_columns = [
            item["parameters"].get("value_column") for item in tool_results
        ]
        assert "age" in value_columns, "Thiếu phân tích cho cột age"
        assert "score" in value_columns, "Thiếu phân tích cho cột score"

    elif "tool_result" in result:
        print(
            "\n===== PYTHON TOOL RESULT (SINGLE-TOOL) ====="
        )
        print(
            result["tool_result"]
        )
    else:
        raise KeyError("Kết quả phải chứa key 'tool_results' hoặc 'tool_result'.")

    # Kiểm tra explanation
    print(
        "\n===== AI EXPLANATION ====="
    )
    explanation = result.get("explanation")
    print(
        explanation
    )

    assert explanation is not None and len(explanation.strip()) > 0, (
        "Explanation không được rỗng."
    )

print("\n>> TEST DATA AGENT MULTI-TOOL PASSED HOÀN TOÀN! <<")