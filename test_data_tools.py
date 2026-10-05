from data_analyzer import load_data

from data_tools import (
    tool_statistics,
    tool_group_analysis,
    tool_max_by_group,
    tool_outliers,
    tool_correlation
)


data = load_data(
    "test.xlsx"
)


if data is None:

    print(
        "Không thể đọc dữ liệu."
    )

else:

    print(
        "\n===== TOOL: STATISTICS ====="
    )

    print(
        tool_statistics(data)
    )


    print(
        "\n===== TOOL: GROUP ANALYSIS ====="
    )

    print(
        tool_group_analysis(
            data,
            "city",
            "score"
        )
    )


    print(
        "\n===== TOOL: OUTLIERS ====="
    )

    print(
        tool_outliers(data)
    )


    print(
        "\n===== TOOL: CORRELATION ====="
    )

    print(
        tool_correlation(data)
    )

# ==========================================
# TEST MAX_BY_GROUP (V4.54)
# ==========================================
from data_agent import run_data_agent

print("\n==========================================")
print("TEST MAX_BY_GROUP: CASE A - NORMAL MAXIMUM")
print("==========================================")
res_a = tool_max_by_group(data, "city", "score")
print("[Result A]:", res_a)
assert res_a["groups"] == ["Hanoi"], f"City lớn nhất phải là Hanoi, thực tế: {res_a['groups']}"
assert res_a["max_value"] == 95, f"Max value phải là 95, thực tế: {res_a['max_value']}"
assert res_a["missing_count"] == 0
assert res_a["missing_group_count"] == 0
print("=> PASS: Case A (Normal Maximum)")

print("\n==========================================")
print("TEST MAX_BY_GROUP: CASE B - TIE (ĐỒNG HẠNG)")
print("==========================================")
data_tie = load_data("test_tie.xlsx")
res_b = tool_max_by_group(data_tie, "city", "score")
print("[Result B]:", res_b)
assert set(res_b["groups"]) == {"Danang", "HCM"}, f"Các nhóm đồng hạng phải là Danang và HCM, thực tế: {res_b['groups']}"
assert len(res_b["groups"]) == 2, f"Phải trả về đủ 2 nhóm đồng hạng, thực tế: {len(res_b['groups'])}"
assert res_b["max_value"] == 95, f"Max value phải là 95, thực tế: {res_b['max_value']}"
print("=> PASS: Case B (Tie)")

print("\n==========================================")
print("TEST MAX_BY_GROUP: CASE C - MISSING VALUES")
print("==========================================")
data_missing = load_data("test_missing.xlsx")
res_c = tool_max_by_group(data_missing, "city", "score")
print("[Result C]:", res_c)
assert res_c["missing_count"] == 1, f"Missing count phải là 1, thực tế: {res_c['missing_count']}"
assert res_c["groups"] == ["Hanoi"], f"Nhóm lớn nhất từ các giá trị hợp lệ phải là Hanoi, thực tế: {res_c['groups']}"
assert res_c["max_value"] == 95.0, f"Max value phải là 95.0, thực tế: {res_c['max_value']}"
print("=> PASS: Case C (Missing Values)")

print("\n==========================================")
print("TEST MAX_BY_GROUP: CASE D - ALL VALUES MISSING")
print("==========================================")
data_all_missing = load_data("test_all_missing.xlsx")
res_d = tool_max_by_group(data_all_missing, "city", "score")
print("[Result D]:", res_d)
assert res_d.get("no_valid_values") is True, f"no_valid_values phải là True, thực tế: {res_d.get('no_valid_values')}"
assert res_d["max_value"] is None, f"max_value phải là None, thực tế: {res_d['max_value']}"
assert res_d["groups"] == [], f"groups phải rỗng [], thực tế: {res_d['groups']}"
assert res_d["missing_count"] == 5, f"missing_count phải là 5, thực tế: {res_d['missing_count']}"
print("=> PASS: Case D (All Values Missing)")

print("\n==========================================")
print("TEST MAX_BY_GROUP: CASE E - MISSING GROUP")
print("==========================================")
data_missing_group = load_data("test_missing_group.xlsx")
res_e = tool_max_by_group(data_missing_group, "city", "score")
print("[Result E]:", res_e)
assert res_e["missing_group_count"] == 1, f"missing_group_count phải là 1, thực tế: {res_e['missing_group_count']}"
assert res_e["groups"] == ["Hanoi"]
assert res_e["max_value"] == 95
print("=> PASS: Case E (Missing Group)")

print("\n==========================================")
print("TEST MAX_BY_GROUP: CASE F - NEGATIVE VALUES")
print("==========================================")
data_negative = load_data("test_negative.xlsx")
res_f = tool_max_by_group(data_negative, "city", "score")
print("[Result F]:", res_f)
assert res_f["max_value"] == -5, f"Max value theo toán học phải là -5, thực tế: {res_f['max_value']}"
assert set(res_f["groups"]) == {"Danang", "HCM"}, f"Nhóm lớn nhất phải là Danang và HCM, thực tế: {res_f['groups']}"
print("=> PASS: Case F (Negative Values)")

print("\n==========================================")
print("TEST MAX_BY_GROUP: CASE G - END-TO-END AGENT")
print("==========================================")
question_g = "Điểm số lớn nhất thuộc về thành phố nào?"
result_g, error_g = run_data_agent(question_g, "test.xlsx")
print("[Agent Error G]:", error_g)
print("[Agent Result G]:", result_g)

assert error_g is None, f"Agent trả về lỗi: {error_g}"
assert result_g is not None, "Result không được là None"
assert "tool_result" in result_g, "Kết quả phải chứa 'tool_result'"
tool_res_g = result_g["tool_result"]
assert tool_res_g["groups"] == ["Hanoi"], f"Nhóm lớn nhất phải là Hanoi, thực tế: {tool_res_g['groups']}"
assert tool_res_g["max_value"] == 95, f"Max value phải là 95, thực tế: {tool_res_g['max_value']}"
assert "Hanoi" in result_g["explanation"], f"Explanation phải nhắc đến Hanoi: {result_g['explanation']}"
assert "95" in result_g["explanation"], f"Explanation phải nhắc đến 95: {result_g['explanation']}"
print("=> PASS: Case G (End-to-End Agent with max_by_group)")

print("\n>> TẤT CẢ CÁC TEST CASES CỦA MAX_BY_GROUP ĐÃ PASS HOÀN TOÀN! <<")