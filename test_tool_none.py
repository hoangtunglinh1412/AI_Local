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

# ==========================================
# TEST GREETING ROUTING (V4.54)
# ==========================================
from tool_planner import create_plan
from tool_parser import parse_tool_plan
from data_agent import run_data_agent

print("\n==========================================")
print("TEST GREETING ROUTING: 'Xin chào'")
print("==========================================")

greeting_question = "Xin chào"

# 1. Kiểm tra Planner
plan_raw = create_plan(greeting_question, data)
print("\n[Raw Plan]:", plan_raw)

parsed_plan, parse_error = parse_tool_plan(plan_raw, data)
print("\n[Parsed Plan]:", parsed_plan)
print("[Parse Error]:", parse_error)

assert parse_error is None, f"Parse error: {parse_error}"
assert len(parsed_plan["steps"]) == 1, "Plan phải có đúng 1 step"
step = parsed_plan["steps"][0]
assert step["tool"] == "none", f"Tool phải là 'none', thực tế là: {step['tool']}"
assert step["parameters"] == {}, f"Parameters phải là {{}}, thực tế là: {step['parameters']}"
print("\n=> PASS: Planner đã định tuyến 'Xin chào' thành tool 'none' với parameters {}")

# 2. Kiểm tra toàn trình qua run_data_agent
result, agent_error = run_data_agent(greeting_question, "test.xlsx")
print("\n[Agent Error]:", agent_error)
print("[Agent Result]:", result)

assert agent_error is None, f"Agent trả về lỗi: {agent_error}"
assert result is not None, "Result không được là None"
explanation = result.get("explanation", "")
expected_explanation = "Xin chào! Tôi có thể giúp bạn phân tích dữ liệu."

assert explanation == expected_explanation, (
    f"Explanation không đúng. Thực tế: '{explanation}', Mong đợi: '{expected_explanation}'"
)
print(f"\n=> PASS: run_data_agent trả về đúng lời chào: '{explanation}'")
print("\n>> GREETING ROUTING V4.54 TEST PASSED HOÀN TOÀN! <<")