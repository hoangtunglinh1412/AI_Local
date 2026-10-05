from data_agent import run_data_agent


question = (
    "Hãy so sánh tuổi trung bình "
    "và điểm trung bình của từng thành phố."
)

result, error = run_data_agent(
    question,
    "test.xlsx"
)

print(
    "\n===== ERROR ====="
)

print(
    error
)

print(
    "\n===== RESULT ====="
)

print(
    result
)