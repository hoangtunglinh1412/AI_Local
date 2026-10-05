import ollama

from tool_registry import TOOLS


MODEL = "qwen3:8b"


def select_tool(question, data):

    tool_description = ""

    for tool_name, tool_info in TOOLS.items():

        tool_description += (
            "\nTool: "
            + tool_name
            + "\n"
        )

        tool_description += (
            "Description: "
            + tool_info["description"]
            + "\n"
        )

        tool_description += (
            "Parameters: "
            + str(tool_info["parameters"])
            + "\n"
        )

    column_information = ""

    for column in data.columns:

        column_information += (
            "- "
            + column
            + "\n"
        )

    prompt = f"""
Bạn là bộ phận lựa chọn tool cho hệ thống
phân tích dữ liệu.

Nhiệm vụ của bạn là đọc câu hỏi của người dùng
và chọn MỘT tool phù hợp nhất trong danh sách.

Danh sách các tool hiện có:

{tool_description}

Các cột trong dataset:

{column_information}

Câu hỏi của người dùng:

{question}

QUY TẮC:

1. Nếu câu hỏi yêu cầu thống kê các cột số
như mean, min, max, count hoặc describe,
hãy chọn tool "statistics".

2. Nếu câu hỏi yêu cầu tính trung bình,
min, max hoặc count theo từng nhóm,
hãy chọn tool "group_analysis".

3. Với "group_analysis":
- group_column là cột dùng để chia nhóm.
- value_column là cột số cần phân tích.
- value_column phải là cột dữ liệu số.

4. Nếu câu hỏi yêu cầu phát hiện giá trị
ngoại lệ hoặc outlier,
hãy chọn tool "outliers".

5. Nếu câu hỏi yêu cầu tương quan,
correlation hoặc mối liên hệ giữa các cột số,
hãy chọn tool "correlation".

6. Nếu câu hỏi chỉ yêu cầu thông tin cơ bản
về dataset như:
- số dòng
- số cột
- tên cột
- cột số
- cột phân loại
- kiểu dữ liệu của cột

hãy chọn tool "none".

7. Nếu câu hỏi có nhiều yêu cầu phân tích,
hãy chọn tool phù hợp nhất với phần phân tích
chính của câu hỏi.

8. Không được tự trả lời câu hỏi của người dùng.

9. Không được giải thích.

10. Chỉ được trả về JSON hợp lệ.

Format bắt buộc:

{{
    "tool": "tên_tool",
    "parameters": {{}}
}}

Ví dụ:

Câu hỏi:
"Điểm trung bình của từng thành phố là bao nhiêu?"

Trả về:

{{
    "tool": "group_analysis",
    "parameters": {{
        "group_column": "city",
        "value_column": "score"
    }}
}}

Câu hỏi:
"Có bao nhiêu dòng dữ liệu?"

Trả về:

{{
    "tool": "none",
    "parameters": {{}}
}}

Không thêm markdown.
Không thêm văn bản bên ngoài JSON.
"""

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "num_ctx": 4096
        }
    )

    answer = response[
        "message"
    ][
        "content"
    ].strip()

    return answer