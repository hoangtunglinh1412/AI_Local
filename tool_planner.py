import ollama

from tool_registry import TOOLS

MODEL = "qwen3:8b"


def create_plan(question, data):

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
Bạn là bộ phận lập kế hoạch
cho hệ thống phân tích dữ liệu.

Nhiệm vụ của bạn là phân tích câu hỏi
và tạo ra danh sách các bước cần thực hiện.

Các tool có thể sử dụng:

{tool_description}

Các cột trong dataset:

{column_information}

Câu hỏi của người dùng:

{question}

QUY TẮC:

QUY TẮC ƯU TIÊN:

Trước tiên, hãy xác định câu hỏi có thực sự yêu cầu phân tích dataset hay không.

1. Nếu câu hỏi chỉ là lời chào, giao tiếp xã giao hoặc câu nói không yêu cầu
phân tích dữ liệu, BẮT BUỘC sử dụng tool "none".

Ví dụ:
- "Xin chào"
- "Chào bạn"
- "Hello"
- "Cảm ơn bạn"
- "Bạn khỏe không?"
- "Ok"
- "Được rồi"

Trong các trường hợp này:
- KHÔNG được sử dụng "clarification".
- KHÔNG được sử dụng bất kỳ tool phân tích nào khác.
- Phải trả về đúng một step với tool "none".
- parameters phải là dictionary rỗng.

2. Nếu câu hỏi thực sự yêu cầu phân tích dữ liệu nhưng thiếu thông tin
cần thiết để xác định tool hoặc parameters, sử dụng "clarification".

Ví dụ:
- "So sánh các nhóm."
- "Phân tích dữ liệu này."
- "Nhóm nào có giá trị cao nhất?"

3. Nếu câu hỏi yêu cầu thông tin cơ bản về dataset như số dòng, số cột,
tên cột, cột số hoặc cột phân loại, sử dụng "none".

4. Chỉ sử dụng "clarification" khi người dùng THỰC SỰ đang yêu cầu
một nhiệm vụ liên quan đến dataset nhưng thông tin cần thiết chưa đủ rõ.

KHÔNG được coi lời chào hoặc giao tiếp xã giao là một câu hỏi thiếu thông tin.

Sau khi áp dụng các quy tắc ưu tiên trên, thực hiện các quy tắc chi tiết bên dưới.

5. Nếu câu hỏi cần nhiều phép phân tích,
hãy tạo nhiều steps.

6. Nếu câu hỏi chỉ yêu cầu thông tin cơ bản
về dataset như:
- số dòng
- số cột
- tên cột
- cột số
- cột phân loại

hãy sử dụng tool "none".

7. Nếu sử dụng tool "none",
parameters phải là dictionary rỗng.

8. Nếu câu hỏi yêu cầu một cột không tồn tại
trong dataset, hãy sử dụng tool "invalid".

9. Nếu sử dụng tool "invalid",
parameters phải chứa "reason".

10. "reason" phải mô tả rõ cột mà người dùng
yêu cầu nhưng không tồn tại trong dataset.

11. Nếu câu hỏi chưa đủ thông tin để xác định
tool hoặc parameters cần sử dụng, hãy sử dụng
tool "clarification".

12. Nếu sử dụng tool "clarification",
parameters phải chứa "question".

13. "question" phải là câu hỏi ngắn gọn để
yêu cầu người dùng cung cấp thông tin còn thiếu.

14. Không được thay thế một cột không tồn tại
bằng một cột khác.

15. Mỗi step chỉ sử dụng một tool.

16. Với group_analysis:
- group_column là cột dùng để chia nhóm.
- value_column là cột số cần phân tích.

17. Chỉ sử dụng tên cột thực sự tồn tại
trong dataset.

18. Không được tự suy đoán tên cột.

19. Không được tự trả lời câu hỏi.

20. Không tạo hai step có cùng tool
và cùng parameters.
Mỗi phép phân tích chỉ được thực hiện một lần.

21. Khi người dùng yêu cầu xác định nhóm
hoặc thành phố có giá trị cao nhất
của một cột số, hãy sử dụng tool
"max_by_group".

Không sử dụng "group_analysis"
để thay thế "max_by_group"
trong trường hợp này.

22. Chỉ trả về JSON hợp lệ.

Format:

{{
    "steps": [
        {{
            "tool": "tên_tool",
            "parameters": {{}}
        }}
    ]
}}

Ví dụ:

Câu hỏi:
"Hãy so sánh tuổi trung bình và điểm trung bình
của từng thành phố."

Trả về:

{{
    "steps": [
        {{
            "tool": "group_analysis",
            "parameters": {{
                "group_column": "city",
                "value_column": "age"
            }}
        }},
        {{
            "tool": "group_analysis",
            "parameters": {{
                "group_column": "city",
                "value_column": "score"
            }}
        }}
    ]
}}

Câu hỏi:
"Dataset có bao nhiêu dòng và bao nhiêu cột?"

Trả về:

{{
    "steps": [
        {{
            "tool": "none",
            "parameters": {{}}
        }}
    ]
}}

Câu hỏi:
"Hãy tính điểm trung bình theo phòng ban."

Trả về:

{{
    "steps": [
        {{
            "tool": "invalid",
            "parameters": {{
                "reason": "Dataset không có cột phòng ban."
            }}
        }}
    ]
}}

Câu hỏi:
"Điểm trung bình theo nhóm là bao nhiêu?"

Trả về:

{{
    "steps": [
        {{
            "tool": "clarification",
            "parameters": {{
                "question": "Bạn muốn nhóm dữ liệu theo cột nào?"
            }}
        }}
    ]
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