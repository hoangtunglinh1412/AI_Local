import ollama

from data_analyzer import load_data

from tool_selector import select_tool

from tool_parser import parse_tool_response
from tool_parser import parse_tool_plan

from tool_executor import execute_tool
from tool_executor import execute_tool_plan

from tool_planner import create_plan

MODEL = "qwen3:8b"


def run_data_agent(
    question,
    filename
):

    # LOAD DATA
    data = load_data(
        filename
    )

    if data is None:
        return None, (
            "Không thể đọc dataset."
        )

    # CREATE TOOL PLAN
    plan_response = create_plan(
        question,
        data
    )

    print(
        "\n[Tool Planner]"
    )

    print(
        plan_response
    )

    # PARSE + VALIDATE PLAN
    plan, error = parse_tool_plan(
        plan_response,
        data
    )

    if error:
        return None, error

    print("\n[Validated Tool Plan]")
    print(plan)

    results = execute_tool_plan(
        plan,
        data
    )

    print("\n[Tool Results]")

    for item in results:

        print(
            "\nTool: "
            + item["tool"]
        )

        if item["tool"] == "max_by_group":

            result = item["result"]

            if isinstance(result, dict):

                if result.get("no_valid_values", False):

                    print(
                        "Không có giá trị hợp lệ "
                        "để tìm maximum."
                    )

                    print(
                        "Missing values: "
                        + str(result["missing_count"])
                    )

                else:

                    print(
                        "Groups: "
                        + str(result["groups"])
                    )

                    print(
                        "Maximum value: "
                        + str(result["max_value"])
                    )

                    print(
                        "Missing values: "
                        + str(result["missing_count"])
                    )

                    print(
                        "Missing group values: "
                        + str(
                            result["missing_group_count"]
                        )
                    )

            else:

                print(result)

        else:

            print(item["result"])

    # DATASET INFORMATION
    dataset_information = f"""
Rows:
{data.shape[0]}

Columns:
{data.shape[1]}

Column names:
{list(data.columns)}

Numeric columns:
{list(data.select_dtypes(include="number").columns)}

Categorical columns:
{list(data.select_dtypes(exclude="number").columns)}
"""

    # SINGLE TOOL
    if len(results) == 1:

        # CLARIFICATION TOOL
        if results[0]["tool"] == "clarification":

            question = results[0]["result"]

            print(
                "\n[DEBUG CLARIFICATION]"
            )

            print(
                "Question = "
                + question
            )

            return {
                "tool_result": None,
                "tool": "clarification",
                "explanation": question
            }, None

        # NONE TOOL
        if results[0]["tool"] == "none":

            greeting_words = [
                "xin chào",
                "chào",
                "hello",
                "hi",
                "cảm ơn",
                "ok",
                "được rồi"
            ]

            question_lower = question.lower().strip()

            is_greeting = False

            for word in greeting_words:
                if question_lower == word:
                    is_greeting = True
                    break

            if is_greeting:
                explanation = "Xin chào! Tôi có thể giúp bạn phân tích dữ liệu."
            else:
                explanation = (
                    "Dataset có "
                    + str(data.shape[0])
                    + " dòng và "
                    + str(data.shape[1])
                    + " cột."
                )

            print("\n[DEBUG NONE]")
            print("Rows = " + str(data.shape[0]))
            print("Columns = " + str(data.shape[1]))
            print("Is greeting = " + str(is_greeting))

            return {
                "tool_result": None,
                "explanation": explanation
            }, None
        
        # INVALID TOOL
        if results[0]["tool"] == "invalid":

            reason = results[0]["parameters"].get(
                "reason",
                "Dataset không có dữ liệu cần thiết."
            )

            explanation = (
                "Không thể thực hiện yêu cầu. "
                + reason
            )

            print(
                "\n[DEBUG INVALID]"
            )

            print(
                "Reason = "
                + reason
            )

            return {
                "tool_result": None,
                "explanation": explanation
            }, None

        # MAX BY GROUP TOOL
        if results[0]["tool"] == "max_by_group":

            result = results[0]["result"]

            if result.get("no_valid_values", False):

                explanation = (
                    "Không có giá trị hợp lệ "
                    "để xác định nhóm có giá trị cao nhất. "
                    "Có "
                    + str(result["missing_count"])
                    + " giá trị bị thiếu."
                )

                return {
                    "tool_result": result,
                    "explanation": explanation
                }, None

            groups = result["groups"]
            max_value = result["max_value"]
            missing_group_count = (
                result["missing_group_count"]
            )

            if len(groups) == 1:

                explanation = (
                    "Nhóm có giá trị cao nhất là "
                    + str(groups[0])
                    + " với giá trị "
                    + str(max_value)
                    + "."
                )

            else:

                group_text = ", ".join(
                    groups[:-1]
                )

                group_text += " và " + groups[-1]

                explanation = (
                    "Các nhóm có giá trị cao nhất là "
                    + group_text
                    + ", cùng đạt "
                    + str(max_value)
                    + "."
                )

            if missing_group_count > 0:

                explanation += (
                    " Có "
                    + str(missing_group_count)
                    + " bản ghi thiếu thông tin nhóm."
                )

            return {
                "tool_result": result,
                "explanation": explanation
            }, None

        result = results[0]["result"]

        tool_result = results[0]

        explanation_prompt = f"""
Bạn là trợ lý phân tích dữ liệu.

Người dùng hỏi:

{question}

Thông tin về dataset:

===== DATASET INFORMATION =====

{dataset_information}

Tool được sử dụng:

{tool_result["tool"]}

Kết quả từ Python:

===== TOOL RESULT =====

{result}

QUY TẮC:

1. Chỉ sử dụng các số liệu có trong
TOOL RESULT hoặc DATASET INFORMATION.

2. Không được tự tạo hoặc thay đổi số liệu.

3. Không được tự tính thêm các số liệu mới.

4. Nếu thông tin được cung cấp không đủ
để trả lời câu hỏi, hãy nói rõ.

5. Nếu kết quả là tương quan,
hãy nhớ rằng tương quan không
đồng nghĩa với quan hệ nhân quả.

6. Nếu số lượng mẫu nhỏ,
hãy nêu rõ đây là một hạn chế
của dữ liệu khi phù hợp.

7. Trả lời bằng tiếng Việt,
ngắn gọn và dễ hiểu.

8. Nếu TOOL được sử dụng là group_analysis,
các giá trị Count, Mean, Minimum và Maximum
phải được hiểu đúng theo ý nghĩa thống kê.

9. Nếu kết quả group_analysis chứa Mean,
và câu hỏi yêu cầu so sánh điểm trung bình,
phải sử dụng từ "điểm trung bình".

10. Không được viết "điểm cao nhất" khi đang
so sánh giá trị Mean.

11. Nếu một nhóm có Mean lớn nhất,
phải diễn đạt là:
"có điểm trung bình cao nhất".

12. Chỉ được dùng cụm "giá trị cao nhất"
hoặc "điểm cao nhất" khi TOOL RESULT thực sự
cung cấp Maximum hoặc kết quả của max_by_group.

13. Không được nhầm Maximum của group_analysis
với Mean của group_analysis.

Chỉ trả lời phần giải thích
cho người dùng.
"""

        response = ollama.chat(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": explanation_prompt
                }
            ],
            options={
                "num_ctx": 4096
            }
        )

        explanation = response[
            "message"
        ][
            "content"
        ].strip()

        return {
            "tool_result": result,
            "explanation": explanation
        }, None

    # MULTI TOOL
    explanation = generate_multi_tool_explanation(
        question,
        data,
        results
    )

    return {
        "tool_results": results,
        "explanation": explanation
    }, None

def generate_multi_tool_explanation(
    question,
    data,
    results
):

    dataset_information = f"""
Rows:
{data.shape[0]}

Columns:
{data.shape[1]}

Column names:
{list(data.columns)}

Numeric columns:
{list(data.select_dtypes(include="number").columns)}

Categorical columns:
{list(data.select_dtypes(exclude="number").columns)}
"""

    tool_results = ""

    for index, item in enumerate(results):

        tool_results += (
            "\n===== TOOL RESULT "
            + str(index + 1)
            + " =====\n"
        )

        tool_results += (
            "Tool: "
            + item["tool"]
            + "\n"
        )

        tool_results += (
            "Parameters: "
            + str(item["parameters"])
            + "\n"
        )

        tool_results += (
            "Result:\n"
            + str(item["result"])
            + "\n"
        )

    prompt = f"""
Bạn là trợ lý phân tích dữ liệu.

Người dùng hỏi:

{question}

Thông tin dataset:

===== DATASET INFORMATION =====

{dataset_information}

Các kết quả phân tích từ Python:

{tool_results}

QUY TẮC BẮT BUỘC:

QUY TẮC BẮT BUỘC:

QUY TẮC ƯU TIÊN CHO max_by_group:

QUY TẮC PHÂN BIỆT group_analysis VÀ max_by_group:

Nếu kết quả có cả group_analysis và max_by_group,
phải coi đây là HAI kết quả độc lập.

group_analysis chỉ dùng để trả lời:
- Count
- Mean
- Minimum
- Maximum của từng nhóm.

max_by_group chỉ dùng để trả lời:
- nhóm có giá trị lớn nhất
- giá trị lớn nhất trong toàn bộ các nhóm.

ĐẶC BIỆT:

Nếu group_analysis cho thấy:
HCM có Mean = 90.0

và max_by_group cho thấy:
Groups = ['Hanoi']
Maximum value = 95

thì bắt buộc phải hiểu:

- HCM có điểm trung bình cao nhất là 90.0.
- Hanoi có điểm cao nhất là 95.

KHÔNG ĐƯỢC viết:
"HCM có điểm cao nhất với giá trị 90.0."

KHÔNG ĐƯỢC dùng Mean của group_analysis
để trả lời câu hỏi về "điểm cao nhất".

Khi câu hỏi chứa cả:
"điểm trung bình"
và
"điểm cao nhất"

phải trả lời thành HAI kết luận riêng biệt:

1. Điểm trung bình:
sử dụng Mean từ group_analysis.

2. Điểm cao nhất:
sử dụng Groups và Maximum value từ max_by_group.

Nếu hai kết quả này xác định hai thành phố khác nhau,
phải nêu đúng cả hai thành phố.

Nếu TOOL RESULT có:

Groups: [...]
Maximum value: ...

thì phải dùng trực tiếp Groups và Maximum value
để trả lời câu hỏi về nhóm có giá trị cao nhất.

Nếu Groups có nhiều nhóm, phải nêu TẤT CẢ
các nhóm trong Groups.

Ví dụ:

Groups: ['Danang', 'HCM']
Maximum value: 99

Phải trả lời rằng Danang và HCM cùng có
giá trị cao nhất là 99.

Missing group values chỉ là thông tin bổ sung
về dữ liệu thiếu. Không được dùng thông tin này
làm lý do để bỏ qua Groups và Maximum value.

Nếu TOOL RESULT đã có Groups và Maximum value,
không được nói rằng không xác định được nhóm
và không được yêu cầu kiểm tra thêm dữ liệu gốc.

Sau quy tắc này, tiếp tục thực hiện các quy tắc
bên dưới.

1. DATASET INFORMATION là nguồn thông tin
chính xác để trả lời các câu hỏi về:
- số dòng
- số cột
- tên cột
- cột số
- cột phân loại.

2. Khi câu hỏi yêu cầu số dòng hoặc số cột,
hãy lấy trực tiếp giá trị từ:
Rows và Columns trong DATASET INFORMATION.

3. Không được tự suy đoán hoặc thay đổi
số dòng, số cột hoặc tên cột.

4. Chỉ được sử dụng thông tin xuất hiện
trực tiếp trong DATASET INFORMATION
hoặc TOOL RESULTS.

5. Không được tự tính toán thêm bất kỳ
chỉ số nào.

6. Không được thực hiện phép phân tích
mà Python chưa thực hiện.

7. Không được suy luận về correlation,
tương quan, quan hệ giữa các biến,
nguyên nhân hoặc xu hướng nếu TOOL RESULTS
không chứa kết quả tương ứng.

8. Không được tạo ra số liệu mới.

9. Phải đọc và sử dụng đúng từng TOOL RESULT
theo đúng tool đã tạo ra kết quả đó.

10. Mỗi tool có một nhiệm vụ riêng:

- group_analysis:
  cung cấp count, mean, min và max
  của value_column theo từng group.

- max_by_group:
  xác định nhóm có giá trị lớn nhất
  của value_column.
  Kết quả của tool này nằm trong:
  "Groups" và "Maximum value".

- Nếu TOOL RESULT của max_by_group có
  "Groups" và "Maximum value", phải sử dụng
  trực tiếp hai thông tin này để trả lời câu hỏi
  về nhóm có giá trị cao nhất.

- Nếu "Groups" chứa nhiều nhóm, phải nêu tất cả
  các nhóm đó.

- "Missing group values" chỉ là thông tin bổ sung
  về dữ liệu thiếu. Không được vì có giá trị group
  bị thiếu mà từ chối sử dụng kết quả
  của max_by_group.

- Không được yêu cầu kiểm tra thêm dữ liệu gốc
  nếu TOOL RESULT đã đủ thông tin để trả lời.

11. Khi có nhiều TOOL RESULT,
không được thay thế kết quả của tool này
bằng kết quả của tool khác.

12. Nếu câu hỏi yêu cầu "điểm trung bình",
hãy sử dụng giá trị mean từ group_analysis.

13. Nếu câu hỏi yêu cầu "số lượng",
hãy sử dụng giá trị count từ group_analysis.

14. Khi người dùng hỏi "so sánh điểm số giữa
các thành phố" và chỉ có TOOL RESULT
của group_analysis, hãy trình bày kết quả
theo các thống kê mà tool cung cấp.

Nếu sử dụng mean để so sánh, bắt buộc phải
gọi là "điểm trung bình".

Không được viết:
"thành phố có điểm cao nhất"
khi đang sử dụng giá trị mean.

Phải viết:
"thành phố có điểm trung bình cao nhất"
nếu đang so sánh mean.

Nếu sử dụng max, phải gọi rõ đó là
"điểm cao nhất trong nhóm" hoặc
"giá trị max của nhóm".

Không được dùng từ "điểm cao nhất" để
mô tả mean.

15. Nếu câu hỏi yêu cầu "điểm cao nhất",
"giá trị cao nhất", "lớn nhất" hoặc
"nhóm nào đứng đầu về giá trị",
hãy sử dụng kết quả của max_by_group
nếu TOOL RESULT có max_by_group.

16. Khi TOOL RESULT có:

Groups: ['Hanoi']
Maximum value: 95

thì phải hiểu rằng:
- nhóm có giá trị cao nhất là Hanoi
- giá trị cao nhất là 95.

Không được thay thế kết quả này bằng
mean hoặc max của một nhóm trong
group_analysis.

17. Đặc biệt, nếu đồng thời có:
- group_analysis
- max_by_group

và câu hỏi hỏi "thành phố nào có điểm cao nhất",
hãy lấy thành phố và giá trị từ max_by_group.

Không được lấy thành phố có mean cao nhất
trong group_analysis để trả lời câu hỏi này.

18. Phân biệt rõ:

- count = số lượng mẫu
- mean = giá trị trung bình
- min = giá trị nhỏ nhất trong nhóm
- max = giá trị lớn nhất trong nhóm
- max_by_group = nhóm có giá trị lớn nhất
  trong toàn bộ các nhóm.

19. Nếu câu hỏi yêu cầu nhiều thông tin,
hãy trả lời từng phần bằng đúng TOOL RESULT
tương ứng.

20. Không được khẳng định rằng một nhóm
"có học sinh giỏi nhất" chỉ dựa trên
một giá trị thống kê nếu dữ liệu không đủ
để chứng minh kết luận đó.

21. Khi chỉ có thể kết luận rằng một thành phố
có giá trị score cao nhất trong dataset,
hãy nói chính xác như vậy và không mở rộng
thành kết luận rộng hơn.

22. Khi một nhóm chỉ có ít mẫu,
có thể nêu đây là một hạn chế của dữ liệu,
nhưng không được tự đánh giá thêm
ngoài thông tin đã có.

23. Trả lời bằng tiếng Việt,
ngắn gọn và dễ hiểu.

Chỉ trả lời phần giải thích cho người dùng.
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

    explanation = response[
        "message"
    ][
        "content"
    ].strip()

    return explanation