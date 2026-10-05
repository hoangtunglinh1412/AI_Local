# AGENTS.md

## 1. Project

Đây là project Local AI Data Agent chạy local trên Windows.

Thư mục project:

D:\local-AI

Công nghệ chính:

- Python
- Ollama
- qwen3:8b
- pandas
- NumPy
- ChromaDB
- RAG
- Data Analysis Agent

Môi trường Python:

.venv

Không tự ý thay đổi môi trường Python nếu task không yêu cầu.

---

## 2. Trạng thái quan trọng

Project hiện đã hoàn thành và kiểm thử thành công đến V4.54.

V4.54 đã PASS các chức năng liên quan đến:

- Tool Planner
- Tool Parser
- Tool Executor
- none
- clarification
- invalid
- max_by_group
- group_analysis
- greeting routing
- dataset information routing

Đặc biệt:

Câu:

"Xin chào"

phải được Planner định tuyến thành:

{
    "steps": [
        {
            "tool": "none",
            "parameters": {}
        }
    ]
}

Sau đó data_agent.py phải nhận diện đây là greeting và trả lời phù hợp.

Không được làm hỏng các chức năng V4.54 đã PASS.

---

## 3. Nguyên tắc sửa code

### 3.1. Không sửa nếu chưa hiểu

Trước khi sửa code:

1. Đọc file liên quan.
2. Hiểu luồng xử lý.
3. Xác định nguyên nhân.
4. Lập kế hoạch thay đổi.
5. Chỉ sau đó mới sửa.

Không được sửa code chỉ dựa trên phỏng đoán.

---

### 3.2. Sửa tối thiểu

Ưu tiên thay đổi nhỏ nhất có thể.

Không tự ý:

- refactor toàn bộ project
- đổi kiến trúc
- đổi tên hàng loạt function
- đổi tên biến hàng loạt
- di chuyển file
- xóa code đang hoạt động
- thay đổi nhiều module không liên quan

Nếu task chỉ yêu cầu sửa một file thì ưu tiên chỉ sửa file đó.

---

### 3.3. Không phá chức năng cũ

Mọi thay đổi mới phải giữ các chức năng đã PASS.

Đặc biệt phải bảo vệ:

- Planner routing
- Parser validation
- Executor
- none
- clarification
- invalid
- group_analysis
- max_by_group
- multi-tool execution
- greeting handling

Nếu thay đổi có nguy cơ ảnh hưởng chức năng cũ, phải báo trước.

---

## 4. Kiến trúc hiện tại

Luồng chính:

Question
    ↓
tool_planner.py
    ↓
Tool Plan
    ↓
tool_parser.py
    ↓
Validated Tool Plan
    ↓
tool_executor.py
    ↓
Tool Results
    ↓
data_agent.py
    ↓
Final Explanation

Các module chính:

### data_agent.py

Điều phối toàn bộ Data Agent.

Có trách nhiệm:

- load dataset
- gọi Planner
- parse plan
- execute tools
- xử lý kết quả
- tạo explanation

---

### tool_planner.py

Chuyển câu hỏi của người dùng thành Tool Plan.

Không được tự trả lời câu hỏi.

Chỉ được trả về JSON hợp lệ.

---

### tool_parser.py

Kiểm tra Tool Plan do Planner tạo ra.

Phải kiểm tra:

- tool hợp lệ
- parameters hợp lệ
- tên cột tồn tại
- kiểu dữ liệu phù hợp

Không được tự thay thế tên cột người dùng yêu cầu bằng cột khác.

---

### tool_executor.py

Thực thi Tool Plan đã được validation.

Không tự ý thay đổi Tool Plan.

---

### tool_registry.py

Đăng ký các tool có thể sử dụng.

---

### data_analyzer.py

Chứa logic phân tích dữ liệu bằng Python/pandas.

---

### data_tools.py

Cung cấp interface cho các tool phân tích.

---

## 5. Quy tắc Planner

Planner phải ưu tiên xác định loại yêu cầu trước khi chọn tool.

### none

Sử dụng `none` khi:

- câu hỏi chỉ là lời chào
- giao tiếp xã giao
- câu nói không yêu cầu phân tích dataset
- yêu cầu thông tin cơ bản về dataset

Ví dụ:

"Xin chào"

"Chào bạn"

"Hello"

"Cảm ơn"

"Dataset có bao nhiêu dòng?"

---

### clarification

Chỉ sử dụng `clarification` khi người dùng thực sự muốn thực hiện nhiệm vụ phân tích dữ liệu nhưng thông tin chưa đủ rõ.

Ví dụ:

"So sánh các nhóm."

"Phân tích dữ liệu này."

"Nhóm nào có giá trị cao nhất?"

Không được coi lời chào là clarification.

---

### invalid

Sử dụng `invalid` khi người dùng yêu cầu cột không tồn tại trong dataset.

Không được tự ý thay thế bằng cột khác.

---

## 6. group_analysis và max_by_group

Hai tool này có ý nghĩa khác nhau.

### group_analysis

Dùng để phân tích thống kê theo nhóm:

- Count
- Mean
- Minimum
- Maximum

Ví dụ:

"So sánh điểm trung bình giữa các thành phố."

---

### max_by_group

Dùng khi người dùng yêu cầu:

- nhóm có giá trị cao nhất
- thành phố có giá trị cao nhất
- nhóm có điểm cao nhất

Không được dùng group_analysis để thay thế max_by_group trong trường hợp này.

---

### Phân biệt Mean và Maximum

Nếu kết quả có cả:

group_analysis

và:

max_by_group

phải coi đây là hai kết quả độc lập.

Ví dụ:

group_analysis:

HCM Mean = 90.0

max_by_group:

Hanoi Maximum = 95

Phải hiểu là:

- HCM có điểm trung bình cao nhất là 90.0.
- Hanoi có điểm cao nhất là 95.

Không được viết:

"HCM có điểm cao nhất với giá trị 90.0."

---

## 7. Quy tắc Dataset

Không được tự suy đoán tên cột.

Chỉ sử dụng các cột thực sự tồn tại trong dataset.

Nếu cần thông tin về dataset:

- đọc dataset
- kiểm tra columns
- kiểm tra dtype
- kiểm tra shape

Không được tự tạo số liệu.

---

## 8. Quy tắc Tool

Mỗi step chỉ sử dụng một tool.

Không tạo hai step có cùng tool và cùng parameters.

Tool Plan phải là JSON hợp lệ.

Không thêm markdown vào Tool Plan.

Không thêm giải thích bên ngoài JSON khi Planner đang được yêu cầu tạo plan.

---

## 9. Quy tắc Test

Sau mỗi thay đổi code:

1. Chạy test liên quan trực tiếp.
2. Nếu thay đổi logic chung, chạy regression test.
3. Kiểm tra output.
4. Chỉ kết luận PASS khi test thực sự chạy thành công.

Không được nói "đã test" nếu chưa thực sự chạy test.

Không được bỏ qua lỗi test.

---

## 10. Quy tắc Git

Git được sử dụng để bảo vệ các phiên bản đã PASS.

Không tự ý:

- git reset
- git checkout
- git clean
- git push
- xóa commit

nếu chưa được người dùng cho phép.

Trước các thay đổi lớn:

- kiểm tra git status
- đảm bảo biết trạng thái hiện tại

Không tự ý sửa lịch sử Git.

---

## 11. Quy tắc File

Không tự ý xóa file.

Không tự ý đổi tên file.

Không tự ý tạo file mới nếu task không yêu cầu.

Không sửa các file ngoài phạm vi task nếu không cần thiết.

Không sửa:

.venv/

__pycache__/

chat_history.db

chroma_db/

Các file trên là dữ liệu môi trường/cache/runtime và đã được loại khỏi Git.

---

## 12. Quy tắc khi thực hiện task

Khi nhận task:

### Bước 1

Đọc các file liên quan.

### Bước 2

Giải thích ngắn gọn vấn đề hiện tại.

### Bước 3

Đưa ra implementation plan.

### Bước 4

Chờ người dùng review nếu task có thay đổi đáng kể hoặc nhiều file.

### Bước 5

Thực hiện thay đổi tối thiểu.

### Bước 6

Chạy test.

### Bước 7

Báo cáo:

- file đã thay đổi
- thay đổi chính
- test đã chạy
- kết quả test
- các vấn đề còn lại

---

## 13. Không tự ý mở rộng phạm vi

Nếu task yêu cầu:

"Sửa lỗi X"

thì chỉ tập trung sửa lỗi X.

Không tự ý:

- refactor
- tối ưu
- thêm tính năng
- thay đổi UI
- đổi model
- đổi database
- đổi kiến trúc

trừ khi người dùng yêu cầu.

---

## 14. Quy tắc an toàn

Nếu không chắc chắn về một thay đổi:

Không được tự suy đoán.

Hãy báo rõ:

- điều chưa chắc chắn
- file liên quan
- phương án đề xuất

Sau đó chờ người dùng quyết định nếu thay đổi có rủi ro.

---

## 15. Mục tiêu phát triển

Project hướng tới một Local AI Data Agent có khả năng:

Question
    ↓
Planner
    ↓
Tool Selection
    ↓
Validation
    ↓
Tool Execution
    ↓
Data Analysis
    ↓
Explanation

Ưu tiên:

1. Đúng
2. Ổn định
3. Có thể kiểm thử
4. Dễ hiểu
5. Dễ mở rộng

Không ưu tiên việc làm code phức tạp chỉ để giảm số dòng code.