from data_tools import (
    tool_statistics,
    tool_group_analysis,
    tool_max_by_group,
    tool_outliers,
    tool_correlation
)

TOOLS = {

    "statistics": {

        "function": tool_statistics,

        "description": (
            "Phân tích thống kê cơ bản "
            "của các cột dữ liệu số."
        ),

        "parameters": [
            "data"
        ]
    },


    "group_analysis": {

        "function": tool_group_analysis,

        "description": (
            "Nhóm dữ liệu theo một cột "
            "và tính count, mean, min, max "
            "cho một cột số."
        ),

        "parameters": [
            "data",
            "group_column",
            "value_column"
        ]
    },

    "max_by_group": {

        "function": tool_max_by_group,

        "description": (
            "Tìm nhóm có giá trị lớn nhất "
            "của một cột số. "
            "Sử dụng khi người dùng yêu cầu "
            "xác định nhóm hoặc thành phố "
            "có giá trị cao nhất."
        ),

        "parameters": [
            "data",
            "group_column",
            "value_column"
        ]
    },

    "outliers": {

        "function": tool_outliers,

        "description": (
            "Phát hiện các giá trị ngoại lệ "
            "bằng phương pháp IQR."
        ),

        "parameters": [
            "data"
        ]
    },


    "correlation": {

        "function": tool_correlation,

        "description": (
            "Tính ma trận tương quan "
            "giữa các cột dữ liệu số."
        ),

        "parameters": [
            "data"
        ]
    },


    "none": {

        "function": None,

        "description": (
            "Không cần thực hiện phép phân tích "
            "bằng Python. Sử dụng khi câu hỏi "
            "chỉ yêu cầu thông tin cơ bản về dataset "
            "như số dòng, số cột hoặc tên cột."
        ),

        "parameters": [
            "data"
        ]
    },

    "invalid": {

        "function": None,

        "description": (
            "Yêu cầu không thể thực hiện vì "
            "dataset không có cột hoặc dữ liệu cần thiết. "
            "Sử dụng khi người dùng yêu cầu một cột "
            "không tồn tại trong dataset."
        ),

        "parameters": [
            "data",
            "reason"
        ]
    },

    "clarification": {

        "function": None,

        "description": (
            "Yêu cầu chưa đủ thông tin hoặc "
            "không đủ rõ ràng để xác định tool "
            "và parameters cần sử dụng. "
            "Sử dụng khi cần hỏi lại người dùng."
        ),

        "parameters": [
            "data",
            "question"
        ]
    }

}