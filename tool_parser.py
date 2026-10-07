import json
import re

from tool_registry import TOOLS


def extract_json_string(text):
    if not isinstance(text, str):
        return ""

    text = text.strip()

    # 1. Thử parse trực tiếp nếu chuỗi đã là JSON sạch
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    # 2. Thử trích xuất từ markdown code block: ```json ... ``` hoặc ``` ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if match:
        candidate = match.group(1).strip()
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    # 3. Thử trích xuất khối giữa ký tự { đầu tiên và } cuối cùng
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1].strip()
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    return text


def parse_tool_response(
    response,
    data
):

    response = extract_json_string(response)

    try:

        result = json.loads(
            response
        )

    except json.JSONDecodeError:

        return None, "JSON không hợp lệ."

    if "tool" not in result:

        return None, "Thiếu tên tool."

    if "parameters" not in result:

        return None, "Thiếu parameters."

    tool_name = result["tool"]

    parameters = result["parameters"]

    if tool_name not in TOOLS:

        return None, (
            "Tool không tồn tại: "
            + tool_name
        )

    if not isinstance(
        parameters,
        dict
    ):

        return None, (
            "Parameters phải là dictionary."
        )

    required_parameters = (
        TOOLS[tool_name]["parameters"]
    )

    missing_parameters = []

    for parameter in required_parameters:

        if parameter == "data":

            continue

        if parameter not in parameters:

            missing_parameters.append(
                parameter
            )

    if len(missing_parameters) > 0:

        return None, (
            "Thiếu parameters: "
            + str(missing_parameters)
        )

    # =========================
    # COLUMN VALIDATION
    # =========================

    if tool_name == "group_analysis":

        group_column = parameters[
            "group_column"
        ]

        value_column = parameters[
            "value_column"
        ]

        if group_column not in data.columns:

            return None, (
                "Không tìm thấy group_column: "
                + group_column
            )

        if value_column not in data.columns:

            return None, (
                "Không tìm thấy value_column: "
                + value_column
            )

        if not data[
            value_column
        ].dtype.kind in "iufc":

            return None, (
                "value_column phải là dữ liệu số: "
                + value_column
            )

    return result, None

def parse_tool_plan(
    response,
    data
):

    response = extract_json_string(response)

    try:
        result = json.loads(
            response
        )
    except json.JSONDecodeError:
        return None, "JSON không hợp lệ."

    if "steps" not in result:
        return None, "Thiếu steps."

    steps = result["steps"]

    if not isinstance(
        steps,
        list
    ):
        return None, "steps phải là danh sách."

    if len(steps) == 0:
        return None, "Plan không có step."

    validated_steps = []

    for step in steps:

        if "tool" not in step:
            return None, "Step thiếu tên tool."

        if "parameters" not in step:
            return None, "Step thiếu parameters."

        tool_name = step["tool"]
        parameters = step["parameters"]

        if tool_name not in TOOLS:
            return None, (
                "Tool không tồn tại: "
                + tool_name
            )

        if not isinstance(
            parameters,
            dict
        ):
            return None, (
                "Parameters phải là dictionary."
            )

        required_parameters = (
            TOOLS[tool_name]["parameters"]
        )

        missing_parameters = []

        for parameter in required_parameters:

            if parameter == "data":
                continue

            if parameter not in parameters:
                missing_parameters.append(
                    parameter
                )

        if len(missing_parameters) > 0:
            return None, (
                "Thiếu parameters: "
                + str(missing_parameters)
            )

        # GROUP ANALYSIS VALIDATION

        if tool_name in [
            "group_analysis",
            "max_by_group"
        ]:

            group_column = parameters[
                "group_column"
            ]

            value_column = parameters[
                "value_column"
            ]

            if group_column not in data.columns:
                return None, (
                    "Không tìm thấy group_column: "
                    + group_column
                )

            if value_column not in data.columns:
                return None, (
                    "Không tìm thấy value_column: "
                    + value_column
                )

            if not data[
                value_column
            ].dtype.kind in "iufc":

                return None, (
                    "value_column phải là dữ liệu số: "
                    + value_column
                )

        validated_steps.append(step)

    # =========================
    # TERMINAL TOOL PRIORITY
    # =========================

    for step in validated_steps:

        if step["tool"] == "clarification":

            validated_steps = [step]
            break

        if step["tool"] == "invalid":

            validated_steps = [step]
            break

        if step["tool"] == "none":

            validated_steps = [step]
            break

    result["steps"] = validated_steps

    return result, None