from tool_registry import TOOLS


def execute_tool(
    tool_result,
    data
):

    tool_name = tool_result[
        "tool"
    ]

    parameters = tool_result[
        "parameters"
    ]


    # =========================
    # NO TOOL
    # =========================
    if tool_name == "none":
        return None

    # INVALID TOOL
    if tool_name == "invalid":
        return None

    # CLARIFICATION TOOL
    if tool_name == "clarification":
        return parameters["question"]

    tool = TOOLS[
        tool_name
    ][
        "function"
    ]


    # =========================
    # GROUP ANALYSIS
    # =========================

    if tool_name == "group_analysis":

        result = tool(
            data,
            parameters[
                "group_column"
            ],
            parameters[
                "value_column"
            ]
        )


    # =========================
    # MAX BY GROUP
    # =========================

    elif tool_name == "max_by_group":

        result = tool(
            data,
            parameters[
                "group_column"
            ],
            parameters[
                "value_column"
            ]
        )


    # =========================
    # OTHER TOOLS
    # =========================

    else:

        result = tool(
            data
        )


    return result

def execute_tool_plan(
    plan,
    data
):

    results = []

    for step in plan["steps"]:

        result = execute_tool(
            step,
            data
        )

        results.append({
            "tool": step["tool"],
            "parameters": step["parameters"],
            "result": result
        })

    return results