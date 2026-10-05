from tool_registry import TOOLS


print(
    "\n===== TOOL REGISTRY ====="
)


for tool_name, tool_info in TOOLS.items():

    print(
        "\nTool:",
        tool_name
    )

    print(
        "Description:",
        tool_info["description"]
    )

    print(
        "Parameters:",
        tool_info["parameters"]
    )


print(
    "\n===== FUNCTION TEST ====="
)

print(
    TOOLS["statistics"]["function"]
)

print(
    TOOLS["group_analysis"]["function"]
)