from benchmarks.models import BenchmarkSuite, BenchmarkTask, ExpectedToolCall

DETERMINISTIC_SUITE = BenchmarkSuite(
    name="deterministic-tools",
    version="1.0.0",
    tasks=[
        BenchmarkTask(
            id="add-017",
            prompt="Use the arithmetic tool to add 17 and 25. Return only the number.",
            expected_answer="42",
            expected_tool_calls=[
                ExpectedToolCall(
                    name="arithmetic",
                    arguments={"operation": "add", "left": 17, "right": 25},
                )
            ],
            tags=["arithmetic", "single-tool"],
        ),
        BenchmarkTask(
            id="multiply-009",
            prompt="Use the arithmetic tool to multiply 9 by 8. Return only the number.",
            expected_answer="72",
            expected_tool_calls=[
                ExpectedToolCall(
                    name="arithmetic",
                    arguments={"operation": "multiply", "left": 9, "right": 8},
                )
            ],
            tags=["arithmetic", "single-tool"],
        ),
        BenchmarkTask(
            id="inventory-keyboard",
            prompt="Look up SKU KEY-104 and return its stock count only.",
            expected_answer="7",
            expected_tool_calls=[
                ExpectedToolCall(name="inventory_lookup", arguments={"sku": "KEY-104"})
            ],
            tags=["lookup", "single-tool"],
        ),
    ],
)


def task_by_id(task_id: str) -> BenchmarkTask:
    for task in DETERMINISTIC_SUITE.tasks:
        if task.id == task_id:
            return task
    raise KeyError(task_id)
