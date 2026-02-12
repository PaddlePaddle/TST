from enum import Enum


class ExecutionMode(str, Enum):
    EAGER = "eager"
    JIT = "jit"


g_execution_mode = ExecutionMode.EAGER
