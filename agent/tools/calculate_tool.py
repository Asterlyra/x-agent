# agent/tools/calculate_tool.py —— 算术计算工具
#
# 安全讲点：绝不用 eval（任意代码执行风险），而是 ast.parse + 白名单节点求值。

import ast
import operator
from typing import Any

from langchain_core.tools import tool

# 白名单：二进制运算符 → 对应的安全求值函数
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_eval(node: ast.AST) -> Any:
    """只接受数字、四则运算、括号（分组）的递归求值器；其余节点一律报错。"""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"不支持的表达式节点: {type(node).__name__}")


@tool
def calculate(expression: str) -> str:
    """用户问算术题/需要数学计算时调用。支持数字与四则运算的安全表达式（如 "2+3*4"、"10/3"、"7%3"、"2**8"）；不支持变量、函数调用或字符串操作（会被拒绝，请直接用文本回答）。示例: 计算 (1+2)*4 → "(1+2)*4 = 12"。"""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree)
    except (SyntaxError, ValueError, ZeroDivisionError) as e:
        raise ValueError(f"无法计算表达式 {expression!r}: {e}") from e
    return f"{expression} = {result}"
