#!/usr/bin/env python3
"""
检查代码中是否包含断言的独立脚本。
作为“Agent-First”工程理念中“品味与审查规范（Golden Rules）”的开端：
- 对于 `tops/` 目录下的业务/算子代码，所有的公有级函数至少应当包含一个断言（包括原生的 `assert` 或者形如 `assert_*` 的工具函数），
- 若公有函数内呈现“零断言（Zero Asserts）”状态，此脚本将发出警告并可被 CI 阻截。
"""

import ast
import os
import sys
from pathlib import Path


class AssertChecker(ast.NodeVisitor):
    def __init__(self):
        self.functions_without_asserts = []

    def visit_FunctionDef(self, node):
        # 忽略私有函数和特殊方法
        if node.name.startswith('_'):
            self.generic_visit(node)
            return

        has_assert = False

        # 遍历函数体，寻找 assert 语句或特定的函数调用
        for child in ast.walk(node):
            if isinstance(child, ast.Assert):
                has_assert = True
                break
            elif isinstance(child, ast.Call):
                # 检查是否调用了以 'assert' 开头的函数，比如 assert_shape_or_none
                func = child.func
                if isinstance(func, ast.Name) and func.id.startswith('assert'):
                    has_assert = True
                    break
                elif isinstance(func, ast.Attribute) and func.attr.startswith('assert'):
                    has_assert = True
                    break

        if not has_assert:
            self.functions_without_asserts.append(node)

        # 继续遍历可能的嵌套函数（虽然通常公有API断言在最外层）
        self.generic_visit(node)


def check_file(filepath: Path) -> list:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
            tree = ast.parse(code, filename=str(filepath))
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return []

    checker = AssertChecker()
    checker.visit(tree)

    return [(filepath, node) for node in checker.functions_without_asserts]


def main():
    # 目标检查目录：当前工程下的 tops 文件夹
    target_dir = Path("tops")
    if not target_dir.exists():
        print(f"Directory {target_dir} not found. Are you at the project root?")
        sys.exit(1)

    all_violations = []

    # 遍历所有.py文件
    for root, _, files in os.walk(target_dir):
        for file in files:
            if file.endswith('.py') and not file == '__init__.py':
                filepath = Path(root) / file
                violations = check_file(filepath)
                all_violations.extend(violations)

    if all_violations:
        print("======== [警告] 零断言 (Zero Asserts) 检查失败 ========")
        print("按照架构设计规范，以下公有函数未发现任何断言 (assert 或 assert_xxx)：\n")

        for filepath, node in all_violations:
            print(f"文件: {filepath} : {node.lineno}")
            print(f"  -> {node.name}() 缺少必要的形状或输入类型断言。")

        print("\n注: 我们推荐 Parse, don't validate。请在这些函数体开头补齐防卫式断言。")
        # 可以设为返回非0让 CI 直接挂掉
        sys.exit(1)
    else:
        print("======== 断言检查通过 ========")
        print("所有检测到的公有函数均至少包含一个断言。棒！")
        sys.exit(0)

if __name__ == "__main__":
    main()
