from pydantic import BaseModel
from typing import List, Optional
import sys


class ImportStatement(BaseModel):
    module: str
    names: List[str] = []
    alias: Optional[str] = None


class Argument(BaseModel):
    name: str
    default: Optional[str] = None
    annotation: Optional[str] = None


class FunctionDefinition(BaseModel):
    name: str
    args: List[Argument] = []
    docstring: Optional[str] = None
    body: str
    decorators: List[str] = []
    returns: Optional[str] = None


class ClassDefinition(BaseModel):
    name: str
    docstring: Optional[str] = None
    methods: List[FunctionDefinition] = []
    decorators: List[str] = []


class PythonModule(BaseModel):
    docstring: Optional[str] = None
    imports: List[ImportStatement] = []
    classes: List[ClassDefinition] = []
    functions: List[FunctionDefinition] = []
    body: Optional[str] = None


class PythonCodeGenerator:
    def generate_code_from_function(self, func: FunctionDefinition) -> str:
        code_lines = []
        for decorator in func.decorators:
            code_lines.append(f"@{decorator}")
        arg_strs = []
        for arg in func.args:
            part = arg.name
            if arg.annotation:
                part += f": {arg.annotation}"
            if arg.default:
                part += f" = {arg.default}"
            arg_strs.append(part)
        args_str = ", ".join(arg_strs)
        return_str = f" -> {func.returns}" if func.returns else ""
        code_lines.append(f"def {func.name}({args_str}){return_str}:")
        if func.docstring:
            code_lines.append(f'    """{func.docstring}"""')
        body_lines = func.body.split("\n")
        if any(line.strip() for line in body_lines):
            for line in body_lines:
                code_lines.append("    " + line)
        else:
            if not func.docstring:
                code_lines.append("    pass")
        return "\n".join(code_lines)

    def generate_code_from_class(self, cls: ClassDefinition) -> str:
        code_lines = []
        for decorator in cls.decorators:
            code_lines.append(f"@{decorator}")
        code_lines.append(f"class {cls.name}:")
        if cls.docstring:
            code_lines.append(f'    """{cls.docstring}"""')
        if cls.methods:
            for method in cls.methods:
                # Determine if method is staticmethod or classmethod
                is_static = any(d == "staticmethod" for d in method.decorators)
                is_class = any(d == "classmethod" for d in method.decorators)

                if is_static:
                    # No 'self' or 'cls' needed
                    pass
                elif is_class:
                    # Ensure first argument is 'cls'
                    if not method.args or method.args[0].name != "cls":
                        method.args.insert(0, Argument(name="cls"))
                else:
                    # Regular instance method, ensure first argument is 'self'
                    if not method.args or method.args[0].name != "self":
                        method.args.insert(0, Argument(name="self"))

                method_code = self.generate_code_from_function(method)
                for line in method_code.split("\n"):
                    code_lines.append("    " + line)
        else:
            if not cls.docstring:
                code_lines.append("    pass")
        return "\n".join(code_lines)

    def generate_pycode_from_module(self, module: PythonModule) -> str:
        code_lines = []
        if module.docstring:
            code_lines.append(f'"""{module.docstring}"""')
            code_lines.append("")
        for imp in module.imports:
            if imp.names:
                if imp.alias and len(imp.names) == 1:
                    code_lines.append(
                        f"from {imp.module} import {imp.names[0]} as {imp.alias}"
                    )
                else:
                    names_str = ", ".join(imp.names)
                    code_lines.append(f"from {imp.module} import {names_str}")
            else:
                if imp.alias:
                    code_lines.append(f"import {imp.module} as {imp.alias}")
                else:
                    code_lines.append(f"import {imp.module}")
        if module.imports:
            code_lines.append("")
        for func in module.functions:
            code_lines.append(self.generate_code_from_function(func))
            code_lines.append("")
        for cls in module.classes:
            code_lines.append(self.generate_code_from_class(cls))
            code_lines.append("")
        if module.body:
            code_lines.append(module.body)
            code_lines.append("")

        return "\n".join(
            line for line in code_lines if line.strip() != "" or line == ""
        )

    def extract_imported_modules(self, module: PythonModule) -> List[str]:
        # Use a set to avoid duplicates
        return list({imp.module for imp in module.imports})

    def filter_packages(self, modules: List[str]) -> List[str]:
        """
        Given a list of module names, return only those that are NOT in the standard library.
        """
        STDLIB_MODULES = set(sys.stdlib_module_names)
        return [m for m in modules if m not in STDLIB_MODULES]
