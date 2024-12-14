from pydantic import BaseModel
from typing import List, Optional


class ImportSpecifier(BaseModel):
    imported: str
    local: Optional[str] = None
    # e.g. import {foo as bar} from 'mod'
    # imported='foo', local='bar'
    # If local is None, local name = imported
    # For default imports: imported='default', local='myDefault'

class ImportStatement(BaseModel):
    source: str
    specifiers: List[ImportSpecifier] = []
    default_import: Optional[str] = None
    namespace_import: Optional[str] = None
    # e.g.
    # import defaultImport from 'some-module'
    # import {foo, bar as baz} from 'another-module'
    # import * as utils from 'utils'

class ExportSpecifier(BaseModel):
    exported: str
    local: Optional[str] = None
    # e.g. export { foo as bar }
    # exported='bar', local='foo'

class ExportStatement(BaseModel):
    specifiers: List[ExportSpecifier] = []
    default_export: bool = False
    # If default_export=True, then it's something like `export default function ...`
    # or `export default class ...`

class Parameter(BaseModel):
    name: str
    default: Optional[str] = None
    # No explicit annotation since JS is dynamically typed, 
    # unless you're parsing TypeScript, then you can add type info.

class FunctionDefinition(BaseModel):
    name: str
    parameters: List[Parameter] = []
    body: str
    async_function: bool = False
    generator: bool = False
    # You could add a flag to distinguish between a function declaration and function expression if needed.
    # For arrow functions, consider a separate model or a boolean flag.

class ClassMethodDefinition(BaseModel):
    name: str
    parameters: List[Parameter] = []
    body: str
    async_method: bool = False
    static: bool = False
    # Could add getters, setters if needed

class ClassFieldDefinition(BaseModel):
    name: str
    value: Optional[str] = None
    static: bool = False
    # For class fields like `class MyClass { static foo = 42; bar = 'hello'; }`

class ClassDefinition(BaseModel):
    name: str
    super_class: Optional[str] = None
    methods: List[ClassMethodDefinition] = []
    fields: List[ClassFieldDefinition] = []

class VariableDeclarator(BaseModel):
    name: str
    value: Optional[str] = None

class VariableDeclaration(BaseModel):
    kind: str  # 'var', 'let', 'const'
    declarations: List[VariableDeclarator]

class JavaScriptModule(BaseModel):
    imports: List[ImportStatement] = []
    exports: List[ExportStatement] = []
    classes: List[ClassDefinition] = []
    functions: List[FunctionDefinition] = []
    variables: List[VariableDeclaration] = []
    body: Optional[str] = None


class JavaScriptCodeGenerator:

    def generate_code_from_import(self, imp: ImportStatement) -> str:
        # For a default import:
        if imp.default_import and not imp.namespace_import and not imp.specifiers:
            # const _ = require('lodash');
            return f"const {imp.default_import} = require('{imp.source}');"
        
        # For namespace imports:
        if imp.namespace_import:
            # const utils = require('utils');
            # emulate `import * as utils from 'utils'`:
            return f"const {imp.namespace_import} = require('{imp.source}');"
        
        # For named imports:
        if imp.specifiers:
            # import {foo, bar as baz} from 'mod';
            # const { foo, bar: baz } = require('mod');
            specs = []
            for s in imp.specifiers:
                if s.local and s.local != s.imported:
                    specs.append(f"{s.imported}: {s.local}")
                else:
                    specs.append(s.imported)
            spec_str = ", ".join(specs)
            return f"const {{ {spec_str} }} = require('{imp.source}');"
        
        # If no default, no specifiers, just a side-effect import:
        return f"require('{imp.source}');"

    def generate_code_from_export(self, exp: ExportStatement) -> str:
        # Handle exports
        if exp.default_export and not exp.specifiers:
            # Placeholder for a default export without specifiers
            return "export default UNDEFINED_DEFAULT_EXPORT;"

        if exp.specifiers:
            spec_list = [
                f"{s.local if s.local else s.exported} as {s.exported}" if s.local else s.exported
                for s in exp.specifiers
            ]
            return f"export {{ {', '.join(spec_list)} }};"
        return ""

    def generate_code_from_function(self, func: FunctionDefinition) -> str:
        async_str = "async " if func.async_function else ""
        gen_str = "*" if func.generator else ""
        params = [f"{p.name}={p.default}" if p.default else p.name for p in func.parameters]
        param_str = ", ".join(params)
        return f"{async_str}function{gen_str} {func.name}({param_str}) {{\n{self.indent_code(func.body)}\n}}"

    def generate_code_from_class(self, cls: ClassDefinition) -> str:
        extend_str = f" extends {cls.super_class}" if cls.super_class else ""
        code_lines = [f"class {cls.name}{extend_str} {{"]

        # Fields
        for field in cls.fields:
            prefix = "static " if field.static else ""
            val = f" = {field.value}" if field.value is not None else ""
            code_lines.append(f"    {prefix}{field.name}{val};")

        # Methods
        for method in cls.methods:
            async_str = "async " if method.async_method else ""
            static_str = "static " if method.static else ""
            params = [f"{p.name}={p.default}" if p.default else p.name for p in method.parameters]
            param_str = ", ".join(params)
            code_lines.append(f"    {async_str}{static_str}{method.name}({param_str}) {{")
            code_lines.append(self.indent_code(method.body, level=2))
            code_lines.append("    }")

        code_lines.append("}")
        return "\n".join(code_lines)

    def generate_code_from_variables(self, vars: List[VariableDeclaration]) -> List[str]:
        lines = []
        for v in vars:
            decls = [f"{d.name} = {d.value}" if d.value is not None else d.name for d in v.declarations]
            lines.append(f"{v.kind} " + ", ".join(decls) + ";")
        return lines

    def indent_code(self, code: str, level: int = 1) -> str:
        prefix = "    " * level
        return "\n".join(prefix + line for line in code.split('\n'))

    def generate_code_from_js_module(self, module: JavaScriptModule) -> str:
        code_lines = []

        # Imports
        for imp in module.imports:
            code_lines.append(self.generate_code_from_import(imp))

        # Variables
        for line in self.generate_code_from_variables(module.variables):
            code_lines.append(line)

        # Functions
        for func in module.functions:
            code_lines.append(self.generate_code_from_function(func))

        # Classes
        for cls in module.classes:
            code_lines.append(self.generate_code_from_class(cls))

        # Exports
        for exp in module.exports:
            export_code = self.generate_code_from_export(exp)
            if export_code:
                code_lines.append(export_code)

        # Body
        if module.body:
            code_lines.append(module.body)

        return "\n".join(code_lines)

    def extract_imported_modules(self, module: JavaScriptModule) -> List[str]:
        """
        Extract a unique list of modules imported by the given JavaScriptModule.
        This is similar to the Python version but uses the 'source' attribute
        from each ImportStatement.
        """
        return list({imp.source for imp in module.imports})