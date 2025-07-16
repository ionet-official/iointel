"""
Advanced prompt templating system for iointel.

This module provides a flexible template engine with variable substitution,
includes, conditional logic, and context-aware prompt generation.
"""

import re
from typing import Dict, Any, Optional, List
from string import Template
from pathlib import Path
from datetime import datetime
import json
from dataclasses import dataclass, field


@dataclass
class TemplateContext:
    """Context object for template rendering."""
    variables: Dict[str, Any] = field(default_factory=dict)
    globals: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize with default global variables."""
        if not self.globals:
            self.globals = {
                "current_datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "current_date": datetime.now().strftime("%Y-%m-%d"),
                "current_time": datetime.now().strftime("%H:%M:%S"),
            }
    
    def update(self, **kwargs):
        """Update variables with new values."""
        self.variables.update(kwargs)
    
    def get_all_variables(self) -> Dict[str, Any]:
        """Get all variables (globals + locals)."""
        return {**self.globals, **self.variables}


class PromptTemplate:
    """
    A flexible prompt template with variable substitution and advanced features.
    """
    
    def __init__(self, template_str: str, name: str = ""):
        self.template_str = template_str
        self.name = name
        self.template = Template(template_str)
        self._includes = self._parse_includes()
        self._conditionals = self._parse_conditionals()
    
    def _parse_includes(self) -> List[str]:
        """Parse include directives in the template."""
        includes = []
        include_pattern = r'{{include\s+([^}]+)}}'
        matches = re.findall(include_pattern, self.template_str)
        for match in matches:
            includes.append(match.strip())
        return includes
    
    def _parse_conditionals(self) -> List[tuple]:
        """Parse conditional blocks in the template."""
        conditionals = []
        conditional_pattern = r'{{if\s+([^}]+)}}(.*?){{endif}}'
        matches = re.findall(conditional_pattern, self.template_str, re.DOTALL)
        for condition, content in matches:
            conditionals.append((condition.strip(), content.strip()))
        return conditionals
    
    def render(self, context: TemplateContext) -> str:
        """Render the template with provided context."""
        try:
            # Process template with advanced features
            processed_template = self._process_advanced_features(context)
            
            # Standard variable substitution
            template_obj = Template(processed_template)
            return template_obj.safe_substitute(context.get_all_variables())
        except Exception as e:
            raise ValueError(f"Error rendering template '{self.name}': {e}")
    
    def _process_advanced_features(self, context: TemplateContext) -> str:
        """Process advanced template features like includes and conditionals."""
        result = self.template_str
        
        # Process conditionals
        for condition, content in self._conditionals:
            if self._evaluate_condition(condition, context):
                # Replace the conditional block with its content
                pattern = r'{{if\s+' + re.escape(condition) + r'}}.*?{{endif}}'
                result = re.sub(pattern, content, result, flags=re.DOTALL)
            else:
                # Remove the conditional block
                pattern = r'{{if\s+' + re.escape(condition) + r'}}.*?{{endif}}'
                result = re.sub(pattern, '', result, flags=re.DOTALL)
        
        # Process includes (simplified - would need template engine reference)
        include_pattern = r'{{include\s+([^}]+)}}'
        result = re.sub(include_pattern, '', result)
        
        return result
    
    def _evaluate_condition(self, condition: str, context: TemplateContext) -> bool:
        """Evaluate a conditional expression."""
        # Simple condition evaluation - can be extended
        variables = context.get_all_variables()
        
        # Handle simple existence checks
        if condition.startswith('exists '):
            var_name = condition[7:].strip()
            return var_name in variables and variables[var_name] is not None
        
        # Handle boolean checks
        if condition in variables:
            return bool(variables[condition])
        
        # Handle comparison operators
        if ' == ' in condition:
            left, right = condition.split(' == ', 1)
            left_val = variables.get(left.strip())
            right_val = right.strip().strip('\'"')
            return str(left_val) == right_val
        
        if ' != ' in condition:
            left, right = condition.split(' != ', 1)
            left_val = variables.get(left.strip())
            right_val = right.strip().strip('\'"')
            return str(left_val) != right_val
        
        return False
    
    @classmethod
    def from_file(cls, file_path: Path) -> "PromptTemplate":
        """Load template from file."""
        with open(file_path, 'r') as f:
            content = f.read()
        return cls(content, name=str(file_path))


class PromptTemplateEngine:
    """
    Advanced prompt template engine with includes, inheritance, and context management.
    """
    
    def __init__(self, templates_dir: Optional[Path] = None):
        self.templates_dir = templates_dir or Path(__file__).parent / "templates"
        self.templates: Dict[str, PromptTemplate] = {}
        self.global_context = TemplateContext()
        self.load_templates()
    
    def load_templates(self):
        """Load all templates from templates directory."""
        if not self.templates_dir.exists():
            self.templates_dir.mkdir(parents=True, exist_ok=True)
            self._create_default_templates()
            return
        
        for template_file in self.templates_dir.glob("*.txt"):
            template = PromptTemplate.from_file(template_file)
            self.templates[template_file.stem] = template
    
    def _create_default_templates(self):
        """Create default template files."""
        default_templates = {
            "reasoning": """
You are working on solving a difficult problem: ${goal}

Based on your previous thoughts and the overall goal, please perform **one reasoning step** that advances you closer to a solution.

{{if previous_context}}
**Your Previous Context:**
${previous_context}
{{endif}}

{{if available_tools}}
**Available Tools:**
${available_tools}
{{endif}}

**Guidelines:**
- Focus on a specific aspect of the problem
- Build on previous steps without repeating them
- Use logical and analytical thinking
- Ensure your solution meets all requirements
- Current time: ${current_datetime}

Complete the task as soon as you have a valid solution.
""",
            "agent_base": """
You are ${agent_name}, an AI assistant specialized in ${domain}.

**Your Role:**
${role_description}

{{if available_tools}}
**Available Tools:**
${tool_list}
{{endif}}

**Guidelines:**
- Be helpful and accurate
- Use tools when appropriate
- Explain your reasoning
- Follow best practices for ${domain}

{{if context}}
**Context:**
${context}
{{endif}}
""",
            "coding_assistant": """
You are a coding assistant helping with ${project_type} development.

{{if coding_tools}}
**Available Tools:**
${coding_tools}
{{endif}}

**Project Context:**
- Language: ${language}
- Framework: ${framework}
- Current task: ${current_task}

**Guidelines:**
- Write clean, maintainable code
- Follow ${language} best practices
- Test your solutions
- Document your code appropriately

{{if requirements}}
**Requirements:**
${requirements}
{{endif}}
""",
            "data_analyst": """
You are a data analyst helping with ${analysis_type} analysis.

**Data Context:**
- Dataset: ${dataset_name}
- Size: ${dataset_size}
- Format: ${data_format}

**Analysis Goals:**
${analysis_goals}

**Available Tools:**
${data_tools}

**Guidelines:**
- Provide clear insights
- Use appropriate visualizations
- Validate your findings
- Explain your methodology
""",
            "research_assistant": """
You are a research assistant helping with ${research_topic}.

**Research Context:**
- Topic: ${research_topic}
- Scope: ${research_scope}
- Deadline: ${deadline}

**Available Sources:**
${research_sources}

**Guidelines:**
- Provide accurate information
- Cite your sources
- Consider multiple perspectives
- Synthesize findings clearly
"""
        }
        
        for name, content in default_templates.items():
            template_file = self.templates_dir / f"{name}.txt"
            with open(template_file, 'w') as f:
                f.write(content)
            self.templates[name] = PromptTemplate(content, name)
    
    def render_template(self, name: str, context: Optional[TemplateContext] = None, **kwargs) -> str:
        """Render a named template with context."""
        if name not in self.templates:
            raise ValueError(f"Template '{name}' not found")
        
        # Create context if not provided
        if context is None:
            context = TemplateContext()
        
        # Add kwargs to context
        context.update(**kwargs)
        
        # Add global context
        context.globals.update(self.global_context.globals)
        
        # Add tool-related context
        try:
            from ...utilities.registries import TOOLS_REGISTRY
            context.update(available_tools=list(TOOLS_REGISTRY.keys()))
        except ImportError:
            pass
        
        return self.templates[name].render(context)
    
    def register_template(self, name: str, template_str: str):
        """Register a template dynamically."""
        self.templates[name] = PromptTemplate(template_str, name)
    
    def list_templates(self) -> List[str]:
        """List all available templates."""
        return list(self.templates.keys())
    
    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """Get a template by name."""
        return self.templates.get(name)
    
    def save_template(self, name: str, template_str: str):
        """Save a template to file and register it."""
        template_file = self.templates_dir / f"{name}.txt"
        with open(template_file, 'w') as f:
            f.write(template_str)
        self.register_template(name, template_str)
    
    def update_global_context(self, **kwargs):
        """Update global context variables."""
        self.global_context.update(**kwargs)
    
    def create_context(self, **kwargs) -> TemplateContext:
        """Create a new template context with provided variables."""
        context = TemplateContext()
        context.update(**kwargs)
        return context


# Global template engine instance
template_engine = PromptTemplateEngine()


def render_template(name: str, **kwargs) -> str:
    """Convenience function to render a template."""
    return template_engine.render_template(name, **kwargs)


def register_template(name: str, template_str: str):
    """Convenience function to register a template."""
    template_engine.register_template(name, template_str)


def list_templates() -> List[str]:
    """Convenience function to list templates."""
    return template_engine.list_templates()