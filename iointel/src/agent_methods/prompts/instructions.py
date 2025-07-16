"""
Enhanced instruction management system for iointel agents.

This module provides both legacy hardcoded instructions and new templated instructions.
"""

from typing import Dict, Any, Optional
from .template_engine import template_engine, TemplateContext


# Legacy hardcoded instructions for backward compatibility
REASONING_INSTRUCTIONS = """
    You are working on solving a difficult problem (the `goal`). Based
    on your previous thoughts and the overall goal, please perform **one
    reasoning step** that advances you closer to a solution. Document
    your thought process and any intermediate steps you take.
    
    After marking this task complete for a single step, you will be
    given a new reasoning task to continue working on the problem. The
    loop will continue until you have a valid solution.
    
    Complete the task as soon as you have a valid solution.
    
    **Guidelines**
    
    - You will not be able to brute force a solution exhaustively. You
        must use your reasoning ability to make a plan that lets you make
        progress.
    - Each step should be focused on a specific aspect of the problem,
        either advancing your understanding of the problem or validating a
        solution.
    - You should build on previous steps without repeating them.
    - Since you will iterate your reasoning, you can explore multiple
        approaches in different steps.
    - Use logical and analytical thinking to reason through the problem.
    - Ensure that your solution is valid and meets all requirements.
    - If you find yourself spinning your wheels, take a step back and
        re-evaluate your approach.
"""


class InstructionManager:
    """
    Manages instruction generation for different agent types and scenarios.
    """
    
    def __init__(self):
        self.template_engine = template_engine
        self._register_instruction_templates()
    
    def _register_instruction_templates(self):
        """Register instruction-specific templates."""
        
        # Enhanced reasoning template
        self.template_engine.register_template("enhanced_reasoning", """
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

{{if constraints}}
**Constraints:**
${constraints}
{{endif}}

Complete the task as soon as you have a valid solution.
""")
        
        # Task-specific instruction templates
        self.template_engine.register_template("task_executor", """
You are ${agent_name}, executing a specific task within a larger workflow.

**Task:** ${task_description}

{{if task_context}}
**Context:** ${task_context}
{{endif}}

{{if available_tools}}
**Available Tools:**
${available_tools}
{{endif}}

**Guidelines:**
- Focus solely on completing this specific task
- Use available tools as needed
- Provide clear results
- Report any issues or blockers
- Be efficient and accurate
""")
        
        # Collaborative agent template
        self.template_engine.register_template("collaborative_agent", """
You are ${agent_name}, part of a collaborative team of agents.

**Your Role:** ${role_description}

**Team Context:**
- Other agents: ${team_members}
- Shared goal: ${shared_goal}
- Your responsibility: ${responsibility}

{{if available_tools}}
**Available Tools:**
${available_tools}
{{endif}}

**Guidelines:**
- Collaborate effectively with other agents
- Share relevant information
- Avoid duplicating others' work
- Focus on your specific responsibilities
- Communicate clearly about progress and blockers
""")
        
        # Domain-specific templates
        self.template_engine.register_template("security_agent", """
You are ${agent_name}, a security-focused AI assistant.

**Security Context:**
- Risk level: ${risk_level}
- Compliance requirements: ${compliance_requirements}
- Security policies: ${security_policies}

**Guidelines:**
- Prioritize security in all recommendations
- Identify potential vulnerabilities
- Suggest security best practices
- Validate input and output for security risks
- Report security concerns immediately

{{if available_tools}}
**Available Security Tools:**
${available_tools}
{{endif}}
""")
    
    def get_instruction(
        self, 
        instruction_type: str, 
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """
        Get formatted instruction for a specific type.
        
        Args:
            instruction_type: Type of instruction (e.g., "reasoning", "task_executor")
            context: Additional context variables
            **kwargs: Additional variables to pass to template
        
        Returns:
            Formatted instruction string
        """
        if instruction_type == "reasoning":
            # Return legacy reasoning instructions for backward compatibility
            return REASONING_INSTRUCTIONS
        
        # Use template engine for new instruction types
        template_context = TemplateContext()
        if context:
            template_context.update(**context)
        template_context.update(**kwargs)
        
        return self.template_engine.render_template(instruction_type, template_context)
    
    def create_custom_instruction(
        self,
        agent_name: str,
        role_description: str,
        domain: str = "general",
        additional_context: Optional[Dict[str, Any]] = None,
        template_name: str = "agent_base"
    ) -> str:
        """
        Create a custom instruction for an agent.
        
        Args:
            agent_name: Name of the agent
            role_description: Description of the agent's role
            domain: Domain of expertise
            additional_context: Additional context variables
            template_name: Base template to use
        
        Returns:
            Formatted instruction string
        """
        context = {
            "agent_name": agent_name,
            "role_description": role_description,
            "domain": domain,
        }
        
        if additional_context:
            context.update(additional_context)
        
        return self.get_instruction(template_name, context)
    
    def register_custom_template(self, name: str, template_str: str):
        """Register a custom instruction template."""
        self.template_engine.register_template(name, template_str)
    
    def list_instruction_types(self) -> list:
        """List all available instruction types."""
        return self.template_engine.list_templates()


# Global instruction manager instance
instruction_manager = InstructionManager()


def get_instruction(instruction_type: str, **kwargs) -> str:
    """Convenience function to get an instruction."""
    return instruction_manager.get_instruction(instruction_type, **kwargs)


def create_custom_instruction(
    agent_name: str,
    role_description: str,
    domain: str = "general",
    **kwargs
) -> str:
    """Convenience function to create a custom instruction."""
    return instruction_manager.create_custom_instruction(
        agent_name, role_description, domain, **kwargs
    )
