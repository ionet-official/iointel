"""
Unified Prompt System with Storage + Search
==========================================

A consolidated prompt management system that provides:
- Centralized prompt building for agents, workflows, feedback
- Searchable prompt repository with semantic search
- Template management and reuse
- Version control for prompt evolution
- Integration with existing prompt collections

This replaces scattered prompt building logic across the codebase.
"""

from typing import Dict, Any, List, Optional, Protocol
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import json
import hashlib
from datetime import datetime
import re

# Import existing prompt collections for compatibility
from iointel.src.agent_methods.data_models.prompt_collections import PromptCollectionManager


class PromptType(str, Enum):
    """Types of prompts in the system."""
    WORKFLOW_GENERATION = "workflow_generation"
    EXECUTION_FEEDBACK = "execution_feedback" 
    TOOL_CATALOG = "tool_catalog"
    AGENT_INSTRUCTIONS = "agent_instructions"
    VALIDATION_ERROR = "validation_error"
    REFINEMENT_GUIDANCE = "refinement_guidance"
    SYSTEM_MESSAGE = "system_message"


@dataclass
class PromptTemplate:
    """A reusable prompt template with variables."""
    id: str
    name: str
    prompt_type: PromptType
    template: str
    variables: List[str] = field(default_factory=list)
    description: str = ""
    tags: List[str] = field(default_factory=list)
    version: int = 1
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    usage_count: int = 0
    
    def render(self, **kwargs) -> str:
        """Render template with provided variables."""
        missing_vars = set(self.variables) - set(kwargs.keys())
        if missing_vars:
            raise ValueError(f"Missing template variables: {missing_vars}")
        
        # Simple template rendering (could be enhanced with Jinja2 later)
        result = self.template
        for var, value in kwargs.items():
            result = result.replace(f"{{{var}}}", str(value))
        
        self.usage_count += 1
        return result


@dataclass 
class PromptInstance:
    """A rendered prompt instance with metadata."""
    id: str
    template_id: Optional[str]
    prompt_type: PromptType
    content: str
    variables_used: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    success: Optional[bool] = None
    response_preview: Optional[str] = None


class PromptBuilder(Protocol):
    """Protocol for prompt builders."""
    def build(self) -> str: ...


class UnifiedPromptSystem:
    """
    Centralized prompt management with storage and search capabilities.
    """
    
    def __init__(self, storage_dir: str = "prompt_repository"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # Initialize subdirectories
        (self.storage_dir / "templates").mkdir(exist_ok=True)
        (self.storage_dir / "instances").mkdir(exist_ok=True)
        
        # Template and instance storage
        self.templates: Dict[str, PromptTemplate] = {}
        self.instances: Dict[str, PromptInstance] = {}
        
        # Integration with existing prompt collections
        self.collection_manager = PromptCollectionManager()
        
        # Load existing templates and instances
        self._load_templates()
        self._load_instances()
        
        # Initialize built-in templates
        self._initialize_builtin_templates()
    
    def create_template(
        self, 
        name: str,
        prompt_type: PromptType,
        template: str,
        variables: Optional[List[str]] = None,
        description: str = "",
        tags: Optional[List[str]] = None
    ) -> PromptTemplate:
        """Create a new prompt template."""
        template_id = self._generate_id(f"{prompt_type}_{name}")
        
        # Extract variables from template if not provided
        if variables is None:
            variables = re.findall(r'\{(\w+)\}', template)
        
        prompt_template = PromptTemplate(
            id=template_id,
            name=name,
            prompt_type=prompt_type,
            template=template,
            variables=variables or [],
            description=description,
            tags=tags or []
        )
        
        self.templates[template_id] = prompt_template
        self._save_template(prompt_template)
        return prompt_template
    
    def render_prompt(
        self,
        template_id: str,
        context: Optional[Dict[str, Any]] = None,
        **variables
    ) -> PromptInstance:
        """Render a prompt from a template."""
        if template_id not in self.templates:
            raise ValueError(f"Template {template_id} not found")
        
        template = self.templates[template_id]
        content = template.render(**variables)
        
        instance_id = self._generate_id(f"{template_id}_{datetime.now().isoformat()}")
        instance = PromptInstance(
            id=instance_id,
            template_id=template_id,
            prompt_type=template.prompt_type,
            content=content,
            variables_used=variables,
            context=context or {}
        )
        
        self.instances[instance_id] = instance
        self._save_instance(instance)
        return instance
    
    def create_direct_prompt(
        self,
        prompt_type: PromptType,
        content: str,
        context: Optional[Dict[str, Any]] = None
    ) -> PromptInstance:
        """Create a prompt directly without a template."""
        instance_id = self._generate_id(f"{prompt_type}_{datetime.now().isoformat()}")
        instance = PromptInstance(
            id=instance_id,
            template_id=None,
            prompt_type=prompt_type,
            content=content,
            context=context or {}
        )
        
        self.instances[instance_id] = instance
        self._save_instance(instance)
        return instance
    
    def search_templates(
        self,
        query: Optional[str] = None,
        prompt_type: Optional[PromptType] = None,
        tags: Optional[List[str]] = None
    ) -> List[PromptTemplate]:
        """Search templates by query, type, or tags."""
        results = list(self.templates.values())
        
        if prompt_type:
            results = [t for t in results if t.prompt_type == prompt_type]
        
        if tags:
            results = [t for t in results if any(tag in t.tags for tag in tags)]
        
        if query:
            query_lower = query.lower()
            results = [
                t for t in results 
                if query_lower in t.name.lower() 
                or query_lower in t.description.lower()
                or query_lower in t.template.lower()
            ]
        
        # Sort by usage count and relevance
        results.sort(key=lambda t: (t.usage_count, t.name), reverse=True)
        return results
    
    def search_instances(
        self,
        query: Optional[str] = None,
        prompt_type: Optional[PromptType] = None,
        success: Optional[bool] = None
    ) -> List[PromptInstance]:
        """Search prompt instances."""
        results = list(self.instances.values())
        
        if prompt_type:
            results = [i for i in results if i.prompt_type == prompt_type]
        
        if success is not None:
            results = [i for i in results if i.success == success]
        
        if query:
            query_lower = query.lower()
            results = [
                i for i in results
                if query_lower in i.content.lower()
            ]
        
        # Sort by creation date (newest first)
        results.sort(key=lambda i: i.created_at, reverse=True)
        return results
    
    def record_prompt_result(
        self,
        instance_id: str,
        success: bool,
        response_preview: Optional[str] = None
    ):
        """Record the result of using a prompt."""
        if instance_id in self.instances:
            instance = self.instances[instance_id]
            instance.success = success
            instance.response_preview = response_preview
            self._save_instance(instance)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the prompt system."""
        return {
            "total_templates": len(self.templates),
            "total_instances": len(self.instances),
            "templates_by_type": {
                ptype.value: len([t for t in self.templates.values() if t.prompt_type == ptype])
                for ptype in PromptType
            },
            "instances_by_type": {
                ptype.value: len([i for i in self.instances.values() if i.prompt_type == ptype])
                for ptype in PromptType
            },
            "success_rate": {
                ptype.value: self._calculate_success_rate(ptype)
                for ptype in PromptType
            }
        }
    
    def _calculate_success_rate(self, prompt_type: PromptType) -> float:
        """Calculate success rate for a prompt type."""
        instances = [i for i in self.instances.values() if i.prompt_type == prompt_type and i.success is not None]
        if not instances:
            return 0.0
        return sum(1 for i in instances if i.success) / len(instances)
    
    def _generate_id(self, base: str) -> str:
        """Generate a unique ID."""
        return hashlib.md5(base.encode()).hexdigest()[:12]
    
    def _save_template(self, template: PromptTemplate):
        """Save template to disk."""
        template_file = self.storage_dir / "templates" / f"{template.id}.json"
        with open(template_file, 'w') as f:
            json.dump(template.__dict__, f, indent=2)
    
    def _save_instance(self, instance: PromptInstance):
        """Save instance to disk."""
        instance_file = self.storage_dir / "instances" / f"{instance.id}.json"
        with open(instance_file, 'w') as f:
            json.dump(instance.__dict__, f, indent=2)
    
    def _load_templates(self):
        """Load templates from disk."""
        templates_dir = self.storage_dir / "templates"
        if not templates_dir.exists():
            return
        
        for template_file in templates_dir.glob("*.json"):
            try:
                with open(template_file) as f:
                    data = json.load(f)
                template = PromptTemplate(**data)
                self.templates[template.id] = template
            except Exception as e:
                print(f"⚠️ Failed to load template {template_file}: {e}")
    
    def _load_instances(self):
        """Load instances from disk."""
        instances_dir = self.storage_dir / "instances"
        if not instances_dir.exists():
            return
        
        for instance_file in instances_dir.glob("*.json"):
            try:
                with open(instance_file) as f:
                    data = json.load(f)
                instance = PromptInstance(**data)
                self.instances[instance.id] = instance
            except Exception as e:
                print(f"⚠️ Failed to load instance {instance_file}: {e}")
    
    def _initialize_builtin_templates(self):
        """Initialize built-in templates by importing from existing prompt files."""
        try:
            # Import existing workflow prompts from absolute path
            from iointel.src.agent_methods.agents.workflow_prompts import get_workflow_planner_instructions, WORKFLOW_PLANNER_INSTRUCTIONS_COMPREHENSIVE as WORKFLOW_PLANNER_INSTRUCTIONS_TEMPLATE
            
            # Store workflow planner instructions as searchable template
            if not any(t.name == "workflow_planner_instructions" for t in self.templates.values()):
                self.create_template(
                    name="workflow_planner_instructions",
                    prompt_type=PromptType.AGENT_INSTRUCTIONS,
                    template=WORKFLOW_PLANNER_INSTRUCTIONS_TEMPLATE,  # Use the raw template, not the processed one
                    variables=["VALID_DATA_SOURCES", "DATA_SOURCE_KNOWLEDGE"],  # Variables that get replaced
                    description="Workflow planner instructions imported from workflow_prompts.py",
                    tags=["workflow_planner", "agent_instructions", "imported"]
                )
                
        except ImportError as e:
            print(f"⚠️ Could not import workflow prompts: {e}")
        
        # Add simple feedback templates
        if not any(t.name == "execution_feedback_success" for t in self.templates.values()):
            self.create_template(
                name="execution_feedback_success",
                prompt_type=PromptType.EXECUTION_FEEDBACK,
                template="🎉 **{workflow_title}** completed in {duration}s\n\n🎯 **Results:** {agent_outputs}\n🛠️ **Tools:** {tool_results}",
                variables=["workflow_title", "duration", "agent_outputs", "tool_results"],
                description="Concise success feedback",
                tags=["execution", "success"]
            )
        
        if not any(t.name == "execution_feedback_failure" for t in self.templates.values()):
            self.create_template(
                name="execution_feedback_failure", 
                prompt_type=PromptType.EXECUTION_FEEDBACK,
                template="❌ **{workflow_title}** failed\n\n🔍 **Issue:** {error_analysis}\n🔧 **Fixes:** {fixes}",
                variables=["workflow_title", "error_analysis", "fixes"],
                description="Concise failure feedback",
                tags=["execution", "failure"]
            )


# Global instance for use across the codebase
unified_prompt_system = UnifiedPromptSystem()


# Convenience functions for easy adoption
def create_prompt_template(name: str, prompt_type: PromptType, template: str, **kwargs) -> PromptTemplate:
    """Convenience function to create a prompt template."""
    return unified_prompt_system.create_template(name, prompt_type, template, **kwargs)


def render_prompt(template_id: str, **kwargs) -> PromptInstance:
    """Convenience function to render a prompt."""
    return unified_prompt_system.render_prompt(template_id, **kwargs)


def search_prompts(query: Optional[str] = None, prompt_type: Optional[PromptType] = None, **kwargs) -> List[PromptTemplate]:
    """Convenience function to search prompts."""
    return unified_prompt_system.search_templates(query, prompt_type, **kwargs)


if __name__ == "__main__":
    # Example usage
    system = UnifiedPromptSystem()
    
    # Create a workflow generation template
    template = system.create_template(
        name="simple_workflow_generation",
        prompt_type=PromptType.WORKFLOW_GENERATION,
        template="Generate a workflow for: {user_request}\n\nAvailable tools: {tool_list}\n\nCreate a {complexity} workflow that {requirements}.",
        variables=["user_request", "tool_list", "complexity", "requirements"],
        description="Simple workflow generation template",
        tags=["workflow", "generation", "simple"]
    )
    
    # Render the template
    instance = system.render_prompt(
        template.id,
        user_request="email processing",
        tool_list="email_parser, sentiment_analyzer",
        complexity="basic",
        requirements="processes emails and analyzes sentiment"
    )
    
    print("Template created:", template.name)
    print("Rendered prompt:", instance.content[:100] + "...")
    
    # Record success
    system.record_prompt_result(instance.id, success=True, response_preview="Generated 3-node workflow successfully")
    
    # Get stats
    stats = system.get_stats()
    print("System stats:", stats)