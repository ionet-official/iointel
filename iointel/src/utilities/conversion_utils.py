"""
Centralized Conversion Utilities - Single Source of Truth
=========================================================

This module consolidates ALL conversion logic from across the codebase into one place.
No more scattered `format_*`, `*_to_*`, `convert_*` methods everywhere!

Core principle: Each conversion has ONE implementation that's used everywhere.

SIMPLE API:
# Auto-detection (inspired by prompt_repository.py)
from iointel.src.utilities.conversion_utils import get
prompt = get(workflow_spec)           # Auto-detects WorkflowSpec
prompt = get(tool_catalog)            # Auto-detects tool catalog  
prompt = get(execution_summary)       # Auto-detects execution summary
prompt = get(validation_errors)       # Auto-detects validation errors

# Specific conversions (when you know the type)
from iointel.src.utilities.conversion_utils import workflow_spec_to_llm_prompt
prompt = workflow_spec_to_llm_prompt(spec)
"""

import json
from typing import Dict, Any, List, Optional, Union
import datetime
from pydantic import BaseModel
import dataclasses

from iointel.src.agent_methods.data_models.workflow_spec import WorkflowSpec, NodeSpec
from iointel.src.web.execution_feedback import WorkflowExecutionSummary, ExecutionStatus


class ConversionUtils:
    """Centralized conversion utilities - single source of truth for all data transformations.
    
    CLEAN API inspired by prompt_repository.py:
    - get(data_object, **kwargs) -> auto-detects type and converts to LLM prompt
    - search(query, type, tags) -> search for relevant prompts/templates
    - All specific conversion methods also available
    """
    
    # ========================================
    # MAIN API - AUTO-DETECTION (from prompt_repository.py)
    # ========================================
    
    @staticmethod
    def get(data_object: Any, **kwargs) -> str:
        """
        Main entry point - automatically detects type and converts to LLM prompt.
        
        Examples:
            ConversionUtils.get(workflow_spec)
            ConversionUtils.get(execution_summary) 
            ConversionUtils.get(tool_catalog, is_data_source=True)
            ConversionUtils.get(validation_errors)
        """
        # WorkflowSpec detection
        if hasattr(data_object, 'nodes') and hasattr(data_object, 'edges'):
            return ConversionUtils.workflow_spec_to_llm_structured(data_object)
        
        # WorkflowExecutionSummary detection
        if hasattr(data_object, 'workflow_title') and hasattr(data_object, 'nodes_executed'):
            return ConversionUtils.execution_summary_to_llm_prompt(data_object)
        
        # Tool catalog detection (dict with tool descriptions)
        if isinstance(data_object, dict) and data_object:
            # Check if it looks like a tool catalog
            first_value = next(iter(data_object.values()))
            if isinstance(first_value, dict) and 'description' in first_value:
                return ConversionUtils.tool_catalog_to_llm_prompt(data_object, **kwargs)
        
        # Validation errors detection (list of lists of strings)
        if (isinstance(data_object, list) and data_object and 
            all(isinstance(item, list) and all(isinstance(e, str) for e in item) for item in data_object)):
            return ConversionUtils.validation_errors_to_llm_prompt(data_object)
        
        # Tool usage results detection
        if (isinstance(data_object, list) and data_object and
            all(hasattr(item, 'tool_name') or (isinstance(item, dict) and 'tool_name' in item) for item in data_object)):
            return ConversionUtils.tool_usage_results_to_llm(data_object)
        
        # Fallback to string representation
        return str(data_object)
    
    # ========================================
    # CORE OBJECT SERIALIZATION
    # ========================================
    
    @staticmethod
    def to_jsonable(obj) -> Any:
        """Convert any object to JSON-serializable format. Used everywhere in UI/API."""
        if isinstance(obj, BaseModel):
            return obj.model_dump()
        elif dataclasses.is_dataclass(obj):
            return ConversionUtils.to_jsonable(dataclasses.asdict(obj))
        elif isinstance(obj, dict):
            return {k: ConversionUtils.to_jsonable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [ConversionUtils.to_jsonable(i) for i in obj]
        elif isinstance(obj, tuple):
            return tuple(ConversionUtils.to_jsonable(i) for i in obj)
        elif isinstance(obj, (datetime.datetime, datetime.date, datetime.time)):
            return obj.isoformat()
        else:
            return obj
    
    # ========================================
    # WORKFLOW SPEC CONVERSIONS
    # ========================================
    
    @staticmethod
    def workflow_spec_to_llm_prompt(spec: WorkflowSpec) -> str:
        """Convert WorkflowSpec to LLM-readable format. Single source of truth."""
        sections = [
            f"# Workflow: {spec.title}",
            f"Description: {spec.description}",
            f"Reasoning: {spec.reasoning}",
            "",
            "## Nodes:",
        ]
        
        for node in spec.nodes:
            sections.append(f"- {node.id} ({node.type}): {node.label}")
            # Only agent and decision nodes have tools
            if node.type in ["agent", "decision"] and hasattr(node.data, 'tools') and node.data.tools:
                sections.append(f"  Tools: {', '.join(node.data.tools)}")
            # Only agent and decision nodes have agent_instructions
            if node.type in ["agent", "decision"] and hasattr(node.data, 'agent_instructions') and node.data.agent_instructions:
                sections.append(f"  Instructions: {node.data.agent_instructions[:100]}...")
            # Data source nodes have source_name
            if node.type == "data_source" and hasattr(node.data, 'source_name'):
                sections.append(f"  Source: {node.data.source_name}")
        
        sections.extend(["", "## Edges:"])
        for edge in spec.edges:
            sections.append(f"- {edge.source} → {edge.target}")
            if hasattr(edge.data, 'route_index') and edge.data.route_index is not None:
                sections.append(f"  Route: {edge.data.route_index}")
        
        return "\n".join(sections)
    
    @staticmethod 
    def workflow_spec_to_yaml(spec: WorkflowSpec, **converter_kwargs) -> str:
        """Convert WorkflowSpec to YAML. Single source of truth."""
        import yaml
        from iointel.src.agent_methods.workflow_converter import WorkflowConverter
        
        converter = WorkflowConverter(**converter_kwargs)
        workflow_def = converter.convert(spec)
        workflow_dict = workflow_def.model_dump(mode="json")
        return yaml.safe_dump(workflow_dict, sort_keys=False)
    
    @staticmethod
    def workflow_spec_to_llm_structured(spec: WorkflowSpec) -> str:
        """
        Convert WorkflowSpec to structured, LLM-friendly representation.
        Moved from WorkflowSpec.to_llm_prompt() - single source of truth.
        
        This includes topology, SLAs, routing logic, and all critical information.
        """
        lines = []
        
        # Header with metadata
        lines.append("📋 WORKFLOW SPECIFICATION")
        lines.append("=" * 50)
        lines.append(f"Title: {spec.title}")
        lines.append(f"Description: {spec.description}")
        lines.append(f"ID: {spec.id}")
        lines.append(f"Version: {spec.rev}")
        lines.append("")
        
        # Topology overview
        lines.append("🏗️ TOPOLOGY OVERVIEW")
        lines.append("-" * 25)
        lines.append(f"Total Nodes: {len(spec.nodes)}")
        lines.append(f"Total Edges: {len(spec.edges)}")
        
        # Categorize nodes
        node_types = {}
        decision_nodes = []
        sla_nodes = []
        
        for node in spec.nodes:
            node_type = node.type
            node_types[node_type] = node_types.get(node_type, 0) + 1
            
            # Check for decision nodes
            if ConversionUtils._is_decision_node(node):
                decision_nodes.append(node.id)
            
            # Check for SLA enforcement
            if ConversionUtils._has_sla_enforcement(node):
                sla_nodes.append(node.id)
        
        for node_type, count in node_types.items():
            lines.append(f"- {node_type}: {count}")
        
        if decision_nodes:
            lines.append(f"- Decision nodes: {', '.join(decision_nodes)}")
        
        if sla_nodes:
            lines.append(f"- SLA enforced: {', '.join(sla_nodes)}")
        
        lines.append("")
        
        # Node details with SLAs
        lines.append("🔍 NODE SPECIFICATIONS")
        lines.append("-" * 25)
        
        for node in spec.nodes:
            lines.extend(ConversionUtils._format_node_details(node))
            lines.append("")
        
        # Edge routing logic
        lines.append("🔀 ROUTING LOGIC")
        lines.append("-" * 20)
        
        if not spec.edges:
            lines.append("No routing edges defined (linear execution)")
        else:
            # Group edges by source
            edges_by_source = {}
            for edge in spec.edges:
                if edge.source not in edges_by_source:
                    edges_by_source[edge.source] = []
                edges_by_source[edge.source].append(edge)
            
            for source_id, edges in edges_by_source.items():
                source_node = next((n for n in spec.nodes if n.id == source_id), None)
                if source_node:
                    lines.append(f"From {source_id} ({source_node.label}):")
                    
                    for edge in edges:
                        target_node = next((n for n in spec.nodes if n.id == edge.target), None)
                        condition_str = f" [condition: {edge.data.condition}]" if edge.data.condition else ""
                        target_label = target_node.label if target_node else edge.target
                        lines.append(f"  → {edge.target} ({target_label}){condition_str}")
                    lines.append("")
        
        # Expected execution patterns
        lines.append("⚡ EXPECTED EXECUTION PATTERNS")
        lines.append("-" * 35)
        
        if decision_nodes:
            lines.append("CONDITIONAL ROUTING EXPECTED:")
            lines.append("- Only ONE path should execute based on conditions")
            lines.append("- Other branches should be skipped (not failures)")
            lines.append("- Efficiency = executed_nodes / nodes_on_chosen_path")
            lines.append("")
        
        if sla_nodes:
            lines.append("SLA ENFORCEMENT ACTIVE:")
            for node_id in sla_nodes:
                node = next((n for n in spec.nodes if n.id == node_id), None)
                if node and hasattr(node, 'sla') and node.sla:
                    sla = node.sla
                    lines.append(f"- {node_id}: {ConversionUtils._format_sla_requirements(sla)}")
            lines.append("")
        
        return "\n".join(lines)
    
    # ========================================
    # HELPER METHODS FOR WORKFLOW CONVERSION
    # ========================================
    
    @staticmethod
    def _format_node_details(node: 'NodeSpec') -> List[str]:
        """Format detailed node information."""
        from iointel.src.agent_methods.data_models.workflow_spec import ROUTING_TOOLS
        
        lines = []
        
        # Node header with type indicator
        node_indicator = "🎯" if ConversionUtils._is_decision_node(node) else "🤖" if node.type == "agent" else "🔧"
        sla_indicator = " [SLA]" if ConversionUtils._has_sla_enforcement(node) else ""
        lines.append(f"{node_indicator} {node.id} - {node.label} ({node.type}){sla_indicator}")
        
        # Instructions/purpose
        if hasattr(node.data, 'agent_instructions') and node.data.agent_instructions:
            lines.append(f"   Purpose: {node.data.agent_instructions[:100]}...")
        elif hasattr(node.data, 'source_name') and node.data.source_name:
            lines.append(f"   Source: {node.data.source_name}")
        
        # Tools available
        if hasattr(node.data, 'tools') and node.data.tools:
            tool_list = []
            for tool in node.data.tools:
                if tool in ROUTING_TOOLS:
                    tool_list.append(f"🔀{tool}")  # Routing tool
                else:
                    tool_list.append(f"🔧{tool}")  # Regular tool
            lines.append(f"   Tools: {', '.join(tool_list)}")
        
        # SLA details
        if hasattr(node, 'sla') and node.sla:
            lines.append(f"   SLA: {ConversionUtils._format_sla_requirements(node.sla)}")
        
        # Configuration
        if hasattr(node.data, 'config') and node.data.config:
            lines.append(f"   Config: {node.data.config}")
        
        return lines
    
    @staticmethod
    def _is_decision_node(node: 'NodeSpec') -> bool:
        """Check if node is a decision/routing node."""
        from iointel.src.agent_methods.data_models.workflow_spec import ROUTING_TOOLS
        
        if node.type == "decision":
            return True
        
        if hasattr(node.data, 'tools') and node.data.tools:
            return any(tool in ROUTING_TOOLS for tool in node.data.tools)
        
        return False
    
    @staticmethod
    def _has_sla_enforcement(node: 'NodeSpec') -> bool:
        """Check if node has SLA enforcement."""
        return (hasattr(node, 'sla') and 
                node.sla and 
                hasattr(node.sla, 'enforce_usage') and 
                node.sla.enforce_usage)
    
    @staticmethod
    def _format_sla_requirements(sla) -> str:
        """Format SLA requirements into readable string."""
        requirements = []
        
        if hasattr(sla, 'tool_usage_required') and sla.tool_usage_required:
            requirements.append("must use tools")
        
        if hasattr(sla, 'min_tool_calls') and sla.min_tool_calls:
            requirements.append(f"min {sla.min_tool_calls} tool calls")
        
        if hasattr(sla, 'required_tools') and sla.required_tools:
            requirements.append(f"required: {', '.join(sla.required_tools)}")
        
        if hasattr(sla, 'final_tool_must_be') and sla.final_tool_must_be:
            requirements.append(f"must end with: {sla.final_tool_must_be}")
        
        return "; ".join(requirements) if requirements else "enforce usage"
    
    # ========================================
    # TOOL USAGE RESULTS CONVERSIONS
    # ========================================
    
    @staticmethod
    def tool_usage_results_to_llm(tool_results: Union[List[Any], List[Dict[str, Any]]]) -> str:
        """Convert tool usage results to LLM-readable format. Single source of truth.
        
        Args:
            tool_results: List of ToolUsageResult objects or dicts with tool data
        """
        if not tool_results:
            return "No tools used."
        
        from ..agent_methods.data_models.datamodels import ToolUsageResult
        
        sections = []
        for i, result in enumerate(tool_results, 1):
            # Handle both ToolUsageResult objects and dicts for backward compatibility
            if isinstance(result, ToolUsageResult):
                # Pydantic model with guaranteed fields
                tool_name = result.tool_name
                tool_result = result.tool_result or 'No result'
                tool_args = result.tool_args
            elif isinstance(result, dict):
                # Dict with keys (backward compatibility)
                tool_name = result.get('tool_name', 'unknown_tool')
                tool_result = result.get('result', result.get('tool_result', 'No result'))
                tool_args = result.get('tool_args', result.get('input', {}))
            else:
                # Fallback
                tool_name = str(result)
                tool_result = 'Unknown result format'
                tool_args = {}
            
            sections.append(f"{i}. **{tool_name}**")
            if tool_args:
                sections.append(f"   Input: {json.dumps(ConversionUtils.to_jsonable(tool_args), indent=2)}")
            if isinstance(tool_result, (dict, list)):
                sections.append(f"   Result: {json.dumps(tool_result, indent=2)}")
            else:
                sections.append(f"   Result: {str(tool_result)[:200]}...")
        
        return "\n".join(sections)
    
    @staticmethod
    def tool_usage_results_to_html(tool_results: Union[List[Any], List[Dict[str, Any]]]) -> str:
        """Convert tool usage results to HTML pills. Single source of truth.
        
        Args:
            tool_results: List of ToolUsageResult objects or dicts with tool data
        """
        if not tool_results:
            return "<p>No tools used.</p>"
        
        from ..agent_methods.data_models.datamodels import ToolUsageResult
        
        html_parts = []
        for result in tool_results:
            # Handle both ToolUsageResult objects and dicts for backward compatibility
            if isinstance(result, ToolUsageResult):
                # Pydantic model with guaranteed fields
                tool_name = result.tool_name
                tool_args = result.tool_args
                tool_result = result.tool_result or ''
            elif isinstance(result, dict):
                # Dict with keys (backward compatibility)
                tool_name = result.get('tool_name', 'unknown_tool')
                tool_args = result.get('tool_args', result.get('input', {}))
                tool_result = result.get('tool_result', result.get('result', ''))
            else:
                # Fallback
                tool_name = str(result)
                tool_args = {}
                tool_result = 'Unknown result format'
            
            pill_html = f"""
            <div class="tool-pill" style="margin-bottom:10px;">
                <div style="font-weight:bold;font-size:1.1em;">🛠️ {tool_name}</div>
                <div style="font-size:0.95em;"><b>Args:</b>
                    <pre style="background:#23272f;color:#ffb300;padding:4px 8px;border-radius:6px;">{
                        json.dumps(ConversionUtils.to_jsonable(tool_args), indent=2)
                    }</pre>
                </div>
                <div style="font-size:0.95em;"><b>Result:</b>
                    <pre style="background:#23272f;color:#ffb300;padding:4px 8px;border-radius:6px;">{
                        json.dumps(ConversionUtils.to_jsonable(tool_result), indent=2) 
                        if not isinstance(tool_result, str) or not ("<" in tool_result and ">" in tool_result)
                        else tool_result
                    }</pre>
                </div>
            </div>
            """
            html_parts.append(pill_html)
        
        return "\n".join(html_parts)
    
    # ========================================
    # EXECUTION SUMMARY CONVERSIONS  
    # ========================================
    
    @staticmethod
    def execution_summary_to_llm_prompt(summary: WorkflowExecutionSummary) -> str:
        """Convert execution summary to LLM prompt format. Single source of truth."""
        sections = [
            f"# Execution Report: {summary.workflow_title}",
            f"Status: {summary.status.value.upper()}",
            f"Duration: {summary.total_duration_seconds:.2f}s",
            f"Nodes Executed: {len(summary.nodes_executed)}",
            f"Nodes Skipped: {len(summary.nodes_skipped)}",
            ""
        ]
        
        if summary.nodes_executed:
            sections.append("## Executed Nodes:")
            for node in summary.nodes_executed:
                status_icon = "✅" if node.status in [ExecutionStatus.SUCCESS, ExecutionStatus.COMPLETED] else "❌"
                sections.append(f"{status_icon} {node.node_label} ({node.node_type})")
                if node.tool_usage:
                    sections.append(f"   Tools: {', '.join(node.tool_usage)}")
                if node.result_preview:
                    preview = node.result_preview[:100] + "..." if len(node.result_preview) > 100 else node.result_preview
                    sections.append(f"   Result: {preview}")
                if node.error_message:
                    sections.append(f"   Error: {node.error_message[:100]}...")
        
        if summary.error_summary:
            sections.extend(["", "## Error Summary:", summary.error_summary])
        
        return "\n".join(sections)
    
    @staticmethod
    def execution_summary_to_html(summary: WorkflowExecutionSummary) -> str:
        """Convert execution summary to HTML display. Single source of truth."""
        status_color = "#28a745" if summary.status == ExecutionStatus.SUCCESS else "#dc3545"
        
        html = f"""
        <div class="execution-summary">
            <h3 style="color: {status_color};">{summary.workflow_title}</h3>
            <p><strong>Status:</strong> <span style="color: {status_color};">{summary.status.value.upper()}</span></p>
            <p><strong>Duration:</strong> {summary.total_duration_seconds:.2f}s</p>
            <p><strong>Nodes:</strong> {len(summary.nodes_executed)} executed, {len(summary.nodes_skipped)} skipped</p>
        """
        
        if summary.nodes_executed:
            html += "<h4>Executed Nodes:</h4><ul>"
            for node in summary.nodes_executed:
                status_icon = "✅" if node.status in [ExecutionStatus.SUCCESS, ExecutionStatus.COMPLETED] else "❌"
                html += f"<li>{status_icon} <strong>{node.node_label}</strong> ({node.node_type})"
                if node.tool_usage:
                    html += f"<br>Tools: {', '.join(node.tool_usage)}"
                html += "</li>"
            html += "</ul>"
        
        html += "</div>"
        return html
    
    # ========================================
    # VALIDATION ERRORS CONVERSIONS
    # ========================================
    
    @staticmethod
    def validation_errors_to_llm_prompt(errors: List[List[str]]) -> str:
        """Convert validation errors to LLM prompt format. Single source of truth."""
        if not errors:
            return "No validation errors."
        
        sections = [
            "🚨🚨🚨 CRITICAL VALIDATION FAILURES - YOUR PREVIOUS ATTEMPTS FAILED 🚨🚨🚨",
            "YOU KEEP MAKING THE SAME MISTAKES! READ AND FIX THESE ERRORS:",
            ""
        ]
        
        # Check for user_input config errors specifically
        has_user_input_error = any(
            "user_input" in error and ("missing required parameters" in error.lower() or "empty config" in error.lower())
            for error_group in errors for error in error_group
        )
        
        if has_user_input_error:
            sections.extend([
                "⚠️⚠️⚠️ USER_INPUT CONFIG ERROR DETECTED ⚠️⚠️⚠️",
                "YOU ARE GENERATING INVALID user_input NODES!",
                "",
                "❌ WRONG (what you keep doing):",
                '{"type": "data_source", "data": {"source_name": "user_input", "config": null}}',
                '{"type": "data_source", "data": {"source_name": "user_input", "config": {}}}',
                "",
                "✅ CORRECT (what you MUST do):",
                '{"type": "data_source", "data": {"source_name": "user_input", "config": {"message": "Enter your input", "default_value": ""}}}',
                "",
                "EVERY user_input MUST have config with BOTH message AND default_value!",
                "DO NOT generate config: null or config: {} - THIS WILL FAIL!",
                "="*80,
                ""
            ])
        
        for i, error_group in enumerate(errors, 1):
            sections.append(f"## Attempt {i} FAILED with these errors:")
            for error in error_group:
                sections.append(f"❌ {error}")
            sections.append("")
        
        sections.extend([
            "## YOU MUST FIX THESE ERRORS:",
            "1. If error says 'missing required parameters' - ADD THOSE PARAMETERS TO CONFIG",
            "2. If error says 'empty config' - ADD A CONFIG OBJECT WITH ALL REQUIRED FIELDS",
            "3. If error mentions 'user_input' - USE THE TEMPLATE ABOVE",
            "",
            "THIS IS YOUR LAST CHANCE - FIX THESE ERRORS OR THE WORKFLOW WILL FAIL!",
            "="*80,
            ""
        ])
        
        return "\n".join(sections)
    
    @staticmethod 
    def validation_errors_to_html(errors: List[List[str]]) -> str:
        """Convert validation errors to HTML display. Single source of truth."""
        if not errors:
            return "<p style='color: green;'>✅ No validation errors</p>"
        
        html = ["<div class='validation-errors' style='color: #dc3545;'>"]
        html.append("<h4>❌ Validation Errors:</h4>")
        
        for i, error_group in enumerate(errors, 1):
            html.append(f"<h5>Error Group {i}:</h5>")
            html.append("<ul>")
            for error in error_group:
                html.append(f"<li>{error}</li>")
            html.append("</ul>")
        
        html.append("</div>")
        return "\n".join(html)
    
    # ========================================
    # TOOL CATALOG CONVERSIONS
    # ========================================
    
    @staticmethod
    def tool_catalog_to_llm_prompt(catalog: Dict[str, Any], title: str = "# Available Tools:", 
                                  usage_note: Optional[str] = None, 
                                  is_data_source: bool = False) -> str:
        """Convert tool catalog to LLM prompt format. Single source of truth.
        
        Args:
            catalog: Tool/data source catalog dictionary
            title: Custom title for the section
            usage_note: Warning/usage note to append
            is_data_source: True if this is data sources (adds config examples)
        
        Returns:
            Complete, ready-to-use prompt section
        """
        if not catalog:
            return f"{title}\n❌ NO ITEMS AVAILABLE"
        
        sections = [f"{title} ({len(catalog)} total):", ""]
        
        # Add data source config examples if needed
        if is_data_source:
            sections.extend([
                "⚠️ MANDATORY CONFIG EXAMPLES - ALWAYS INCLUDE ALL REQUIRED FIELDS:",
                '  user_input: {"source_name": "user_input", "type": "data_source", "config": {"message": "Enter your input", "default_value": "Default response"}}',
                '  prompt_tool: {"source_name": "prompt_tool", "type": "data_source", "config": {"message": "Enter contextual info", "default_value": "Default context"}}',
                ""
            ])
        
        for i, (item_name, item_info) in enumerate(catalog.items()):
            # Check format type
            if 'params' in item_info:
                # Concise format
                desc = item_info.get('description', '')
                params = item_info.get('params', [])
                param_str = f"({', '.join(params)})" if params else "()"
                
                if is_data_source:
                    sections.append(f"source_name: {item_name}{param_str} - {desc}")
                else:
                    sections.append(f"tool: {item_name}{param_str} - {desc}")
            else:
                # Verbose format
                sections.append(f"📦 {item_name}")
                sections.append(f"   Description: {item_info.get('description', 'No description')}")
                
                parameters = item_info.get('parameters', {})
                if parameters:
                    sections.append("   Parameters:")
                    for param, param_info in parameters.items():
                        if isinstance(param_info, dict):
                            param_type = param_info.get('type', 'any')
                            required = param_info.get('required', False)
                            desc = param_info.get('description', '')
                            req_str = " (required)" if required else " (optional)"
                            sections.append(f"     • {param} ({param_type}){req_str}: {desc}")
                        else:
                            sections.append(f"     • {param}: {param_info}")
                else:
                    sections.append("   Parameters: None")
                
                # Add usage example
                if is_data_source:
                    if item_name == 'user_input':
                        usage = '{"source_name": "user_input", "type": "data_source", "config": {"prompt": "Enter your question", "default_value": "What is the weather?"}}'
                    elif item_name == 'prompt_tool':
                        usage = '{"source_name": "prompt_tool", "type": "data_source", "config": {"prompt": "Enter context", "default_value": "Default"}}'
                    else:
                        usage = f'{{"source_name": "{item_name}", "config": {{"param": "value"}}}}'
                else:
                    usage = f'"{item_name}"'
                sections.append(f"   Usage: {usage}")
            
            # Add separator between items (except last)
            if i < len(catalog) - 1:
                sections.append("===")
            
            sections.append("")
        
        # Add usage note if provided
        if usage_note:
            sections.append(f"🚨 {usage_note}. Any other names will cause failure.")
        
        return "\n".join(sections)


# ========================================
# CONVENIENCE FUNCTIONS (GLOBAL SCOPE)
# ========================================

# ========================================
# MAIN API - SIMPLE AND CLEAN
# ========================================

# Primary interface - auto-detects type and converts
def get(data_object: Any, **kwargs) -> str:
    """Auto-detect type and convert to LLM prompt. Main entry point."""
    return ConversionUtils.get(data_object, **kwargs)

# ========================================
# SPECIFIC CONVERSION FUNCTIONS  
# ========================================

# Single source of truth exports - use these everywhere
to_jsonable = ConversionUtils.to_jsonable
workflow_spec_to_llm_prompt = ConversionUtils.workflow_spec_to_llm_prompt
workflow_spec_to_llm_structured = ConversionUtils.workflow_spec_to_llm_structured
workflow_spec_to_yaml = ConversionUtils.workflow_spec_to_yaml
tool_usage_results_to_llm = ConversionUtils.tool_usage_results_to_llm
tool_usage_results_to_html = ConversionUtils.tool_usage_results_to_html
execution_summary_to_llm_prompt = ConversionUtils.execution_summary_to_llm_prompt
execution_summary_to_html = ConversionUtils.execution_summary_to_html
validation_errors_to_llm_prompt = ConversionUtils.validation_errors_to_llm_prompt
validation_errors_to_html = ConversionUtils.validation_errors_to_html
tool_catalog_to_llm_prompt = ConversionUtils.tool_catalog_to_llm_prompt


# ========================================
# LEGACY COMPATIBILITY ALIASES
# ========================================
# Keep these during migration, remove after all code updated

def format_result_for_html(result_dict: Dict[str, Any]) -> str:
    """Legacy alias - use tool_usage_results_to_html instead."""
    tool_results = result_dict.get("tool_usage_results", [])
    main_result = result_dict.get("result", "")
    
    html_parts = []
    if main_result:
        html_parts.append(f'<div style="margin-bottom:1em;"><b>Agent:</b> {main_result}</div>')
    
    html_parts.append(tool_usage_results_to_html(tool_results))
    return "\n".join(html_parts)

def spec_to_yaml(spec: WorkflowSpec, **kwargs) -> str:
    """Legacy alias - use workflow_spec_to_yaml instead."""
    return workflow_spec_to_yaml(spec, **kwargs)

def spec_to_definition(spec: WorkflowSpec, **kwargs):
    """Legacy alias - imports from workflow_converter."""
    from iointel.src.agent_methods.workflow_converter import spec_to_definition as _spec_to_definition
    return _spec_to_definition(spec, **kwargs)