"""
IO.net themed logger with beautiful formatting that matches the design ethos.

This logger provides structured, emoji-rich output that's both human-readable
and machine-parseable, following the cyberpunk/IO.net aesthetic.
"""

from datetime import datetime
from typing import Dict, Any, Optional, List, ContextManager
from enum import Enum
from contextlib import contextmanager
from dataclasses import dataclass, field
import threading
import json


class LogLevel(Enum):
    """Log levels with corresponding emojis and IO.net cyberpunk colors."""
    DEBUG = ("▪", "DEBUG", "\033[38;5;240m")      # Dark gray
    INFO = ("◆", "INFO", "\033[38;5;51m")         # Electric cyan  
    SUCCESS = ("◉", "SUCCESS", "\033[38;5;76m")  # Electric green
    WARNING = ("◈", "WARN", "\033[38;5;214m")    # Electric orange
    ERROR = ("◇", "ERROR", "\033[38;5;196m")     # Electric red
    CRITICAL = ("◎", "CRIT", "\033[38;5;201m")   # Electric magenta


@dataclass
class LogContext:
    """Context for a grouped logging session."""
    component: str
    execution_id: Optional[str] = None
    indent_level: int = 0
    parent: Optional['LogContext'] = None
    start_time: float = field(default_factory=lambda: datetime.now().timestamp())
    suppress_header: bool = False


class IOLogger:
    """
    IO.net themed logger with structured, beautiful output.
    
    Features:
    - Emoji indicators for different log levels
    - Structured data support for complex objects
    - Component-based logging (workflow, execution, agent, etc.)
    - Timeline formatting for execution tracking
    - Optional JSON structured logging for machine parsing
    - Context management for grouped/hierarchical logging
    """
    
    def __init__(self, component: str = "system", structured: bool = False):
        self.component = component.upper()
        self.structured = structured
        self._contexts = threading.local()
        self._use_grouping = False  # Enable grouped logging mode
    
    def enable_grouping(self):
        """Enable grouped/hierarchical logging mode."""
        self._use_grouping = True
        return self
    
    @property
    def current_context(self) -> Optional[LogContext]:
        """Get the current logging context."""
        if not hasattr(self._contexts, 'stack'):
            self._contexts.stack = []
        return self._contexts.stack[-1] if self._contexts.stack else None
    
    @contextmanager
    def group(self, title: str, execution_id: Optional[str] = None, suppress_header: bool = False) -> ContextManager[LogContext]:
        """
        Create a grouped logging context.
        
        Usage:
            with logger.group("Building DAG"):
                logger.info("Processing nodes")
                with logger.group("Node Execution"):
                    logger.info("Executing node X")
        """
        # Create new context
        parent = self.current_context
        new_context = LogContext(
            component=title,
            execution_id=execution_id or (parent.execution_id if parent else None),
            indent_level=(parent.indent_level + 1) if parent else 0,
            parent=parent,
            suppress_header=suppress_header
        )
        
        # Push to stack
        if not hasattr(self._contexts, 'stack'):
            self._contexts.stack = []
        self._contexts.stack.append(new_context)
        
        # Print header if grouping is enabled
        if self._use_grouping and not suppress_header:
            self._print_group_header(new_context)
        
        try:
            yield new_context
        finally:
            # Print footer if grouping is enabled
            if self._use_grouping and not suppress_header:
                self._print_group_footer(new_context)
            
            # Pop from stack
            self._contexts.stack.pop()
    
    def _print_group_header(self, context: LogContext):
        """Print group header."""
        amber_glow = "\033[38;5;214m"
        cyan_accent = "\033[38;5;51m"
        gray_dim = "\033[38;5;240m"
        reset = "\033[0m"
        
        indent = "  " * context.indent_level
        
        # Build header
        header = f"\n{indent}{gray_dim}┌─{reset} {amber_glow}[{context.component}]{reset}"
        if context.execution_id:
            header += f" {gray_dim}exec:{cyan_accent}{context.execution_id[:8]}{reset}"
        header += f" {gray_dim}@ {cyan_accent}{datetime.now().strftime('%H:%M:%S.%f')[:-3]}{reset}"
        
        print(header)
    
    def _print_group_footer(self, context: LogContext):
        """Print group footer with timing."""
        gray_dim = "\033[38;5;240m"
        cyan_accent = "\033[38;5;51m"
        reset = "\033[0m"
        
        indent = "  " * context.indent_level
        duration = datetime.now().timestamp() - context.start_time
        
        footer = f"{indent}{gray_dim}└─ completed in {cyan_accent}{duration:.3f}s{reset}\n"
        print(footer)
        
    def _format_message(
        self, 
        level: LogLevel, 
        message: str, 
        data: Optional[Dict[str, Any]] = None,
        execution_id: Optional[str] = None
    ) -> str:
        """Format log message with IO.net cyberpunk styling."""
        
        if self.structured:
            # JSON structured logging for machine parsing
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "level": level.value[1],
                "component": self.component,
                "message": message,
                "execution_id": execution_id,
                "data": data or {}
            }
            return json.dumps(log_entry)
        
        # Check if we're in grouped mode and have a context
        context = self.current_context if self._use_grouping else None
        indent = ("  " * context.indent_level) if context else ""
        
        # IO.net cyberpunk styling with glowing amber text
        emoji, level_name, color = level.value
        reset = "\033[0m"
        amber_glow = "\033[38;5;214m"  # Glowing amber for text
        amber_dim = "\033[38;5;172m"   # Dimmer amber for secondary
        cyan_accent = "\033[38;5;51m"  # Electric cyan accents
        gray_dim = "\033[38;5;240m"    # Dark gray for structure
        
        if self._use_grouping and context:
            # Grouped format - simplified without redundant headers
            base_msg = f"{indent}{gray_dim}│{reset} {color}{emoji}{reset} {amber_glow}{message}{reset}"
        else:
            # Original format for backward compatibility
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            
            # Execution context with cyberpunk separators
            context_parts = [self.component]
            if execution_id:
                context_parts.append(f"exec:{execution_id[:8]}")
            context_str = "▸".join(context_parts)
            
            # Main log line with cyberpunk structure
            base_msg = f"{gray_dim}┌─[{reset}{cyan_accent}{timestamp}{reset}{gray_dim}]─{reset} {color}{emoji} {level_name:7}{reset} {gray_dim}[{reset}{amber_glow}{context_str}{reset}{gray_dim}]{reset}\n"
            base_msg += f"{gray_dim}│{reset} {amber_glow}{message}{reset}"
        
        # Add structured data with cyberpunk tree structure
        if data:
            data_lines = []
            items = list(data.items())
            for i, (key, value) in enumerate(items):
                is_last = i == len(items) - 1
                branch = "└─" if is_last else "├─"
                
                if isinstance(value, dict):
                    # Nested object with cyberpunk formatting
                    data_lines.append(f"{indent}{gray_dim}│   {branch} {reset}{amber_dim}{key}:{reset}")
                    for j, (subkey, subval) in enumerate(value.items()):
                        is_sub_last = j == len(value) - 1
                        sub_prefix = "    " if is_last else "│   "
                        sub_branch = "└─" if is_sub_last else "├─"
                        data_lines.append(f"{indent}{gray_dim}│   {sub_prefix}{sub_branch} {reset}{amber_glow}{subkey}: {cyan_accent}{subval}{reset}")
                elif isinstance(value, list):
                    if len(value) <= 3:
                        # Show short lists inline
                        data_lines.append(f"{indent}{gray_dim}│   {branch} {reset}{amber_dim}{key}: {cyan_accent}{value}{reset}")
                    else:
                        # Show long lists with count
                        data_lines.append(f"{indent}{gray_dim}│   {branch} {reset}{amber_dim}{key}: {cyan_accent}[{len(value)} items]{reset}")
                else:
                    # Simple key-value with glowing amber
                    data_lines.append(f"{indent}{gray_dim}│   {branch} {reset}{amber_dim}{key}: {amber_glow}{value}{reset}")
            
            if data_lines:
                base_msg += "\n" + "\n".join(data_lines)
        
        return base_msg
    
    def debug(self, message: str, data: Optional[Dict[str, Any]] = None, execution_id: Optional[str] = None):
        """Log debug information."""
        print(self._format_message(LogLevel.DEBUG, message, data, execution_id))
    
    def info(self, message: str, data: Optional[Dict[str, Any]] = None, execution_id: Optional[str] = None):
        """Log general information."""
        print(self._format_message(LogLevel.INFO, message, data, execution_id))
    
    def success(self, message: str, data: Optional[Dict[str, Any]] = None, execution_id: Optional[str] = None):
        """Log success events."""
        print(self._format_message(LogLevel.SUCCESS, message, data, execution_id))
    
    def warning(self, message: str, data: Optional[Dict[str, Any]] = None, execution_id: Optional[str] = None):
        """Log warnings."""
        print(self._format_message(LogLevel.WARNING, message, data, execution_id))
    
    def error(self, message: str, data: Optional[Dict[str, Any]] = None, execution_id: Optional[str] = None):
        """Log errors."""
        print(self._format_message(LogLevel.ERROR, message, data, execution_id))
    
    def critical(self, message: str, data: Optional[Dict[str, Any]] = None, execution_id: Optional[str] = None):
        """Log critical errors."""
        print(self._format_message(LogLevel.CRITICAL, message, data, execution_id))
    
    def execution_plan(self, title: str, batches: List[List[str]], parallelism: int = 1):
        """Log an execution plan with visual structure."""
        amber_glow = "\033[38;5;214m"
        amber_dim = "\033[38;5;172m"
        cyan_accent = "\033[38;5;51m"
        gray_dim = "\033[38;5;240m"
        reset = "\033[0m"
        
        indent = ("  " * self.current_context.indent_level) if self._use_grouping and self.current_context else ""
        
        # Title line
        print(f"{indent}{gray_dim}│{reset} {amber_glow}🎯 {title}: {cyan_accent}{len(batches)} batches{reset}, max parallelism: {cyan_accent}{parallelism}{reset}")
        
        # Batch details
        for i, batch in enumerate(batches):
            is_last = i == len(batches) - 1
            branch = "└─" if is_last else "├─"
            batch_type = "parallel" if len(batch) > 1 and parallelism > 1 else "sequential"
            
            print(f"{indent}{gray_dim}│   {branch} 📦 Batch {i}: {cyan_accent}{batch}{reset} ({amber_dim}{batch_type}{reset})")
    
    def execution_report(
        self, 
        title: str, 
        report_data: Dict[str, Any], 
        execution_id: Optional[str] = None
    ):
        """Log execution reports with cyberpunk IO.net styling."""
        
        # IO.net cyberpunk colors
        amber_glow = "\033[38;5;214m"
        cyan_accent = "\033[38;5;51m" 
        gray_dim = "\033[38;5;240m"
        green_accent = "\033[38;5;76m"
        reset = "\033[0m"
        
        lines = []
        
        # Cyberpunk header with angular design
        header_line = "▼" * (len(title) + 30)
        lines.append(f"{cyan_accent}{header_line}{reset}")
        lines.append(f"{gray_dim}◢{reset} {amber_glow}EXECUTION ANALYSIS REPORT{reset} {gray_dim}◣{reset}")
        lines.append(f"{gray_dim}◥{reset} {amber_glow}{title}{reset} {gray_dim}◤{reset}")
        lines.append(f"{cyan_accent}{header_line}{reset}")
        lines.append("")
        
        for key, value in report_data.items():
            if key == "workflow_spec" and hasattr(value, 'to_llm_prompt'):
                # Special cyberpunk formatting for workflow specs
                lines.append(f"{gray_dim}┌─{reset} {cyan_accent}◆ WORKFLOW TOPOLOGY{reset} {gray_dim}─────────────────────────────────────{reset}")
                spec_lines = value.to_llm_prompt().split('\n')
                for spec_line in spec_lines:
                    if spec_line.strip():
                        lines.append(f"{gray_dim}│{reset} {amber_glow}{spec_line}{reset}")
                lines.append(f"{gray_dim}└──────────────────────────────────────────────────────────────{reset}")
                lines.append("")
                
            elif key == "execution_summary":
                lines.append(f"{gray_dim}┌─{reset} {green_accent}◉ EXECUTION ANALYSIS{reset} {gray_dim}──────────────────────────────{reset}")
                lines.append(f"{gray_dim}│{reset} {amber_glow}{str(value)}{reset}")
                lines.append(f"{gray_dim}└──────────────────────────────────────────────────────────────{reset}")
                lines.append("")
                
            elif key == "feedback_prompt":
                lines.append(f"{gray_dim}┌─{reset} {cyan_accent}◈ FEEDBACK PROMPT ANALYSIS{reset} {gray_dim}───────────────────────{reset}")
                lines.append(f"{gray_dim}│ ├─{reset} {amber_glow}Length: {len(str(value))} characters{reset}")
                
                # Check for key sections with cyberpunk indicators
                if "WORKFLOW SPECIFICATION" in str(value):
                    lines.append(f"{gray_dim}│ ├─{reset} {green_accent}◉ Contains workflow specification{reset}")
                if "EXECUTION RESULTS" in str(value):
                    lines.append(f"{gray_dim}│ ├─{reset} {green_accent}◉ Contains execution results{reset}")
                if "EXPECTED EXECUTION PATTERNS" in str(value):
                    lines.append(f"{gray_dim}│ └─{reset} {green_accent}◉ Contains expected patterns analysis{reset}")
                else:
                    lines.append(f"{gray_dim}│ └─{reset} {amber_glow}Analysis complete{reset}")
                    
                lines.append(f"{gray_dim}└──────────────────────────────────────────────────────────────{reset}")
                lines.append("")
                
            else:
                # Standard key-value with cyberpunk styling
                lines.append(f"{gray_dim}▸{reset} {amber_glow}{key.upper().replace('_', ' ')}: {cyan_accent}{value}{reset}")
        
        report_text = "\n".join(lines)
        self.info("Generated execution analysis report", execution_id=execution_id)
        print(report_text)


# Global logger instances for different components
workflow_logger = IOLogger("WORKFLOW")
execution_logger = IOLogger("EXECUTION") 
agent_logger = IOLogger("AGENT")
system_logger = IOLogger("SYSTEM")

# Structured logger for machine parsing
structured_logger = IOLogger("STRUCTURED", structured=True)


# ===== PROMPT LOGGING SYSTEM =====
# Global prompt history for debugging and analysis
trace_history: list[Dict[str, Any]] = []

def log_trace(
    prompt_type: str, 
    prompt: str, 
    response: Optional[str] = None, 
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Log a trace for debugging purposes.
    
    Args:
        prompt_type: Type of prompt (workflow_generation, agent_instruction, etc.)
        prompt: The actual prompt sent to the LLM
        response: The response from the LLM (optional)
        metadata: Additional metadata about the prompt
        
    Returns:
        Unique ID for the logged prompt
    """
    import uuid
    
    prompt_entry = {
        "timestamp": datetime.now().isoformat(),
        "type": prompt_type,
        "prompt": prompt,
        "response": response,
        "metadata": metadata or {},
        "id": str(uuid.uuid4())
    }
    
    trace_history.append(prompt_entry)
    
    # Log to console with cyberpunk styling
    agent_logger.info(f"🤖 Logged {prompt_type} prompt", data={
        "prompt_id": prompt_entry["id"],
        "prompt_length": len(prompt),
        "has_response": response is not None,
        "metadata_keys": list(metadata.keys()) if metadata else []
    })
    
    return prompt_entry["id"]

def get_trace_history() -> list[Dict[str, Any]]:
    """Get all logged prompts."""
    return trace_history.copy()

def clear_trace_history() -> int:
    """Clear all logged traces and return count of cleared traces."""
    global trace_history
    count = len(trace_history)
    trace_history.clear()
    agent_logger.info(f"🧹 Cleared {count} prompts from history")
    return count

def get_trace_by_id(prompt_id: str) -> Optional[Dict[str, Any]]:
    """Get a specific prompt by its ID."""
    for prompt in trace_history:
        if prompt["id"] == prompt_id:
            return prompt.copy()
    return None


def get_component_logger(component: str, grouped: bool = False) -> IOLogger:
    """Get a logger for a specific component."""
    logger = IOLogger(component)
    if grouped:
        logger.enable_grouping()
    return logger