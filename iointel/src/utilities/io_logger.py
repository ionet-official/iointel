"""
IO.net themed logger with beautiful formatting that matches the design ethos.

This logger provides structured, emoji-rich output that's both human-readable
and machine-parseable, following the cyberpunk/IO.net aesthetic.
"""

from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum
import json


class LogLevel(Enum):
    """Log levels with corresponding emojis and IO.net cyberpunk colors."""
    DEBUG = ("▪", "DEBUG", "\033[38;5;240m")      # Dark gray
    INFO = ("◆", "INFO", "\033[38;5;51m")         # Electric cyan  
    SUCCESS = ("◉", "SUCCESS", "\033[38;5;76m")  # Electric green
    WARNING = ("◈", "WARN", "\033[38;5;214m")    # Electric orange
    ERROR = ("◇", "ERROR", "\033[38;5;196m")     # Electric red
    CRITICAL = ("◎", "CRIT", "\033[38;5;201m")   # Electric magenta


class IOLogger:
    """
    IO.net themed logger with structured, beautiful output.
    
    Features:
    - Emoji indicators for different log levels
    - Structured data support for complex objects
    - Component-based logging (workflow, execution, agent, etc.)
    - Timeline formatting for execution tracking
    - Optional JSON structured logging for machine parsing
    """
    
    def __init__(self, component: str = "system", structured: bool = False):
        self.component = component.upper()
        self.structured = structured
        
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
        
        # IO.net cyberpunk styling with glowing amber text
        emoji, level_name, color = level.value
        reset = "\033[0m"
        amber_glow = "\033[38;5;214m"  # Glowing amber for text
        amber_dim = "\033[38;5;172m"   # Dimmer amber for secondary
        cyan_accent = "\033[38;5;51m"  # Electric cyan accents
        gray_dim = "\033[38;5;240m"    # Dark gray for structure
        
        # Timestamp with cyberpunk angular brackets
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        # Execution context with cyberpunk separators
        context_parts = [self.component]
        if execution_id:
            context_parts.append(f"exec:{execution_id[:8]}")
        context = "▸".join(context_parts)
        
        # Main log line with cyberpunk structure
        base_msg = f"{gray_dim}┌─[{reset}{cyan_accent}{timestamp}{reset}{gray_dim}]─{reset} {color}{emoji} {level_name:7}{reset} {gray_dim}[{reset}{amber_glow}{context}{reset}{gray_dim}]{reset}\n"
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
                    data_lines.append(f"{gray_dim}│ {branch} {reset}{amber_dim}{key}:{reset}")
                    for j, (subkey, subval) in enumerate(value.items()):
                        is_sub_last = j == len(value) - 1
                        sub_prefix = "   " if is_last else "│  "
                        sub_branch = "└─" if is_sub_last else "├─"
                        data_lines.append(f"{gray_dim}│ {sub_prefix} {sub_branch} {reset}{amber_glow}{subkey}: {cyan_accent}{subval}{reset}")
                elif isinstance(value, list):
                    data_lines.append(f"{gray_dim}│ {branch} {reset}{amber_dim}{key}: {cyan_accent}[{len(value)} items]{reset}")
                else:
                    # Simple key-value with glowing amber
                    data_lines.append(f"{gray_dim}│ {branch} {reset}{amber_dim}{key}: {amber_glow}{value}{reset}")
            
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
prompt_history: list[Dict[str, Any]] = []

def log_prompt(
    prompt_type: str, 
    prompt: str, 
    response: Optional[str] = None, 
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Log a prompt for debugging purposes.
    
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
    
    prompt_history.append(prompt_entry)
    
    # Log to console with cyberpunk styling
    agent_logger.info(f"🤖 Logged {prompt_type} prompt", data={
        "prompt_id": prompt_entry["id"],
        "prompt_length": len(prompt),
        "has_response": response is not None,
        "metadata_keys": list(metadata.keys()) if metadata else []
    })
    
    return prompt_entry["id"]

def get_prompt_history() -> list[Dict[str, Any]]:
    """Get all logged prompts."""
    return prompt_history.copy()

def clear_prompt_history() -> int:
    """Clear all logged prompts and return count of cleared prompts."""
    global prompt_history
    count = len(prompt_history)
    prompt_history.clear()
    agent_logger.info(f"🧹 Cleared {count} prompts from history")
    return count

def get_prompt_by_id(prompt_id: str) -> Optional[Dict[str, Any]]:
    """Get a specific prompt by its ID."""
    for prompt in prompt_history:
        if prompt["id"] == prompt_id:
            return prompt.copy()
    return None


def get_component_logger(component: str) -> IOLogger:
    """Get a logger for a specific component."""
    return IOLogger(component)