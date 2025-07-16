"""
UI Configuration for Text Input Tools
=====================================

This module provides enhanced UI configurations for text input tools
WITHOUT modifying the actual tool functions. Tools remain as simple
functions, but get enhanced UI components in the frontend.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class TextInputUIConfig:
    """Configuration for enhanced text input UI components."""
    tool_name: str
    input_type: str = "textarea"  # "textarea" or "text"
    height: str = "120px"
    min_height: str = "80px"
    scrollable: bool = True
    resizable: bool = True
    show_character_count: bool = True
    has_run_button: bool = False
    is_readonly: bool = False
    placeholder: str = "Type here..."
    prompt_label: str = "Enter your text:"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "tool_name": self.tool_name,
            "input_type": self.input_type,
            "height": self.height,
            "min_height": self.min_height,
            "scrollable": self.scrollable,
            "resizable": self.resizable,
            "show_character_count": self.show_character_count,
            "has_run_button": self.has_run_button,
            "is_readonly": self.is_readonly,
            "placeholder": self.placeholder,
            "prompt_label": self.prompt_label
        }

# UI configurations for different text input tools
TEXT_INPUT_UI_CONFIGS = {
    "prompt_tool": TextInputUIConfig(
        tool_name="prompt_tool",
        input_type="textarea",
        height="120px",
        has_run_button=False,  # Just displays text, no execution needed
        is_readonly=False,     # Can be edited in workflow designer
        placeholder="Enter your prompt or message...",
        prompt_label="Prompt Message",
        show_character_count=True
    ),
    
    "user_input": TextInputUIConfig(
        tool_name="user_input", 
        input_type="textarea",
        height="120px",
        has_run_button=True,   # Needs run button to execute workflow
        is_readonly=False,     # User can type
        placeholder="Enter your input...",
        prompt_label="User Input",
        show_character_count=True
    )
}

def get_text_input_ui_config(tool_name: str, node_config: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """
    Get UI configuration for a text input tool.
    
    Args:
        tool_name: Name of the tool
        node_config: Optional node configuration to override defaults
        
    Returns:
        UI configuration dict or None if not a text input tool
    """
    if tool_name not in TEXT_INPUT_UI_CONFIGS:
        return None
    
    config = TEXT_INPUT_UI_CONFIGS[tool_name]
    ui_config = config.to_dict()
    
    # Override with node-specific config if provided
    if node_config:
        ui_config["placeholder"] = node_config.get("placeholder", config.placeholder)
        ui_config["prompt_label"] = node_config.get("prompt", config.prompt_label)
        ui_config["height"] = node_config.get("height", config.height)
    
    return ui_config

def is_text_input_tool(tool_name: str) -> bool:
    """Check if a tool is a text input tool with enhanced UI."""
    return tool_name in TEXT_INPUT_UI_CONFIGS

def get_all_text_input_tools() -> Dict[str, TextInputUIConfig]:
    """Get all text input tool configurations."""
    return TEXT_INPUT_UI_CONFIGS.copy()