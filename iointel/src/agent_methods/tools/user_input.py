"""
User Input Tool - Simple interface for collecting user inputs in workflows.
"""

from typing import Dict, Any, Optional
from iointel.src.utilities.decorators import register_tool


@register_tool
def user_input(
    prompt: str,
    input_type: str = "text",
    placeholder: Optional[str] = None,
    options: Optional[list] = None,
    default_value: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Collect user input with a simple interface.
    
    This tool creates an interactive input field that pauses workflow execution
    until the user provides input. The input is then available for subsequent
    workflow steps.
    
    Args:
        prompt: The question or prompt to show the user
        input_type: Type of input - "text", "number", "select", "textarea" 
        placeholder: Placeholder text for the input field
        options: List of options for select type inputs
        default_value: Default value to pre-populate
        
    Returns:
        Dictionary with form definition for UI rendering, or the actual user input value
        
    Example:
        user_input(
            prompt="What city do you want weather for?",
            placeholder="Enter city name..."
        )
    """
    print(f"📝 Creating user input field: {prompt}")
    
    # Check if we have user inputs from execution metadata
    execution_metadata = kwargs.get('execution_metadata', {})
    user_inputs = execution_metadata.get('user_inputs', {})
    
    # Also get the task_id/node_id from execution_metadata if available
    node_id = execution_metadata.get('node_id') or execution_metadata.get('task_id')
    
    # Generate form_id to match against
    form_id = prompt.lower().replace(" ", "_").replace("?", "")[:30]
    
    print(f"🔍 Checking for user inputs. Available keys: {list(user_inputs.keys()) if user_inputs else 'None'}")
    print(f"🔍 Looking for form_id: '{form_id}'")
    print(f"🔍 Looking for node_id: '{node_id}'")
    print(f"🔍 Original prompt: '{prompt}'")
    print(f"🔍 DEBUG: user_inputs contents: {user_inputs}")
    print(f"🔍 DEBUG: execution_metadata keys: {list(execution_metadata.keys())}")
    
    if user_inputs:
        # Try multiple strategies to find the user input value
        user_value = None
        
        # Strategy 1: Look for generic 'user_input' key
        if 'user_input' in user_inputs:
            user_value = user_inputs['user_input']
            print(f"✅ Found user input with generic key: {user_value}")
        
        # Strategy 2: Look for form_id match
        elif form_id in user_inputs:
            user_value = user_inputs[form_id]
            print(f"✅ Found user input with form_id '{form_id}': {user_value}")
        
        # Strategy 3: Look for node_id match
        elif node_id and node_id in user_inputs:
            user_value = user_inputs[node_id]
            print(f"✅ Found user input with node_id '{node_id}': {user_value}")
        
        # Strategy 4: Look for any key containing 'user_input'
        else:
            for key, value in user_inputs.items():
                if 'user_input' in key.lower():
                    user_value = value
                    print(f"✅ Found user input with key '{key}': {user_value}")
                    break
        
        # Strategy 5: If only one input, use it
        if not user_value and len(user_inputs) == 1:
            key, user_value = next(iter(user_inputs.items()))
            print(f"✅ Found single user input with key '{key}': {user_value}")
        
        if user_value is not None:
            return {
                "tool_type": "user_input",
                "status": "completed",
                "user_input": user_value,
                "message": f"User provided: {user_value}"
            }
    
    print(f"⚠️ No user input value found in execution metadata")
    
    # Form ID already generated above
    
    return {
        "tool_type": "user_input",
        "form_id": form_id,
        "prompt": prompt,
        "input_type": input_type,
        "placeholder": placeholder or f"Enter {input_type}...",
        "options": options,
        "default_value": default_value,
        "ui_action": "show_input_form",
        "status": "awaiting_input",
        "message": f"Waiting for user input: {prompt}"
    }


@register_tool
def prompt_tool(message: str, **kwargs) -> Dict[str, Any]:
    """
    A simple tool that passes a prompt message to the next step in the workflow.
    
    This tool doesn't require user interaction - it just formats and passes
    the message along to subsequent workflow steps. Useful for injecting
    prompts, instructions, or context into workflow execution.
    
    Args:
        message: The message/prompt to pass to the next step
        
    Returns:
        Dictionary with the formatted message
        
    Example:
        prompt_tool(message="Please analyze the following data for trends")
    """
    print(f"📝 Prompt tool executing with message: {message}")
    
    return {
        "tool_type": "prompt",
        "status": "completed",
        "message": message,
        "prompt": message,
        "output": message
    }