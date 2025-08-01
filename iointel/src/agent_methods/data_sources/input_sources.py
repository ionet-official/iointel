"""
Input Source Implementations
============================

Clean, standardized implementations of data sources using Pydantic models.
These are NOT tools - they provide input data to workflows.
"""

from typing import Dict, Any, Optional
from .models import DataSourceRequest, DataSourceResponse
from .registry import register_data_source
from ...utilities.io_logger import get_component_logger

logger = get_component_logger("DATA_SOURCES")


def resolve_user_input_value(user_inputs: Dict[str, Any], node_id: Optional[str] = None) -> Optional[str]:
    """
    Simple, straightforward user input resolution.
    
    Args:
        user_inputs: Dictionary of user inputs from execution metadata
        node_id: Node ID from workflow execution (for logging only)
        
    Returns:
        The user input value or None if not found
    """
    if not user_inputs:
        logger.debug("No runtime user inputs provided - will use config defaults")
        return None
    
    logger.info("🔍 Runtime user inputs available", data={
        "available_keys": list(user_inputs.keys()),
        "input_count": len(user_inputs)
    })
    
    # If user provided input, use it. Period.
    if len(user_inputs) == 1:
        key, value = next(iter(user_inputs.items()))
        logger.success("✅ Using user input", data={
            "key": key,
            "value_preview": str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
        })
        return value
    
    # Multiple inputs? Use the first one (this shouldn't happen in normal usage)
    if len(user_inputs) > 1:
        key, value = next(iter(user_inputs.items()))
        logger.warning("Multiple user inputs found, using first", data={
            "selected_key": key,
            "total_inputs": len(user_inputs)
        })
        return value
    
    return None


@register_data_source("user_input")
def user_input_source(request: DataSourceRequest, **kwargs) -> DataSourceResponse:
    """
    Interactive user input data source.
    
    This creates an interactive input field that pauses workflow execution
    until the user provides input.
    """
    logger.info("📝 User input source initiated", data={
        "message": request.message,
        "has_default": request.default_value is not None
    })
    
    # Check if we have user inputs from execution metadata
    execution_metadata = kwargs.get('execution_metadata', {})
    user_inputs = execution_metadata.get('user_inputs', {})
    node_id = execution_metadata.get('node_id') or execution_metadata.get('task_id')
    
    # Use our simple resolver function
    user_value = resolve_user_input_value(user_inputs, node_id)
    
    if user_value is not None:
        logger.success("✅ User input resolved successfully", data={
            "value_preview": user_value[:50] + "..." if len(user_value) > 50 else user_value
        })
        
        return DataSourceResponse(
            source_type="user_input",
            message=user_value,
            status="completed"
        )
    
    # No user input yet - use default_value if available, otherwise return form for UI
    if request.default_value:
        logger.info("📝 Using default value for user input", data={
            "default_preview": request.default_value[:50] + "..." if len(request.default_value) > 50 else request.default_value
        })
        
        return DataSourceResponse(
            source_type="user_input",
            message=request.default_value,
            status="completed"
        )
    
    # Generate form_id for UI matching
    form_id = request.message.lower().replace(" ", "_").replace("?", "")[:30]
    
    logger.info("📋 Returning input form for user interaction", data={
        "form_id": form_id,
        "message": request.message
    })
    
    return DataSourceResponse(
        source_type="user_input",
        message=request.message,
        status="awaiting_input",
        form_id=form_id,
        ui_action="show_input_form",
        placeholder=f"Enter response to: {request.message}",
        input_type="text",
        default_value=request.default_value
    )


@register_data_source("prompt_tool")
def prompt_source(request: DataSourceRequest, **kwargs) -> DataSourceResponse:
    """
    Static prompt/message data source.
    
    This passes a static message to the workflow without user interaction.
    Useful for injecting prompts, instructions, or context.
    """
    logger.info("📝 Prompt source executing", data={
        "message_preview": request.message[:50] + "..." if len(request.message) > 50 else request.message,
        "has_default": request.default_value is not None
    })
    
    # Use the message from the request
    message = request.message
    
    # If no message but has default_value, use that
    if not message and request.default_value:
        message = request.default_value
        logger.info("📝 Using default value as message", data={
            "default_preview": message[:50] + "..." if len(message) > 50 else message
        })
    
    # Check for user input overrides
    execution_metadata = kwargs.get('execution_metadata', {})
    user_inputs = execution_metadata.get('user_inputs', {})
    node_id = execution_metadata.get('node_id') or execution_metadata.get('task_id')
    
    if user_inputs and node_id and node_id in user_inputs:
        override_message = user_inputs[node_id]
        if override_message is not None:  # Allow empty strings
            message = override_message
            logger.success("✅ Using user input override", data={
                "node_id": node_id,
                "override_preview": message[:50] + "..." if len(message) > 50 else message
            })
    
    logger.success("✅ Prompt source completed", data={
        "message_preview": message[:50] + "..." if len(message) > 50 else message
    })
    
    return DataSourceResponse(
        source_type="prompt_tool",
        message=message,
        status="completed"
    )