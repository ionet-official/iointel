"""
User Input Tool - Simple interface for collecting user inputs in workflows.
"""

from typing import Dict, Any, Optional
from iointel.src.utilities.decorators import register_tool
from iointel.src.agent_methods.data_models.prompt_collections import (
    prompt_collection_manager
)
from iointel.src.utilities.io_logger import get_component_logger

# Beautiful IOLogger for structured output
logger = get_component_logger("USER_INPUT_TOOL")


def resolve_user_input_value(user_inputs: Dict[str, Any], node_id: Optional[str] = None) -> Optional[str]:
    """
    Simple, straightforward user input resolution.
    
    The reality is: if there's user input, just use it. The web UI sends exactly
    what the user typed. No need for complex fallback strategies.
    
    Args:
        user_inputs: Dictionary of user inputs from execution metadata
        node_id: Node ID from workflow execution (for logging only)
        
    Returns:
        The user input value or None if not found
    """
    if not user_inputs:
        logger.debug("No user inputs provided")
        return None
    
    logger.info("🔍 User input available", data={
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
        logger.warning("⚠️ Multiple inputs found, using first", data={
            "selected_key": key,
            "available_keys": list(user_inputs.keys()),
            "value_preview": str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
        })
        return value
    
    # This should never happen (covered by first check)
    logger.error("❌ No user input found", data={
        "user_inputs": user_inputs
    })
    return None


@register_tool
def user_input(
    prompt: str,
    input_type: str = "text",
    placeholder: Optional[str] = None,
    options: Optional[list] = None,
    default_value: Optional[str] = None,
    collection_id: Optional[str] = None,
    load_suggestions: bool = True,
    save_to_collection: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Collect user input with a simple interface and optional collection support.
    
    This tool creates an interactive input field that pauses workflow execution
    until the user provides input. The input is then available for subsequent
    workflow steps. Supports loading suggestions from saved collections and
    saving inputs to collections for reuse.
    
    Args:
        prompt: The question or prompt to show the user
        input_type: Type of input - "text", "number", "select", "textarea" 
        placeholder: Placeholder text for the input field
        options: List of options for select type inputs
        default_value: Default value to pre-populate
        collection_id: ID of collection to load suggestions from
        load_suggestions: Whether to load suggestions from popular collections
        save_to_collection: Name of collection to save this input to (creates if doesn't exist)
        
    Returns:
        Dictionary with form definition for UI rendering, or the actual user input value
        
    Example:
        user_input(
            prompt="What city do you want weather for?",
            placeholder="Enter city name...",
            collection_id="city_names",
            save_to_collection="weather_queries"
        )
    """
    logger.info("📝 User input tool initiated", data={
        "prompt": prompt,
        "input_type": input_type,
        "has_placeholder": placeholder is not None,
        "has_options": options is not None,
        "has_default": default_value is not None
    })
    
    # Check if we have user inputs from execution metadata
    execution_metadata = kwargs.get('execution_metadata', {})
    user_inputs = execution_metadata.get('user_inputs', {})
    
    # Debug: Log all kwargs to see what's being passed
    logger.debug("🔍 All kwargs received", data={
        "kwargs_keys": list(kwargs.keys()),
        "has_execution_metadata": 'execution_metadata' in kwargs,
        "execution_metadata_keys": list(execution_metadata.keys()) if execution_metadata else [],
        "user_inputs_available": user_inputs
    })
    
    # Get task_id/node_id from execution_metadata if available
    node_id = execution_metadata.get('node_id') or execution_metadata.get('task_id')
    
    # Generate form_id to match against
    form_id = prompt.lower().replace(" ", "_").replace("?", "")[:30]
    
    # Use our simple resolver function
    user_value = resolve_user_input_value(user_inputs, node_id)
    
    if user_value is not None:
        # Save to collection if requested
        if save_to_collection and user_value.strip():
            try:
                # Try to find existing collection by name
                existing_collection = None
                for collection in prompt_collection_manager.list_collections():
                    if collection.name == save_to_collection:
                        existing_collection = collection
                        break
                
                if existing_collection:
                    existing_collection.add_record(user_value)
                    prompt_collection_manager.save_collection(existing_collection) 
                    logger.success("💾 Saved to existing collection", data={
                        "collection_name": save_to_collection,
                        "value_preview": user_value[:50] + "..." if len(user_value) > 50 else user_value
                    })
                else:
                    # Create new collection
                    prompt_collection_manager.create_collection_from_records(
                        name=save_to_collection,
                        records=[user_value],
                        description=f"Collection for {prompt}",
                        tags=["user_input", "auto_generated"]
                    )
                    logger.success("💾 Created new collection", data={
                        "collection_name": save_to_collection,
                        "value_preview": user_value[:50] + "..." if len(user_value) > 50 else user_value
                    })
                    
            except Exception as e:
                logger.error("⚠️ Collection save failed", data={
                    "collection_name": save_to_collection,
                    "error": str(e)
                })
        
        logger.success("✅ User input resolved successfully", data={
            "value_preview": user_value[:50] + "..." if len(user_value) > 50 else user_value,
            "saved_to_collection": save_to_collection is not None
        })
        
        return {
            "tool_type": "user_input",
            "status": "completed",
            "user_input": user_value,
            "message": f"User provided: {user_value}"
        }
    
    # No user input found - need to show input form
    logger.info("🔄 No user input found, showing input form", data={
        "form_id": form_id,
        "node_id": node_id,
        "will_load_suggestions": load_suggestions or collection_id is not None
    })
    
    # Load suggestions from collections if requested
    suggestions = []
    collection_data = None
    
    if load_suggestions or collection_id:
        try:
            if collection_id:
                # Load specific collection
                collection = prompt_collection_manager.load_collection(collection_id)
                if collection:
                    suggestions = collection.records
                    collection_data = {
                        "id": collection.id,
                        "name": collection.name,
                        "description": collection.description,
                        "records": collection.records
                    }
                    collection.mark_used()  # Update usage stats
                    prompt_collection_manager.save_collection(collection)
                    logger.success("📚 Loaded specific collection suggestions", data={
                        "collection_name": collection.name,
                        "suggestion_count": len(suggestions)
                    })
            
            if not suggestions and load_suggestions:
                # Load popular suggestions across all collections
                popular_records = prompt_collection_manager.get_popular_records(
                    tool_filter="user_input", 
                    limit=10
                )
                suggestions = [record["record"] for record in popular_records]
                logger.success("📚 Loaded popular suggestions", data={
                    "suggestion_count": len(suggestions)
                })
                
        except Exception as e:
            logger.error("⚠️ Error loading suggestions", data={
                "collection_id": collection_id,
                "error": str(e)
            })
    
    logger.info("📋 Returning input form for user interaction", data={
        "form_id": form_id,
        "input_type": input_type,
        "has_suggestions": len(suggestions) > 0,
        "suggestion_count": len(suggestions)
    })
    
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
        "message": f"Waiting for user input: {prompt}",
        # Collection support
        "suggestions": suggestions,
        "collection_data": collection_data,
        "save_to_collection": save_to_collection
    }


@register_tool
def prompt_tool(
    message: str, 
    collection_id: Optional[str] = None,
    save_to_collection: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    A simple tool that passes a prompt message to the next step in the workflow.
    
    This tool doesn't require user interaction - it just formats and passes
    the message along to subsequent workflow steps. Useful for injecting
    prompts, instructions, or context into workflow execution. Can optionally
    save prompts to collections for reuse.
    
    Args:
        message: The message/prompt to pass to the next step
        collection_id: ID of collection to load the message from (overrides message)
        save_to_collection: Name of collection to save this prompt to
        
    Returns:
        Dictionary with the formatted message
        
    Example:
        prompt_tool(
            message="Please analyze the following data for trends",
            save_to_collection="analysis_prompts"
        )
    """
    logger.info("📝 Prompt tool executing", data={
        "message_preview": message[:50] + "..." if len(message) > 50 else message,
        "has_collection_id": collection_id is not None,
        "will_save_to_collection": save_to_collection is not None
    })
    
    # Check for user input overrides first (similar to user_input tool)
    execution_metadata = kwargs.get('execution_metadata', {})
    user_inputs = execution_metadata.get('user_inputs', {})
    node_id = execution_metadata.get('node_id') or execution_metadata.get('task_id')
    
    user_input_override = None
    if user_inputs and node_id:
        # Look for user input override for this node
        if node_id in user_inputs:
            override_message = user_inputs[node_id]
            if override_message is not None:  # Allow empty strings
                user_input_override = override_message
                message = override_message
                logger.success("✅ Using user input override", data={
                    "node_id": node_id,
                    "override_preview": message[:50] + "..." if len(message) > 50 else message
                })
    
    # Load from collection if specified (but only if no user input override)
    if collection_id and user_input_override is None:
        try:
            collection = prompt_collection_manager.load_collection(collection_id)
            if collection and collection.records:
                # Use the first record as the message
                message = collection.records[0]
                collection.mark_used()
                prompt_collection_manager.save_collection(collection)
                logger.success("📚 Loaded message from collection", data={
                    "collection_name": collection.name,
                    "message_preview": message[:50] + "..." if len(message) > 50 else message
                })
        except Exception as e:
            logger.error("⚠️ Error loading from collection", data={
                "collection_id": collection_id,
                "error": str(e)
            })
    
    # Save to collection if requested
    if save_to_collection and message.strip():
        try:
            # Try to find existing collection by name
            existing_collection = None
            for collection in prompt_collection_manager.list_collections():
                if collection.name == save_to_collection:
                    existing_collection = collection
                    break
            
            if existing_collection:
                existing_collection.add_record(message)
                prompt_collection_manager.save_collection(existing_collection)
                logger.success("💾 Added prompt to existing collection", data={
                    "collection_name": save_to_collection
                })
            else:
                # Create new collection
                prompt_collection_manager.create_collection_from_records(
                    name=save_to_collection,
                    records=[message],
                    description="Collection for prompts",
                    tags=["prompt_tool", "auto_generated"]
                )
                logger.success("💾 Created new collection with prompt", data={
                    "collection_name": save_to_collection
                })
                
        except Exception as e:
            logger.error("⚠️ Error saving to collection", data={
                "collection_name": save_to_collection,
                "error": str(e)
            })
    
    return {
        "tool_type": "prompt",
        "status": "completed",
        "message": message,
        "prompt": message,
        "output": message
    }