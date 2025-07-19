"""
User Input Tool - Simple interface for collecting user inputs in workflows.
"""

from typing import Dict, Any, Optional, List
from iointel.src.utilities.decorators import register_tool
from iointel.src.agent_methods.data_models.prompt_collections import (
    prompt_collection_manager, ListRecords
)


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
                        print(f"💾 Added '{user_value}' to existing collection '{save_to_collection}'")
                    else:
                        # Create new collection
                        new_collection = prompt_collection_manager.create_collection_from_records(
                            name=save_to_collection,
                            records=[user_value],
                            description=f"Collection for {prompt}",
                            tags=["user_input", "auto_generated"]
                        )
                        print(f"💾 Created new collection '{save_to_collection}' with record '{user_value}'")
                        
                except Exception as e:
                    print(f"⚠️ Error saving to collection: {e}")
            
            return {
                "tool_type": "user_input",
                "status": "completed",
                "user_input": user_value,
                "message": f"User provided: {user_value}"
            }
    
    print(f"⚠️ No user input value found in execution metadata")
    
    # Form ID already generated above
    
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
                    print(f"📚 Loaded {len(suggestions)} suggestions from collection '{collection.name}'")
            
            if not suggestions and load_suggestions:
                # Load popular suggestions across all collections
                popular_records = prompt_collection_manager.get_popular_records(
                    tool_filter="user_input", 
                    limit=10
                )
                suggestions = [record["record"] for record in popular_records]
                print(f"📚 Loaded {len(suggestions)} popular suggestions")
                
        except Exception as e:
            print(f"⚠️ Error loading suggestions: {e}")
    
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
    print(f"📝 Prompt tool executing with message: {message}")
    
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
                print(f"✅ Using user input override: {message}")
    
    # Load from collection if specified (but only if no user input override)
    if collection_id and user_input_override is None:
        try:
            collection = prompt_collection_manager.load_collection(collection_id)
            if collection and collection.records:
                # Use the first record as the message
                message = collection.records[0]
                collection.mark_used()
                prompt_collection_manager.save_collection(collection)
                print(f"📚 Loaded message from collection '{collection.name}': {message}")
        except Exception as e:
            print(f"⚠️ Error loading from collection: {e}")
    
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
                print(f"💾 Added prompt to existing collection '{save_to_collection}'")
            else:
                # Create new collection
                new_collection = prompt_collection_manager.create_collection_from_records(
                    name=save_to_collection,
                    records=[message],
                    description=f"Collection for prompts",
                    tags=["prompt_tool", "auto_generated"]
                )
                print(f"💾 Created new collection '{save_to_collection}' with prompt")
                
        except Exception as e:
            print(f"⚠️ Error saving to collection: {e}")
    
    return {
        "tool_type": "prompt",
        "status": "completed",
        "message": message,
        "prompt": message,
        "output": message
    }