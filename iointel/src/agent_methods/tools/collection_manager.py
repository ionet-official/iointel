"""
Collection Manager Tool - API for managing prompt collections
"""

from typing import Dict, Any, Optional, List
from iointel.src.utilities.decorators import register_tool
from iointel.src.agent_methods.data_models.prompt_collections import (
    prompt_collection_manager, ListRecords
)


@register_tool
def create_collection(
    name: str,
    records: List[str],
    description: Optional[str] = None,
    tags: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Create a new prompt collection.
    
    Args:
        name: Name for the collection
        records: List of prompts/inputs to add to the collection
        description: Optional description
        tags: Optional tags for categorization
        
    Returns:
        Dictionary with collection details
    """
    try:
        collection = prompt_collection_manager.create_collection_from_records(
            name=name,
            records=records,
            description=description,
            tags=tags or []
        )
        
        return {
            "tool_type": "collection_manager",
            "action": "create",
            "status": "success",
            "collection_id": collection.id,
            "collection_name": collection.name,
            "records_count": len(collection.records),
            "message": f"Created collection '{name}' with {len(records)} records"
        }
    except Exception as e:
        return {
            "tool_type": "collection_manager",
            "action": "create",
            "status": "error",
            "message": f"Error creating collection: {str(e)}"
        }


@register_tool
def list_collections(
    tool_filter: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    List all available prompt collections.
    
    Args:
        tool_filter: Optional filter by tool compatibility
        
    Returns:
        Dictionary with collections list
    """
    try:
        collections = prompt_collection_manager.list_collections()
        
        if tool_filter:
            collections = [c for c in collections if tool_filter in c.compatible_tools]
        
        collection_summaries = []
        for collection in collections:
            collection_summaries.append({
                "id": collection.id,
                "name": collection.name,
                "description": collection.description,
                "records_count": len(collection.records),
                "tags": collection.tags,
                "usage_count": collection.usage_count,
                "last_used": collection.last_used.isoformat() if collection.last_used else None,
                "created_at": collection.created_at.isoformat(),
                "updated_at": collection.updated_at.isoformat()
            })
        
        return {
            "tool_type": "collection_manager",
            "action": "list",
            "status": "success",
            "collections": collection_summaries,
            "total_count": len(collection_summaries),
            "message": f"Found {len(collection_summaries)} collections"
        }
    except Exception as e:
        return {
            "tool_type": "collection_manager",
            "action": "list",
            "status": "error",
            "message": f"Error listing collections: {str(e)}"
        }


@register_tool
def get_collection(
    collection_id: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Get details of a specific collection.
    
    Args:
        collection_id: ID of the collection to retrieve
        
    Returns:
        Dictionary with collection details
    """
    try:
        collection = prompt_collection_manager.load_collection(collection_id)
        
        if not collection:
            return {
                "tool_type": "collection_manager",
                "action": "get",
                "status": "error",
                "message": f"Collection '{collection_id}' not found"
            }
        
        return {
            "tool_type": "collection_manager",
            "action": "get",
            "status": "success",
            "collection": {
                "id": collection.id,
                "name": collection.name,
                "description": collection.description,
                "records": collection.records,
                "tags": collection.tags,
                "usage_count": collection.usage_count,
                "last_used": collection.last_used.isoformat() if collection.last_used else None,
                "created_at": collection.created_at.isoformat(),
                "updated_at": collection.updated_at.isoformat(),
                "compatible_tools": collection.compatible_tools,
                "stats": collection.get_stats()
            },
            "message": f"Retrieved collection '{collection.name}'"
        }
    except Exception as e:
        return {
            "tool_type": "collection_manager",
            "action": "get",
            "status": "error",
            "message": f"Error retrieving collection: {str(e)}"
        }


@register_tool
def search_collections(
    query: str,
    tool_filter: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Search across all collections for matching records.
    
    Args:
        query: Search query
        tool_filter: Optional filter by tool compatibility
        
    Returns:
        Dictionary with search results
    """
    try:
        results = prompt_collection_manager.search_all_collections(query, tool_filter)
        
        return {
            "tool_type": "collection_manager",
            "action": "search",
            "status": "success",
            "query": query,
            "results": results,
            "total_matches": len(results),
            "message": f"Found {len(results)} collections with matches for '{query}'"
        }
    except Exception as e:
        return {
            "tool_type": "collection_manager",
            "action": "search",
            "status": "error",
            "message": f"Error searching collections: {str(e)}"
        }


@register_tool
def add_to_collection(
    collection_id: str,
    record: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Add a record to an existing collection.
    
    Args:
        collection_id: ID of the collection
        record: Record to add
        
    Returns:
        Dictionary with operation result
    """
    try:
        collection = prompt_collection_manager.load_collection(collection_id)
        
        if not collection:
            return {
                "tool_type": "collection_manager",
                "action": "add_record",
                "status": "error",
                "message": f"Collection '{collection_id}' not found"
            }
        
        collection.add_record(record)
        prompt_collection_manager.save_collection(collection)
        
        return {
            "tool_type": "collection_manager",
            "action": "add_record",
            "status": "success",
            "collection_id": collection_id,
            "record": record,
            "total_records": len(collection.records),
            "message": f"Added record to collection '{collection.name}'"
        }
    except Exception as e:
        return {
            "tool_type": "collection_manager",
            "action": "add_record",
            "status": "error",
            "message": f"Error adding record: {str(e)}"
        }


@register_tool
def delete_collection(
    collection_id: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Delete a collection.
    
    Args:
        collection_id: ID of the collection to delete
        
    Returns:
        Dictionary with operation result
    """
    try:
        success = prompt_collection_manager.delete_collection(collection_id)
        
        if success:
            return {
                "tool_type": "collection_manager",
                "action": "delete",
                "status": "success",
                "collection_id": collection_id,
                "message": f"Deleted collection '{collection_id}'"
            }
        else:
            return {
                "tool_type": "collection_manager",
                "action": "delete",
                "status": "error",
                "message": f"Collection '{collection_id}' not found"
            }
    except Exception as e:
        return {
            "tool_type": "collection_manager",
            "action": "delete",
            "status": "error",
            "message": f"Error deleting collection: {str(e)}"
        }


@register_tool
def get_popular_records(
    tool_filter: Optional[str] = None,
    limit: int = 10,
    **kwargs
) -> Dict[str, Any]:
    """
    Get the most popular records across all collections.
    
    Args:
        tool_filter: Optional filter by tool compatibility
        limit: Maximum number of records to return
        
    Returns:
        Dictionary with popular records
    """
    try:
        popular_records = prompt_collection_manager.get_popular_records(tool_filter, limit)
        
        return {
            "tool_type": "collection_manager",
            "action": "get_popular",
            "status": "success",
            "records": popular_records,
            "total_count": len(popular_records),
            "message": f"Retrieved {len(popular_records)} popular records"
        }
    except Exception as e:
        return {
            "tool_type": "collection_manager",
            "action": "get_popular",
            "status": "error",
            "message": f"Error retrieving popular records: {str(e)}"
        }