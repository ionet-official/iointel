"""
Prompt Collections Data Models
=============================

Pydantic models for managing collections of prompts and inputs that can be 
saved, loaded, and searched across the workflow system.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import uuid
import json
from pathlib import Path


class ListRecords(BaseModel):
    """
    A collection of prompt/input records that can be saved, loaded, and searched.
    
    This model holds a list of strings representing prompts, test queries, or 
    user inputs that can be reused across different workflow executions.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for the collection")
    name: str = Field(..., description="Human-readable name for the collection")
    description: Optional[str] = Field(None, description="Description of what this collection contains")
    
    # Core data
    records: List[str] = Field(default_factory=list, description="List of prompt/input strings")
    
    # Metadata
    tags: List[str] = Field(default_factory=list, description="Tags for categorization and search")
    created_at: datetime = Field(default_factory=datetime.now, description="When the collection was created")
    updated_at: datetime = Field(default_factory=datetime.now, description="When the collection was last updated")
    
    # Usage tracking
    usage_count: int = Field(default=0, description="How many times records from this collection have been used")
    last_used: Optional[datetime] = Field(None, description="When a record from this collection was last used")
    
    # Tool integration
    compatible_tools: List[str] = Field(default_factory=lambda: ["prompt_tool", "user_input"], 
                                       description="Tools that can use this collection")
    
    def add_record(self, record: str) -> None:
        """Add a new record to the collection."""
        if record not in self.records:
            self.records.append(record)
            self.updated_at = datetime.now()
    
    def remove_record(self, record: str) -> bool:
        """Remove a record from the collection. Returns True if removed."""
        if record in self.records:
            self.records.remove(record)
            self.updated_at = datetime.now()
            return True
        return False
    
    def search_records(self, query: str, case_sensitive: bool = False) -> List[str]:
        """Search for records containing the query string."""
        if not case_sensitive:
            query = query.lower()
            return [record for record in self.records 
                   if query in record.lower()]
        return [record for record in self.records if query in record]
    
    def mark_used(self) -> None:
        """Mark this collection as used (updates usage stats)."""
        self.usage_count += 1
        self.last_used = datetime.now()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about this collection."""
        return {
            "total_records": len(self.records),
            "usage_count": self.usage_count,
            "last_used": self.last_used,
            "age_days": (datetime.now() - self.created_at).days,
            "tags": self.tags
        }


class PromptCollectionManager:
    """
    Manager class for handling prompt collections storage and retrieval.
    """
    
    def __init__(self, storage_dir: str = "saved_prompt_collections"):
        """Initialize the collection manager with a storage directory."""
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self._collections_cache: Dict[str, ListRecords] = {}
    
    def save_collection(self, collection: ListRecords) -> str:
        """Save a collection to disk and return its ID."""
        collection.updated_at = datetime.now()
        
        # Save to file
        filepath = self.storage_dir / f"{collection.id}.json"
        with open(filepath, 'w') as f:
            json.dump(collection.model_dump(), f, indent=2, default=str)
        
        # Update cache
        self._collections_cache[collection.id] = collection
        
        return collection.id
    
    def load_collection(self, collection_id: str) -> Optional[ListRecords]:
        """Load a collection by ID."""
        # Check cache first
        if collection_id in self._collections_cache:
            return self._collections_cache[collection_id]
        
        # Load from file
        filepath = self.storage_dir / f"{collection_id}.json"
        if not filepath.exists():
            return None
        
        try:
            with open(filepath) as f:
                data = json.load(f)
            
            collection = ListRecords(**data)
            self._collections_cache[collection_id] = collection
            return collection
        except Exception as e:
            print(f"Error loading collection {collection_id}: {e}")
            return None
    
    def list_collections(self) -> List[ListRecords]:
        """List all available collections."""
        collections = []
        
        for filepath in self.storage_dir.glob("*.json"):
            collection_id = filepath.stem
            collection = self.load_collection(collection_id)
            if collection:
                collections.append(collection)
        
        return sorted(collections, key=lambda x: x.updated_at, reverse=True)
    
    def search_all_collections(self, query: str, tool_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search across all collections for records matching the query.
        
        Returns a list of matches with collection metadata.
        """
        matches = []
        
        for collection in self.list_collections():
            # Filter by tool compatibility if specified
            if tool_filter and tool_filter not in collection.compatible_tools:
                continue
            
            # Search within this collection
            matching_records = collection.search_records(query)
            
            if matching_records:
                matches.append({
                    "collection_id": collection.id,
                    "collection_name": collection.name,
                    "matching_records": matching_records,
                    "total_records": len(collection.records),
                    "tags": collection.tags,
                    "last_used": collection.last_used
                })
        
        return matches
    
    def get_popular_records(self, tool_filter: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the most popular records across all collections."""
        all_records = []
        
        for collection in self.list_collections():
            if tool_filter and tool_filter not in collection.compatible_tools:
                continue
            
            for record in collection.records:
                all_records.append({
                    "record": record,
                    "collection_name": collection.name,
                    "collection_id": collection.id,
                    "usage_count": collection.usage_count,
                    "last_used": collection.last_used
                })
        
        # Sort by usage count and last used
        all_records.sort(key=lambda x: (x["usage_count"], x["last_used"] or datetime.min), reverse=True)
        
        return all_records[:limit]
    
    def delete_collection(self, collection_id: str) -> bool:
        """Delete a collection."""
        filepath = self.storage_dir / f"{collection_id}.json"
        
        if filepath.exists():
            filepath.unlink()
            if collection_id in self._collections_cache:
                del self._collections_cache[collection_id]
            return True
        
        return False
    
    def create_collection_from_records(self, name: str, records: List[str], 
                                     description: Optional[str] = None,
                                     tags: Optional[List[str]] = None) -> ListRecords:
        """Create a new collection from a list of records."""
        collection = ListRecords(
            name=name,
            description=description,
            records=records,
            tags=tags or []
        )
        
        self.save_collection(collection)
        return collection


# Global instance for easy access
prompt_collection_manager = PromptCollectionManager()