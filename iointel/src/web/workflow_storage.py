"""
Workflow Storage - Persistent storage for user-generated workflows.

This module handles saving, loading, and managing user-created workflows
that can be reused across sessions.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid
from ..agent_methods.data_models.workflow_spec import WorkflowSpec


class WorkflowStorage:
    """Manages persistent storage of user workflows."""
    
    def __init__(self, storage_dir: str = "saved_workflows"):
        """
        Initialize workflow storage.
        
        Args:
            storage_dir: Directory to store workflow files
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.yaml_dir = self.storage_dir / "yaml"
        self.json_dir = self.storage_dir / "json"
        self.metadata_dir = self.storage_dir / "metadata"
        
        for dir_path in [self.yaml_dir, self.json_dir, self.metadata_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def save_workflow(
        self, 
        workflow_spec: WorkflowSpec, 
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Save a workflow to persistent storage.
        
        Args:
            workflow_spec: The workflow specification to save
            name: Custom name for the workflow (defaults to workflow title)
            description: Custom description (defaults to workflow description)
            tags: Optional tags for categorization
            
        Returns:
            str: The unique workflow ID
        """
        # Generate unique ID if not present
        workflow_id = str(workflow_spec.id) if workflow_spec.id else str(uuid.uuid4())
        
        # Use provided name/description or fall back to workflow spec
        save_name = name or workflow_spec.title or f"Workflow_{workflow_id[:8]}"
        save_description = description or workflow_spec.description or "No description"
        save_tags = tags or []
        
        # Create metadata
        metadata = {
            "id": workflow_id,
            "name": save_name,
            "description": save_description,
            "tags": save_tags,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "source": "user_generated",
            "original_title": workflow_spec.title,
            "node_count": len(workflow_spec.nodes),
            "edge_count": len(workflow_spec.edges),
            "node_types": list(set(node.type for node in workflow_spec.nodes)),
            "complexity": self._calculate_complexity(workflow_spec)
        }
        
        # Save files
        safe_name = self._sanitize_filename(save_name)
        
        # Save YAML
        yaml_content = workflow_spec.to_yaml()
        yaml_file = self.yaml_dir / f"{safe_name}_{workflow_id[:8]}.yaml"
        yaml_file.write_text(yaml_content, encoding='utf-8')
        
        # Save JSON (WorkflowSpec format)
        json_content = workflow_spec.model_dump_json(indent=2)
        json_file = self.json_dir / f"{safe_name}_{workflow_id[:8]}.json"
        json_file.write_text(json_content, encoding='utf-8')
        
        # Save metadata
        metadata_file = self.metadata_dir / f"{safe_name}_{workflow_id[:8]}.json"
        metadata_file.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        
        print(f"💾 Saved workflow: {save_name} (ID: {workflow_id[:8]})")
        return workflow_id
    
    def load_workflow(self, workflow_id: str) -> Optional[WorkflowSpec]:
        """
        Load a workflow by ID.
        
        Args:
            workflow_id: The workflow ID to load
            
        Returns:
            WorkflowSpec or None if not found
        """
        # Find the workflow file
        for json_file in self.json_dir.glob(f"*_{workflow_id[:8]}.json"):
            try:
                content = json_file.read_text(encoding='utf-8')
                workflow_data = json.loads(content)
                return WorkflowSpec.model_validate(workflow_data)
            except Exception as e:
                print(f"❌ Error loading workflow {workflow_id}: {e}")
                continue
        
        return None
    
    def list_workflows(self, tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        List all saved workflows with metadata.
        
        Args:
            tags: Optional tags to filter by
            
        Returns:
            List of workflow metadata dictionaries
        """
        workflows = []
        
        for metadata_file in self.metadata_dir.glob("*.json"):
            try:
                content = metadata_file.read_text(encoding='utf-8')
                metadata = json.loads(content)
                
                # Filter by tags if provided
                if tags:
                    workflow_tags = set(metadata.get("tags", []))
                    if not any(tag in workflow_tags for tag in tags):
                        continue
                
                workflows.append(metadata)
            except Exception as e:
                print(f"❌ Error reading metadata {metadata_file}: {e}")
                continue
        
        # Sort by updated_at descending (most recent first)
        workflows.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return workflows
    
    def search_workflows(self, query: str) -> List[Dict[str, Any]]:
        """
        Search workflows by name, description, or tags.
        
        Args:
            query: Search query string
            
        Returns:
            List of matching workflow metadata
        """
        query_lower = query.lower()
        all_workflows = self.list_workflows()
        
        matching_workflows = []
        for workflow in all_workflows:
            # Check name, description, and tags
            name_match = query_lower in workflow.get("name", "").lower()
            desc_match = query_lower in workflow.get("description", "").lower()
            tag_match = any(query_lower in tag.lower() for tag in workflow.get("tags", []))
            original_title_match = query_lower in workflow.get("original_title", "").lower()
            
            if name_match or desc_match or tag_match or original_title_match:
                matching_workflows.append(workflow)
        
        return matching_workflows
    
    def delete_workflow(self, workflow_id: str) -> bool:
        """
        Delete a workflow by ID.
        
        Args:
            workflow_id: The workflow ID to delete
            
        Returns:
            bool: True if deleted successfully
        """
        deleted = False
        
        # Delete all files for this workflow
        for directory in [self.yaml_dir, self.json_dir, self.metadata_dir]:
            for file_path in directory.glob(f"*_{workflow_id[:8]}.*"):
                try:
                    file_path.unlink()
                    deleted = True
                    print(f"🗑️ Deleted: {file_path}")
                except Exception as e:
                    print(f"❌ Error deleting {file_path}: {e}")
        
        return deleted
    
    def update_workflow(
        self, 
        workflow_id: str, 
        workflow_spec: WorkflowSpec,
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """
        Update an existing workflow.
        
        Args:
            workflow_id: The workflow ID to update
            workflow_spec: Updated workflow specification
            name: Updated name
            description: Updated description
            tags: Updated tags
            
        Returns:
            bool: True if updated successfully
        """
        # Load existing metadata to preserve creation time
        existing_metadata = None
        for metadata_file in self.metadata_dir.glob(f"*_{workflow_id[:8]}.json"):
            try:
                content = metadata_file.read_text(encoding='utf-8')
                existing_metadata = json.loads(content)
                break
            except Exception:
                continue
        
        if not existing_metadata:
            # If no existing metadata, treat as new save
            return self.save_workflow(workflow_spec, name, description, tags) == workflow_id
        
        # Delete old files
        self.delete_workflow(workflow_id)
        
        # Save with preserved creation time
        save_name = name or existing_metadata.get("name") or workflow_spec.title
        save_description = description or existing_metadata.get("description") or workflow_spec.description
        save_tags = tags if tags is not None else existing_metadata.get("tags", [])
        
        # Create updated metadata
        metadata = {
            "id": workflow_id,
            "name": save_name,
            "description": save_description,
            "tags": save_tags,
            "created_at": existing_metadata.get("created_at", datetime.now().isoformat()),
            "updated_at": datetime.now().isoformat(),
            "source": "user_generated",
            "original_title": workflow_spec.title,
            "node_count": len(workflow_spec.nodes),
            "edge_count": len(workflow_spec.edges),
            "node_types": list(set(node.type for node in workflow_spec.nodes)),
            "complexity": self._calculate_complexity(workflow_spec)
        }
        
        # Save files with new content
        safe_name = self._sanitize_filename(save_name)
        
        # Save YAML
        yaml_content = workflow_spec.to_yaml()
        yaml_file = self.yaml_dir / f"{safe_name}_{workflow_id[:8]}.yaml"
        yaml_file.write_text(yaml_content, encoding='utf-8')
        
        # Save JSON
        json_content = workflow_spec.model_dump_json(indent=2)
        json_file = self.json_dir / f"{safe_name}_{workflow_id[:8]}.json"
        json_file.write_text(json_content, encoding='utf-8')
        
        # Save metadata
        metadata_file = self.metadata_dir / f"{safe_name}_{workflow_id[:8]}.json"
        metadata_file.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        
        print(f"📝 Updated workflow: {save_name} (ID: {workflow_id[:8]})")
        return True
    
    def export_workflow_yaml(self, workflow_id: str) -> Optional[str]:
        """
        Get the YAML content for a workflow.
        
        Args:
            workflow_id: The workflow ID
            
        Returns:
            YAML content string or None if not found
        """
        for yaml_file in self.yaml_dir.glob(f"*_{workflow_id[:8]}.yaml"):
            try:
                return yaml_file.read_text(encoding='utf-8')
            except Exception as e:
                print(f"❌ Error reading YAML {yaml_file}: {e}")
        
        return None
    
    def get_workflow_stats(self) -> Dict[str, Any]:
        """
        Get statistics about saved workflows.
        
        Returns:
            Dictionary with workflow statistics
        """
        workflows = self.list_workflows()
        
        if not workflows:
            return {
                "total_count": 0,
                "complexity_breakdown": {},
                "node_type_usage": {},
                "tags_usage": {},
                "creation_timeline": []
            }
        
        # Calculate statistics
        complexity_counts = {}
        node_type_counts = {}
        tag_counts = {}
        
        for workflow in workflows:
            # Complexity breakdown
            complexity = workflow.get("complexity", "Unknown")
            complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
            
            # Node type usage
            for node_type in workflow.get("node_types", []):
                node_type_counts[node_type] = node_type_counts.get(node_type, 0) + 1
            
            # Tag usage
            for tag in workflow.get("tags", []):
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        return {
            "total_count": len(workflows),
            "complexity_breakdown": complexity_counts,
            "node_type_usage": node_type_counts,
            "tags_usage": tag_counts,
            "creation_timeline": [w.get("created_at") for w in workflows[-10:]]  # Last 10
        }
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for filesystem safety."""
        # Remove or replace invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        
        # Limit length and strip whitespace
        return filename.strip()[:50]
    
    def _calculate_complexity(self, workflow_spec: WorkflowSpec) -> str:
        """Calculate workflow complexity."""
        node_count = len(workflow_spec.nodes)
        edge_count = len(workflow_spec.edges)
        
        if node_count == 1:
            return "Simple"
        elif node_count <= 3 and edge_count <= 3:
            return "Basic"
        elif node_count <= 6 and edge_count <= 8:
            return "Intermediate"
        else:
            return "Advanced"