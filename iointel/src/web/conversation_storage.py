"""
Conversation Storage System
===========================

Manages conversation IDs and versioning for the web interface to prevent
corrupted context loading and track conversation versions.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict
from uuid import uuid4


@dataclass
class ConversationMetadata:
    """Metadata for a conversation session."""
    
    conversation_id: str
    version: str
    created_at: str
    last_used_at: str
    session_type: str  # "web_interface", "api", "cli", etc.
    user_agent: Optional[str] = None
    total_messages: int = 0
    workflow_count: int = 0
    execution_count: int = 0
    status: str = "active"  # "active", "archived", "corrupted"
    notes: Optional[str] = None


class ConversationStorage:
    """Manages conversation sessions with versioning and metadata tracking."""
    
    def __init__(self, storage_dir: str = "conversations"):
        """Initialize conversation storage."""
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.metadata_file = self.storage_dir / "conversation_metadata.json"
        
        # Load existing metadata
        self.conversations: Dict[str, ConversationMetadata] = {}
        self._load_metadata()
    
    def _load_metadata(self):
        """Load conversation metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file) as f:
                    data = json.load(f)
                    self.conversations = {
                        conv_id: ConversationMetadata(**conv_data)
                        for conv_id, conv_data in data.items()
                    }
                # Only print in non-test environments
                if not os.getenv("TESTING_MODE"):
                    print(f"📚 Loaded {len(self.conversations)} conversation records")
            except Exception as e:
                print(f"⚠️ Failed to load conversation metadata: {e}")
    
    def _save_metadata(self):
        """Save conversation metadata to disk."""
        try:
            data = {
                conv_id: asdict(conv_meta)
                for conv_id, conv_meta in self.conversations.items()
            }
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"⚠️ Failed to save conversation metadata: {e}")
    
    def create_conversation(
        self,
        session_type: str = "web_interface",
        version: Optional[str] = None,
        user_agent: Optional[str] = None,
        notes: Optional[str] = None
    ) -> str:
        """Create a new conversation session."""
        conversation_id = f"{session_type}_{int(time.time())}_{str(uuid4())[:8]}"
        
        if version is None:
            # Auto-generate version based on existing sessions
            existing_versions = [
                conv.version for conv in self.conversations.values() 
                if conv.session_type == session_type
            ]
            version_num = len([v for v in existing_versions if v.startswith("v")]) + 1
            version = f"v{version_num}"
        
        metadata = ConversationMetadata(
            conversation_id=conversation_id,
            version=version,
            created_at=datetime.now().isoformat(),
            last_used_at=datetime.now().isoformat(),
            session_type=session_type,
            user_agent=user_agent,
            notes=notes
        )
        
        self.conversations[conversation_id] = metadata
        self._save_metadata()
        
        print(f"🆕 Created conversation: {conversation_id} ({version})")
        return conversation_id
    
    def get_conversation(self, conversation_id: str) -> Optional[ConversationMetadata]:
        """Get conversation metadata by ID."""
        return self.conversations.get(conversation_id)
    
    def update_conversation_usage(
        self,
        conversation_id: str,
        message_delta: int = 0,
        workflow_delta: int = 0,
        execution_delta: int = 0
    ):
        """Update conversation usage statistics."""
        if conversation_id in self.conversations:
            conv = self.conversations[conversation_id]
            conv.last_used_at = datetime.now().isoformat()
            conv.total_messages += message_delta
            conv.workflow_count += workflow_delta
            conv.execution_count += execution_delta
            self._save_metadata()
    
    def mark_conversation_corrupted(self, conversation_id: str, reason: str):
        """Mark a conversation as corrupted."""
        if conversation_id in self.conversations:
            conv = self.conversations[conversation_id]
            conv.status = "corrupted"
            conv.notes = f"Corrupted: {reason}"
            self._save_metadata()
            print(f"🚨 Marked conversation as corrupted: {conversation_id}")
    
    def archive_conversation(self, conversation_id: str):
        """Archive a conversation (keep metadata but mark as archived)."""
        if conversation_id in self.conversations:
            conv = self.conversations[conversation_id]
            conv.status = "archived"
            conv.last_used_at = datetime.now().isoformat()
            self._save_metadata()
            print(f"📦 Archived conversation: {conversation_id}")
    
    def get_active_web_conversation(self) -> str:
        """Get the most recent active web interface conversation or create a new one."""
        # Find the most recent active web interface conversation
        web_conversations = [
            conv for conv in self.conversations.values()
            if conv.session_type == "web_interface" and conv.status == "active"
        ]
        
        if web_conversations:
            # Sort by last_used_at and get the most recent
            latest = max(web_conversations, key=lambda c: c.last_used_at)
            
            # Check if it's recent (less than 24 hours old)
            last_used = datetime.fromisoformat(latest.last_used_at)
            hours_since_use = (datetime.now() - last_used).total_seconds() / 3600
            
            if hours_since_use < 24:
                print(f"🔄 Using existing conversation: {latest.conversation_id} ({latest.version})")
                return latest.conversation_id
        
        # Create a new conversation
        return self.create_conversation(
            session_type="web_interface",
            notes="Auto-created for web interface session"
        )
    
    def list_conversations(self, session_type: Optional[str] = None, status: Optional[str] = None) -> List[ConversationMetadata]:
        """List conversations with optional filtering."""
        conversations = list(self.conversations.values())
        
        if session_type:
            conversations = [c for c in conversations if c.session_type == session_type]
        
        if status:
            conversations = [c for c in conversations if c.status == status]
        
        # Sort by last_used_at descending
        conversations.sort(key=lambda c: c.last_used_at, reverse=True)
        return conversations
    
    def cleanup_old_conversations(self, days_old: int = 30):
        """Archive conversations older than specified days."""
        cutoff = datetime.now().timestamp() - (days_old * 24 * 3600)
        
        archived_count = 0
        for conv_id, conv in self.conversations.items():
            last_used = datetime.fromisoformat(conv.last_used_at)
            if last_used.timestamp() < cutoff and conv.status == "active":
                self.archive_conversation(conv_id)
                archived_count += 1
        
        print(f"📦 Archived {archived_count} old conversations")
        return archived_count


# Global conversation storage instance
_conversation_storage = None

def get_conversation_storage() -> ConversationStorage:
    """Get the global conversation storage instance."""
    global _conversation_storage
    if _conversation_storage is None:
        _conversation_storage = ConversationStorage()
    return _conversation_storage