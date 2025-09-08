"""
Unified Conversation Storage System
===================================

Single SQLite database for all conversation data - metadata and messages.
Replaces both ConversationStorage (JSON) and SimpleConversation (SQLite).
"""

import sqlite3
import json
import os
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass


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


class UnifiedConversationStorage:
    """
    Unified conversation storage using a single SQLite database.
    Handles both conversation metadata and message content.
    """
    
    def __init__(self, db_path: str = "conversations.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize the database with unified schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Conversations table (metadata)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id TEXT PRIMARY KEY,
                version TEXT NOT NULL,
                created_at DATETIME NOT NULL,
                last_used_at DATETIME NOT NULL,
                session_type TEXT NOT NULL,
                user_agent TEXT,
                total_messages INTEGER DEFAULT 0,
                workflow_count INTEGER DEFAULT 0,
                execution_count INTEGER DEFAULT 0,
                status TEXT DEFAULT 'active',
                notes TEXT
            )
        """)
        
        # Messages table (actual chat content)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                user_input TEXT NOT NULL,
                agent_response TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
            )
        """)
        
        # Create indexes for better performance
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_conversation_id 
            ON messages(conversation_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_timestamp 
            ON messages(timestamp)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversations_session_type 
            ON conversations(session_type)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversations_status 
            ON conversations(status)
        """)
        
        conn.commit()
        conn.close()
    
    def create_conversation(
        self,
        session_type: str = "web_interface",
        version: Optional[str] = None,
        user_agent: Optional[str] = None,
        notes: Optional[str] = None
    ) -> str:
        """Create a new conversation session."""
        import time
        from uuid import uuid4
        
        conversation_id = f"{session_type}_{int(time.time())}_{str(uuid4())[:8]}"
        
        if version is None:
            # Auto-generate version based on existing sessions
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM conversations 
                WHERE session_type = ? AND version LIKE 'v%'
            """, (session_type,))
            version_num = cursor.fetchone()[0] + 1
            version = f"v{version_num}"
            conn.close()
        
        now = datetime.now(timezone.utc).isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO conversations (
                conversation_id, version, created_at, last_used_at, 
                session_type, user_agent, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (conversation_id, version, now, now, session_type, user_agent, notes))
        conn.commit()
        conn.close()
        
        print(f"🆕 Created conversation: {conversation_id} ({version})")
        return conversation_id
    
    def get_conversation(self, conversation_id: str) -> Optional[ConversationMetadata]:
        """Get conversation metadata by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT conversation_id, version, created_at, last_used_at,
                   session_type, user_agent, total_messages, workflow_count,
                   execution_count, status, notes
            FROM conversations WHERE conversation_id = ?
        """, (conversation_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return ConversationMetadata(
                conversation_id=row[0],
                version=row[1],
                created_at=row[2],
                last_used_at=row[3],
                session_type=row[4],
                user_agent=row[5],
                total_messages=row[6],
                workflow_count=row[7],
                execution_count=row[8],
                status=row[9],
                notes=row[10]
            )
        return None
    
    def add_message(
        self, 
        conversation_id: str, 
        user_input: str, 
        agent_response: str
    ) -> bool:
        """Add a message to a conversation."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Add the message
            cursor.execute("""
                INSERT INTO messages (conversation_id, user_input, agent_response, timestamp)
                VALUES (?, ?, ?, ?)
            """, (conversation_id, user_input, agent_response, datetime.now(timezone.utc)))
            
            # Update conversation metadata
            cursor.execute("""
                UPDATE conversations 
                SET total_messages = total_messages + 1,
                    last_used_at = ?
                WHERE conversation_id = ?
            """, (datetime.now(timezone.utc).isoformat(), conversation_id))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error adding message: {e}")
            return False
    
    def get_messages(self, conversation_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get conversation messages."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT user_input, agent_response, timestamp
                FROM messages
                WHERE conversation_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (conversation_id, limit))
            
            results = cursor.fetchall()
            conn.close()
            
            messages = []
            for user_input, agent_response, timestamp in results:
                messages.append({
                    "user_input": user_input,
                    "agent_response": agent_response,
                    "timestamp": timestamp
                })
            
            return messages
        except Exception as e:
            print(f"Error getting messages: {e}")
            return []
    
    def update_conversation_usage(
        self,
        conversation_id: str,
        message_delta: int = 0,
        workflow_delta: int = 0,
        execution_delta: int = 0
    ):
        """Update conversation usage statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE conversations 
            SET total_messages = total_messages + ?,
                workflow_count = workflow_count + ?,
                execution_count = execution_count + ?,
                last_used_at = ?
            WHERE conversation_id = ?
        """, (message_delta, workflow_delta, execution_delta, 
              datetime.now(timezone.utc).isoformat(), conversation_id))
        conn.commit()
        conn.close()
    
    def mark_conversation_corrupted(self, conversation_id: str, reason: str):
        """Mark a conversation as corrupted."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE conversations 
            SET status = 'corrupted', notes = ?
            WHERE conversation_id = ?
        """, (f"Corrupted: {reason}", conversation_id))
        conn.commit()
        conn.close()
        print(f"🚨 Marked conversation as corrupted: {conversation_id}")
    
    def archive_conversation(self, conversation_id: str):
        """Archive a conversation."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE conversations 
            SET status = 'archived', last_used_at = ?
            WHERE conversation_id = ?
        """, (datetime.now(timezone.utc).isoformat(), conversation_id))
        conn.commit()
        conn.close()
        print(f"📦 Archived conversation: {conversation_id}")
    
    def get_active_web_conversation(self) -> str:
        """Get the most recent active web interface conversation or create a new one."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT conversation_id, last_used_at
            FROM conversations
            WHERE session_type = 'web_interface' AND status = 'active'
            ORDER BY last_used_at DESC
            LIMIT 1
        """)
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            conversation_id, last_used_at = row
            last_used = datetime.fromisoformat(last_used_at)
            hours_since_use = (datetime.now(timezone.utc) - last_used).total_seconds() / 3600
            
            if hours_since_use < 24:
                print(f"🔄 Using existing conversation: {conversation_id}")
                return conversation_id
        
        # Create a new conversation
        return self.create_conversation(
            session_type="web_interface",
            notes="Auto-created for web interface session"
        )
    
    def list_conversations(
        self, 
        session_type: Optional[str] = None, 
        status: Optional[str] = None
    ) -> List[ConversationMetadata]:
        """List conversations with optional filtering."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = """
            SELECT conversation_id, version, created_at, last_used_at,
                   session_type, user_agent, total_messages, workflow_count,
                   execution_count, status, notes
            FROM conversations
        """
        params = []
        
        conditions = []
        if session_type:
            conditions.append("session_type = ?")
            params.append(session_type)
        if status:
            conditions.append("status = ?")
            params.append(status)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY last_used_at DESC"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        conversations = []
        for row in rows:
            conversations.append(ConversationMetadata(
                conversation_id=row[0],
                version=row[1],
                created_at=row[2],
                last_used_at=row[3],
                session_type=row[4],
                user_agent=row[5],
                total_messages=row[6],
                workflow_count=row[7],
                execution_count=row[8],
                status=row[9],
                notes=row[10]
            ))
        
        return conversations
    
    def get_recent_conversations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversation IDs and their latest activity."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT conversation_id, last_used_at, total_messages
            FROM conversations
            ORDER BY last_used_at DESC
            LIMIT ?
        """, (limit,))
        
        results = cursor.fetchall()
        conn.close()
        
        conversations = []
        for conv_id, last_activity, message_count in results:
            conversations.append({
                "conversation_id": conv_id,
                "last_activity": last_activity,
                "message_count": message_count
            })
        
        return conversations
    
    def clear_conversation(self, conversation_id: str) -> bool:
        """Clear all messages for a specific conversation."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Delete messages
            cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
            
            # Reset message count
            cursor.execute("""
                UPDATE conversations 
                SET total_messages = 0
                WHERE conversation_id = ?
            """, (conversation_id,))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error clearing conversation: {e}")
            return False
    
    def cleanup_old_conversations(self, days_old: int = 30):
        """Archive conversations older than specified days."""
        cutoff = datetime.now(timezone.utc).timestamp() - (days_old * 24 * 3600)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE conversations 
            SET status = 'archived'
            WHERE status = 'active' 
            AND datetime(last_used_at) < datetime(?, 'unixepoch')
        """, (cutoff,))
        
        archived_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        print(f"📦 Archived {archived_count} old conversations")
        return archived_count


# Global unified conversation storage instance
_unified_conversation_storage = None

def get_unified_conversation_storage() -> UnifiedConversationStorage:
    """Get the global unified conversation storage instance."""
    global _unified_conversation_storage
    if _unified_conversation_storage is None:
        _unified_conversation_storage = UnifiedConversationStorage()
    return _unified_conversation_storage


# Migration function
def migrate_from_old_systems():
    """Migrate data from old ConversationStorage (JSON) and SimpleConversation (SQLite) to unified system."""
    print("🔄 Starting migration from old conversation systems...")
    
    unified_storage = UnifiedConversationStorage("conversations.db")
    
    # Migrate from ConversationStorage (JSON)
    json_file = Path("conversations/conversation_metadata.json")
    if json_file.exists():
        print("📄 Migrating conversation metadata from JSON...")
        try:
            with open(json_file) as f:
                data = json.load(f)
            
            conn = sqlite3.connect(unified_storage.db_path)
            cursor = conn.cursor()
            
            for conv_id, conv_data in data.items():
                cursor.execute("""
                    INSERT OR REPLACE INTO conversations (
                        conversation_id, version, created_at, last_used_at,
                        session_type, user_agent, total_messages, workflow_count,
                        execution_count, status, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    conv_id,
                    conv_data.get("version", "v1"),
                    conv_data.get("created_at", datetime.now(timezone.utc).isoformat()),
                    conv_data.get("last_used_at", datetime.now(timezone.utc).isoformat()),
                    conv_data.get("session_type", "web_interface"),
                    conv_data.get("user_agent"),
                    conv_data.get("total_messages", 0),
                    conv_data.get("workflow_count", 0),
                    conv_data.get("execution_count", 0),
                    conv_data.get("status", "active"),
                    conv_data.get("notes")
                ))
            
            conn.commit()
            conn.close()
            print(f"✅ Migrated {len(data)} conversation metadata records")
        except Exception as e:
            print(f"❌ Error migrating JSON data: {e}")
    
    # Migrate from SimpleConversation (SQLite)
    old_db_path = "simple_conversations.db"
    if os.path.exists(old_db_path):
        print("🗄️ Migrating messages from old SQLite database...")
        try:
            old_conn = sqlite3.connect(old_db_path)
            old_cursor = old_conn.cursor()
            
            new_conn = sqlite3.connect(unified_storage.db_path)
            new_cursor = new_conn.cursor()
            
            # Get all messages from old database
            old_cursor.execute("""
                SELECT conversation_id, user_input, agent_response, timestamp
                FROM conversations
                ORDER BY timestamp
            """)
            
            messages = old_cursor.fetchall()
            migrated_count = 0
            
            for conv_id, user_input, agent_response, timestamp in messages:
                # Ensure conversation exists in new database
                new_cursor.execute("""
                    INSERT OR IGNORE INTO conversations (
                        conversation_id, version, created_at, last_used_at, session_type
                    ) VALUES (?, 'v1', ?, ?, 'web_interface')
                """, (conv_id, timestamp, timestamp))
                
                # Insert message
                new_cursor.execute("""
                    INSERT INTO messages (conversation_id, user_input, agent_response, timestamp)
                    VALUES (?, ?, ?, ?)
                """, (conv_id, user_input, agent_response, timestamp))
                
                migrated_count += 1
            
            new_conn.commit()
            old_conn.close()
            new_conn.close()
            print(f"✅ Migrated {migrated_count} messages")
        except Exception as e:
            print(f"❌ Error migrating SQLite data: {e}")
    
    print("🎉 Migration completed!")
    return unified_storage


if __name__ == "__main__":
    # Run migration
    migrate_from_old_systems()
