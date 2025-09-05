"""
Simple conversation storage - separate from prompt storage.
Just stores user input and agent response for context.
"""
import json
import sqlite3
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from pathlib import Path


class SimpleConversation:
    """
    Simple conversation storage that only stores user input and agent response.
    No system prompts, no tool calls, no debugging info - just clean conversation.
    """
    
    def __init__(self, db_path: str = "simple_conversations.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize the database with a simple conversation table."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                user_input TEXT NOT NULL,
                agent_response TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create index separately
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversation_id ON conversations(conversation_id)
        """)
        
        conn.commit()
        conn.close()
    
    def add(self, conversation_id: str, user_input: str, agent_response: str) -> bool:
        """
        Add a simple conversation turn.
        
        Args:
            conversation_id: Unique identifier for the conversation
            user_input: What the user said
            agent_response: What the agent responded
            
        Returns:
            bool: True if successful
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO conversations (conversation_id, user_input, agent_response, timestamp)
                VALUES (?, ?, ?, ?)
            """, (conversation_id, user_input, agent_response, datetime.now(timezone.utc)))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error adding conversation: {e}")
            return False
    
    def get_conversation(self, conversation_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get conversation history for a specific conversation ID.
        
        Args:
            conversation_id: The conversation to retrieve
            limit: Maximum number of recent turns to return
            
        Returns:
            List of conversation turns with user_input, agent_response, timestamp
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT user_input, agent_response, timestamp
                FROM conversations
                WHERE conversation_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (conversation_id, limit))
            
            results = cursor.fetchall()
            conn.close()
            
            # Convert to list of dicts, most recent first
            conversation = []
            for user_input, agent_response, timestamp in results:
                conversation.append({
                    "user_input": user_input,
                    "agent_response": agent_response,
                    "timestamp": timestamp
                })
            
            return conversation
        except Exception as e:
            print(f"Error getting conversation: {e}")
            return []
    
    def get_recent_conversations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent conversation IDs and their latest activity.
        
        Args:
            limit: Maximum number of conversations to return
            
        Returns:
            List of conversation summaries
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT conversation_id, MAX(timestamp) as last_activity, COUNT(*) as message_count
                FROM conversations
                GROUP BY conversation_id
                ORDER BY last_activity DESC
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
        except Exception as e:
            print(f"Error getting recent conversations: {e}")
            return []
    
    def clear_conversation(self, conversation_id: str) -> bool:
        """
        Clear all messages for a specific conversation.
        
        Args:
            conversation_id: The conversation to clear
            
        Returns:
            bool: True if successful
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM conversations WHERE conversation_id = ?", (conversation_id,))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error clearing conversation: {e}")
            return False


# Global instance for easy access
simple_conversation = SimpleConversation()
