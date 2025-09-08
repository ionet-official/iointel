#!/usr/bin/env python3
"""
Fresh Start Script for Unified Conversation Storage
==================================================

Initialize the new unified conversation system without migrating old data.
This creates a clean slate for testing the new architecture.
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from iointel.src.web.unified_conversation_storage import UnifiedConversationStorage

def start_fresh():
    """Initialize fresh unified conversation storage."""
    print("🚀 Starting fresh with unified conversation storage...")
    
    # Backup old database files if they exist
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"backup_conversations_{timestamp}"
    
    old_files = [
        "simple_conversations.db",
        "unified_conversations.db",
        "conversations"
    ]
    
    if any(os.path.exists(f) for f in old_files):
        os.makedirs(backup_dir, exist_ok=True)
        print(f"📦 Creating backup in: {backup_dir}")
        
        for file_path in old_files:
            if os.path.exists(file_path):
                if os.path.isfile(file_path):
                    import shutil
                    shutil.copy2(file_path, backup_dir)
                    print(f"💾 Backed up file: {file_path}")
                elif os.path.isdir(file_path):
                    import shutil
                    shutil.copytree(file_path, f"{backup_dir}/{file_path}")
                    print(f"💾 Backed up directory: {file_path}")
    else:
        print("📭 No old conversation files found to backup")
    
    # Initialize new unified storage
    print("📦 Creating new unified conversation database...")
    storage = UnifiedConversationStorage("conversations.db")
    
    # Create a test conversation
    print("🧪 Creating test conversation...")
    test_conv_id = storage.create_conversation(
        session_type="web_interface",
        notes="Test conversation for unified system"
    )
    
    # Add a test message
    print("💬 Adding test message...")
    success = storage.add_message(
        test_conv_id,
        "Hello, this is a test message!",
        "Hi! This is the unified conversation system working correctly."
    )
    
    if success:
        print("✅ Test message added successfully")
    else:
        print("❌ Failed to add test message")
    
    # Verify the data
    print("🔍 Verifying data...")
    conv = storage.get_conversation(test_conv_id)
    messages = storage.get_messages(test_conv_id)
    
    print(f"📊 Conversation: {conv.conversation_id} (v{conv.version})")
    print(f"📊 Messages: {len(messages)}")
    print(f"📊 Status: {conv.status}")
    
    if messages:
        print(f"💬 Latest agent response: {messages[0]['agent_response'][:50]}...")
    
    print("\n🎉 Fresh unified conversation system is ready!")
    print("📁 Database: unified_conversations.db")
    print("🔧 Ready to update API endpoints")
    
    return storage

if __name__ == "__main__":
    start_fresh()