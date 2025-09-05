import json


from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
    SystemPromptPart,
    ToolCallPart,
    ToolReturnPart,
    RetryPromptPart,
)

from datetime import datetime, timezone
from sqlalchemy import Column, String, Text, DateTime, select
from sqlalchemy.orm import declarative_base

# For async support
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker


import inspect


def parse_timestamp(ts_str: str) -> datetime:
    """Convert an ISO timestamp string (with trailing 'Z') into a datetime object."""
    if ts_str.endswith("Z"):
        ts_str = ts_str[:-1] + "+00:00"
    return datetime.fromisoformat(ts_str)


def parse_part(part: dict):
    part_kind = part.get("part_kind")
    if part_kind == "system-prompt":
        return SystemPromptPart(**part)
    elif part_kind == "user-prompt":
        if "timestamp" in part and part["timestamp"]:
            part["timestamp"] = parse_timestamp(part["timestamp"])
        return UserPromptPart(**part)
    elif part_kind == "retry-prompt":
        return RetryPromptPart(**part)
    elif part_kind == "text":
        return TextPart(**part)
    elif part_kind == "tool-call":
        return ToolCallPart(**part)
    elif part_kind == "tool-return":
        # Fix: Handle timestamp conversion for ToolReturnPart
        if "timestamp" in part and part["timestamp"]:
            part["timestamp"] = parse_timestamp(part["timestamp"])
        return ToolReturnPart(**part)
    else:
        raise ValueError(f"Unknown part kind: {part_kind}")


def parse_request(item: dict) -> ModelRequest:
    """Parse a dictionary representing a request into a ModelRequest instance."""
    parts_data = item.get("parts", [])
    parts = [parse_part(p) for p in parts_data]
    return ModelRequest(parts=parts, kind=item.get("kind"))


def parse_response(item: dict) -> ModelResponse:
    """Parse a dictionary representing a response into a ModelResponse instance."""
    parts_data = item.get("parts", [])
    parts = [parse_part(p) for p in parts_data]
    ts = item.get("timestamp")
    if ts:
        ts = parse_timestamp(ts)
    return ModelResponse(
        parts=parts,
        model_name=item.get("model_name"),
        timestamp=ts,
        kind=item.get("kind"),
    )


Base = declarative_base()


class ConversationHistory(Base):
    __tablename__ = "conversation_history"
    conversation_id = Column(String, primary_key=True, index=True)
    messages_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.now(timezone.utc))


class AsyncMemory:
    def __init__(self, connection_string: str):
        """
        Initialize the async memory module.
        :param connection_string: A SQLAlchemy async-compatible database URL.
            For SQLite (async): "sqlite+aiosqlite:///path/to/db.sqlite3"
            For Postgres (async): "postgresql+asyncpg://user:password@host:port/dbname"
        """
        self.engine = create_async_engine(connection_string, future=True)
        self.SessionLocal = async_sessionmaker(bind=self.engine, expire_on_commit=False)

    async def init_models(self):
        """
        Create the database tables asynchronously.
        """
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    async def store_run_history(self, conversation_id: str, result) -> bool:
        print(f"\nStoring run history for conversation {conversation_id}")
        async with self.SessionLocal() as session:
            try:
                result_obj = await session.execute(
                    select(ConversationHistory).filter_by(
                        conversation_id=conversation_id
                    )
                )
                existing_conversation = result_obj.scalars().first()
                print(f"Found existing conversation: {bool(existing_conversation)}")

                existing_messages_raw = (
                    existing_conversation.messages_json
                    if existing_conversation
                    else "[]"
                )
                if isinstance(existing_messages_raw, bytes):
                    existing_messages_raw = existing_messages_raw.decode("utf-8")
                print(f"Existing messages raw type: {type(existing_messages_raw)}")

                existing_messages = json.loads(existing_messages_raw)
                print(f"Loaded {len(existing_messages)} existing messages")

                # Ensure result.all_messages_json() is awaited if async, decoded, and parsed correctly
                new_messages_raw = (
                    result.all_messages_json()
                    if hasattr(result, "all_messages_json")
                    else "[]"
                )
                print(f"New messages raw type: {type(new_messages_raw)}")

                # Fix: explicitly handle potential coroutine and decode bytes
                if inspect.isawaitable(new_messages_raw):
                    print("Awaiting new_messages_raw coroutine")
                    new_messages_raw = await new_messages_raw
                if isinstance(new_messages_raw, bytes):
                    print("Decoding new_messages_raw bytes")
                    new_messages_raw = new_messages_raw.decode("utf-8")

                new_messages = json.loads(new_messages_raw)
                print(f"Loaded {len(new_messages)} new messages")

                combined_messages = existing_messages + new_messages
                messages_json = json.dumps(combined_messages)
                print(f"Combined message count: {len(combined_messages)}")

                if existing_conversation:
                    print("Updating existing conversation")
                    existing_conversation.messages_json = messages_json
                    existing_conversation.created_at = datetime.now(timezone.utc)
                else:
                    print("Creating new conversation")
                    conversation = ConversationHistory(
                        conversation_id=conversation_id,
                        messages_json=messages_json,
                        created_at=datetime.now(timezone.utc),
                    )
                    session.add(conversation)

                await session.commit()
                print("Successfully committed to database")
                return True

            except Exception as e:
                print(f"Error storing run history: {e}")
                print(f"Error type: {type(e)}")
                import traceback
                print(f"Traceback: {traceback.format_exc()}")
                return False
            finally:
                await session.close()
                print("Session closed")
        
    

    async def get_history(self, conversation_id: str) -> str:
        """
        Asynchronously retrieve stored conversation history as a JSON string for a given conversation_id.
        """
        async with self.SessionLocal() as session:
            result = await session.execute(
                select(ConversationHistory).filter_by(conversation_id=conversation_id)
            )
            conversation = result.scalars().first()
            print(f"--- Found conversation: {bool(conversation)} using conversation_id: {conversation_id}")
            if conversation:
                print(f"--- Messages JSON length: {len(conversation.messages_json)}")
            return conversation.messages_json if conversation else None

    async def get_message_history(self, conversation_id: str, MAX_MESSAGES=100):
        print(f"--- Getting message history for conversation_id: {conversation_id}")
        raw = await self.get_history(conversation_id)
        print(f"--- Got raw history: {bool(raw)}")
        if raw:
            print(f"--- Raw type: {type(raw)}")
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
                print("--- Decoded bytes to string")
            try:
                history_list = json.loads(raw)
                print(f"--- Parsed JSON, got {len(history_list)} messages")
            except Exception as e:
                print("Error parsing JSON from stored history:", e)
                return None
            filtered_history_list = history_list[-MAX_MESSAGES:]
            print(f"--- Filtered to last {len(filtered_history_list)} messages")
            parsed_history = []
            
            # Check if this is a structured output conversation (like workflow planner)
            is_structured_output = self._is_structured_output_conversation(conversation_id, history_list)
            print(f"--- Detected structured output conversation: {is_structured_output}")
            
            for item in filtered_history_list:
                kind = item.get("kind")
                parts = item.get("parts", [])
                
                # For structured output conversations, preserve tool-call/tool-return messages
                # For regular conversations, filter them out as before
                if is_structured_output:
                    # Keep all parts for structured output conversations
                    filtered_parts = parts
                else:
                    # Filter out tool-call/tool-return parts for regular conversations
                    filtered_parts = [
                        part
                        for part in parts
                        if part.get("part_kind")
                        not in {"tool-call", "tool-return", "retry-prompt"}
                    ]
                
                if not filtered_parts:
                    continue

                item["parts"] = filtered_parts
                if kind == "request":
                    parsed_history.append(parse_request(item))
                elif kind == "response":
                    parsed_history.append(parse_response(item))
            print(f"--- Returning {len(parsed_history)} parsed messages")
            return parsed_history
        return None

    def _is_structured_output_conversation(self, conversation_id: str, history_list: list) -> bool:
        """
        Detect if this is a structured output conversation (like workflow planner).
        
        Criteria:
        1. Conversation ID contains 'workflow' or 'planner'
        2. Messages contain tool-call/tool-return parts with structured output tools
        3. Messages contain 'final_result' or similar structured output tool calls
        """
        # Check conversation ID patterns
        if any(keyword in conversation_id.lower() for keyword in ['workflow', 'planner']):
            return True
        
        # Check message content for structured output patterns
        for item in history_list:
            parts = item.get("parts", [])
            for part in parts:
                part_kind = part.get("part_kind")
                if part_kind == "tool-call":
                    tool_name = part.get("tool_name", "")
                    # Check for structured output tools
                    if any(keyword in tool_name.lower() for keyword in ['final_result', 'structured', 'workflow']):
                        return True
                elif part_kind == "tool-return":
                    # Check for structured output in tool returns
                    content = part.get("content", "")
                    if isinstance(content, str) and any(keyword in content.lower() for keyword in ['workflow', 'nodes', 'edges']):
                        return True
        
        return False

    async def list_conversation_ids(self) -> list[str]:
        """
        Asynchronously list all unique conversation IDs stored in the database.
        """
        async with self.SessionLocal() as session:
            try:
                result = await session.execute(
                    select(ConversationHistory.conversation_id)
                )
                ids = result.scalars().all()
                return ids
            except Exception as e:
                print(f"Error listing conversation IDs: {e}")
                return []
