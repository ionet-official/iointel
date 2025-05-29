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

    async def store_run_history(self, conversation_id: str, result) -> None:
        async with self.SessionLocal() as session:
            try:
                result_obj = await session.execute(
                    select(ConversationHistory).filter_by(
                        conversation_id=conversation_id
                    )
                )
                existing_conversation = result_obj.scalars().first()

                existing_messages_raw = (
                    existing_conversation.messages_json
                    if existing_conversation
                    else "[]"
                )
                if isinstance(existing_messages_raw, bytes):
                    existing_messages_raw = existing_messages_raw.decode("utf-8")

                existing_messages = json.loads(existing_messages_raw)

                # Ensure result.all_messages_json() is awaited if async, decoded, and parsed correctly
                new_messages_raw = (
                    result.all_messages_json()
                    if hasattr(result, "all_messages_json")
                    else "[]"
                )

                # Fix: explicitly handle potential coroutine and decode bytes
                if inspect.isawaitable(new_messages_raw):
                    new_messages_raw = await new_messages_raw
                if isinstance(new_messages_raw, bytes):
                    new_messages_raw = new_messages_raw.decode("utf-8")

                new_messages = json.loads(new_messages_raw)

                combined_messages = existing_messages + new_messages
                messages_json = json.dumps(combined_messages)

                if existing_conversation:
                    existing_conversation.messages_json = messages_json
                    existing_conversation.created_at = datetime.now(timezone.utc)
                else:
                    conversation = ConversationHistory(
                        conversation_id=conversation_id,
                        messages_json=messages_json,
                        created_at=datetime.now(timezone.utc),
                    )
                    session.add(conversation)

                await session.commit()

            except Exception as e:
                print(f"Error storing run history: {e}")
            finally:
                await session.close()

    async def get_history(self, conversation_id: str) -> str:
        """
        Asynchronously retrieve stored conversation history as a JSON string for a given conversation_id.
        """
        async with self.SessionLocal() as session:
            result = await session.execute(
                select(ConversationHistory).filter_by(conversation_id=conversation_id)
            )
            conversation = result.scalars().first()
            return conversation.messages_json if conversation else None

    async def get_message_history(self, conversation_id: str, MAX_MESSAGES=100):
        raw = await self.get_history(conversation_id)
        if raw:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            try:
                history_list = json.loads(raw)
            except Exception as e:
                print("Error parsing JSON from stored history:", e)
                return None
            filtered_history_list = history_list[-MAX_MESSAGES:]
            parsed_history = []
            for item in filtered_history_list:
                kind = item.get("kind")
                parts = item.get("parts", [])
                # Explicitly filter out tool-call/tool-return parts
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
            return parsed_history
        return None
