from controlflow.memory.memory import Memory, MemoryProvider
from controlflow.memory.async_memory import AsyncMemory, AsyncMemoryProvider
from controlflow.memory.providers.postgres import PostgresMemory, AsyncPostgresMemory
from typing import Optional
from langchain_openai import OpenAIEmbeddings
import os


class Memory(Memory):
    """
    Simple wrapper class of cf.Memory to store and retrieve data from memory via a MemoryModule.
    A class to store and retrieve data from memory via a MemoryModule.

    provider = PostgresMemory(
        database_url="<database str>",
        embedding_dimension=1536,
        embedding_fn=OpenAIEmbeddings(),
        table_name="vector_db",
    )
    # Create a memory module for user preferences
    user_preferences = Memory(
        key="user_preferences",
        instructions="Store and retrieve user preferences.",
        provider=provider,
    )

    # Create an agent with access to the memory
    agent = Agent(memories=[user_preferences])
    (tasks("My text to process")
        .custom(
            name="do-fancy-thing",
            objective="Perform a fancy custom step on the text",
            agents=[agent],
            instructions="Analyze the text in a fancy custom way",
            custom_key="some_extra_value",
        )
        ...
       )

    results = tasks.run_tasks()
    print(results)
    """

    def __init__(        
        self,
        key: str,
        instructions: str,
        provider: MemoryProvider = None,
    ):
        super().__init__(key, instructions, provider)


class AsyncMemory(AsyncMemory):
    """
    Simple wrapper class of cf.AsyncMemory to store and retrieve data from memory via a MemoryModule.
    A class to store and retrieve data from memory via a MemoryModule.

     provider = AsyncPostgresMemory(
        database_url="<database str>",
        embedding_dimension=1536,
        embedding_fn=OpenAIEmbeddings(),
        table_name="vector_db",
    )
    # Create a memory module for user preferences
    user_preferences = AsyncMemory(
        key="user_preferences",
        instructions="Store and retrieve user preferences.",
        provider=provider,
    )

    # Create an agent with access to the memory
    agent = Agent(memories=[user_preferences])
    (tasks("My text to process")
        .custom(
            name="do-fancy-thing",
            objective="Perform a fancy custom step on the text",
            agents=[agent],
            instructions="Analyze the text in a fancy custom way",
            custom_key="some_extra_value",
        )
        ...
       )

    results = tasks.run_tasks(client_mode = False, run_async=True)
    print(results)   
    """

    def __init__(        
        self,
        key: str,
        instructions: str,
        provider: AsyncMemoryProvider = None,
    ):
        super().__init__(key, instructions, provider)


class PostgresMemoryProvider(PostgresMemory):
    """
    A class to store and retrieve data from a PostgreSQL database.
    """

    def __init__(
        self,
        database_url: str = None,
        embedding_dimension: float = 1536,
        embedding_fn: Optional[OpenAIEmbeddings] = None,
        table_name:str = None,
        **kwargs
    ):
        if isinstance(embedding_fn, OpenAIEmbeddings):
            embedding_fn = embedding_fn
        
        else:
            embed_kwargs = {}
            for key, env_name in {
                "api_key": "OPENAI_API_KEY",
                "model": "OPENAI_API_MODEL",
                "base_url": "OPENAI_API_BASE_URL",
            }.items():
                if value := os.environ.get(env_name):
                    embed_kwargs[key] = value
                embedding_fn = OpenAIEmbeddings(dimensions=embedding_dimension, **embed_kwargs)
        super().__init__(database_url, embedding_dimension, embedding_fn, table_name, **kwargs)

class AsyncPostgresMemoryProvider(AsyncPostgresMemory):
    """
    A class to store and retrieve data from a PostgreSQL database.
    """

    def __init__(
        self,
        database_url: str = None,
        embedding_dimension: float = 1536,
        embedding_fn: Optional[OpenAIEmbeddings] = None,
        table_name:str = None,
        **kwargs
    ):
        if isinstance(embedding_fn, OpenAIEmbeddings):
            embedding_fn = embedding_fn
        
        else:
            embed_kwargs = {}
            for key, env_name in {
                "api_key": "OPENAI_API_KEY",
                "model": "OPENAI_API_MODEL",
                "base_url": "OPENAI_API_BASE_URL",
            }.items():
                if value := os.environ.get(env_name):
                    embed_kwargs[key] = value
                embedding_fn = OpenAIEmbeddings(dimensions=embedding_dimension, **embed_kwargs)

        super().__init__(database_url, embedding_dimension, embedding_fn, table_name, **kwargs)