import os
import json
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Callable
from r2r import R2RClient
from ...utilities.decorators import register_tool


class RAG:
    """
    A wrapper for the R2RClient that exposes common operations for ingestion,
    extraction, graph management, deletion, and retrieval.
    """

    def __init__(self, host: str = "http://localhost:7272") -> None:
        """Initialize the R2R client.

        Args:
            host (str): The host URL for the R2R service.
        """
        self.client: R2RClient = R2RClient(host)

    # -----------------------------
    # Document Ingestion Operations
    # -----------------------------
    @register_tool(name="rag_ingest_document")
    def ingest_document(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        ingestion_mode: str = "hi-res",
        ingestion_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Ingest a document using one of the available modes.

        Args:
            file_path (str): Path to the document file.
            metadata (Optional[Dict[str, Any]]): Additional metadata for the document.
            ingestion_mode (str): Mode of ingestion ("hi-res", "fast", or "custom").
            ingestion_config (Optional[Dict[str, Any]]): Additional configuration if using custom mode.

        Returns:
            Dict[str, Any]: Response from the ingestion endpoint.
        """
        kwargs: Dict[str, Any] = {
            "file_path": file_path,
            "ingestion_mode": ingestion_mode,
        }
        if metadata:
            kwargs["metadata"] = metadata
        if ingestion_config:
            kwargs["ingestion_config"] = ingestion_config
        return self.client.documents.create(**kwargs)

    @register_tool(name="rag_ingest_documents_parallel")
    def ingest_documents_parallel(
        self,
        data_list: List[Dict[str, Any]],
        ingestion_mode: str = "hi-res",
        metadata: Optional[Dict[str, Any]] = None,
        metadata_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        ingestion_config: Optional[Dict[str, Any]] = None,
        max_workers: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        Parallelize the ingestion of multiple documents.

        Each document's data is written to a temporary file (round-trip to disk)
        and then ingested concurrently using a ThreadPoolExecutor.

        Args:
            data_list (List[Dict[str, Any]]): A list of data objects that are JSON serializable.
            ingestion_mode (str): Ingestion mode to use ("hi-res", "fast", or "custom").
            metadata (Optional[Dict[str, Any]]): A static metadata dictionary to include with every ingestion.
            metadata_fn (Optional[Callable[[Dict[str, Any]], Dict[str, Any]]]): A function that takes a data object and returns metadata. If provided, this function will be used to extract metadata from each document instead of the static metadata.
            ingestion_config (Optional[Dict[str, Any]]): Custom ingestion configuration for "custom" mode.
            max_workers (int): Maximum number of parallel workers.

        Returns:
            List[Dict[str, Any]]: A list of responses from the ingestion endpoint (one per document).
        """

        def ingest_single(data: Dict[str, Any]) -> Dict[str, Any]:
            # Create a unique temporary file name.
            temp_dir: str = tempfile.gettempdir()
            file_name: str = f"{uuid.uuid4()}.json"
            file_path: str = os.path.join(temp_dir, file_name)

            # Determine the metadata to use for this document:
            current_metadata: Optional[Dict[str, Any]] = (
                metadata_fn(data) if metadata_fn else metadata
            )

            try:
                # Write the data to disk.
                with open(file_path, "w") as f:
                    f.write(json.dumps(data))

                # Ingest the file.
                response: Dict[str, Any] = self.ingest_document(
                    file_path=file_path,
                    metadata=current_metadata,
                    ingestion_mode=ingestion_mode,
                    ingestion_config=ingestion_config,
                )
                return response
            finally:
                # Ensure the temporary file is removed.
                if os.path.exists(file_path):
                    os.remove(file_path)

        results: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit ingestion tasks for each data item.
            futures = [executor.submit(ingest_single, data) for data in data_list]
            for future in as_completed(futures):
                try:
                    result: Dict[str, Any] = future.result()
                    results.append(result)
                except Exception as e:
                    print("Error during ingestion:", e)
        return results

    # -----------------------------
    # Document Extraction & Listing
    # -----------------------------
    def _extract_document(self, document_id: str) -> Dict[str, Any]:
        """
        Extract entities and relationships from a document.

        Args:
            document_id (str): The ID of the document.

        Returns:
            Dict[str, Any]: Response from the extraction endpoint.
        """
        return self.client.documents.extract(document_id)

    @register_tool(name="rag_extract_document")
    def extract_documents(
        self, document_ids: List[str], max_workers: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Parallelize the extraction of entities and relationships from multiple documents.

        Args:
            document_ids (List[str]): A list of document IDs to extract.
            max_workers (int): Maximum number of parallel workers.

        Returns:
            List[Dict[str, Any]]: A list of responses from the extraction endpoint (one per document).
        """
        results: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._extract_document, doc_id)
                for doc_id in document_ids
            ]
            for future in as_completed(futures):
                try:
                    result: Dict[str, Any] = future.result()
                    results.append(result)
                except Exception as e:
                    print("Error during extraction:", e)
        return results

    @register_tool(name="rag_list_document_entities")
    def list_document_entities(self, document_id: str) -> Dict[str, Any]:
        """
        List the entities extracted from a document.

        Args:
            document_id (str): The ID of the document.

        Returns:
            Dict[str, Any]: List of entities.
        """
        return self.client.documents.list_entities(document_id)

    @register_tool(name="rag_list_document_relationships")
    def list_document_relationships(self, document_id: str) -> Dict[str, Any]:
        """
        List the relationships extracted from a document.

        Args:
            document_id (str): The ID of the document.

        Returns:
            Dict[str, Any]: List of relationships.
        """
        return self.client.documents.list_relationships(document_id)

    # -----------------------------
    # Collection & Graph Operations
    # -----------------------------

    @register_tool(name="rag_list_collections")
    def list_collections(self) -> Dict[str, Any]:
        """
        List all collections.

        Returns:
            Dict[str, Any]: Collections list.
        """
        return self.client.collections.list()

    def create_collection(self, name: str, description: str) -> Dict[str, Any]:
        """
        Create a new collection.

        Args:
            name (str): The collection's name.
            description (str): The collection's description.

        Returns:
            Dict[str, Any]: Response from the collection creation endpoint.
        """
        return self.client.collections.create(name=name, description=description)

    @register_tool(name="rag_add_document_to_collection")
    def add_document_to_collection(
        self, collection_id: str, document_id: str
    ) -> Dict[str, Any]:
        """
        Add a document to a collection.

        Args:
            collection_id (str): The collection's ID.
            document_id (str): The document's ID.

        Returns:
            Dict[str, Any]: Response from the add document to collection endpoint.
        """
        return self.client.collections.add_document(collection_id, document_id)

    @register_tool(name="rag_remove_document_from_collection")
    def remove_document_from_collection(
        self, collection_id: str, document_id: str
    ) -> Dict[str, Any]:
        """
        Remove a document from a collection.

        Args:
            collection_id (str): The collection's ID.
            document_id (str): The document's ID.

        Returns:
            Dict[str, Any]: Response from the remove document from collection endpoint.
        """
        return self.client.collections.remove_document(collection_id, document_id)

    @register_tool(name="rag_get_collection")
    def get_collection(self, collection_id: str) -> Dict[str, Any]:
        """
        Retrieve a collection by ID.

        Args:
            collection_id (str): The collection's ID.

        Returns:
            Dict[str, Any]: Collection details.
        """
        return self.client.collections.retrieve(collection_id)

    @register_tool(name="rag_list_graph_entities")
    def list_graph_entities(self, collection_id: str) -> Dict[str, Any]:
        """
        List graph entities for a given collection.

        Args:
            collection_id (str): The collection's ID.

        Returns:
            Dict[str, Any]: Graph entities.
        """
        return self.client.graphs.list_entities(collection_id)

    @register_tool(name="rag_list_graph_relationships")
    def list_graph_relationships(self, collection_id: str) -> Dict[str, Any]:
        """
        List graph relationships for a given collection.

        Args:
            collection_id (str): The collection's ID.

        Returns:
            Dict[str, Any]: Graph relationships.
        """
        return self.client.graphs.list_relationships(collection_id)

    @register_tool(name="rag_pull_graph")
    def pull_graph(self, collection_id: str) -> Dict[str, Any]:
        """
        Pull all entities and relationships from a collection into the graph.

        Args:
            collection_id (str): The collection's ID.

        Returns:
            Dict[str, Any]: Response from the graph pull operation.
        """
        return self.client.graphs.pull(collection_id=collection_id)

    @register_tool(name="rag_build_graph")
    def build_graph(self, collection_id: str) -> Dict[str, Any]:
        """
        Build communities in the graph for a collection.

        Args:
            collection_id (str): The collection's ID.

        Returns:
            Dict[str, Any]: Response from the graph build operation.
        """
        return self.client.graphs.build(collection_id=collection_id)

    # -----------------------------
    # Deletion Operations
    # -----------------------------
    @register_tool(name="rag_delete_document")
    def delete_document(self, document_id: str) -> Dict[str, Any]:
        """
        Delete a document.

        Args:
            document_id (str): The ID of the document to delete.

        Returns:
            Dict[str, Any]: Response from the deletion endpoint.
        """
        return self.client.documents.delete(document_id)

    def delete_chunk(self, chunk_id: str) -> Dict[str, Any]:
        """
        Delete a chunk from a document.

        Args:
            chunk_id (str): The ID of the chunk to delete.

        Returns:
            Dict[str, Any]: Response from the deletion endpoint.
        """
        return self.client.chunks.delete(chunk_id)

    # -----------------------------
    # Retrieval Operations
    # -----------------------------
    @register_tool(name="rag_retrieval")
    def retrieval_rag(
        self,
        query: str,
        rag_generation_config: Optional[Dict[str, Any]] = None,
        search_settings: Optional[Dict[str, Any]] = None,
        task_prompt_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Perform a Retrieval-Augmented Generation (RAG) query.

        Args:
            query (str): The query string.
            rag_generation_config (Optional[Dict[str, Any]]): Configuration for generation (e.g., model, temperature).
            search_settings (Optional[Dict[str, Any]]): Settings for the document/chunk search.
            task_prompt_override (Optional[str]): Custom prompt template.

        Returns:
            Dict[str, Any]: RAG query response.
        """
        response = self.client.retrieval.rag(
            query=query,
            rag_generation_config=rag_generation_config,
            search_settings=search_settings,
            task_prompt_override=task_prompt_override,
        )
        return response["results"]["completion"]["choices"][0]["message"]["content"]

    @register_tool(name="rag_retrieval_search")
    def retrieval_search(
        self, query: str, search_settings: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform a search query.

        Args:
            query (str): The search query.
            search_settings (Optional[Dict[str, Any]]): Additional search settings.

        Returns:
            Dict[str, Any]: Search response.
        """
        return self.client.retrieval.search(query, search_settings)

    # -----------------------------
    # System Settings - TODO NOT SURE  BOUT ADDING THESE TO REGISTRY
    # -----------------------------

    def get_system_settings(self) -> Dict[str, Any]:
        """
        Retrieve system settings.

        Returns:
            Dict[str, Any]: System settings.
        """
        return self.client.system.settings()

    # -----------------------------
    # Vector indexes
    # -----------------------------
    def list_vector_indexes(self, offset: int, limit: int) -> Dict[str, Any]:
        """
        List all vector indexes.

        Args:
            offset (int): The offset.
            limit (int): The limit.
        Returns:
            Dict[str, Any]: Vector indexes list.
        """
        return self.client.indices.list(offset=0, limit=10)

    def create_vector_index(
        self,
        config: Optional[Dict[str, Any]],
        run_with_orchestration: Optional[bool] = True,
    ) -> Dict[str, Any]:
        """
        Create a new vector index.

        Args:
            name (str): The index's name.
            config (Dict[str, Any]): The index's configuration.
            run_with_orchestration (bool): Run the index creation with orchestration.


            #create hnsw index for efficient vector search
            config={
                "table_name": "chunks",  # The table containing vector embeddings
                "index_method": "hnsw",   # Hierarchical Navigable Small World graph
                "index_measure": "cosine_distance",  # Similarity measure
                "index_arguments": {
                    "m": 16,              # Number of connections per layer
                    "ef_construction": 64,# Size of dynamic candidate list for construction
                    "ef": 40,            # Size of dynamic candidate list for search
                },
                "index_name": "my_document_embeddings_idx",
                "index_column": "embedding",
                "concurrently": True     # Build index without blocking table writes
            },

            # Create an IVF-Flat index for balanced performance
            config={
                "table_name": "chunks",
                "index_method": "ivf_flat", # Inverted File with Flat storage
                "index_measure": "l2_distance",
                "index_arguments": {
                    "lists": 100,         # Number of cluster centroids
                    "probe": 10,          # Number of clusters to search
                },
                "index_name": "my_ivf_embeddings_idx",
                "index_column": "embedding",
                "concurrently": True
            }

        Returns:
            Dict[str, Any]: Response from the index creation endpoint.
        """
        return self.client.indices.create(config, run_with_orchestration)

    def delete_vector_index(
        self, index_name: str, table_name: str, run_with_orchestration: bool
    ) -> Dict[str, Any]:
        """
        Delete a vector index.

        Args:
            index_name (str): The index's name.

        Returns:
            Dict[str, Any]: Response from the index deletion endpoint.
        """
        return self.client.indices.delete(
            index_name, table_name, run_with_orchestration
        )
