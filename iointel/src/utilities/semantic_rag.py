"""
Factory-based semantic RAG system using usearch for vector similarity search.

This module provides a modular, reusable RAG capability that can work with:
- Pydantic models (list[BaseModel])
- List of lists of strings (list[list[str]])
- DataFrames

Key features:
- Field/column selection for encoding specific parts of data
- Multiple vector indices per collection
- Borda count reranking across multiple vectors
- Single record insertion
- Factory pattern for different data types
"""

import os
import json
import uuid
from typing import List, Dict, Any, Union, Optional, Callable
from dataclasses import dataclass, field
import pandas as pd
from pydantic import BaseModel

try:
    from usearch.index import Index as UsearchIndex, MetricKind
    import numpy as np
    import hashlib
    # Lazy import for sentence transformers to avoid slow startup
    SentenceTransformer = None
except ImportError as e:
    raise ImportError(f"Required dependencies missing: {e}. Install with: uv add usearch sentence-transformers numpy")


class FastHashEncoder:
    """Super fast hash-based encoder for prototyping. No ML, just consistent hashing."""
    
    def __init__(self, dim: int = 128):
        self.dim = dim
        
    def encode(self, texts):
        """Hash-based encoding - fast but not semantic."""
        if isinstance(texts, str):
            texts = [texts]
            
        encodings = []
        for text in texts:
            # Use MD5 hash for consistency
            hash_obj = hashlib.md5(text.encode())
            # Convert to numbers and normalize
            hash_bytes = hash_obj.digest()
            # Take first bytes and convert to floats
            nums = [b / 255.0 for b in hash_bytes[:self.dim//4]]
            # Repeat to fill dimension
            while len(nums) < self.dim:
                nums.extend(nums[:min(len(nums), self.dim - len(nums))])
            encodings.append(nums[:self.dim])
            
        return np.array(encodings) if len(encodings) > 1 else np.array(encodings[0])


def _get_sentence_transformer(model_name: str):
    """Lazy load sentence transformer."""
    global SentenceTransformer
    if SentenceTransformer is None:
        from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


@dataclass 
class VectorIndex:
    """A single vector index for a specific field/encoding."""
    name: str
    index: UsearchIndex
    field_extractor: Callable[[Any], str]
    embedding_dim: int
    
    
@dataclass
class Record:
    """A record in the RAG collection."""
    id: str
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    embeddings: Dict[str, List[float]] = field(default_factory=dict)
    

class SemanticRAGCollection:
    """
    A collection of records with multiple vector indices for semantic search.
    
    Supports:
    - Multiple vector indices per collection (different fields/encodings)
    - Field selection at init or runtime
    - Single record insertion
    - Borda count reranking across multiple vectors
    """
    
    def __init__(
        self,
        name: str,
        encoder_model: str = "all-MiniLM-L6-v2",
        index_path: Optional[str] = None,
        fast_mode: bool = True  # Use fast hash encoder by default for prototyping
    ):
        self.name = name
        self.encoder_model = encoder_model
        self.index_path = index_path
        self.fast_mode = fast_mode
        
        # Lazy initialization of encoder
        self._encoder = None
        
        # Storage
        self.records: Dict[str, Record] = {}
        self.vector_indices: Dict[str, VectorIndex] = {}
        self.record_ids: List[str] = []  # Ordered list for index mapping
        
        # Get embedding dimension
        if fast_mode:
            self.embedding_dim = 128  # Fixed for hash encoder
        else:
            sample_embedding = self.encoder.encode("sample")
            self.embedding_dim = len(sample_embedding)
    
    @property
    def encoder(self):
        """Lazy load encoder when first accessed."""
        if self._encoder is None:
            if self.fast_mode:
                print("🚀 Using FastHashEncoder for prototyping (fast startup, no semantic accuracy)")
                self._encoder = FastHashEncoder(dim=128)
            else:
                print(f"⏳ Loading SentenceTransformer model: {self.encoder_model}")
                self._encoder = _get_sentence_transformer(self.encoder_model)
        return self._encoder
    
    def create_index(
        self, 
        index_name: str, 
        field_extractor: Callable[[Any], str]
    ) -> VectorIndex:
        """Create a new vector index for a specific field extraction."""
        index = UsearchIndex(ndim=self.embedding_dim, metric=MetricKind.Cos)
        
        vector_index = VectorIndex(
            name=index_name,
            index=index,
            field_extractor=field_extractor,
            embedding_dim=self.embedding_dim
        )
        
        self.vector_indices[index_name] = vector_index
        
        # If we have existing records, encode them for this new index
        if self.records:
            self._rebuild_index(index_name)
            
        return vector_index
    
    def add_record(
        self, 
        data: Any, 
        record_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a single record to the collection."""
        if record_id is None:
            record_id = str(uuid.uuid4())
            
        record = Record(
            id=record_id,
            data=data,
            metadata=metadata or {}
        )
        
        # Add to records
        self.records[record_id] = record
        if record_id not in self.record_ids:
            self.record_ids.append(record_id)
        
        # Encode for all existing indices
        record_idx = self.record_ids.index(record_id)
        for index_name, vector_index in self.vector_indices.items():
            text = vector_index.field_extractor(data)
            embedding = self.encoder.encode(text)
            record.embeddings[index_name] = embedding.tolist()
            vector_index.index.add(record_idx, embedding)
            
        return record_id
    
    def add_records_bulk(
        self, 
        data_list: List[Any],
        metadata_list: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """Add multiple records efficiently."""
        record_ids = []
        metadata_list = metadata_list or [{}] * len(data_list)
        
        for i, (data, metadata) in enumerate(zip(data_list, metadata_list)):
            record_id = self.add_record(data, metadata=metadata)
            record_ids.append(record_id)
            
        return record_ids
    
    def _rebuild_index(self, index_name: str):
        """Rebuild a specific index for all existing records."""
        vector_index = self.vector_indices[index_name]
        
        # Clear the index
        vector_index.index = UsearchIndex(ndim=self.embedding_dim, metric=MetricKind.Cos)
        
        # Re-encode all records
        for i, record_id in enumerate(self.record_ids):
            record = self.records[record_id]
            text = vector_index.field_extractor(record.data)
            embedding = self.encoder.encode(text)
            record.embeddings[index_name] = embedding.tolist()
            vector_index.index.add(i, embedding)
    
    def search_single_index(
        self,
        query: str,
        index_name: str,
        top_k: int = 10,
        filter_fn: Optional[Callable[[Record], bool]] = None
    ) -> List[Dict[str, Any]]:
        """Search using a single vector index."""
        if index_name not in self.vector_indices:
            raise ValueError(f"Index '{index_name}' not found")
            
        vector_index = self.vector_indices[index_name]
        query_embedding = self.encoder.encode(query)
        
        # Search with extra results for filtering
        search_k = top_k * 3 if filter_fn else top_k
        matches = vector_index.index.search(query_embedding, search_k)
        
        results = []
        for match in matches:
            record_idx = match.key
            if record_idx < len(self.record_ids):
                record_id = self.record_ids[record_idx]
                record = self.records[record_id]
                
                # Apply filter if provided
                if filter_fn and not filter_fn(record):
                    continue
                
                similarity = 1 - float(match.distance)  # Convert cosine distance to similarity
                
                result = {
                    "record_id": record_id,
                    "data": record.data,
                    "metadata": record.metadata,
                    "similarity": similarity,
                    "distance": float(match.distance),
                    "index_used": index_name
                }
                results.append(result)
                
                if len(results) >= top_k:
                    break
                    
        return results
    
    def search_multi_index(
        self,
        query: str,
        index_names: List[str],
        top_k: int = 10,
        rerank_method: str = "borda",
        filter_fn: Optional[Callable[[Record], bool]] = None
    ) -> List[Dict[str, Any]]:
        """Search across multiple indices with reranking."""
        if not index_names:
            raise ValueError("Must provide at least one index name")
            
        # Get results from each index
        all_results = {}
        for index_name in index_names:
            results = self.search_single_index(query, index_name, top_k * 2, filter_fn)
            for result in results:
                record_id = result["record_id"]
                if record_id not in all_results:
                    all_results[record_id] = {
                        "record_id": record_id,
                        "data": result["data"],
                        "metadata": result["metadata"],
                        "scores": {},
                        "ranks": {},
                        "indices_used": []
                    }
                all_results[record_id]["scores"][index_name] = result["similarity"]
                all_results[record_id]["indices_used"].append(index_name)
        
        # Calculate ranks for each index
        for index_name in index_names:
            # Sort by score for this index
            sorted_results = sorted(
                [r for r in all_results.values() if index_name in r["scores"]],
                key=lambda x: x["scores"][index_name],
                reverse=True
            )
            # Assign ranks
            for rank, result in enumerate(sorted_results):
                result["ranks"][index_name] = rank
        
        # Apply reranking
        if rerank_method == "borda":
            final_results = self._borda_count_rerank(list(all_results.values()), index_names)
        elif rerank_method == "avg_score":
            final_results = self._avg_score_rerank(list(all_results.values()), index_names)
        else:
            raise ValueError(f"Unknown rerank method: {rerank_method}")
            
        return final_results[:top_k]
    
    def _borda_count_rerank(self, results: List[Dict], index_names: List[str]) -> List[Dict]:
        """Rerank using Borda count method."""
        for result in results:
            borda_score = 0
            num_indices = len([idx for idx in index_names if idx in result["ranks"]])
            
            for index_name in index_names:
                if index_name in result["ranks"]:
                    # Higher rank = lower number, so invert for Borda count
                    rank = result["ranks"][index_name]
                    max_rank = len(results) - 1
                    borda_score += max_rank - rank
                    
            result["final_score"] = borda_score / num_indices if num_indices > 0 else 0
            result["rerank_method"] = "borda_count"
            
        return sorted(results, key=lambda x: x["final_score"], reverse=True)
    
    def _avg_score_rerank(self, results: List[Dict], index_names: List[str]) -> List[Dict]:
        """Rerank using average similarity scores."""
        for result in results:
            scores = [result["scores"][idx] for idx in index_names if idx in result["scores"]]
            result["final_score"] = sum(scores) / len(scores) if scores else 0
            result["rerank_method"] = "avg_score"
            
        return sorted(results, key=lambda x: x["final_score"], reverse=True)
    
    def switch_to_semantic_mode(self, encoder_model: Optional[str] = None):
        """Switch from fast hash mode to semantic embeddings. Requires rebuilding indices."""
        if not self.fast_mode:
            print("Already in semantic mode")
            return
            
        print("⚠️  Switching to semantic mode - will rebuild all indices")
        self.fast_mode = False
        if encoder_model:
            self.encoder_model = encoder_model
        
        # Clear current encoder
        self._encoder = None
        
        # Update embedding dimension
        sample_embedding = self.encoder.encode("sample")
        self.embedding_dim = len(sample_embedding)
        
        # Rebuild all indices
        for index_name in list(self.vector_indices.keys()):
            self._rebuild_index(index_name)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        encoder_info = "FastHashEncoder" if self.fast_mode else f"SentenceTransformer:{self.encoder_model}"
        return {
            "name": self.name,
            "total_records": len(self.records),
            "vector_indices": list(self.vector_indices.keys()),
            "encoder_type": encoder_info,
            "embedding_dim": self.embedding_dim,
            "fast_mode": self.fast_mode
        }
    
    def save(self, path: str):
        """Save collection to disk."""
        os.makedirs(path, exist_ok=True)
        
        # Save vector indices
        for index_name, vector_index in self.vector_indices.items():
            index_path = os.path.join(path, f"{index_name}.usearch")
            vector_index.index.save(index_path)
        
        # Save metadata and records
        metadata = {
            "name": self.name,
            "record_ids": self.record_ids,
            "records": {
                record_id: {
                    "data": record.data,
                    "metadata": record.metadata,
                    "embeddings": record.embeddings
                }
                for record_id, record in self.records.items()
            },
            "stats": self.get_stats()
        }
        
        with open(os.path.join(path, "collection.json"), 'w') as f:
            json.dump(metadata, f, indent=2, default=str)


class RAGFactory:
    """Factory for creating semantic RAG collections from different data types."""
    
    @staticmethod
    def from_pydantic(
        models: List[BaseModel],
        collection_name: str = "pydantic_collection",
        field_encodings: Optional[Dict[str, Union[str, List[str]]]] = None,
        **kwargs
    ) -> SemanticRAGCollection:
        """
        Create RAG collection from Pydantic models.
        
        Args:
            models: List of Pydantic model instances
            collection_name: Name for the collection
            field_encodings: Dict mapping index names to field names or list of field names
                Example: {"title": "title", "content": ["description", "reasoning"]}
        """
        collection = SemanticRAGCollection(collection_name, **kwargs)
        
        # Default field encodings if not provided
        if field_encodings is None and models:
            sample_model = models[0]
            field_names = list(sample_model.model_fields.keys()) if hasattr(sample_model, 'model_fields') else []
            field_encodings = {field: field for field in field_names if isinstance(getattr(sample_model, field, None), str)}
        
        # Create indices for each field encoding
        for index_name, fields in (field_encodings or {}).items():
            if isinstance(fields, str):
                fields = [fields]
                
            def make_extractor(field_list):
                def extractor(model):
                    parts = []
                    for field_name in field_list:
                        value = getattr(model, field_name, None)
                        if value is not None:
                            parts.append(str(value))
                    return " | ".join(parts)
                return extractor
            
            collection.create_index(index_name, make_extractor(fields))
        
        # Add all models
        collection.add_records_bulk(models)
        return collection
    
    @staticmethod
    def from_lists(
        data: List[List[str]],
        collection_name: str = "list_collection", 
        column_encodings: Optional[Dict[str, Union[int, List[int]]]] = None,
        **kwargs
    ) -> SemanticRAGCollection:
        """
        Create RAG collection from list of lists.
        
        Args:
            data: List of string lists
            collection_name: Name for the collection
            column_encodings: Dict mapping index names to column indices
                Example: {"title": 0, "content": [1, 2, 3]}
        """
        collection = SemanticRAGCollection(collection_name, **kwargs)
        
        # Default encodings if not provided
        if column_encodings is None and data:
            max_cols = max(len(row) for row in data) if data else 0
            column_encodings = {f"col_{i}": i for i in range(min(max_cols, 5))}  # First 5 columns
            if max_cols > 1:
                column_encodings["all"] = list(range(max_cols))  # All columns combined
        
        # Create indices
        for index_name, cols in (column_encodings or {}).items():
            if isinstance(cols, int):
                cols = [cols]
                
            def make_extractor(col_list):
                def extractor(row_list):
                    parts = []
                    for col_idx in col_list:
                        if col_idx < len(row_list) and row_list[col_idx]:
                            parts.append(str(row_list[col_idx]))
                    return " | ".join(parts)
                return extractor
            
            collection.create_index(index_name, make_extractor(cols))
        
        # Add all lists
        collection.add_records_bulk(data)
        return collection
    
    @staticmethod
    def from_dataframe(
        df: pd.DataFrame,
        collection_name: str = "dataframe_collection",
        column_encodings: Optional[Dict[str, Union[str, List[str]]]] = None,
        **kwargs
    ) -> SemanticRAGCollection:
        """
        Create RAG collection from DataFrame.
        
        Args:
            df: pandas DataFrame
            collection_name: Name for the collection
            column_encodings: Dict mapping index names to column names
                Example: {"title": "name", "content": ["description", "details"]}
        """
        collection = SemanticRAGCollection(collection_name, **kwargs)
        
        # Default encodings if not provided
        if column_encodings is None:
            text_cols = [col for col in df.columns if df[col].dtype == 'object']
            column_encodings = {col: col for col in text_cols[:5]}  # First 5 text columns
            if len(text_cols) > 1:
                column_encodings["all_text"] = text_cols  # All text columns combined
        
        # Create indices
        for index_name, cols in (column_encodings or {}).items():
            if isinstance(cols, str):
                cols = [cols]
                
            def make_extractor(col_list):
                def extractor(row_series):
                    parts = []
                    for col_name in col_list:
                        if col_name in row_series and pd.notna(row_series[col_name]):
                            parts.append(str(row_series[col_name]))
                    return " | ".join(parts)
                return extractor
            
            collection.create_index(index_name, make_extractor(cols))
        
        # Add all rows
        rows = [row for _, row in df.iterrows()]
        collection.add_records_bulk(rows)
        return collection


# Convenience functions
def create_pydantic_rag(models: List[BaseModel], **kwargs) -> SemanticRAGCollection:
    """Convenience function to create Pydantic RAG collection."""
    return RAGFactory.from_pydantic(models, **kwargs)

def create_list_rag(data: List[List[str]], **kwargs) -> SemanticRAGCollection:
    """Convenience function to create list RAG collection."""
    return RAGFactory.from_lists(data, **kwargs)

def create_dataframe_rag(df: pd.DataFrame, **kwargs) -> SemanticRAGCollection:
    """Convenience function to create DataFrame RAG collection."""
    return RAGFactory.from_dataframe(df, **kwargs)


# Example usage
if __name__ == "__main__":
    print("🚀 Fast RAG System Demo")
    
    # Example with list of lists and custom field encodings
    sample_data = [
        ["Stock Trading Bot", "financial", "Analyzes stocks and makes trades", "high complexity"],
        ["Weather Alerts", "weather", "Sends weather notifications", "low complexity"],
        ["Data Pipeline", "data", "Processes CSV files", "medium complexity"],
        ["Web Scraper", "web", "Extracts data from websites", "medium complexity"],
        ["Email Bot", "automation", "Sends automated emails", "low complexity"],
    ]
    
    # Create collection with fast mode (default)
    print("\n1. Creating collection in fast mode...")
    rag = RAGFactory.from_lists(
        sample_data,
        collection_name="workflows",
        column_encodings={
            "title": 0,
            "category": 1, 
            "description": 2,
            "full": [0, 2, 3]  # title + description + complexity
        },
        fast_mode=True  # Fast startup, no semantic accuracy
    )
    
    print(f"Stats: {rag.get_stats()}")
    
    # Search single index
    print("\n2. Search 'title' index for 'trading':")
    results = rag.search_single_index("trading", "title", top_k=2)
    for r in results:
        print(f"  {r['similarity']:.3f}: {r['data'][0]}")
    
    # Search multiple indices with Borda count
    print("\n3. Search multiple indices for 'financial data':")
    results = rag.search_multi_index("financial data", ["title", "category", "full"], top_k=2)
    for r in results:
        print(f"  {r['final_score']:.3f}: {r['data'][0]} (indices: {r['indices_used']})")
    
    # Add single record
    print("\n4. Adding single record:")
    new_id = rag.add_record(["Chat Bot", "ai", "Conversational AI assistant", "high complexity"])
    print(f"Added record: {new_id}")
    
    # Switch to semantic mode when needed
    print("\n5. To switch to semantic mode when ready:")
    print("   rag.switch_to_semantic_mode('all-MiniLM-L6-v2')")
    print("   # This will download model and rebuild indices with real semantic embeddings")