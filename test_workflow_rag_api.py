#!/usr/bin/env python3
"""
Test script for Workflow RAG API
================================

Demonstrates searching saved workflows using semantic similarity.
"""

import requests
import json
from typing import List, Dict, Any


class WorkflowRAGClient:
    """Client for interacting with Workflow RAG API."""
    
    def __init__(self, base_url: str = "http://localhost:8101"):
        self.base_url = base_url
    
    def search(self, query: str, top_k: int = 5, indices: List[str] = None) -> Dict[str, Any]:
        """Search workflows by semantic similarity."""
        endpoint = f"{self.base_url}/search"
        
        params = {
            "query": query,
            "top_k": top_k
        }
        
        if indices:
            params["indices"] = ",".join(indices)
        
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        return response.json()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG collection statistics."""
        response = requests.get(f"{self.base_url}/stats")
        response.raise_for_status()
        return response.json()
    
    def refresh_index(self) -> Dict[str, Any]:
        """Refresh the RAG index."""
        response = requests.post(f"{self.base_url}/refresh")
        response.raise_for_status()
        return response.json()


def main():
    """Test the Workflow RAG API."""
    print("🔍 Workflow RAG API Test")
    print("=" * 50)
    
    client = WorkflowRAGClient()
    
    # Get stats
    print("\n📊 RAG Collection Stats:")
    try:
        stats = client.get_stats()
        print(json.dumps(stats, indent=2))
    except Exception as e:
        print(f"❌ Error getting stats: {e}")
        print("Make sure the RAG service is running: uv run python -m iointel.src.web.workflow_rag_service")
        return
    
    # Test queries
    test_queries = [
        ("stock trading", ["both"]),
        ("weather analysis", ["title"]),
        ("data processing", ["description"]),
        ("conditional routing", ["both"]),
        ("email notification", None),  # Use default
    ]
    
    print("\n🔍 Testing Semantic Search:")
    for query, indices in test_queries:
        print(f"\n{'='*50}")
        print(f"Query: '{query}'")
        if indices:
            print(f"Indices: {indices}")
        
        try:
            results = client.search(query, top_k=3, indices=indices)
            print(f"Found {results['total_found']} results\n")
            
            for i, result in enumerate(results['results'], 1):
                workflow = result['workflow_spec']
                print(f"{i}. {workflow['title']} (score: {result['similarity_score']:.3f})")
                print(f"   Description: {workflow['description']}")
                print(f"   ID: {workflow['id']}")
                print(f"   Matched on: {result['matched_fields']['indices_used']}")
                
        except Exception as e:
            print(f"❌ Error searching: {e}")
    
    # Test multi-index search
    print(f"\n{'='*50}")
    print("🔍 Multi-Index Search Test:")
    print("Query: 'analyze financial data'")
    print("Indices: ['title', 'description']")
    
    try:
        results = client.search("analyze financial data", top_k=5, indices=["title", "description"])
        print(f"\nFound {results['total_found']} results using Borda count reranking\n")
        
        for i, result in enumerate(results['results'], 1):
            workflow = result['workflow_spec']
            print(f"{i}. {workflow['title']} (score: {result['similarity_score']:.3f})")
            
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()