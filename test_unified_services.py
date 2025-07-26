#!/usr/bin/env python3
"""
Test the unified services integration
"""

import requests
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_service_endpoints(base_url="http://localhost:8000"):
    """Test all integrated service endpoints."""
    
    print("🧪 Testing Unified Services Integration")
    print("=" * 50)
    print(f"Base URL: {base_url}")
    
    # Test main page
    print("\n1. Testing main page...")
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            print("✅ Main page accessible")
        else:
            print(f"❌ Main page failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Main page error: {e}")
    
    # Test test analytics endpoints
    print("\n2. Testing test analytics endpoints...")
    test_endpoints = [
        "/api/test-analytics/health",
        "/api/test-analytics/coverage", 
        "/api/test-analytics/dashboard-summary",
        "/api/test-analytics/layers",
        "/api/test-analytics/categories"
    ]
    
    for endpoint in test_endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}")
            if response.status_code == 200:
                print(f"✅ {endpoint}")
                if "coverage" in endpoint:
                    data = response.json()
                    print(f"   Coverage: {data.get('coverage_percentage', 0):.1f}%")
            else:
                print(f"❌ {endpoint}: {response.status_code}")
        except Exception as e:
            print(f"❌ {endpoint}: {e}")
    
    # Test workflow RAG endpoints
    print("\n3. Testing workflow RAG endpoints...")
    rag_endpoints = [
        "/api/workflow-rag/health",
        "/api/workflow-rag/stats",
        "/api/workflow-rag/"
    ]
    
    for endpoint in rag_endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}")
            if response.status_code == 200:
                print(f"✅ {endpoint}")
                if "stats" in endpoint:
                    data = response.json()
                    print(f"   Workflows: {data.get('total_records', 0)}")
            else:
                print(f"❌ {endpoint}: {response.status_code}")
        except Exception as e:
            print(f"❌ {endpoint}: {e}")
    
    # Test search functionality
    print("\n4. Testing search functionality...")
    
    # Test analytics search
    try:
        response = requests.get(f"{base_url}/api/test-analytics/search?query=gate pattern&limit=3")
        if response.status_code == 200:
            results = response.json()
            print(f"✅ Test search: found {len(results)} results")
            if results:
                print(f"   Top result: {results[0]['test_name']}")
        else:
            print(f"❌ Test search failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Test search error: {e}")
    
    # Test workflow RAG search
    try:
        response = requests.get(f"{base_url}/api/workflow-rag/search?query=stock trading&top_k=3")
        if response.status_code == 200:
            results = response.json()
            print(f"✅ Workflow search: found {results.get('total_found', 0)} results")
            if results.get('results'):
                print(f"   Top result: {results['results'][0]['workflow_spec']['title']}")
        else:
            print(f"❌ Workflow search failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Workflow search error: {e}")
    
    # Test static files
    print("\n5. Testing static files...")
    try:
        response = requests.get(f"{base_url}/test-analytics")
        if response.status_code == 200:
            print("✅ Test analytics panel accessible")
        else:
            print(f"❌ Test analytics panel: {response.status_code}")
    except Exception as e:
        print(f"❌ Test analytics panel error: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 Integration test complete!")
    print("\nTo access the services:")
    print(f"• Main page: {base_url}/")
    print(f"• Test Analytics: {base_url}/test-analytics")
    print(f"• API Docs: {base_url}/docs")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test unified services")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL for testing")
    args = parser.parse_args()
    
    test_service_endpoints(args.url)