#!/usr/bin/env python3
"""
Quick demo and test of the test analytics service
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from iointel.src.utilities.test_analytics_service import TestAnalyticsService, get_system_coverage, search_tests_quick


def main():
    print("🧪 Test Analytics Service Demo")
    print("=" * 50)
    
    try:
        # Initialize service
        print("🔧 Initializing analytics service...")
        service = TestAnalyticsService()
        
        # Get system coverage
        print("\n📊 System Coverage Metrics:")
        coverage = service.get_system_coverage_metrics()
        print(f"  • Total workflows: {coverage.total_workflows}")
        print(f"  • Tested workflows: {coverage.tested_workflows}")
        print(f"  • Coverage percentage: {coverage.coverage_percentage:.1f}%")
        print(f"  • Success rate: {coverage.success_rate:.1f}%")
        print(f"  • Total tests: {coverage.passing_tests + coverage.failing_tests}")
        
        print("\n📋 Test Distribution by Layer:")
        for layer, count in coverage.test_count_by_layer.items():
            print(f"  • {layer}: {count}")
        
        print("\n📂 Test Distribution by Category:")
        for category, count in coverage.test_count_by_category.items():
            print(f"  • {category}: {count}")
        
        # Test search functionality
        print("\n🔍 Testing Search Functionality:")
        search_queries = [
            "gate pattern",
            "routing",
            "sla enforcement",
            "decision"
        ]
        
        for query in search_queries:
            print(f"\n  Searching for: '{query}'")
            results = service.search_tests(query, limit=3)
            print(f"  Found {len(results)} results")
            for result in results:
                print(f"    • {result.test_case.name} (score: {result.relevance_score:.1f})")
                print(f"      Layer: {result.test_case.layer.value}, Category: {result.test_case.category}")
        
        # Get quality scores
        print("\n🏆 Workflow Quality Scores:")
        quality_scores = service.get_workflow_quality_scores()
        print(f"  Found {len(quality_scores)} workflows")
        for score in quality_scores[:5]:  # Show top 5
            print(f"  • {score.workflow_title}")
            print(f"    Overall: {score.overall_score:.1f}, Coverage: {score.coverage_score:.1f}, Quality: {score.quality_score:.1f}")
            print(f"    Status: {score.status}, Tests: {score.test_count}")
        
        # Get gaps analysis
        print("\n📈 Gaps Analysis:")
        gaps = service.get_test_gaps_analysis()
        print(f"  Total gaps identified: {gaps['summary']['total_gaps']}")
        print(f"  Priority areas: {gaps['summary']['priority_areas']}")
        
        for rec in gaps['recommendations']:
            print(f"  • {rec['type']}: {rec['message']}")
        
        print("\n✅ Analytics service working correctly!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")


if __name__ == "__main__":
    main()