"""
Test Analytics API
==================

Web API endpoints for test analytics, coverage metrics, and search functionality.
Provides data for the test analytics panel in the web interface.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional, Dict, Any
from pydantic import BaseModel

from ..utilities.test_analytics_service import (
    TestAnalyticsService, 
    TestCoverageMetrics, 
    WorkflowQualityScore,
    TestSearchResult
)
from ..utilities.helpers import make_logger

logger = make_logger(__name__)

# Create router for test analytics endpoints
test_analytics_router = APIRouter(prefix="/api/test-analytics", tags=["test-analytics"])

# Initialize analytics service
analytics_service = TestAnalyticsService()


# Pydantic models for API responses
class CoverageMetricsResponse(BaseModel):
    """Response model for coverage metrics."""
    total_workflows: int
    tested_workflows: int
    untested_workflows: int
    coverage_percentage: float
    test_count_by_layer: Dict[str, int]
    test_count_by_category: Dict[str, int]
    passing_tests: int
    failing_tests: int
    success_rate: float


class QualityScoreResponse(BaseModel):
    """Response model for workflow quality scores."""
    workflow_id: str
    workflow_title: str
    test_count: int
    coverage_score: float
    quality_score: float
    overall_score: float
    last_tested: Optional[str]
    status: str
    recommendations: List[str]


class TestSearchResponse(BaseModel):
    """Response model for test search results."""
    test_id: str
    test_name: str
    test_description: str
    test_layer: str
    test_category: str
    test_tags: List[str]
    relevance_score: float
    matching_fields: List[str]
    snippet: str
    created_at: Optional[str]


class GapsAnalysisResponse(BaseModel):
    """Response model for gaps analysis."""
    layer_analysis: Dict[str, Any]
    category_analysis: Dict[str, Any]
    recommendations: List[Dict[str, Any]]
    summary: Dict[str, Any]


@test_analytics_router.get("/coverage", response_model=CoverageMetricsResponse)
async def get_coverage_metrics():
    """
    Get system-wide test coverage metrics.
    
    Returns comprehensive coverage statistics including:
    - Total and tested workflows
    - Test distribution by layer and category
    - Pass/fail rates and success metrics
    """
    try:
        metrics = analytics_service.get_system_coverage_metrics()
        return CoverageMetricsResponse(
            total_workflows=metrics.total_workflows,
            tested_workflows=metrics.tested_workflows,
            untested_workflows=metrics.untested_workflows,
            coverage_percentage=metrics.coverage_percentage,
            test_count_by_layer=metrics.test_count_by_layer,
            test_count_by_category=metrics.test_count_by_category,
            passing_tests=metrics.passing_tests,
            failing_tests=metrics.failing_tests,
            success_rate=metrics.success_rate
        )
    except Exception as e:
        logger.error(f"Error getting coverage metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@test_analytics_router.get("/quality-scores", response_model=List[QualityScoreResponse])
async def get_quality_scores():
    """
    Get workflow quality scores based on test coverage and results.
    
    Returns quality analysis for all workflows including:
    - Coverage and quality scores (0-100)
    - Test count and validation status
    - Recommendations for improvement
    """
    try:
        scores = analytics_service.get_workflow_quality_scores()
        return [
            QualityScoreResponse(
                workflow_id=score.workflow_id,
                workflow_title=score.workflow_title,
                test_count=score.test_count,
                coverage_score=score.coverage_score,
                quality_score=score.quality_score,
                overall_score=score.overall_score,
                last_tested=score.last_tested.isoformat() if score.last_tested else None,
                status=score.status,
                recommendations=score.recommendations
            )
            for score in scores
        ]
    except Exception as e:
        logger.error(f"Error getting quality scores: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@test_analytics_router.get("/search", response_model=List[TestSearchResponse])
async def search_tests(
    query: str = Query(..., description="Search query for tests"),
    layer: Optional[str] = Query(None, description="Filter by test layer"),
    category: Optional[str] = Query(None, description="Filter by test category"),
    limit: int = Query(10, ge=1, le=50, description="Maximum number of results")
):
    """
    RAG-based search through test cases.
    
    Supports natural language queries and returns relevant tests with:
    - Relevance scoring and ranking
    - Highlighted matching fields
    - Context snippets from matches
    """
    try:
        results = analytics_service.search_tests(
            query=query,
            layer=layer,
            category=category,
            limit=limit
        )
        
        return [
            TestSearchResponse(
                test_id=result.test_case.id,
                test_name=result.test_case.name,
                test_description=result.test_case.description,
                test_layer=result.test_case.layer.value,
                test_category=result.test_case.category,
                test_tags=result.test_case.tags,
                relevance_score=result.relevance_score,
                matching_fields=result.matching_fields,
                snippet=result.snippet,
                created_at=result.test_case.created_at.isoformat() if result.test_case.created_at and hasattr(result.test_case.created_at, 'isoformat') else str(result.test_case.created_at) if result.test_case.created_at else None
            )
            for result in results
        ]
    except Exception as e:
        logger.error(f"Error searching tests: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@test_analytics_router.get("/gaps-analysis", response_model=GapsAnalysisResponse)
async def get_gaps_analysis():
    """
    Get test coverage gaps analysis.
    
    Identifies areas needing more test coverage including:
    - Layer and category coverage gaps
    - Recommendations for improvement
    - Priority areas for test development
    """
    try:
        analysis = analytics_service.get_test_gaps_analysis()
        return GapsAnalysisResponse(**analysis)
    except Exception as e:
        logger.error(f"Error getting gaps analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@test_analytics_router.get("/layers")
async def get_available_layers():
    """Get list of available test layers for filtering."""
    try:
        return {
            "layers": ["logical", "agentic", "orchestration", "feedback"],
            "descriptions": {
                "logical": "Structure validation, spec validation, edge cases (no LLM calls)",
                "agentic": "LLM workflow generation tests (generate_only calls)",
                "orchestration": "End-to-end execution tests (plan_and_execute calls)",
                "feedback": "User feedback and refinement workflow tests"
            }
        }
    except Exception as e:
        logger.error(f"Error getting layers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@test_analytics_router.get("/categories")
async def get_available_categories():
    """Get list of available test categories for filtering."""
    try:
        all_tests = analytics_service.repo.get_all_tests()
        categories = sorted(set(test.category for test in all_tests))
        
        # Add descriptions for common categories
        category_descriptions = {
            "routing_validation": "Tests for conditional logic and routing",
            "sla_enforcement": "Tests for SLA compliance and tool usage",
            "data_flow": "Tests for data passing between workflow nodes",
            "tool_integration": "Tests for tool usage and integration",
            "workflow_generation": "Tests for workflow planner functionality",
            "gate_pattern": "Tests for conditional gating patterns"
        }
        
        return {
            "categories": categories,
            "descriptions": category_descriptions
        }
    except Exception as e:
        logger.error(f"Error getting categories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@test_analytics_router.post("/export-report")
async def export_analytics_report():
    """
    Export comprehensive analytics report to JSON file.
    
    Generates a complete analytics report including all metrics,
    quality scores, gaps analysis, and test summaries.
    """
    try:
        report_path = analytics_service.export_analytics_report()
        return {
            "success": True,
            "report_path": report_path,
            "message": "Analytics report exported successfully"
        }
    except Exception as e:
        logger.error(f"Error exporting report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@test_analytics_router.get("/dashboard-summary")
async def get_dashboard_summary():
    """
    Get summary data for test analytics dashboard.
    
    Returns key metrics and highlights for dashboard display.
    """
    try:
        coverage = analytics_service.get_system_coverage_metrics()
        quality_scores = analytics_service.get_workflow_quality_scores()
        gaps = analytics_service.get_test_gaps_analysis()
        
        # Calculate summary statistics
        total_tests = coverage.passing_tests + coverage.failing_tests
        avg_quality_score = sum(score.overall_score for score in quality_scores) / len(quality_scores) if quality_scores else 0
        
        # Find top and bottom performers
        top_workflows = sorted(quality_scores, key=lambda x: x.overall_score, reverse=True)[:3]
        bottom_workflows = sorted(quality_scores, key=lambda x: x.overall_score)[:3]
        
        return {
            "overview": {
                "total_tests": total_tests,
                "success_rate": coverage.success_rate,
                "coverage_percentage": coverage.coverage_percentage,
                "avg_quality_score": round(avg_quality_score, 1)
            },
            "layer_distribution": coverage.test_count_by_layer,
            "category_distribution": coverage.test_count_by_category,
            "top_workflows": [
                {
                    "id": wf.workflow_id,
                    "title": wf.workflow_title,
                    "score": round(wf.overall_score, 1),
                    "status": wf.status
                }
                for wf in top_workflows
            ],
            "bottom_workflows": [
                {
                    "id": wf.workflow_id,
                    "title": wf.workflow_title,
                    "score": round(wf.overall_score, 1),
                    "status": wf.status
                }
                for wf in bottom_workflows
            ],
            "priority_gaps": gaps["summary"]["priority_areas"][:5],
            "recommendations_count": len(gaps["recommendations"]),
            "recent_activity": {
                "tests_added_this_week": 0,  # TODO: Calculate from test timestamps
                "last_test_run": None  # TODO: Get from test execution logs
            }
        }
    except Exception as e:
        logger.error(f"Error getting dashboard summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Health check endpoint
@test_analytics_router.get("/health")
async def health_check():
    """Health check for test analytics service."""
    try:
        # Basic service validation
        test_count = len(analytics_service.repo.get_all_tests())
        return {
            "status": "healthy",
            "service": "test_analytics",
            "test_repository_available": True,
            "total_tests": test_count,
            "timestamp": analytics_service.repo._get_current_timestamp()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": analytics_service.repo._get_current_timestamp()
        }