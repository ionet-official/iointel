"""
Test Analytics Service
====================

Provides analytics, coverage metrics, and search functionality for the unified test system.
Supports RAG-based test search and comprehensive coverage reporting.
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from collections import Counter

from iointel.src.agent_methods.data_models.workflow_spec import WorkflowSpec
from iointel.src.utilities.workflow_test_repository import WorkflowTestRepository, WorkflowTestCase


@dataclass
class TestCoverageMetrics:
    """Test coverage metrics for a workflow or system component."""
    total_workflows: int
    tested_workflows: int
    untested_workflows: int
    coverage_percentage: float
    test_count_by_layer: Dict[str, int]
    test_count_by_category: Dict[str, int]
    passing_tests: int
    failing_tests: int
    success_rate: float


@dataclass
class WorkflowQualityScore:
    """Quality score for a workflow based on test coverage and results."""
    workflow_id: str
    workflow_title: str
    test_count: int
    coverage_score: float  # 0-100 based on test coverage
    quality_score: float   # 0-100 based on test results
    overall_score: float   # Combined score
    last_tested: Optional[datetime]
    status: str           # "untested", "passing", "failing", "mixed"
    recommendations: List[str]


@dataclass
class TestSearchResult:
    """Result from RAG-based test search."""
    test_case: WorkflowTestCase
    relevance_score: float
    matching_fields: List[str]
    snippet: str


class TestAnalyticsService:
    """
    Service for test analytics, coverage metrics, and intelligent search.
    
    Provides comprehensive insights into test quality, coverage gaps,
    and workflow validation status.
    """
    
    def __init__(self, test_repository_path: str = "smart_test_repository"):
        self.repo = WorkflowTestRepository(test_repository_path)
        self.repository_path = Path(test_repository_path)
        
    def get_system_coverage_metrics(self) -> TestCoverageMetrics:
        """Get overall system test coverage metrics."""
        all_tests = self.repo.get_all_tests()
        
        # Count tests by layer and category
        test_count_by_layer = Counter()
        test_count_by_category = Counter()
        passing_tests = 0
        failing_tests = 0
        
        for test in all_tests:
            test_count_by_layer[test.layer.value] += 1
            test_count_by_category[test.category] += 1
            
            # Analyze test results if available
            if hasattr(test, 'last_result') and test.last_result:
                if test.last_result.get('passed', False):
                    passing_tests += 1
                else:
                    failing_tests += 1
        
        total_tests = len(all_tests)
        success_rate = (passing_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Calculate meaningful coverage for current test structure
        # Since tests aren't yet linked to specific workflows, calculate based on test patterns
        unique_categories = {test.category for test in all_tests}
        {test.layer.value for test in all_tests}
        
        # Use test categories as proxy for "workflow types" covered
        tested_workflows = len(unique_categories)
        total_estimated_workflows = max(tested_workflows, 15)  # Estimate based on categories
        coverage_percentage = (tested_workflows / total_estimated_workflows * 100) if total_estimated_workflows > 0 else 0
        
        return TestCoverageMetrics(
            total_workflows=total_estimated_workflows,
            tested_workflows=tested_workflows,
            untested_workflows=total_estimated_workflows - tested_workflows,
            coverage_percentage=coverage_percentage,
            test_count_by_layer=dict(test_count_by_layer),
            test_count_by_category=dict(test_count_by_category),
            passing_tests=passing_tests,
            failing_tests=failing_tests,
            success_rate=success_rate
        )
    
    def get_workflow_quality_scores(self, workflows: Optional[List[WorkflowSpec]] = None) -> List[WorkflowQualityScore]:
        """
        Calculate quality scores for workflows based on test coverage and results.
        
        Args:
            workflows: List of WorkflowSpec objects. If None, uses workflow IDs from tests.
        """
        quality_scores = []
        
        if workflows is None:
            # Extract workflow IDs from test contexts
            all_tests = self.repo.get_all_tests()
            workflow_ids = set()
            for test in all_tests:
                if test.context and 'workflow_id' in test.context:
                    workflow_ids.add(test.context['workflow_id'])
            
            # Create minimal WorkflowSpec objects for scoring
            workflows = [
                WorkflowSpec(id=wf_id, title=f"Workflow {wf_id}", description="", nodes=[], edges=[])
                for wf_id in workflow_ids
            ]
        
        for workflow in workflows:
            score = self._calculate_workflow_quality_score(workflow)
            quality_scores.append(score)
        
        # Sort by overall score descending
        quality_scores.sort(key=lambda x: x.overall_score, reverse=True)
        return quality_scores
    
    def _calculate_workflow_quality_score(self, workflow: WorkflowSpec) -> WorkflowQualityScore:
        """Calculate quality score for a single workflow."""
        # Get tests associated with this workflow
        workflow_tests = []
        all_tests = self.repo.get_all_tests()
        
        for test in all_tests:
            if (test.context and 
                test.context.get('workflow_id') == workflow.id):
                workflow_tests.append(test)
        
        test_count = len(workflow_tests)
        
        # Calculate coverage score (0-100)
        # Based on test count and layer diversity
        coverage_score = min(test_count * 10, 100)  # 10 points per test, max 100
        
        # Bonus for covering multiple layers
        layers_covered = {test.layer.value for test in workflow_tests}
        layer_bonus = len(layers_covered) * 5  # 5 points per layer
        coverage_score = min(coverage_score + layer_bonus, 100)
        
        # Calculate quality score based on test results
        if hasattr(workflow, 'test_alignment') and workflow.test_alignment.test_results:
            test_results = workflow.test_alignment.test_results
            passed_tests = sum(1 for result in test_results if result.passed)
            total_results = len(test_results)
            quality_score = (passed_tests / total_results * 100) if total_results > 0 else 0
            last_tested = max(result.executed_at for result in test_results) if test_results else None
            
            # Determine status
            if total_results == 0:
                status = "untested"
            elif passed_tests == total_results:
                status = "passing"
            elif passed_tests == 0:
                status = "failing"
            else:
                status = "mixed"
        else:
            quality_score = 0
            last_tested = None
            status = "untested"
        
        # Calculate overall score (weighted average)
        overall_score = (coverage_score * 0.4 + quality_score * 0.6)
        
        # Generate recommendations
        recommendations = []
        if test_count == 0:
            recommendations.append("Add basic test coverage")
        if test_count < 3:
            recommendations.append("Increase test count for better coverage")
        if len(layers_covered) < 2:
            recommendations.append("Add tests across multiple layers (logical, agentic, orchestration)")
        if quality_score < 80:
            recommendations.append("Fix failing tests to improve quality")
        if not last_tested or (datetime.now() - last_tested).days > 7:
            recommendations.append("Run recent test validation")
        
        return WorkflowQualityScore(
            workflow_id=workflow.id,
            workflow_title=workflow.title,
            test_count=test_count,
            coverage_score=coverage_score,
            quality_score=quality_score,
            overall_score=overall_score,
            last_tested=last_tested,
            status=status,
            recommendations=recommendations
        )
    
    def search_tests(self, query: str, layer: Optional[str] = None, 
                    category: Optional[str] = None, limit: int = 10) -> List[TestSearchResult]:
        """
        RAG-based search through test cases.
        
        Args:
            query: Search query (natural language)
            layer: Optional layer filter
            category: Optional category filter
            limit: Maximum number of results
            
        Returns:
            List of TestSearchResult objects with relevance scores
        """
        all_tests = self.repo.get_all_tests()
        
        # Apply filters
        filtered_tests = []
        for test in all_tests:
            if layer and test.layer.value != layer:
                continue
            if category and test.category != category:
                continue
            filtered_tests.append(test)
        
        # Simple text-based relevance scoring
        query_lower = query.lower()
        query_terms = query_lower.split()
        
        results = []
        for test in filtered_tests:
            relevance_score, matching_fields, snippet = self._calculate_test_relevance(
                test, query_lower, query_terms
            )
            
            if relevance_score > 0:
                results.append(TestSearchResult(
                    test_case=test,
                    relevance_score=relevance_score,
                    matching_fields=matching_fields,
                    snippet=snippet
                ))
        
        # Sort by relevance and limit
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results[:limit]
    
    def _calculate_test_relevance(self, test: WorkflowTestCase, query_lower: str, 
                                 query_terms: List[str]) -> Tuple[float, List[str], str]:
        """Calculate relevance score for a test case against search query."""
        score = 0.0
        matching_fields = []
        snippets = []
        
        # Check name (high weight)
        if query_lower in test.name.lower():
            score += 10.0
            matching_fields.append("name")
            snippets.append(f"Name: {test.name}")
        
        # Check description (medium weight)
        if query_lower in test.description.lower():
            score += 5.0
            matching_fields.append("description")
            # Create snippet with context
            desc_lower = test.description.lower()
            start_idx = max(0, desc_lower.find(query_lower) - 20)
            end_idx = min(len(test.description), start_idx + len(query_lower) + 40)
            snippet = test.description[start_idx:end_idx]
            if start_idx > 0:
                snippet = "..." + snippet
            if end_idx < len(test.description):
                snippet = snippet + "..."
            snippets.append(f"Description: {snippet}")
        
        # Check tags (medium weight)
        for tag in test.tags:
            if query_lower in tag.lower():
                score += 3.0
                matching_fields.append("tags")
                snippets.append(f"Tag: {tag}")
        
        # Check category (low weight)
        if query_lower in test.category.lower():
            score += 2.0
            matching_fields.append("category")
            snippets.append(f"Category: {test.category}")
        
        # Check user prompt if available (medium weight)
        if test.user_prompt and query_lower in test.user_prompt.lower():
            score += 4.0
            matching_fields.append("user_prompt")
            # Create snippet
            prompt_lower = test.user_prompt.lower()
            start_idx = max(0, prompt_lower.find(query_lower) - 15)
            end_idx = min(len(test.user_prompt), start_idx + len(query_lower) + 30)
            snippet = test.user_prompt[start_idx:end_idx]
            if start_idx > 0:
                snippet = "..." + snippet
            if end_idx < len(test.user_prompt):
                snippet = snippet + "..."
            snippets.append(f"Prompt: {snippet}")
        
        # Bonus for multiple term matches
        for term in query_terms:
            term_matches = 0
            searchable_text = f"{test.name} {test.description} {' '.join(test.tags)}".lower()
            term_matches += searchable_text.count(term)
            score += term_matches * 0.5
        
        # Create combined snippet
        combined_snippet = " | ".join(snippets[:3])  # Limit to first 3 matches
        
        return score, matching_fields, combined_snippet
    
    def get_test_gaps_analysis(self) -> Dict[str, Any]:
        """
        Analyze gaps in test coverage and suggest areas for improvement.
        
        Returns:
            Dictionary with gap analysis and recommendations
        """
        all_tests = self.repo.get_all_tests()
        
        # Analyze by layer
        layer_counts = Counter(test.layer.value for test in all_tests)
        layer_gaps = []
        
        expected_layers = ["logical", "agentic", "orchestration", "feedback"]
        for layer in expected_layers:
            count = layer_counts.get(layer, 0)
            if count < 5:  # Arbitrary threshold
                layer_gaps.append({
                    "layer": layer,
                    "current_count": count,
                    "recommended_minimum": 5,
                    "priority": "high" if count == 0 else "medium"
                })
        
        # Analyze by category
        category_counts = Counter(test.category for test in all_tests)
        category_gaps = []
        
        important_categories = [
            "routing_validation", "sla_enforcement", "data_flow", 
            "tool_integration", "workflow_generation", "gate_pattern"
        ]
        
        for category in important_categories:
            count = category_counts.get(category, 0)
            if count < 3:  # Arbitrary threshold
                category_gaps.append({
                    "category": category,
                    "current_count": count,
                    "recommended_minimum": 3,
                    "priority": "high" if count == 0 else "medium"
                })
        
        # Generate recommendations
        recommendations = []
        
        if layer_gaps:
            recommendations.append({
                "type": "layer_coverage",
                "message": f"Add more tests in {len(layer_gaps)} layer(s)",
                "details": layer_gaps
            })
        
        if category_gaps:
            recommendations.append({
                "type": "category_coverage", 
                "message": f"Improve coverage in {len(category_gaps)} category(s)",
                "details": category_gaps
            })
        
        # Check for recent test activity
        recent_tests = []
        for test in all_tests:
            if test.created_at:
                try:
                    # Handle both datetime objects and string timestamps
                    if isinstance(test.created_at, str):
                        test_date = datetime.fromisoformat(test.created_at.replace('Z', '+00:00'))
                    else:
                        test_date = test.created_at
                    
                    if (datetime.now() - test_date).days <= 7:
                        recent_tests.append(test)
                except (ValueError, TypeError):
                    # Skip tests with invalid timestamps
                    continue
        
        if len(recent_tests) < 2:
            recommendations.append({
                "type": "test_activity",
                "message": "Low recent test activity - consider adding new tests",
                "details": {"recent_tests": len(recent_tests), "recommended": 5}
            })
        
        return {
            "layer_analysis": {
                "gaps": layer_gaps,
                "distribution": dict(layer_counts)
            },
            "category_analysis": {
                "gaps": category_gaps,
                "distribution": dict(category_counts)
            },
            "recommendations": recommendations,
            "summary": {
                "total_gaps": len(layer_gaps) + len(category_gaps),
                "priority_areas": [gap["layer"] for gap in layer_gaps if gap["priority"] == "high"] +
                                [gap["category"] for gap in category_gaps if gap["priority"] == "high"]
            }
        }
    
    def export_analytics_report(self, output_path: str = "test_analytics_report.json") -> str:
        """
        Export comprehensive analytics report to JSON file.
        
        Args:
            output_path: Path for output file
            
        Returns:
            Path to generated report file
        """
        report = {
            "generated_at": datetime.now().isoformat(),
            "system_metrics": self.get_system_coverage_metrics().__dict__,
            "quality_scores": [score.__dict__ for score in self.get_workflow_quality_scores()],
            "gaps_analysis": self.get_test_gaps_analysis(),
            "test_summary": {
                "total_tests": len(self.repo.get_all_tests()),
                "layers": list({test.layer.value for test in self.repo.get_all_tests()}),
                "categories": list({test.category for test in self.repo.get_all_tests()}),
            }
        }
        
        # Convert datetime objects to strings for JSON serialization
        def datetime_handler(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object {obj} is not JSON serializable")
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=datetime_handler)
        
        return output_path


# Convenience functions for quick access
def get_system_coverage() -> TestCoverageMetrics:
    """Quick access to system coverage metrics."""
    service = TestAnalyticsService()
    return service.get_system_coverage_metrics()


def search_tests_quick(query: str, limit: int = 5) -> List[TestSearchResult]:
    """Quick test search function."""
    service = TestAnalyticsService()
    return service.search_tests(query, limit=limit)


def generate_coverage_report(output_file: str = "coverage_report.json") -> str:
    """Generate and save coverage report."""
    service = TestAnalyticsService()
    return service.export_analytics_report(output_file)