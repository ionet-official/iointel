"""
Workflow-Test Alignment System
==============================

This module provides automatic filtering and validation of workflows based on
their test alignment status. Only workflows that pass their tests are considered
"production-ready" and available in the app.

Key Features:
- Automatic test result recording
- Production-ready workflow filtering
- Test-to-workflow linking
- Validation status tracking
"""

from typing import List, Dict, Optional
from iointel.src.agent_methods.data_models.workflow_spec import WorkflowSpec, TestResult
from iointel.src.web.workflow_storage import WorkflowStorage
from iointel.src.utilities.workflow_test_repository import WorkflowTestRepository


class WorkflowAlignmentService:
    """Service for managing workflow-test alignment and filtering."""
    
    def __init__(self, workflow_storage: Optional[WorkflowStorage] = None):
        self.workflow_storage = workflow_storage or WorkflowStorage()
        self.test_repository = WorkflowTestRepository()
    
    def link_workflow_to_tests(self, workflow_id: str, test_ids: List[str]):
        """Link a workflow to specific tests that validate it."""
        workflow = self.workflow_storage.load_workflow(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        alignment = workflow.get_test_alignment()
        for test_id in test_ids:
            alignment.test_ids.add(test_id)
        
        workflow.metadata['test_alignment'] = alignment.model_dump()
        self.workflow_storage.save_workflow(workflow)
    
    def record_test_result(self, workflow_id: str, test_result: TestResult):
        """Record a test result for a workflow."""
        workflow = self.workflow_storage.load_workflow(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow.add_test_result(
            test_id=test_result.test_id,
            test_name=test_result.test_name,
            passed=test_result.passed,
            execution_details=test_result.execution_details,
            error_message=test_result.error_message
        )
        
        self.workflow_storage.save_workflow(workflow)
    
    def get_production_ready_workflows(self) -> List[WorkflowSpec]:
        """Get all workflows that are production-ready (pass all tests)."""
        all_workflows = self.workflow_storage.list_workflows()
        return [wf for wf in all_workflows if wf.is_production_ready()]
    
    def get_workflows_by_validation_status(self, status: str) -> List[WorkflowSpec]:
        """Get workflows filtered by validation status."""
        all_workflows = self.workflow_storage.list_workflows()
        return [wf for wf in all_workflows if wf.get_validation_status() == status]
    
    def get_failing_workflows_summary(self) -> Dict[str, List[TestResult]]:
        """Get summary of workflows with failing tests."""
        all_workflows = self.workflow_storage.list_workflows()
        failing_summary = {}
        
        for workflow in all_workflows:
            failing_tests = workflow.get_failing_tests()
            if failing_tests:
                failing_summary[str(workflow.id)] = failing_tests
        
        return failing_summary
    
    def auto_link_workflows_to_tests(self, workflow_pattern: Optional[str] = None) -> Dict[str, int]:
        """Automatically link workflows to tests based on patterns."""
        # Get all workflows
        workflows = self.workflow_storage.list_workflows()
        
        # Get all tests
        all_tests = []
        from ..utilities.workflow_test_repository import TestLayer
        for layer in TestLayer:
            all_tests.extend(self.test_repository.get_tests_by_layer(layer))
        
        links_created = {}
        
        for workflow in workflows:
            linked_count = 0
            
            # Auto-link based on workflow title/description patterns
            for test in all_tests:
                should_link = False
                
                # Link gate pattern tests to gate pattern workflows
                if ('gate_pattern' in (test.tags or []) and 
                    ('gate' in workflow.title.lower() or 'conditional' in workflow.title.lower())):
                    should_link = True
                
                # Link stock tests to stock workflows  
                if (any(tag in ['stock', 'trading'] for tag in (test.tags or [])) and
                    any(word in workflow.title.lower() for word in ['stock', 'trading', 'market'])):
                    should_link = True
                
                # Link routing tests to routing workflows
                if ('routing' in (test.tags or []) and
                    any(word in workflow.title.lower() for word in ['routing', 'decision', 'conditional'])):
                    should_link = True
                
                if should_link:
                    workflow.link_to_test(test.id)
                    linked_count += 1
            
            if linked_count > 0:
                self.workflow_storage.save_workflow(workflow)
                links_created[str(workflow.id)] = linked_count
        
        return links_created
    
    def run_validation_sweep(self) -> Dict[str, any]:
        """Run all tests against their linked workflows and update alignment."""
        # This would integrate with the test runner to execute tests
        # and automatically update workflow alignment metadata
        
        validation_results = {
            'workflows_tested': 0,
            'tests_executed': 0,
            'newly_production_ready': [],
            'newly_failing': [],
            'errors': []
        }
        
        # Get workflows with test links
        workflows = self.workflow_storage.list_workflows()
        linked_workflows = [wf for wf in workflows if wf.get_test_alignment().test_ids]
        
        validation_results['workflows_tested'] = len(linked_workflows)
        
        # This would trigger actual test execution - placeholder for now
        # Real implementation would use the test runner to execute tests
        # and call record_test_result() for each result
        
        return validation_results


class ProductionWorkflowFilter:
    """Filter for UI to only show production-ready workflows."""
    
    def __init__(self, alignment_service: WorkflowAlignmentService):
        self.alignment_service = alignment_service
    
    def get_app_workflows(self, include_untested: bool = False) -> List[WorkflowSpec]:
        """
        Get workflows that should be available in the app.
        
        Args:
            include_untested: Whether to include untested workflows (default: False)
        
        Returns:
            List of production-ready workflows, optionally including untested ones
        """
        if include_untested:
            # Include both production-ready and untested workflows
            production_ready = self.alignment_service.get_production_ready_workflows()
            untested = self.alignment_service.get_workflows_by_validation_status("untested")
            return production_ready + untested
        else:
            # Only production-ready workflows
            return self.alignment_service.get_production_ready_workflows()
    
    def filter_workflow_list(self, workflows: List[WorkflowSpec], 
                           filter_mode: str = "production_only") -> List[WorkflowSpec]:
        """
        Filter a list of workflows based on test alignment.
        
        Args:
            workflows: List of workflows to filter
            filter_mode: "production_only", "include_untested", or "all"
        
        Returns:
            Filtered list of workflows
        """
        if filter_mode == "all":
            return workflows
        
        if filter_mode == "include_untested":
            return [wf for wf in workflows 
                   if wf.get_validation_status() in ["passing", "untested"]]
        
        # "production_only" (default)
        return [wf for wf in workflows if wf.is_production_ready()]