"""
Global pytest configuration and fixtures for the IOIntel test suite.

This file registers all workflow test fixtures and custom markers
for the centralized testing system.
"""


# Import all fixtures from the centralized fixture system
from fixtures.workflow_test_fixtures import *

def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "layer(name): mark test as belonging to a specific test layer (logical, agentic, orchestration, feedback)")
    config.addinivalue_line("markers", "category(name): mark test as belonging to a specific category")
    config.addinivalue_line("markers", "tags(name): mark test with specific tags")


def pytest_collection_modifyitems(config, items):
    """Modify test items during collection."""
    # Add custom behavior if needed
    pass