"""
formatting.py: Utilities for rendering agent and tool results as HTML for UI display.

MIGRATED TO CENTRALIZED CONVERSION UTILITIES
===========================================
All conversion logic has been moved to iointel.src.utilities.conversion_utils.py
This file now just imports from the centralized location for backward compatibility.
"""

# Import from centralized conversion utilities
from iointel.src.utilities.conversion_utils import (
    to_jsonable,
    format_result_for_html,
    tool_usage_results_to_html
)

# Legacy compatibility - remove after full migration
__all__ = ['to_jsonable', 'format_result_for_html', 'tool_usage_results_to_html']
