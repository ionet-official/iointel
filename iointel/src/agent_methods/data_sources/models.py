"""
Data Source Models
=================

Standardized Pydantic models for data source inputs and outputs.
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field


class DataSourceRequest(BaseModel):
    """Standardized request model for all data sources."""
    message: str = Field(..., description="The message/prompt to display or process")
    default_value: str = Field(None, description="Default value if no user input provided")
    
    class Config:
        use_enum_values = True


class DataSourceResponse(BaseModel):
    """Standardized response model for all data sources."""
    source_type: str = Field(..., description="Type of data source (user_input, prompt_tool)")
    message: str = Field(..., description="The actual message/content provided")
    status: str = Field(..., description="Status of the data source execution")
    
    # UI-specific fields (only populated for interactive sources)
    form_id: Optional[str] = Field(None, description="Unique form identifier")
    ui_action: Optional[str] = Field(None, description="Action for UI to perform")
    placeholder: Optional[str] = Field(None, description="Placeholder text")
    input_type: Optional[str] = Field(None, description="Input type for forms")
    options: Optional[List[str]] = Field(None, description="Select options")
    default_value: Optional[str] = Field(None, description="Default form value")
    suggestions: Optional[List[str]] = Field(None, description="Input suggestions")
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        use_enum_values = True