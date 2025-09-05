import os
from functools import cache
from typing import Optional

_IO_INTEL_API = "https://api.intelligence-dev.io.solutions/api/v1"
_IO_INTEL_BASE_MODEL = "meta-llama/Llama-3.3-70B-Instruct"


def _get_env_var(suffix, default=None):
    for prefix in ("IO_API", "OPENAI_API"):
        if value := os.getenv(f"{prefix}_{suffix}", ""):
            return value
    return default


@cache
def get_api_url() -> str:
    return _get_env_var("BASE_URL", _IO_INTEL_API).rstrip("/")


@cache
def get_base_model() -> str:
    return _get_env_var("MODEL", _IO_INTEL_BASE_MODEL)


@cache
def get_api_key() -> str:
    return _get_env_var("KEY")


def get_available_models() -> list[str]:
    """Get list of all available LLM models."""
    return [
        "gpt-4o",
        "meta-llama/Llama-3.3-70B-Instruct", 
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
    ]

def get_model_config(model: Optional[str] = None, api_key: Optional[str] = None, base_url: Optional[str] = None) -> dict:
    """
    Centralized function to get the correct API configuration for any model.
    
    Args:
        model: Model name (e.g., "gpt-4o", "meta-llama/Llama-3.3-70B-Instruct")
        api_key: Override API key
        base_url: Override base URL
        
    Returns:
        dict with keys: model, api_key, base_url, is_openai
    """
    import os
    
    # Default model if none provided
    resolved_model = model or "gpt-4o"
    
    # Determine if this is an OpenAI model
    is_openai_model = (
        resolved_model.startswith("gpt-") or 
        "openai" in resolved_model.lower() or
        resolved_model in ["gpt-4o", "gpt-4", "gpt-3.5-turbo"]
    )
    
    if is_openai_model:
        # OpenAI configuration
        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        resolved_base_url = base_url or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    else:
        # IO Intel configuration  
        resolved_api_key = api_key or get_api_key()
        resolved_base_url = base_url or get_api_url()
    
    return {
        "model": resolved_model,
        "api_key": resolved_api_key,
        "base_url": resolved_base_url,
        "is_openai": is_openai_model
    }
