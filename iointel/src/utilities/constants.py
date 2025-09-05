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
    # Check for IO_API_BASE first (what's actually in creds.env)
    url = os.getenv("IO_API_BASE") or os.getenv("IO_API_BASE_URL") or _get_env_var("BASE_URL", _IO_INTEL_API)
    return url.rstrip("/")


@cache
def get_base_model() -> str:
    return _get_env_var("MODEL", _IO_INTEL_BASE_MODEL)


@cache
def get_api_key() -> str:
    return _get_env_var("KEY")


def get_available_models() -> list[str]:
    """Get list of all available LLM models that support tool calling."""
    return [
        # OpenAI models (fully supported)
        "gpt-4o",
        "gpt-4o-mini",
        
        # IO Intel models (fully supported)
        "meta-llama/Llama-3.3-70B-Instruct", 
        "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        "Qwen/Qwen3-235B-A22B-Thinking-2507",
        "Intel/Qwen3-Coder-480B-A35B-Instruct-int4-mixed-ar",
        "mistralai/Mistral-Nemo-Instruct-2407",
        
        # Note: GPT-OSS models excluded until server configuration is fixed
        # "openai/gpt-oss-120b",  # BLOCKED: Server needs vLLM flags
        # "openai/gpt-oss-20b",   # BLOCKED: Server needs vLLM flags
        
        # Note: CohereForAI models excluded (no tool choice support)
        # "CohereForAI/aya-expanse-32b",  # Chat only, no tool calling
    ]

def get_available_models_with_tool_calling() -> list[str]:
    """Get list of models that support tool calling (same as get_available_models for now)."""
    return get_available_models()

def get_chat_only_models() -> list[str]:
    """Get list of models that only support chat (no tool calling)."""
    return [
        "CohereForAI/aya-expanse-32b",  # No tool choice support
    ]

def get_blocked_models() -> list[str]:
    """Get list of models that are blocked due to server configuration issues."""
    return [
        "openai/gpt-oss-120b",  # Server needs --tool-call-parser openai --enable-auto-tool-choice
        "openai/gpt-oss-20b",   # Server needs --tool-call-parser openai --enable-auto-tool-choice
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
    # Note: GPT-OSS models (even with openai/ prefix) are NOT OpenAI models, they're served through IO Intel
    is_openai_model = (
        (resolved_model.startswith("gpt-") and not "gpt-oss" in resolved_model.lower()) or 
        ("openai" in resolved_model.lower() and not "gpt-oss" in resolved_model.lower()) or
        resolved_model in ["gpt-5", "gpt-4o", "gpt-4", "gpt-3.5-turbo"]
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
