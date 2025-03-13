import os
from functools import cache

_IO_INTEL_API = "https://api.intelligence.io.solutions/api/v1"
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
