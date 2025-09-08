from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field, validator


class Settings(BaseSettings):
    app_name: str = "RL Model Evaluation API"
    version: str = "1.0.0"
    
    # API Configuration
    api_host: str = Field("0.0.0.0", description="API host")
    api_port: int = Field(8000, description="API port")
    api_workers: int = Field(1, description="Number of API workers")
    
    # Model API Configuration
    io_api_key: Optional[str] = Field(None, description="IO API key")
    io_base_url: Optional[str] = Field(None, description="IO base URL")
    
    # Evaluation Configuration
    default_num_tasks: int = Field(3, description="Default number of tasks")
    default_timeout: int = Field(120, description="Default timeout in seconds")
    max_timeout: int = Field(600, description="Maximum timeout in seconds")
    max_concurrent_evaluations: int = Field(5, description="Max concurrent evaluations")
    
    # Logging Configuration
    log_level: str = Field("INFO", description="Logging level")
    log_format: str = Field(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string"
    )
    
    # Security Configuration
    cors_origins: List[str] = Field(
        default=["*"],
        env="CORS_ORIGINS"
    )
    api_key_header: str = Field(default="X-API-Key", env="API_KEY_HEADER")
    require_api_key: bool = Field(default=False, env="REQUIRE_API_KEY")
    api_keys: List[str] = Field(default_factory=list, env="API_KEYS")
    
    # Rate Limiting
    rate_limit_enabled: bool = Field(default=True, env="RATE_LIMIT_ENABLED")
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_period: int = Field(default=3600, env="RATE_LIMIT_PERIOD")  # seconds
    
    # Task Storage
    task_retention_hours: int = Field(default=24, env="TASK_RETENTION_HOURS")
    max_stored_tasks: int = Field(default=1000, env="MAX_STORED_TASKS")
    
    # Models Configuration
    models_requiring_settings: List[str] = Field(
        default=[
            "deepseek-ai/DeepSeek-R1-0528",
            "meta-llama/Llama-3.3-70B-Instruct",
        ],
        env="MODELS_REQUIRING_SETTINGS"
    )
    
    recommended_models: List[str] = Field(
        default=[
            # IO/OS Models
            "meta-llama/Llama-3.3-70B-Instruct",
            "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
            "Qwen/Qwen2.5-Coder-32B-Instruct",
            "microsoft/phi-4",
            "nvidia/AceMath-7B-Instruct",
            "watt-ai/watt-tool-70B",
            # OpenAI Models
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-5",
        ],
        env="RECOMMENDED_MODELS"
    )
    
    @validator("api_keys", pre=True)
    def parse_api_keys(cls, v):
        if isinstance(v, str):
            return [key.strip() for key in v.split(",") if key.strip()]
        return v
    
    @validator("cors_origins", pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v
    
    @validator("models_requiring_settings", pre=True)
    def parse_models_requiring_settings(cls, v):
        if isinstance(v, str):
            return [model.strip() for model in v.split(",") if model.strip()]
        return v
    
    @validator("recommended_models", pre=True)
    def parse_recommended_models(cls, v):
        if isinstance(v, str):
            return [model.strip() for model in v.split(",") if model.strip()]
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()