"""
Stores all configurations
"""

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BaseConfig(BaseSettings):
    """
    Main source of configurations
    """

    model_config = SettingsConfigDict(
        env_prefix="NL2SQLVERIFICATION_",
        env_ignore_empty=True,
        env_file=".env",
        env_file_encoding="utf-8",
        cli_parse_args=False,
        extra="ignore",
    )

    DATASET_BATCHING_ROWS: int = 0
    EXPERIMENT_EXECUTION_MODE: Literal["SEQUENTIAL", "PARALLEL"] = "PARALLEL"
    GENAI_DEFAULT_LLM: str = "gemini-2.5-pro"
    GENAI_DEFAULT_TEMPERATURE: float = 0.2
    GENAI_RETRY_DELAY_MAX: int = 60
    GENAI_RETRY_MAX_ATTEMPTS: int = 50
    GENAI_RETRY_DELAY_MULTIPLIER: float = 1
    GENAI_HTTP_API_VERSION: str = "v1"
    GOOGLE_CLOUD_PROJECT: str = Field(alias="GOOGLE_CLOUD_PROJECT")
    GOOGLE_CLOUD_LOCATION: str = Field("us-central1", alias="GOOGLE_CLOUD_LOCATION")


BASE_CONFIG = BaseConfig()
