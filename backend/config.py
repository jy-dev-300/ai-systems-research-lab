import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv(
        "DATABASE_URL", "postgresql+psycopg://optimo:optimo@localhost:5432/optimo"
    )
    backend: str = os.getenv("OPTIMO_BACKEND", "mock")
    vllm_base_url: str = os.getenv("VLLM_BASE_URL", "http://localhost:8001/v1")
    vllm_model: str = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")


settings = Settings()
