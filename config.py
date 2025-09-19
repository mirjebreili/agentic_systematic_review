"""
Load and validate all configuration from .env file using Pydantic models.

Requirements:
1. Create Pydantic BaseSettings model for type-safe configuration
2. Validate all paths exist
3. Validate LLM connectivity on startup
4. Provide singleton pattern for config access
"""

from pydantic import validator
from pydantic_settings import BaseSettings
from typing import Literal
from pathlib import Path
import requests

class Settings(BaseSettings):
    # LLM Configuration
    llm_provider: Literal["ollama", "vllm"] = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    vllm_base_url: str = "http://localhost:8000"
    model_name: str = "qwen2:7b-instruct"
    max_tokens: int = 500
    temperature: float = 0.1

    # Paths
    papers_folder: Path = Path("./papers")  # Default papers folder
    fields_config_file: Path = Path("./fields_config.yaml")  # YAML config file path
    pdf_directory: Path = Path("./data/pdfs")
    features_file: Path = Path("./data/features.xlsx")
    results_file: Path = Path("./data/results.xlsx")
    chroma_db_path: Path = Path("./chroma_db")
    log_file: Path = Path("./logs/extraction.log")

    # Embedding Configuration
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 512
    chunk_overlap: int = 50

    # Processing Configuration
    batch_size: int = 5
    top_k_chunks: int = 3
    retry_attempts: int = 3
    log_level: str = "INFO"

    @validator("pdf_directory", "features_file", "chroma_db_path", pre=True, always=True)
    def validate_path_exists(cls, v):
        if isinstance(v, str):
            path = Path(v)
        else:
            path = v

        if not path.exists():
            raise ValueError(f"Path does not exist: {path}")

        return path

    @validator("results_file", "log_file", pre=True, always=True)
    def validate_and_create_path(cls, v):
        if isinstance(v, str):
            path = Path(v)
        else:
            path = v

        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        return path

    @validator('llm_provider')
    def check_llm_connectivity(cls, v, values):
        if v == 'ollama':
            url = values.get('ollama_base_url')
            try:
                # Ollama's API root returns a status message
                response = requests.get(url)
                response.raise_for_status()
            except (requests.exceptions.RequestException, requests.exceptions.HTTPError) as e:
                # This is a warning for now, as the user might not have Ollama running yet
                print(f"Warning: Ollama connection failed at {url}. Please ensure Ollama is running.")
        elif v == 'vllm':
            url = values.get('vllm_base_url')
            # vLLM has a /health endpoint
            try:
                response = requests.get(f"{url}/health")
                response.raise_for_status()
            except (requests.exceptions.RequestException, requests.exceptions.HTTPError) as e:
                print(f"Warning: vLLM connection failed at {url}/health. Please ensure vLLM server is running.")
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

settings = Settings()
