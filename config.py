"""
Load and validate all configuration from .env file using Pydantic models.
"""

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal, Any
from pathlib import Path
import requests

class Settings(BaseSettings):
    # Pydantic V2 model configuration
    model_config = SettingsConfigDict(
        protected_namespaces=('settings_',),
        env_file=".env",
        env_file_encoding='utf-8'
    )

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

    @field_validator("pdf_directory", "chroma_db_path", mode='before')
    @classmethod
    def validate_or_create_path(cls, v: Any) -> Path:
        """Create paths if they don't exist instead of failing validation"""
        if isinstance(v, str):
            path = Path(v)
        else:
            path = v

        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {path}")

        return path

    @field_validator("papers_folder", "fields_config_file", mode='before')
    @classmethod
    def validate_required_paths(cls, v: Any) -> Path:
        """Only validate that these critical paths exist, don't create them"""
        if isinstance(v, str):
            path = Path(v)
        else:
            path = v

        if not path.exists():
            print(f"Warning: Required path does not exist: {path}")
            print("Please create this path before running the application.")

        return path

    @field_validator("results_file", "log_file", mode='before')
    @classmethod
    def validate_and_create_path(cls, v: Any) -> Path:
        if isinstance(v, str):
            path = Path(v)
        else:
            path = v

        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        return path

    @field_validator('llm_provider')
    @classmethod
    def check_llm_connectivity(cls, v: str, values: 'Settings') -> str:
        if v == 'ollama':
            url = values.data.get('ollama_base_url')
            try:
                response = requests.get(url)
                response.raise_for_status()
            except (requests.exceptions.RequestException, requests.exceptions.HTTPError) as e:
                print(f"Warning: Ollama connection failed at {url}. Please ensure Ollama is running.")
        elif v == 'vllm':
            url = values.data.get('vllm_base_url')
            try:
                response = requests.get(f"{url}/health")
                response.raise_for_status()
            except (requests.exceptions.RequestException, requests.exceptions.HTTPError) as e:
                print(f"Warning: vLLM connection failed at {url}/health. Please ensure vLLM server is running.")
        return v

settings = Settings()
