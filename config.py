"""
Load and validate all configuration from .env file using Pydantic models.
"""

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal, Any
from pathlib import Path
import requests

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        protected_namespaces=('settings_',),
        env_file=".env",
        env_file_encoding='utf-8',
        case_sensitive=False,
        env_prefix='',  # No prefix needed
        extra='ignore'  # Ignore unknown env vars
    )

    # LLM Configuration
    llm_provider: Literal["ollama", "vllm"] = Field(default="ollama", alias="LLM_PROVIDER")
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    vllm_base_url: str = Field(default="http://localhost:8000", alias="VLLM_BASE_URL")
    model_name: str = Field(default="gemma2:latest", alias="MODEL_NAME")
    max_tokens: int = Field(default=500, alias="MAX_TOKENS")
    temperature: float = Field(default=0.1, alias="TEMPERATURE")

    # Paths
    papers_folder: Path = Field(default=Path("./papers"), alias="PAPERS_FOLDER")
    fields_config_file: Path = Field(default=Path("./fields_config.yaml"), alias="FIELDS_CONFIG_FILE")
    pdf_directory: Path = Field(default=Path("./data/pdfs"), alias="PDF_DIRECTORY")
    features_file: Path = Field(default=Path("./data/features.xlsx"), alias="FEATURES_FILE")
    results_file: Path = Field(default=Path("./data/results.xlsx"), alias="RESULTS_FILE")
    chroma_db_path: Path = Field(default=Path("./chroma_db"), alias="CHROMA_DB_PATH")
    log_file: Path = Field(default=Path("./logs/extraction.log"), alias="LOG_FILE")

    # Embedding Configuration
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")
    chunk_size: int = Field(default=512, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, alias="CHUNK_OVERLAP")

    # Processing Configuration
    batch_size: int = Field(default=5, alias="BATCH_SIZE")
    top_k_chunks: int = Field(default=3, alias="TOP_K_CHUNKS")
    retry_attempts: int = Field(default=3, alias="RETRY_ATTEMPTS")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

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

    @field_validator('llm_provider', mode='after')
    @classmethod
    def check_llm_connectivity(cls, v: str) -> str:
        # Get settings instance to access other fields
        if v == 'ollama':
            url = "http://localhost:11434"  # Default fallback
            try:
                response = requests.get(url, timeout=5)
                response.raise_for_status()
                print(f"✅ Ollama connection successful at {url}")
            except Exception as e:
                print(f"⚠️ Warning: Ollama connection failed at {url}. Please ensure Ollama is running.")
        elif v == 'vllm':
            url = "http://localhost:8000"  # Default fallback
            try:
                response = requests.get(f"{url}/health", timeout=5)
                response.raise_for_status()
                print(f"✅ vLLM connection successful at {url}")
            except Exception as e:
                print(f"⚠️ Warning: vLLM connection failed at {url}/health. Please ensure vLLM server is running.")
        return v

settings = Settings()
