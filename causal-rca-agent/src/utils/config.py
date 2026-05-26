from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Ollama — two model tiers: small for simple tasks, large for reasoning
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="tinyllama", alias="OLLAMA_MODEL")
    ollama_model_large: str = Field(default="phi3", alias="OLLAMA_MODEL_LARGE")

    # Memory store
    chroma_persist_dir: str = Field(default="./chroma_db", alias="CHROMA_PERSIST_DIR")

    # Pipeline behaviour
    max_retries: int = Field(default=3, alias="MAX_RETRIES")
    max_tokens: int = Field(default=512, alias="MAX_TOKENS")
    temperature: float = Field(default=0.1, alias="TEMPERATURE")
    confidence_threshold: float = Field(default=0.70, alias="CONFIDENCE_THRESHOLD")
    memory_similarity_threshold: float = Field(default=0.80, alias="MEMORY_SIMILARITY_THRESHOLD")

    # Timeouts (seconds)
    llm_timeout: int = Field(default=120, alias="LLM_TIMEOUT")

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    model_config = {
        "populate_by_name": True,
        "env_file": ".env",
        "extra": "ignore",
    }


settings = Settings()
