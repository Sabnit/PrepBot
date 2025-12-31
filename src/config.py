"""
Configuration module for PrepBot.
Loads environment variables and provides centralized configuration.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
import os
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # AWS Bedrock Configuration
    aws_region: str = Field(default="us-east-1", env="AWS_REGION")
    aws_access_key_id: Optional[str] = Field(default=None, env="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(default=None, env="AWS_SECRET_ACCESS_KEY")
    aws_session_token: Optional[str] = Field(default=None, env="AWS_SESSION_TOKEN")
    
    # Bedrock Model IDs
    bedrock_llm_model_id: str = Field(
        default="amazon.nova-lite-v1:0",
        env="BEDROCK_LLM_MODEL_ID"
    )
    bedrock_embedding_model_id: str = Field(
        default="amazon.titan-embed-text-v2:0",
        env="BEDROCK_EMBEDDING_MODEL_ID"
    )
    
    # LangSmith Configuration
    langchain_tracing_v2: bool = Field(default=True, env="LANGCHAIN_TRACING_V2")
    langchain_endpoint: str = Field(
        default="https://api.smith.langchain.com",
        env="LANGCHAIN_ENDPOINT"
    )
    langsmith_api_key: Optional[str] = Field(default=None, env="LANGSMITH_API_KEY")
    langchain_project: str = Field(
        default="question-asker-bot",
        env="LANGCHAIN_PROJECT"
    )
    
    # Document Processing Settings
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")
    
    # Application Settings
    max_questions_per_session: int = Field(
        default=20,
        env="MAX_QUESTIONS_PER_SESSION"
    )
    
    # Paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "data")
    uploads_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent / "data" / "uploads"
    )
    vector_db_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent / "data" / "vector_db"
    )
    database_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent / "database"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._create_directories()
        self._setup_langsmith()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        for directory in [
            self.data_dir,
            self.uploads_dir,
            self.vector_db_dir,
            self.database_dir
        ]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _setup_langsmith(self):
        """Setup LangSmith environment variables for tracing."""
        if self.langchain_tracing_v2 and self.langsmith_api_key:
            os.environ["LANGCHAIN_TRACING_V2"] = str(self.langchain_tracing_v2)
            os.environ["LANGCHAIN_ENDPOINT"] = self.langchain_endpoint
            os.environ["LANGCHAIN_API_KEY"] = self.langsmith_api_key
            os.environ["LANGCHAIN_PROJECT"] = self.langchain_project
            print("✓ LangSmith tracing enabled")
        else:
            print("⚠ LangSmith tracing disabled (no API key provided)")
    
    @property
    def database_url(self) -> str:
        """Get SQLite database URL."""
        return f"sqlite:///{self.database_dir / 'sessions.db'}"
    
    def get_bedrock_client_config(self) -> dict:
        """Get boto3 client configuration for Bedrock."""
        config = {
            "region_name": self.aws_region,
            "service_name": "bedrock-runtime"
        }
        
        if self.aws_access_key_id and self.aws_secret_access_key:
            config["aws_access_key_id"] = self.aws_access_key_id
            config["aws_secret_access_key"] = self.aws_secret_access_key
            
            # Add session token if available (for temporary credentials)
            if self.aws_session_token:
                config["aws_session_token"] = self.aws_session_token
        
        return config


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings