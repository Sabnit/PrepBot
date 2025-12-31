import boto3
from typing import Optional
from langchain_aws import ChatBedrock, BedrockEmbeddings

from config import settings


class BedrockModels:
    """
    Singleton class to manage Bedrock LLM and Embedding models.
    Ensures we only create one instance of each model.
    """
    
    _instance = None
    _bedrock_client = None
    _llm = None
    _embeddings = None
    
    def __new__(cls):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super(BedrockModels, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize Bedrock models (only once)."""
        if self._bedrock_client is None:
            self._initialize_bedrock_client()
            self._initialize_llm()
            self._initialize_embeddings()
    
    def _initialize_bedrock_client(self):
        """Create the boto3 Bedrock runtime client."""
        config = settings.get_bedrock_client_config()
        
        # Build client kwargs
        client_kwargs = {
            "service_name": config["service_name"],
            "region_name": config["region_name"],
        }
        
        # Add credentials if provided
        if config.get("aws_access_key_id"):
            client_kwargs["aws_access_key_id"] = config["aws_access_key_id"]
        if config.get("aws_secret_access_key"):
            client_kwargs["aws_secret_access_key"] = config["aws_secret_access_key"]
        if config.get("aws_session_token"):
            client_kwargs["aws_session_token"] = config["aws_session_token"]
        
        # Create bedrock-runtime client
        self._bedrock_client = boto3.client(**client_kwargs)
        
        print(f"✓ Bedrock client initialized (region: {config['region_name']})")
    
    def _initialize_llm(self):
        """Initialize the ChatBedrock LLM (Amazon Nova Lite)."""
        self._llm = ChatBedrock(
            client=self._bedrock_client,
            model_id=settings.bedrock_llm_model_id,
            model_kwargs={
                "temperature": 0.7,  # Balanced creativity for questions
                "top_p": 0.9,
                "max_tokens": 2000,
            }
        )
        
        print(f"✓ LLM initialized: {settings.bedrock_llm_model_id}")
    
    def _initialize_embeddings(self):
        """Initialize Bedrock Embeddings (Titan Text Embeddings V2)."""
        self._embeddings = BedrockEmbeddings(
            client=self._bedrock_client,
            model_id=settings.bedrock_embedding_model_id,
        )
        
        print(f"✓ Embeddings initialized: {settings.bedrock_embedding_model_id}")
    
    @property
    def bedrock_client(self):
        """Get the Bedrock runtime client."""
        return self._bedrock_client
    
    @property
    def llm(self) -> ChatBedrock:
        """Get the LLM instance."""
        return self._llm
    
    @property
    def embeddings(self) -> BedrockEmbeddings:
        """Get the embeddings instance."""
        return self._embeddings
    
    def get_llm_with_temperature(self, temperature: float) -> ChatBedrock:
        """
        Get a new LLM instance with custom temperature.
        Useful for different use cases (e.g., strict evaluation vs creative questions).
        
        Args:
            temperature: Temperature value (0.0 to 1.0)
            
        Returns:
            ChatBedrock instance with specified temperature
        """
        return ChatBedrock(
            client=self._bedrock_client,
            model_id=settings.bedrock_llm_model_id,
            model_kwargs={
                "temperature": temperature,
                "top_p": 0.9,
                "max_tokens": 2000,
            }
        )


# Global instance - import this throughout the app
bedrock_models = BedrockModels()


# Convenience functions for easy access
def get_llm(temperature: Optional[float] = None) -> ChatBedrock:
    """
    Get the LLM instance.
    
    Args:
        temperature: Optional custom temperature (default uses 0.7)
        
    Returns:
        ChatBedrock LLM instance
    """
    if temperature is not None:
        return bedrock_models.get_llm_with_temperature(temperature)
    return bedrock_models.llm


def get_embeddings() -> BedrockEmbeddings:
    """
    Get the embeddings instance.
    
    Returns:
        BedrockEmbeddings instance
    """
    return bedrock_models.embeddings


def get_bedrock_client():
    """
    Get the raw boto3 Bedrock client.
    
    Returns:
        boto3 bedrock-runtime client
    """
    return bedrock_models.bedrock_client