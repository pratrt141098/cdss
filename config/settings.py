from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    gemini_api_key: str

    # Paths
    mimic_dir: str = "./data/mimic"
    synthetic_dir: str = "./data/synthetic"
    chroma_persist_dir: str = "./chroma_db"

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 50

    # Retrieval
    retrieval_top_k: int = 5

    # Models
    embedding_model: str = "models/text-embedding-004"
    generation_model: str = "models/gemini-1.5-flash"

    class Config:
        env_file = ".env"


settings = Settings()