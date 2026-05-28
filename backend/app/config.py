from __future__ import annotations
from pydantic_settings import BaseSettings
from pydantic import field_validator
import json


class Settings(BaseSettings):
    deepseek_api_key: str
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-flash"
    database_url: str = "sqlite+aiosqlite:///./data/db.sqlite3"
    chroma_path: str = "./data/chroma"
    upload_dir: str = "./uploads"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    chunk_token_size: int = 384
    chunk_token_overlap: int = 64
    # Final chunks injected into prompt context.
    rag_top_k: int = 12
    rag_retrieval_mode: str = "dense"  # dense | bm25 | hybrid | hybrid_rerank
    # Candidate pool before final ranking/fusion.
    rag_dense_candidate_k: int = 100
    rag_bm25_candidate_k: int = 100
    rag_rrf_k: int = 60
    rag_rerank_candidate_k: int = 50
    rag_reranker_model: str = "BAAI/bge-reranker-v2-m3"
    # Pricing is USD per 1M tokens, used for request-level cost estimation logs.
    deepseek_input_price_per_1m: float = 0.27
    deepseek_output_price_per_1m: float = 1.10
    cors_origins: list[str] = ["http://localhost:3000"]

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors(cls, v: str | list) -> list[str]:
        if isinstance(v, str):
            return json.loads(v)
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
