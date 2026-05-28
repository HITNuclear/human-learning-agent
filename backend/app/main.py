from __future__ import annotations
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.database import init_db
from app.api import trees, nodes, rag, study, metrics
from app.services.rag_service import get_rag_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure data directories exist
    os.makedirs("data", exist_ok=True)
    os.makedirs(settings.upload_dir, exist_ok=True)

    await init_db()

    # Initialize RAG service singleton (loads embedding model ~3s)
    get_rag_service()

    yield


app = FastAPI(title="Human Learning Agent", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(trees.router)
app.include_router(nodes.router)
app.include_router(rag.router)
app.include_router(study.router)
app.include_router(metrics.router)


@app.get("/api/health")
async def health():
    return {"status": "ok"}
