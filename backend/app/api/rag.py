from __future__ import annotations
import uuid
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from app.services.rag_service import get_rag_service
from app.services import document_processor

router = APIRouter(prefix="/api/rag", tags=["rag"])


class TextPayload(BaseModel):
    content: str
    name: str = "Pasted text"
    tree_id: str = "global"


class UrlPayload(BaseModel):
    url: str
    tree_id: str = "global"


class SourceResponse(BaseModel):
    source_id: str
    source_name: str
    tree_id: str


@router.get("/sources", response_model=list[SourceResponse])
async def list_sources():
    rag = get_rag_service()
    return rag.list_sources()


@router.post("/upload", response_model=dict)
async def upload_pdf(
    file: UploadFile = File(...),
    tree_id: str = Form("global"),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    content = await file.read()
    chunks = await document_processor.process_pdf(content)
    if not chunks:
        raise HTTPException(status_code=422, detail="No text could be extracted from the PDF")

    source_id = str(uuid.uuid4())
    rag = get_rag_service()
    rag.add_chunks(chunks, source_id=source_id, source_name=file.filename, tree_id=tree_id)

    return {"source_id": source_id, "chunks": len(chunks), "name": file.filename}


@router.post("/text", response_model=dict)
async def add_text(payload: TextPayload):
    chunks = document_processor.process_text(payload.content)
    if not chunks:
        raise HTTPException(status_code=422, detail="No text content provided")

    source_id = str(uuid.uuid4())
    rag = get_rag_service()
    rag.add_chunks(chunks, source_id=source_id, source_name=payload.name, tree_id=payload.tree_id)

    return {"source_id": source_id, "chunks": len(chunks), "name": payload.name}


@router.post("/url", response_model=dict)
async def add_url(payload: UrlPayload):
    try:
        chunks = await document_processor.process_url(payload.url)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to fetch URL: {e}")

    if not chunks:
        raise HTTPException(status_code=422, detail="No text could be extracted from the URL")

    source_id = str(uuid.uuid4())
    name = payload.url[:80]
    rag = get_rag_service()
    rag.add_chunks(chunks, source_id=source_id, source_name=name, tree_id=payload.tree_id)

    return {"source_id": source_id, "chunks": len(chunks), "name": name}


@router.delete("/sources/{source_id}", status_code=204)
async def delete_source(source_id: str):
    rag = get_rag_service()
    rag.delete_source(source_id)
