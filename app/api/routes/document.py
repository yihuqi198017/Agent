# -*- coding: utf-8 -*-
"""文档管理 API：上传与列表。"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.etl import ETLPipeline
from app.infrastructure.embedding import EmbeddingProvider
from app.infrastructure.database.models import Document, DocumentChunk
from app.infrastructure.database.session import get_async_session
from app.infrastructure.vectordb import MilvusManager
from app.models.schemas import DocumentInfo, DocumentUploadResponse

router = APIRouter(tags=["documents"])


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(..., description="上传文件"),
    source_org: str | None = Form(default=None),
    guideline_name: str | None = Form(default=None),
    version: str | None = Form(default=None),
    publish_date: str | None = Form(default=None),
    specialty: str | None = Form(default=None),
    session: AsyncSession = Depends(get_async_session),
) -> DocumentUploadResponse:
    """上传文档并执行 ETL 分块后写入数据库。"""
    settings = get_settings()
    upload_root = Path("uploads")
    upload_root.mkdir(parents=True, exist_ok=True)

    doc_id = str(uuid.uuid4())
    safe_name = file.filename or "unnamed"
    dest = upload_root / f"{doc_id}_{safe_name}"

    try:
        raw = await file.read()
        await asyncio.to_thread(dest.write_bytes, raw)
    except Exception as exc:
        logger.exception("保存上传文件失败: {}", exc)
        raise HTTPException(status_code=500, detail=f"保存文件失败: {exc!s}") from exc

    pipeline = ETLPipeline()
    try:
        etl = await pipeline.run_bytes(
            raw,
            filename=safe_name,
            mime_type=file.content_type,
        )
    except Exception as exc:
        logger.exception("ETL 失败: {}", exc)
        raise HTTPException(status_code=422, detail=f"文档解析失败: {exc!s}") from exc

    meta: dict[str, str | int] = {"chunk_count": len(etl.chunks)}
    optional_meta = {
        "source_org": source_org,
        "guideline_name": guideline_name,
        "version": version,
        "publish_date": publish_date,
        "specialty": specialty,
    }
    for k, v in optional_meta.items():
        if v:
            meta[k] = v

    doc = Document(
        id=doc_id,
        filename=safe_name,
        mime_type=file.content_type,
        storage_path=str(dest),
        status="processing",
        meta=meta,
    )
    session.add(doc)

    chunk_records: list[DocumentChunk] = []
    for i, chunk_text in enumerate(etl.chunks):
        chunk = DocumentChunk(
            id=str(uuid.uuid4()),
            document_id=doc_id,
            chunk_index=i,
            content=chunk_text[:65000],
            vector_id=None,
            meta={"source_org": source_org, "guideline_name": guideline_name} if (source_org or guideline_name) else None,
        )
        session.add(chunk)
        chunk_records.append(chunk)

    vector_uploaded = False
    vector_error: str | None = None
    if settings.enable_embedding_upload and chunk_records:
        try:
            embedder = EmbeddingProvider(settings)
            vectors = await embedder.embed_documents([c.content for c in chunk_records])
            if vectors:
                milvus = MilvusManager(
                    host=settings.milvus_host,
                    port=str(settings.milvus_port),
                    **(
                        {"user": settings.milvus_user, "password": settings.milvus_password}
                        if settings.milvus_user
                        else {}
                    ),
                )
                dim = len(vectors[0])
                await milvus.create_collection(settings.milvus_collection_name, dim=dim)
                vector_ids = await milvus.insert(
                    settings.milvus_collection_name,
                    vectors=vectors,
                    metadata=[{"id": c.id} for c in chunk_records],
                )
                for c, vid in zip(chunk_records, vector_ids, strict=True):
                    c.vector_id = vid
                vector_uploaded = True
                meta["vector_count"] = len(vector_ids)
                meta["embedding_model"] = embedder.model_name
        except Exception as exc:  # noqa: BLE001
            vector_error = str(exc)
            logger.warning("文档向量入库失败，回退文本检索: {}", exc)

    meta["vector_uploaded"] = vector_uploaded
    if vector_error:
        meta["vector_error"] = vector_error[:300]
    doc.meta = meta
    doc.status = "ready"
    await session.commit()

    msg = "上传并分块成功"
    if settings.enable_embedding_upload:
        if vector_uploaded:
            msg = "上传、分块并向量入库成功"
        else:
            msg = "上传并分块成功（向量入库失败，已回退文本检索）"

    return DocumentUploadResponse(
        document_id=doc_id,
        filename=safe_name,
        status="ready",
        chunk_count=len(etl.chunks),
        message=msg,
    )


@router.get("/documents", response_model=list[DocumentInfo])
async def list_documents(
    session: AsyncSession = Depends(get_async_session),
) -> list[DocumentInfo]:
    """列出已入库文档元数据。"""
    try:
        result = await session.execute(select(Document).order_by(Document.created_at.desc()))
        rows = result.scalars().all()
        out: list[DocumentInfo] = []
        for d in rows:
            out.append(
                DocumentInfo(
                    id=d.id,
                    filename=d.filename,
                    mime_type=d.mime_type,
                    status=d.status,
                    created_at=d.created_at.isoformat() if d.created_at else None,
                )
            )
        return out
    except Exception as exc:
        logger.exception("查询文档列表失败: {}", exc)
        raise HTTPException(status_code=500, detail=f"查询失败: {exc!s}") from exc
