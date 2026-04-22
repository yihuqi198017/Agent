# -*- coding: utf-8 -*-
"""医疗分诊 API。"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.medical.triage_policy import PROTOCOL_UPDATED_AT, PROTOCOL_VERSION
from app.core.medical.triage_service import TriageService
from app.infrastructure.database.session import get_async_session
from app.infrastructure.trace.tracer import Tracer
from app.models.schemas import TriageChatRequest, TriageChatResponse, TriageProtocolInfo

router = APIRouter(tags=["triage"])

_service = TriageService()
_tracer = Tracer()


@router.get("/triage/protocol/version", response_model=TriageProtocolInfo)
async def triage_protocol_version() -> TriageProtocolInfo:
    """返回分诊协议版本。"""
    return TriageProtocolInfo(
        protocol_version=PROTOCOL_VERSION,
        updated_at=PROTOCOL_UPDATED_AT,
    )


@router.post("/triage/chat", response_model=TriageChatResponse)
async def triage_chat(
    request: TriageChatRequest,
    session: AsyncSession = Depends(get_async_session),
) -> TriageChatResponse:
    """医疗分诊主入口。"""
    trace_id = str(uuid.uuid4())
    span = _tracer.start_trace(trace_id, "triage_chat")

    try:
        response = await _service.chat(request, session, trace_id=trace_id)
        _tracer.end_span(
            span,
            result={
                "triage_level": response.triage_level.value,
                "risk_flags": [f.code for f in response.risk_flags],
            },
        )
        return response
    except Exception as exc:
        logger.exception("triage_chat 失败: {}", exc)
        _tracer.end_span(span, error=str(exc))
        raise HTTPException(status_code=500, detail=f"分诊失败: {exc!s}") from exc
