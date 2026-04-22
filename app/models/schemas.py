# -*- coding: utf-8 -*-
"""API 请求与响应模型。"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from app.models.enums import MessageRole


class ChatMessage(BaseModel):
    """单条对话消息。"""

    role: str = Field(description="角色：system | user | assistant")
    content: str = Field(description="文本内容")


class ChatRequest(BaseModel):
    """对话请求。"""

    messages: list[ChatMessage] = Field(min_length=1, description="OpenAI 风格消息列表")
    model: str | None = Field(default=None, description="优先使用的模型 ID")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1)
    conversation_id: str | None = Field(default=None, description="可选会话 ID")


class ChatResponse(BaseModel):
    """非流式对话响应。"""

    id: str = Field(description="响应 ID")
    model: str = Field(description="实际使用的模型")
    content: str = Field(description="助手回复正文")
    trace_id: str | None = Field(default=None, description="链路追踪 ID")
    usage: dict[str, Any] | None = Field(default=None, description="Token 用量")


class TriageLevel(str, Enum):
    """分诊级别。"""

    EMERGENCY = "emergency"
    URGENT = "urgent"
    ROUTINE = "routine"
    SELF_CARE = "self_care"


class RiskFlag(BaseModel):
    """高风险规则命中项。"""

    code: str
    label: str
    severity: str = Field(default="medium", description="low|medium|high|critical")
    matched_text: str | None = None


class TriageChatRequest(BaseModel):
    """医疗分诊请求。"""

    symptom_text: str = Field(min_length=1, max_length=4000, description="症状描述")
    age: int | None = Field(default=None, ge=0, le=120)
    sex: str | None = Field(default=None, description="male|female|other")
    duration: str | None = Field(default=None, description="症状持续时间")
    temperature: float | None = Field(default=None, ge=30.0, le=45.0)
    pregnancy_status: str | None = Field(default=None)
    history: list[str] = Field(default_factory=list)
    current_medications: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    session_id: str | None = Field(default=None, description="会话 ID")
    model: str | None = Field(default=None, description="可选模型 ID")


class DocumentUploadResponse(BaseModel):
    """文档上传响应。"""

    document_id: str
    filename: str
    status: str
    chunk_count: int = 0
    message: str = "ok"


class DocumentInfo(BaseModel):
    """文档列表项。"""

    id: str
    filename: str
    mime_type: str | None = None
    status: str
    created_at: str | None = None


class DocumentUploadRequest(BaseModel):
    """文档上传附加元数据（可选）。"""

    tags: list[str] = Field(default_factory=list)


class Message(BaseModel):
    """对话消息（记忆/RAG 内部使用）。"""

    role: MessageRole
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryItem(BaseModel):
    """长期记忆召回条目。"""

    id: str
    content: str
    score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryContext(BaseModel):
    """短期 + 长期记忆合并上下文。"""

    session_id: str
    short_term_messages: list[Message]
    long_term_items: list[MemoryItem]


class RetrievalResult(BaseModel):
    """检索单条结果。"""

    id: str
    content: str
    score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    source: str = "vector"


class Citation(BaseModel):
    """答案中的引用标注。"""

    index: int
    result_id: str
    snippet: str


class RAGResponse(BaseModel):
    """RAG 生成结果。"""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    raw_contexts: list[RetrievalResult] = Field(default_factory=list)
    model: str | None = None


class TriageProtocolInfo(BaseModel):
    """分诊协议版本信息。"""

    protocol_version: str
    updated_at: str


class TriageChatResponse(BaseModel):
    """医疗分诊响应。"""

    triage_level: TriageLevel
    advice: str
    risk_flags: list[RiskFlag] = Field(default_factory=list)
    suggest_handoff: bool = False
    disclaimer: str
    citations: list[Citation] = Field(default_factory=list)
    trace_id: str
    usage: dict[str, Any] | None = None
