# -*- coding: utf-8 -*-
"""医疗分诊服务：输入校验、风险识别、RAG 生成与输出审查。"""

from __future__ import annotations

import math
from typing import Any

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.medical.prompt_templates import (
    MEDICAL_TRIAGE_SYSTEM_PROMPT,
    build_emergency_advice,
    build_triage_query,
)
from app.core.medical.red_flags import RedFlagHit, detect_red_flags
from app.core.medical.safety_guard import sanitize_input_text, sanitize_output_text
from app.core.medical.triage_policy import (
    DEFAULT_DISCLAIMER,
    infer_triage_level,
    should_suggest_handoff,
)
from app.core.rag.generator import RAGGenerator
from app.core.rag.retriever import MultiRetriever
from app.infrastructure.database.models import Document, DocumentChunk
from app.infrastructure.llm.model_router import ModelConfig, ModelRouter
from app.models.schemas import Citation, RiskFlag, TriageChatRequest, TriageChatResponse, TriageLevel


class _RouterRAGAdapter:
    """把 ModelRouter 适配为 RAGGenerator 所需的 ainvoke 接口。"""

    def __init__(
        self,
        router: ModelRouter,
        model_preference: str | None,
        *,
        temperature: float = 0.2,
        max_tokens: int | None = 800,
    ) -> None:
        self._router = router
        self._model_preference = model_preference
        self._temperature = temperature
        self._max_tokens = max_tokens
        self.last_usage: dict[str, Any] | None = None
        self.last_model: str | None = None

    async def ainvoke(self, input: Any, **kwargs: Any) -> str:
        if not isinstance(input, list):
            raise TypeError("RAG 输入必须是 OpenAI 风格 messages 列表")

        resp = await self._router.chat(
            input,
            model_preference=self._model_preference,
            temperature=kwargs.get("temperature", self._temperature),
            max_tokens=kwargs.get("max_tokens", self._max_tokens),
        )
        self.last_usage = resp.usage
        self.last_model = resp.model_id
        return resp.content


class _HashEmbeddingModel:
    """轻量 embedding 回退：用于 MultiRetriever 的向量分支兜底。"""

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    def embed_query(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        s = text.strip()
        if not s:
            return vec
        for i in range(len(s) - 1):
            idx = (ord(s[i]) * 31 + ord(s[i + 1])) % self._dim
            vec[idx] += 1.0
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec


class _LocalEntity(dict):
    def to_dict(self) -> dict[str, Any]:
        return dict(self)


class _LocalHit:
    def __init__(self, *, doc_id: str, text: str, distance: float) -> None:
        self.id = doc_id
        self.distance = distance
        self.entity = _LocalEntity({"id": doc_id, "text": text})


class _LocalMilvus:
    """本地向量检索兼容层：提供与 Milvus 类似的 search 接口。"""

    def __init__(self, *, id_to_text: dict[str, str], embedder: _HashEmbeddingModel) -> None:
        self._id_to_text = id_to_text
        self._embedder = embedder
        self._doc_vecs = {doc_id: self._embedder.embed_query(text) for doc_id, text in id_to_text.items()}

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def search(
        self,
        data: list[list[float]],
        anns_field: str,
        param: dict[str, Any],
        limit: int,
        output_fields: list[str] | None = None,
        **kwargs: Any,
    ) -> list[list[Any]]:
        _ = (anns_field, param, output_fields, kwargs)
        if not data:
            return [[]]
        qvec = data[0]
        ranked = sorted(
            self._doc_vecs.items(),
            key=lambda kv: self._cosine(qvec, kv[1]),
            reverse=True,
        )[: max(1, limit)]
        hits = [
            _LocalHit(
                doc_id=doc_id,
                text=self._id_to_text.get(doc_id, ""),
                distance=float(1.0 - self._cosine(qvec, vec)),
            )
            for doc_id, vec in ranked
        ]
        return [hits]


class TriageService:
    """医疗分诊主服务。"""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._top_k = max(1, int(self._settings.triage_top_k))
        self._max_chunks = max(100, int(self._settings.triage_max_chunks))

    def _build_router(self) -> ModelRouter | None:
        if not self._settings.openai_api_key:
            return None

        cfg = ModelConfig(
            model_id=self._settings.openai_model,
            api_key=self._settings.openai_api_key,
            base_url=self._settings.openai_api_base or None,
            priority=0,
            weight=1.0,
        )
        return ModelRouter([cfg])

    async def _load_documents(self, session: AsyncSession) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
        stmt = (
            select(DocumentChunk.id, DocumentChunk.content, Document.filename, Document.meta)
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(Document.status == "ready")
            .order_by(Document.created_at.desc(), DocumentChunk.chunk_index.asc())
            .limit(self._max_chunks)
        )
        result = await session.execute(stmt)

        id_to_text: dict[str, str] = {}
        id_to_meta: dict[str, dict[str, Any]] = {}
        for row in result.all():
            rid = str(row.id)
            text = str(row.content or "").strip()
            if not text:
                continue
            id_to_text[rid] = text
            meta = dict(row.meta or {})
            if row.filename and "filename" not in meta:
                meta["filename"] = row.filename
            id_to_meta[rid] = meta
        return id_to_text, id_to_meta

    async def _retrieve_contexts(self, query: str, session: AsyncSession) -> list[Any]:
        id_to_text, id_to_meta = await self._load_documents(session)
        if not id_to_text:
            return []

        embedder = _HashEmbeddingModel()
        retriever = MultiRetriever(_LocalMilvus(id_to_text=id_to_text, embedder=embedder), embedder)
        retriever.register_keyword_documents(id_to_text)

        try:
            contexts = await retriever.retrieve(query, top_k=self._top_k, mode="hybrid")
        except Exception as exc:
            logger.warning("hybrid 检索失败，回退 keyword: {}", exc)
            contexts = await retriever.keyword_search(query, self._top_k)

        for item in contexts:
            if item.id in id_to_meta:
                item.metadata = {**item.metadata, **id_to_meta[item.id]}
        return contexts

    @staticmethod
    def _to_risk_flags(hits: list[RedFlagHit]) -> list[RiskFlag]:
        return [
            RiskFlag(
                code=h.code,
                label=h.label,
                severity=h.severity,
                matched_text=h.matched_text,
            )
            for h in hits
        ]

    @staticmethod
    def _fallback_advice(level: str, hit_labels: list[str], has_context: bool) -> str:
        if level in {"emergency", "urgent"}:
            mapped = TriageLevel.EMERGENCY if level == "emergency" else TriageLevel.URGENT
            return build_emergency_advice(level=mapped, hit_labels=hit_labels)
        base = "建议先补充休息、饮水与体温/症状观察记录，若 24-48 小时无改善请线下就医。"
        if not has_context:
            base += " 当前知识库暂无可用参考文档，建议优先咨询线下医生。"
        return base

    @staticmethod
    def _fallback_citations(contexts: list[Any]) -> list[Citation]:
        citations: list[Citation] = []
        for i, item in enumerate(contexts[:2], start=1):
            snippet = item.content[:200] + ("..." if len(item.content) > 200 else "")
            citations.append(Citation(index=i, result_id=item.id, snippet=snippet))
        return citations

    async def chat(self, request: TriageChatRequest, session: AsyncSession, trace_id: str) -> TriageChatResponse:
        symptom_text, redactions = sanitize_input_text(request.symptom_text)
        red_hits = detect_red_flags(symptom_text)

        triage_level = infer_triage_level(
            risk_severities=[h.severity for h in red_hits],
            age=request.age,
            temperature=request.temperature,
            symptom_text=symptom_text,
        )
        suggest_handoff = should_suggest_handoff(triage_level)

        hit_labels = [h.label for h in red_hits]
        query = build_triage_query(request, symptom_text)

        contexts = await self._retrieve_contexts(query, session)
        citations: list[Citation] = []
        usage: dict[str, Any] | None = None

        answer: str
        if triage_level.value == "emergency":
            answer = build_emergency_advice(triage_level, hit_labels)
            citations = self._fallback_citations(contexts)
        else:
            router = self._build_router()
            if router is not None and contexts:
                try:
                    llm_adapter = _RouterRAGAdapter(router, request.model, temperature=0.2, max_tokens=800)
                    generator = RAGGenerator(
                        llm=llm_adapter,
                        system_prompt=MEDICAL_TRIAGE_SYSTEM_PROMPT,
                        model_name=self._settings.openai_model,
                    )
                    rag = await generator.generate(query=query, contexts=contexts, chat_history=[])
                    answer = rag.answer
                    citations = rag.citations or self._fallback_citations(contexts)
                    usage = llm_adapter.last_usage
                except Exception as exc:
                    logger.warning("RAG 生成失败，回退模板建议: {}", exc)
                    answer = self._fallback_advice(triage_level.value, hit_labels, bool(contexts))
                    citations = self._fallback_citations(contexts)
            else:
                answer = self._fallback_advice(triage_level.value, hit_labels, bool(contexts))
                citations = self._fallback_citations(contexts)

        audit = sanitize_output_text(answer)
        final_advice = audit.text.strip()
        if DEFAULT_DISCLAIMER not in final_advice:
            final_advice = f"{final_advice}\n\n{DEFAULT_DISCLAIMER}"

        risk_flags = self._to_risk_flags(red_hits)
        if redactions:
            risk_flags.append(
                RiskFlag(
                    code="privacy_redaction",
                    label="输入已脱敏",
                    severity="low",
                    matched_text=",".join(redactions),
                )
            )

        return TriageChatResponse(
            triage_level=triage_level,
            advice=final_advice,
            risk_flags=risk_flags,
            suggest_handoff=suggest_handoff,
            disclaimer=DEFAULT_DISCLAIMER,
            citations=citations,
            trace_id=trace_id,
            usage=usage,
        )
