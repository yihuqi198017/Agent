# -*- coding: utf-8 -*-
"""Embedding 编码器：OpenAI 优先，哈希向量回退。"""

from __future__ import annotations

import asyncio
import math
from typing import Any

from loguru import logger

from app.config import Settings

try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:  # pragma: no cover
    OpenAIEmbeddings = None  # type: ignore[assignment]


class HashEmbeddings:
    """无外部依赖时使用的哈希 embedding。"""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    @staticmethod
    def _normalize(vec: list[float]) -> list[float]:
        norm = math.sqrt(sum(x * x for x in vec))
        if norm <= 0:
            return vec
        return [x / norm for x in vec]

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        s = text.strip()
        if not s:
            return vec
        for i in range(max(1, len(s) - 1)):
            a = ord(s[i])
            b = ord(s[i + 1]) if i + 1 < len(s) else 17
            idx = (a * 31 + b) % self.dim
            vec[idx] += 1.0
        return self._normalize(vec)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)


class EmbeddingProvider:
    """统一 embedding provider。"""

    def __init__(self, settings: Settings) -> None:
        self.model_name = settings.openai_embedding_model
        self.backend: Any

        if settings.openai_api_key and OpenAIEmbeddings is not None:
            self.backend = OpenAIEmbeddings(
                model=settings.openai_embedding_model,
                api_key=settings.openai_api_key,
                base_url=settings.openai_api_base,
            )
            logger.info("Embedding provider: OpenAI ({})", settings.openai_embedding_model)
        else:
            self.backend = HashEmbeddings()
            self.model_name = "hash-embedding"
            logger.warning("Embedding provider 回退到 hash-embedding")

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        aembed = getattr(self.backend, "aembed_documents", None)
        if callable(aembed):
            return await aembed(texts)

        embed = getattr(self.backend, "embed_documents", None)
        if callable(embed):
            return await asyncio.to_thread(embed, texts)

        raise TypeError("embedding backend 未实现 embed_documents")

    async def embed_query(self, text: str) -> list[float]:
        aembed = getattr(self.backend, "aembed_query", None)
        if callable(aembed):
            return await aembed(text)

        embed = getattr(self.backend, "embed_query", None)
        if callable(embed):
            return await asyncio.to_thread(embed, text)

        raise TypeError("embedding backend 未实现 embed_query")
