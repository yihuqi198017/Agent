# -*- coding: utf-8 -*-
"""文档分块：基于 LangChain TextSplitter，支持 fixed/recursive/paragraph。"""

from __future__ import annotations

from enum import Enum
from typing import List

import tiktoken
from loguru import logger

try:
    from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover - 运行时降级到本地逻辑
    CharacterTextSplitter = None  # type: ignore[assignment]
    RecursiveCharacterTextSplitter = None  # type: ignore[assignment]


class ChunkStrategy(str, Enum):
    """分块策略枚举。"""

    FIXED = "fixed"
    RECURSIVE = "recursive"
    PARAGRAPH = "paragraph"


class DocumentChunker:
    """文档分块器：优先使用 LangChain TextSplitter。"""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        encoding_name: str = "cl100k_base",
    ) -> None:
        self.chunk_size = max(32, chunk_size)
        self.chunk_overlap = max(0, min(chunk_overlap, self.chunk_size - 1))
        try:
            self._encoding = tiktoken.get_encoding(encoding_name)
        except Exception as exc:
            logger.warning("tiktoken 编码不可用，回退字符长度计算: {}", exc)
            self._encoding = None

    def _len(self, text: str) -> int:
        if self._encoding is None:
            return len(text)
        return len(self._encoding.encode(text))

    def _split_with_langchain(self, text: str, strategy: ChunkStrategy) -> list[str] | None:
        if CharacterTextSplitter is None or RecursiveCharacterTextSplitter is None:
            return None

        if strategy == ChunkStrategy.FIXED:
            splitter = CharacterTextSplitter(
                separator="",
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=self._len,
            )
            return [c for c in splitter.split_text(text) if c.strip()]

        if strategy == ChunkStrategy.PARAGRAPH:
            splitter = CharacterTextSplitter(
                separator="\n\n",
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=self._len,
            )
            return [c for c in splitter.split_text(text) if c.strip()]

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self._len,
            separators=["\n\n", "\n", "。", ". ", " "],
        )
        return [c for c in splitter.split_text(text) if c.strip()]

    def chunk(self, text: str, strategy: ChunkStrategy = ChunkStrategy.RECURSIVE) -> List[str]:
        """按策略将全文切分为块列表。"""
        text = text.strip()
        if not text:
            return []

        chunks = self._split_with_langchain(text, strategy)
        if chunks is not None:
            return chunks

        # LangChain 不可用时的降级策略
        logger.warning("langchain_text_splitters 不可用，回退固定窗口分块")
        return self._chunk_fixed(text)

    def _chunk_fixed(self, text: str) -> List[str]:
        """固定窗口降级分块。"""
        out: list[str] = []
        start = 0
        n = len(text)
        while start < n:
            end = min(n, start + self.chunk_size)
            out.append(text[start:end])
            if end >= n:
                break
            start = max(0, end - self.chunk_overlap)
        return out
