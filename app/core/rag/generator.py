# -*- coding: utf-8 -*-
"""RAG 答案生成：基于 LangChain Prompt + Runnable。"""

from __future__ import annotations

import re
from typing import Any, Protocol, runtime_checkable

from loguru import logger

from app.models.schemas import Citation, Message, RAGResponse, RetrievalResult

try:
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda
except ImportError:  # pragma: no cover
    AIMessage = Any  # type: ignore[assignment]
    BaseMessage = Any  # type: ignore[assignment]
    HumanMessage = Any  # type: ignore[assignment]
    SystemMessage = Any  # type: ignore[assignment]
    StrOutputParser = None  # type: ignore[assignment]
    ChatPromptTemplate = None  # type: ignore[assignment]
    RunnableLambda = None  # type: ignore[assignment]


@runtime_checkable
class RAGLLMProtocol(Protocol):
    """支持异步生成的 LLM 接口（如 LangChain Runnable）。"""

    async def ainvoke(self, input: Any, **kwargs: Any) -> Any:
        ...


class RAGGenerator:
    """RAG 生成器：基于检索上下文生成答案并抽取引用。"""

    def __init__(
        self,
        llm: Any,
        system_prompt: str | None = None,
        model_name: str | None = None,
    ) -> None:
        self._llm = llm
        self._system_prompt = system_prompt or (
            "你是严谨的知识助手。仅根据提供上下文作答；若上下文不足请说明。"
            "回答中引用来源时使用 [1]、[2]，并与上下文编号一致。"
        )
        self._model_name = model_name

    def _build_messages_payload(
        self,
        query: str,
        contexts: list[RetrievalResult],
        chat_history: list[Any],
    ) -> dict[str, str]:
        ctx_lines = []
        for i, c in enumerate(contexts, start=1):
            ctx_lines.append(f"[{i}] (id={c.id})\n{c.content}")
        context_block = "\n\n".join(ctx_lines) if ctx_lines else "（无检索上下文）"

        history_lines: list[str] = []
        for msg in chat_history:
            if isinstance(msg, Message):
                history_lines.append(f"{msg.role.value}: {msg.content}")
            elif isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                history_lines.append(f"{role}: {content}")
            else:
                history_lines.append(str(msg))

        user_content = (
            f"## 检索上下文\n{context_block}\n\n"
            f"## 历史对话（摘要）\n{chr(10).join(history_lines) if history_lines else '（无）'}\n\n"
            f"## 用户问题\n{query}\n\n"
            "请作答并在必要时使用 [1]、[2] 引用上下文编号。"
        )
        return {"system_prompt": self._system_prompt, "user_content": user_content}

    @staticmethod
    def _to_openai_messages(messages: list[Any]) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        for m in messages:
            role = "user"
            content = ""
            if hasattr(m, "content"):
                content = str(getattr(m, "content", ""))
            elif isinstance(m, dict):
                role = str(m.get("role", "user"))
                content = str(m.get("content", ""))
                out.append({"role": role, "content": content})
                continue
            else:
                content = str(m)

            cls_name = m.__class__.__name__.lower()
            if "system" in cls_name:
                role = "system"
            elif "ai" in cls_name or "assistant" in cls_name:
                role = "assistant"
            else:
                role = "user"
            out.append({"role": role, "content": content})
        return out

    async def _invoke_llm(self, messages: list[Any]) -> Any:
        if isinstance(self._llm, RAGLLMProtocol):
            try:
                return await self._llm.ainvoke(messages)
            except Exception:
                # 兼容只接受 OpenAI dict messages 的适配器
                return await self._llm.ainvoke(self._to_openai_messages(messages))

        fn = getattr(self._llm, "ainvoke", None)
        if callable(fn):
            try:
                return await fn(messages)
            except Exception:
                return await fn(self._to_openai_messages(messages))

        raise TypeError("llm 需实现异步 ainvoke")

    def _extract_citations(self, answer: str, contexts: list[RetrievalResult]) -> list[Citation]:
        refs = [int(x) for x in re.findall(r"\[(\d+)\]", answer)]
        citations: list[Citation] = []
        seen: set[int] = set()
        for idx in refs:
            if idx in seen or idx < 1 or idx > len(contexts):
                continue
            seen.add(idx)
            r = contexts[idx - 1]
            snippet = r.content[:200] + ("..." if len(r.content) > 200 else "")
            citations.append(Citation(index=idx, result_id=r.id, snippet=snippet))
        return citations

    @staticmethod
    def _parse_llm_output(raw: Any) -> str:
        if raw is None:
            return ""
        if isinstance(raw, str):
            return raw
        if hasattr(raw, "content"):
            return str(getattr(raw, "content", ""))
        if isinstance(raw, dict) and "content" in raw:
            return str(raw["content"])
        return str(raw)

    async def generate(self, query: str, contexts: list[RetrievalResult], chat_history: list[Any]) -> RAGResponse:
        payload = self._build_messages_payload(query, contexts, chat_history)

        try:
            if ChatPromptTemplate is not None and RunnableLambda is not None:
                prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", "{system_prompt}"),
                        ("human", "{user_content}"),
                    ]
                )

                async def _run(messages: list[Any]) -> str:
                    raw = await self._invoke_llm(messages)
                    text = self._parse_llm_output(raw)
                    if StrOutputParser is not None:
                        return StrOutputParser().invoke(text)
                    return text

                chain = prompt | RunnableLambda(lambda _: "", afunc=_run)
                answer = await chain.ainvoke(payload)
            else:
                # LangChain 不可用时保留最小功能
                messages = [
                    {"role": "system", "content": payload["system_prompt"]},
                    {"role": "user", "content": payload["user_content"]},
                ]
                raw = await self._invoke_llm(messages)
                answer = self._parse_llm_output(raw)
        except Exception as e:
            logger.exception("RAG 生成调用失败: {}", e)
            raise RuntimeError(f"生成失败: {e}") from e

        answer = str(answer).strip()
        citations = self._extract_citations(answer, contexts)
        return RAGResponse(
            answer=answer,
            citations=citations,
            raw_contexts=list(contexts),
            model=self._model_name,
        )
