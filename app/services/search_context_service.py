from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI
from opik import start_as_current_span
from pydantic import BaseModel, Field

from app.core.config import settings
from app.utils.logger import custom_logger


class CondensedSearchContext(BaseModel):
    summary: str = Field(description="질문과 직접 관련된 검색 결과 요약")
    key_points: list[str] = Field(default_factory=list, description="핵심 사실 목록")
    cautions: list[str] = Field(default_factory=list, description="불확실성 또는 주의사항")


class SearchContextService:
    def __init__(self) -> None:
        self.enabled = settings.SEARCH_SUMMARIZER_ENABLED
        self.max_docs = settings.SEARCH_SUMMARIZER_MAX_DOCS
        self.max_chars_per_doc = settings.SEARCH_SUMMARIZER_MAX_CHARS_PER_DOC
        self._llm: ChatOpenAI | None = None

    async def build_context(
        self,
        *,
        question: str,
        documents: list[dict[str, Any]],
        context_type: str,
    ) -> str:
        limited_documents = self._prepare_documents(documents)
        raw_chars = self._documents_char_count(documents)
        prepared_chars = self._documents_char_count(limited_documents)

        with start_as_current_span(
            name="search_context_preparation",
            type="tool",
            input={
                "question": question,
                "context_type": context_type,
                "raw_doc_count": len(documents),
                "prepared_doc_count": len(limited_documents),
                "raw_chars": raw_chars,
                "prepared_chars": prepared_chars,
                "summarizer_enabled": self.enabled,
            },
            project_name=self._get_opik_project_name(),
        ):
            if not limited_documents:
                return self._build_fallback_context(
                    question=question,
                    documents=[],
                    context_type=context_type,
                    reason="no_documents",
                )

            if not self.enabled:
                return self._build_fallback_context(
                    question=question,
                    documents=limited_documents,
                    context_type=context_type,
                    reason="summarizer_disabled",
                )

            try:
                condensed = await self._summarize(
                    question=question,
                    documents=limited_documents,
                    context_type=context_type,
                    raw_chars=raw_chars,
                    prepared_chars=prepared_chars,
                )
                formatted = self._format_condensed_context(
                    question=question,
                    documents=limited_documents,
                    context_type=context_type,
                    condensed=condensed,
                )
                self._record_result_span(
                    question=question,
                    context_type=context_type,
                    mode="compressed",
                    doc_count=len(limited_documents),
                    raw_chars=raw_chars,
                    prepared_chars=prepared_chars,
                    output_chars=len(formatted),
                )
                return formatted
            except Exception as exc:
                custom_logger.warning("Search context summarization failed: %s", exc)
                return self._build_fallback_context(
                    question=question,
                    documents=limited_documents,
                    context_type=context_type,
                    reason=f"summarization_error:{type(exc).__name__}",
                    raw_chars=raw_chars,
                    prepared_chars=prepared_chars,
                )

    def _prepare_documents(self, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        prepared: list[dict[str, Any]] = []
        for doc in documents[: self.max_docs]:
            prepared.append(
                {
                    "c_id": doc.get("c_id"),
                    "creation_year": doc.get("creation_year"),
                    "domain": doc.get("domain"),
                    "source": doc.get("source"),
                    "source_spec": doc.get("source_spec"),
                    "content": self._trim_text(doc.get("content", "")),
                }
            )
        return prepared

    def _trim_text(self, text: str) -> str:
        normalized = " ".join(text.split())
        if len(normalized) <= self.max_chars_per_doc:
            return normalized
        return normalized[: self.max_chars_per_doc].rstrip() + "..."

    def _documents_char_count(self, documents: list[dict[str, Any]]) -> int:
        return sum(len(str(doc.get("content", ""))) for doc in documents)

    def _get_llm(self) -> ChatOpenAI:
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=settings.SEARCH_SUMMARIZER_MODEL,
                api_key=settings.OPENAI_API_KEY,
                temperature=0,
                max_tokens=settings.SEARCH_SUMMARIZER_MAX_OUTPUT_TOKENS,
            )
        return self._llm

    async def _summarize(
        self,
        *,
        question: str,
        documents: list[dict[str, Any]],
        context_type: str,
        raw_chars: int,
        prepared_chars: int,
    ) -> CondensedSearchContext:
        with start_as_current_span(
            name="search_context_compression",
            type="llm",
            input={
                "question": question,
                "context_type": context_type,
                "model": settings.SEARCH_SUMMARIZER_MODEL,
                "doc_count": len(documents),
                "raw_chars": raw_chars,
                "prepared_chars": prepared_chars,
                "max_output_tokens": settings.SEARCH_SUMMARIZER_MAX_OUTPUT_TOKENS,
            },
            project_name=self._get_opik_project_name(),
        ):
            extractor = self._get_llm().with_structured_output(CondensedSearchContext)
            result = await extractor.ainvoke(
                [
                    (
                        "system",
                        "너는 Elasticsearch 검색 결과를 메인 답변 모델에 전달하기 전에 압축하는 역할이다. "
                        "질문과 직접 관련된 정보만 남기고, 군더더기 문장과 반복 설명은 제거하라. "
                        "확인할 수 없는 내용은 추측하지 말고 cautions에 남겨라. "
                        "key_points는 짧은 bullet 형태의 문장으로 작성하라.",
                    ),
                    (
                        "human",
                        f"질문: {question}\n"
                        f"컨텍스트 종류: {context_type}\n"
                        f"검색 결과: {documents}",
                    ),
                ]
            )
            return result

    def _format_condensed_context(
        self,
        *,
        question: str,
        documents: list[dict[str, Any]],
        context_type: str,
        condensed: CondensedSearchContext,
    ) -> str:
        header = "[의학 검색 요약]" if context_type == "medical" else "[지역/병원 검색 요약]"
        lines = [
            header,
            f"사용자 질문: {question}",
            f"요약: {condensed.summary.strip()}",
            "핵심 포인트:",
        ]

        if condensed.key_points:
            for point in condensed.key_points:
                lines.append(f"- {point.strip()}")
        else:
            lines.append("- 관련 핵심 포인트를 추출하지 못했습니다.")

        if condensed.cautions:
            lines.append("주의사항:")
            for caution in condensed.cautions:
                lines.append(f"- {caution.strip()}")

        lines.append("참고 문서:")
        for index, doc in enumerate(documents, start=1):
            lines.append(self._format_doc_reference(index=index, doc=doc))

        return "\n".join(lines)

    def _build_fallback_context(
        self,
        *,
        question: str,
        documents: list[dict[str, Any]],
        context_type: str,
        reason: str = "fallback",
        raw_chars: int | None = None,
        prepared_chars: int | None = None,
    ) -> str:
        header = "[의학 검색 요약]" if context_type == "medical" else "[지역/병원 검색 요약]"
        lines = [
            header,
            f"사용자 질문: {question}",
            "핵심 근거:",
        ]

        if not documents:
            lines.append("- 검색 결과가 없습니다.")
            return "\n".join(lines)

        for index, doc in enumerate(documents, start=1):
            lines.append(self._format_doc_reference(index=index, doc=doc, include_excerpt=True))
        formatted = "\n".join(lines)
        self._record_result_span(
            question=question,
            context_type=context_type,
            mode="fallback",
            doc_count=len(documents),
            raw_chars=raw_chars if raw_chars is not None else self._documents_char_count(documents),
            prepared_chars=prepared_chars if prepared_chars is not None else self._documents_char_count(documents),
            output_chars=len(formatted),
            fallback_reason=reason,
        )
        return formatted

    def _record_result_span(
        self,
        *,
        question: str,
        context_type: str,
        mode: str,
        doc_count: int,
        raw_chars: int,
        prepared_chars: int,
        output_chars: int,
        fallback_reason: str | None = None,
    ) -> None:
        payload = {
            "question": question,
            "context_type": context_type,
            "mode": mode,
            "doc_count": doc_count,
            "raw_chars": raw_chars,
            "prepared_chars": prepared_chars,
            "output_chars": output_chars,
        }
        if fallback_reason:
            payload["fallback_reason"] = fallback_reason

        with start_as_current_span(
            name="search_context_result",
            type="tool",
            input=payload,
            project_name=self._get_opik_project_name(),
        ):
            return

    def _get_opik_project_name(self) -> str | None:
        if settings.OPIK is None:
            return None
        return settings.OPIK.PROJECT

    def _format_doc_reference(
        self,
        *,
        index: int,
        doc: dict[str, Any],
        include_excerpt: bool = False,
    ) -> str:
        parts = [
            f"- 문서 {index}",
            f"c_id={doc.get('c_id')}",
            f"year={doc.get('creation_year')}",
        ]
        if doc.get("source_spec"):
            parts.append(f"source_spec={doc.get('source_spec')}")
        elif doc.get("source"):
            parts.append(f"source={doc.get('source')}")

        line = ", ".join(parts)
        if include_excerpt and doc.get("content"):
            return f"{line}, excerpt={doc.get('content')}"
        return line
