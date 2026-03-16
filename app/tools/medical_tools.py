from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from app.core.config import settings
from app.services.elasticsearch_service import ElasticsearchService


_search_service: ElasticsearchService | None = None
_tool_llm: ChatOpenAI | None = None


class SymptomExtraction(BaseModel):
    symptoms: list[str] = Field(default_factory=list, description="질문에서 파악한 증상 목록")


class DurationExtraction(BaseModel):
    durations: list[str] = Field(default_factory=list, description="질문에서 파악한 지속 기간 표현 목록")


def configure_medical_tools(search_service: ElasticsearchService) -> None:
    global _search_service
    _search_service = search_service


def get_medical_tools(search_service: ElasticsearchService) -> list[BaseTool]:
    configure_medical_tools(search_service)
    return [medical_search, hospital_search, symptom_duration_parser, symptom_parser]


def _get_search_service() -> ElasticsearchService:
    if _search_service is None:
        raise RuntimeError("ElasticsearchService is not configured for medical tools.")
    return _search_service


def _get_tool_llm() -> ChatOpenAI:
    global _tool_llm
    if _tool_llm is None:
        _tool_llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=0,
        )
    return _tool_llm


@tool
async def medical_search(query: str) -> str:
    """의학 문서 인덱스에서 질병, 진단, 치료 등 설명형 정보를 검색합니다."""
    documents = await _get_search_service().search(query)
    if not documents:
        return "의학 지식 검색 결과가 없습니다."
    return build_medical_context(documents)


@tool
async def hospital_search(query: str, region: str = "") -> str:
    """같은 edu-collection 인덱스에서 지역과 병원 관련 문맥을 우선 검색합니다."""
    documents = await _get_search_service().search(build_location_query(query=query, region=region))
    if not documents:
        return "지역 또는 병원 관련 검색 결과가 없습니다."
    return build_hospital_context(documents)


@tool
async def symptom_duration_parser(query: str) -> str:
    """질문에서 지속 기간 정보를 추출해 구조화합니다."""
    return format_symptom_duration(await parse_symptom_duration(query))


@tool
async def symptom_parser(query: str) -> str:
    """질문에서 증상 표현을 추출해 구조화합니다."""
    return format_symptoms(await parse_symptoms(query))


def build_location_query(query: str, region: str = "") -> str:
    parts = [query]
    if region:
        parts.append(region)
    parts.extend(["병원", "의료기관", "위치", "지역"])
    return " ".join(part for part in parts if part)


async def parse_symptom_duration(question: str) -> dict[str, Any]:
    extractor = _get_tool_llm().with_structured_output(DurationExtraction)
    result = await extractor.ainvoke(
        [
            (
                "system",
                "사용자 질문에서 지속 기간 표현만 추출하라. 없으면 빈 리스트를 반환하라. "
                "의미를 보존한 원문 표현을 최대한 유지하라.",
            ),
            ("human", question),
        ]
    )
    return {
        "question": question,
        "durations": result.durations,
        "has_duration_context": bool(result.durations),
    }


def format_symptom_duration(parsed: dict[str, Any]) -> str:
    durations = ", ".join(parsed.get("durations", [])) or "명시되지 않음"
    return "\n".join(
        [
            "[지속기간 분석]",
            f"원문 질문: {parsed.get('question', '')}",
            f"추출된 지속기간: {durations}",
        ]
    )


async def parse_symptoms(question: str) -> dict[str, Any]:
    extractor = _get_tool_llm().with_structured_output(SymptomExtraction)
    result = await extractor.ainvoke(
        [
            (
                "system",
                "사용자 질문에서 증상 표현만 추출하라. 없으면 빈 리스트를 반환하라. "
                "동의어는 대표 증상명으로 정규화하라. 예: '이가 아파'는 '치통'.",
            ),
            ("human", question),
        ]
    )
    return {
        "question": question,
        "symptoms": result.symptoms,
        "has_symptom_context": bool(result.symptoms),
    }


def format_symptoms(parsed: dict[str, Any]) -> str:
    symptoms = ", ".join(parsed.get("symptoms", [])) or "명시되지 않음"
    return "\n".join(
        [
            "[증상 분석]",
            f"원문 질문: {parsed.get('question', '')}",
            f"추출된 증상: {symptoms}",
        ]
    )


def build_medical_context(documents: list[dict[str, Any]]) -> str:
    chunks = []
    for index, doc in enumerate(documents, start=1):
        chunks.append(
            "\n".join(
                [
                    f"[문서 {index}]",
                    f"c_id: {doc.get('c_id')}",
                    f"creation_year: {doc.get('creation_year')}",
                    f"domain: {doc.get('domain')}",
                    f"source: {doc.get('source')}",
                    f"source_spec: {doc.get('source_spec')}",
                    f"content: {doc.get('content')}",
                ]
            )
        )
    return "\n\n".join(chunks)


def build_hospital_context(documents: list[dict[str, Any]]) -> str:
    chunks = []
    for index, doc in enumerate(documents, start=1):
        chunks.append(
            "\n".join(
                [
                    f"[지역/병원 문맥 {index}]",
                    f"c_id: {doc.get('c_id')}",
                    f"creation_year: {doc.get('creation_year')}",
                    f"content: {doc.get('content')}",
                ]
            )
        )
    return "\n\n".join(chunks)


def build_sources(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sources = []
    for doc in documents:
        sources.append(
            {
                "c_id": doc.get("c_id"),
                "score": doc.get("score"),
                "creation_year": doc.get("creation_year"),
                "domain": doc.get("domain"),
                "source": doc.get("source"),
                "source_spec": doc.get("source_spec"),
            }
        )
    return sources

