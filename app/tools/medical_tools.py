import re
from typing import Any

from langchain_core.tools import BaseTool, tool

from app.services.elasticsearch_service import ElasticsearchService


SYMPTOM_TERMS = (
    "기침",
    "발열",
    "열",
    "가래",
    "호흡곤란",
    "두통",
    "복통",
    "인후통",
    "콧물",
    "오한",
    "흉통",
    "설사",
    "구토",
    "피로",
    "체중감소",
    "치통",
    "이가 아파",
    "이가 아프",
    "목이 아파",
    "배가 아파",
)

_search_service: ElasticsearchService | None = None


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
def symptom_duration_parser(query: str) -> str:
    """질문에서 지속 기간 정보를 추출해 구조화합니다."""
    return format_symptom_duration(parse_symptom_duration(query))


@tool
def symptom_parser(query: str) -> str:
    """질문에서 증상 표현을 추출해 구조화합니다."""
    return format_symptoms(parse_symptoms(query))


def build_location_query(query: str, region: str = "") -> str:
    parts = [query]
    if region:
        parts.append(region)
    parts.extend(["병원", "의료기관", "위치", "지역"])
    return " ".join(part for part in parts if part)


def parse_symptom_duration(question: str) -> dict[str, Any]:
    duration_patterns = [
        r"(\d+\s*시간(?:째)?)",
        r"(\d+\s*일(?:째)?)",
        r"(\d+\s*주(?:째)?)",
        r"(\d+\s*달(?:째)?)",
        r"(\d+\s*개월(?:째)?)",
        r"(며칠(?:째)?)",
        r"(몇\s*주(?:째)?)",
        r"(몇\s*달(?:째)?)",
        r"(오래\s*지속)",
        r"(계속)",
    ]
    duration_matches: list[str] = []
    for pattern in duration_patterns:
        duration_matches.extend(match.strip() for match in re.findall(pattern, question))

    return {
        "question": question,
        "durations": duration_matches,
        "has_duration_context": bool(duration_matches),
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


def parse_symptoms(question: str) -> dict[str, Any]:
    matched_terms = [term for term in SYMPTOM_TERMS if term in question]
    normalized_symptoms: list[str] = []
    for term in matched_terms:
        normalized = normalize_symptom(term)
        if normalized not in normalized_symptoms:
            normalized_symptoms.append(normalized)

    return {
        "question": question,
        "symptoms": normalized_symptoms,
        "has_symptom_context": bool(normalized_symptoms),
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


def normalize_symptom(term: str) -> str:
    if term in ("이가 아파", "이가 아프", "치통"):
        return "치통"
    if term == "목이 아파":
        return "인후통"
    if term == "배가 아파":
        return "복통"
    return term
