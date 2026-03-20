import re
from typing import Any

from opik import start_as_current_span
from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from app.core.config import settings
from app.services.elasticsearch_service import ElasticsearchService
from app.services.search_context_service import SearchContextService


_search_service: ElasticsearchService | None = None
_tool_llm: ChatOpenAI | None = None
_search_context_service: SearchContextService | None = None
_triage_llm: ChatOpenAI | None = None


class SymptomExtraction(BaseModel):
    symptoms: list[str] = Field(default_factory=list, description="질문에서 파악한 증상 목록")


class DurationExtraction(BaseModel):
    durations: list[str] = Field(default_factory=list, description="질문에서 파악한 지속 기간 표현 목록")


class EmergencySymptomAssessment(BaseModel):
    question: str
    symptoms: list[str] = Field(default_factory=list)
    concerning_symptoms: list[str] = Field(default_factory=list)
    urgency_level: str = "low"
    needs_clarification: bool = False
    clarifying_questions: list[str] = Field(default_factory=list)
    safety_message: str | None = None


class EmergencyFollowUpResolution(BaseModel):
    should_seek_emergency_care: bool = False
    message: str | None = None
    reason: str | None = None


class TriageLLMOutput(BaseModel):
    concerning_symptoms: list[str] = Field(default_factory=list)
    urgency_level: str = "low"
    needs_clarification: bool = False
    clarifying_questions: list[str] = Field(default_factory=list)
    safety_message: str | None = None


class SubjectValidityAssessment(BaseModel):
    is_valid_subject: bool = True
    reason: str | None = None
    response: str | None = None


class SubjectValidityLLMOutput(BaseModel):
    is_valid_subject: bool = True
    subject_type: str = "human_or_body"
    mentioned_location: str | None = None
    reason: str | None = None
    response: str | None = None


EMERGENCY_SYMPTOM_RULES: dict[str, dict[str, Any]] = {
    "출혈": {
        "keywords": {"출혈", "피", "혈변", "토혈", "객혈", "혈뇨", "질출혈", "코피", "피가 난다"},
        "urgency": "high",
        "questions": [
            "피가 나는 곳이 어디인지 알려주세요.",
            "출혈이 많거나 계속 나는지 알려주세요.",
            "어지럽거나 숨이 차지는 않는지 알려주세요.",
        ],
        "safety_message": "출혈이 많거나 멈추지 않거나 어지럼증이 함께 있으면 즉시 진료가 필요할 수 있습니다.",
    },
    "호흡곤란": {
        "keywords": {"호흡곤란", "숨참", "숨이 차다", "숨막힘", "호흡 곤란", "숨을 쉬기 힘들", "숨쉬기 힘들"},
        "urgency": "high",
        "questions": [
            "숨찬 정도가 어느 정도인지 알려주세요.",
            "가만히 있어도 숨이 찬지 알려주세요.",
            "가슴 통증이 함께 있는지 알려주세요.",
        ],
        "safety_message": "호흡이 많이 불편하거나 흉통, 청색증이 있으면 즉시 응급 진료가 필요할 수 있습니다.",
    },
    "흉통": {
        "keywords": {"흉통", "가슴통증", "가슴이 아프다", "가슴 답답", "가슴 통증"},
        "urgency": "high",
        "questions": [
            "가슴 통증이 언제부터 있었는지 알려주세요.",
            "통증이 심해지는지 알려주세요.",
            "숨차거나 식은땀이 함께 있는지 알려주세요.",
        ],
        "safety_message": "심한 흉통이나 호흡곤란, 식은땀이 함께 있으면 즉시 진료가 필요할 수 있습니다.",
    },
    "의식저하": {
        "keywords": {"의식저하", "실신", "기절", "혼미", "반응이 느리다"},
        "urgency": "high",
        "questions": [
            "쓰러지거나 잠깐 의식을 잃은 적이 있는지 알려주세요.",
            "지금도 깨우기 어렵거나 반응이 느린지 알려주세요.",
        ],
        "safety_message": "의식 저하나 실신이 있으면 즉시 응급 평가가 필요할 수 있습니다.",
    },
    "구토": {
        "keywords": {"구토", "토함", "토했다", "구역질"},
        "urgency": "medium",
        "questions": [
            "구토를 몇 번 했는지 알려주세요.",
            "물을 마셔도 못 넘길 정도인지 알려주세요.",
            "피가 섞였거나 심한 복통이 있는지 알려주세요.",
        ],
        "safety_message": "구토가 반복되거나 피가 섞이거나 수분 섭취가 어려우면 빠른 진료가 필요할 수 있습니다.",
    },
    "설사": {
        "keywords": {"설사", "물설사", "묽은 변"},
        "urgency": "medium",
        "questions": [
            "설사가 하루에 몇 번 정도인지 알려주세요.",
            "피가 섞였는지 알려주세요.",
            "입마름이나 어지럼증이 있는지 알려주세요.",
        ],
        "safety_message": "설사가 심하거나 혈변, 탈수 증상이 있으면 빠른 진료가 필요할 수 있습니다.",
    },
    "메스꺼움": {
        "keywords": {"메스꺼움", "오심", "속이 메스껍다", "울렁거림"},
        "urgency": "medium",
        "questions": [
            "메스꺼움이 언제부터 시작됐는지 알려주세요.",
            "구토나 복통이 함께 있는지 알려주세요.",
            "음식이나 물을 마실 수 있는지 알려주세요.",
        ],
        "safety_message": "메스꺼움이 심해 먹거나 마시기 어렵거나 다른 증상이 함께 있으면 진료가 필요할 수 있습니다.",
    },
}


FOLLOW_UP_SEVERE_INDICATORS: dict[str, set[str]] = {
    "출혈": {"많", "계속", "멈추지", "어지럼", "식은땀", "숨이 차", "숨차"},
    "호흡곤란": {"가만히 있어도", "말하기 힘", "숨이 안", "계속", "가슴 통증", "흉통", "식은땀", "어지럼"},
    "흉통": {"심해", "계속", "가슴 통증", "흉통", "숨차", "호흡곤란", "식은땀", "어지럼"},
    "의식저하": {"쓰러", "의식", "반응이 느", "깨우기 어렵"},
}

BLEEDING_LOCATION_HINTS: tuple[str, ...] = (
    "귀",
    "코",
    "입",
    "잇몸",
    "목",
    "가래",
    "기침",
    "변",
    "소변",
    "질",
    "항문",
    "상처",
)

NON_HUMAN_SUBJECT_HINTS: tuple[str, ...] = (
    "노트북",
    "컴퓨터",
    "pc",
    "휴지통",
    "쓰레기통",
    "책상",
    "의자",
    "가방",
    "신발",
    "자동차",
    "차",
    "핸드폰",
    "휴대폰",
    "폰",
    "냉장고",
    "세탁기",
    "에어컨",
    "로봇",
    "인형",
    "장난감",
    "TV",
    "텔레비전",
)


def configure_medical_tools(search_service: ElasticsearchService) -> None:
    global _search_service, _search_context_service
    _search_service = search_service
    _search_context_service = SearchContextService()


def get_medical_tools(search_service: ElasticsearchService) -> list[BaseTool]:
    configure_medical_tools(search_service)
    return [
        medical_search,
        hospital_search,
        symptom_duration_parser,
        symptom_parser,
        subject_validity_check,
        emergency_symptom_triage,
    ]


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


def _get_search_context_service() -> SearchContextService:
    if _search_context_service is None:
        raise RuntimeError("SearchContextService is not configured for medical tools.")
    return _search_context_service


def _get_triage_llm() -> ChatOpenAI:
    global _triage_llm
    if _triage_llm is None:
        _triage_llm = ChatOpenAI(
            model=settings.TRIAGE_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=0,
        )
    return _triage_llm


def _get_opik_project_name() -> str | None:
    if settings.OPIK is None:
        return None
    return settings.OPIK.PROJECT


@tool
async def medical_search(query: str) -> str:
    """의학 문서 인덱스에서 질병, 진단, 치료 등 설명형 정보를 검색합니다."""
    with start_as_current_span(
        name="medical_search",
        type="tool",
        input={"query": query},
        project_name=_get_opik_project_name(),
    ):
        documents = await _get_search_service().search(query)
        if not documents:
            return "의학 지식 검색 결과가 없습니다."
        return await build_medical_context(question=query, documents=documents)


@tool
async def hospital_search(query: str, region: str = "") -> str:
    """위치 기반 병원 질문에는 직접 병원 정보를 제공하지 않고 검색 또는 119 문의를 안내합니다."""
    with start_as_current_span(
        name="hospital_search",
        type="tool",
        input={"query": query, "region": region},
        project_name=_get_opik_project_name(),
    ):
        return build_location_guidance(query=query, region=region)


@tool
async def symptom_duration_parser(query: str) -> str:
    """질문에서 지속 기간 정보를 추출해 구조화합니다."""
    with start_as_current_span(
        name="symptom_duration_parser",
        type="tool",
        input={"query": query},
        project_name=_get_opik_project_name(),
    ):
        return format_symptom_duration(await parse_symptom_duration(query))


@tool
async def symptom_parser(query: str) -> str:
    """질문에서 증상 표현을 추출해 구조화합니다."""
    with start_as_current_span(
        name="symptom_parser",
        type="tool",
        input={"query": query},
        project_name=_get_opik_project_name(),
    ):
        return format_symptoms(await parse_symptoms(query))


@tool
async def emergency_symptom_triage(query: str) -> str:
    """질문의 증상 중 위급 가능성이 있는 항목을 찾아 추가 확인 질문이 필요한지 판단합니다."""
    with start_as_current_span(
        name="emergency_symptom_triage",
        type="tool",
        input={"query": query},
        project_name=_get_opik_project_name(),
    ):
        symptoms = await parse_symptoms(query)
        assessment = await assess_emergency_symptoms_with_llm(
            question=query,
            symptoms=symptoms.get("symptoms", []),
        )
        return format_emergency_symptom_triage(assessment)


@tool
async def subject_validity_check(query: str) -> str:
    """증상 질문의 대상이 사람의 신체인지 확인하고, 아니면 답변이 어렵다고 안내합니다."""
    with start_as_current_span(
        name="subject_validity_check",
        type="tool",
        input={"query": query},
        project_name=_get_opik_project_name(),
    ):
        assessment = await assess_subject_validity(query=query)
        return format_subject_validity_check(assessment)


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


async def assess_subject_validity(query: str) -> SubjectValidityAssessment:
    normalized_query = query.strip()
    extracted_location = _extract_subject_location(normalized_query)
    if _looks_like_non_human_location(extracted_location):
        return SubjectValidityAssessment(
            is_valid_subject=False,
            reason="non_human_subject",
            response=(
                "[질문 대상 확인]\n"
                "사람의 신체가 아닌 대상의 증상은 의료적으로 답변하기 어렵습니다. "
                "실제 사람에게 나타난 증상이라면 증상이 생긴 부위를 다시 알려주세요."
            ),
        )

    try:
        extractor = _get_triage_llm().with_structured_output(SubjectValidityLLMOutput)
        result = await extractor.ainvoke(
            [
                (
                    "system",
                    "너는 의료 질문의 대상이 사람의 신체인지 판별하는 분류 모델이다. "
                    "특히 '~에서', '~쪽에서', '~부위에서'처럼 위치 표현이 나오면 그 위치가 사람의 신체 부위인지, "
                    "사람이 아닌 물건/무생물/장치인지 문맥으로 판단하라. "
                    "예: '귀에서 피가 나요'는 사람 신체 부위이므로 valid, "
                    "'휴지통에서 피가 나요', '노트북에서 피가 나요'는 non_human이므로 invalid. "
                    "질문이 실제 사람의 신체 증상에 대한 것이면 is_valid_subject=true, subject_type='human_or_body'로 반환하라. "
                    "사람이 아닌 물건, 무생물, 장치, 비유적 대상의 증상 문의라면 is_valid_subject=false, "
                    "subject_type='non_human'으로 반환하라. 애매하면 subject_type='unclear'로 두고 false로 반환하라. "
                    "mentioned_location에는 질문에서 판단의 핵심이 된 위치나 대상을 짧게 적어라. "
                    "false일 때만 한국어 response를 작성하라. response는 "
                    "'사람의 신체가 아닌 대상의 증상은 의료적으로 답변하기 어렵습니다. 실제 사람에게 나타난 증상이라면 증상이 생긴 부위를 다시 알려주세요.' "
                    "처럼 짧고 직접적으로 작성하라.",
                ),
                ("human", normalized_query),
            ]
        )
        if _looks_like_non_human_location(result.mentioned_location):
            response = result.response or (
                "[질문 대상 확인]\n"
                "사람의 신체가 아닌 대상의 증상은 의료적으로 답변하기 어렵습니다. "
                "실제 사람에게 나타난 증상이라면 증상이 생긴 부위를 다시 알려주세요."
            )
            return SubjectValidityAssessment(
                is_valid_subject=False,
                reason=result.reason or "non_human_subject",
                response=response,
            )

        if not result.is_valid_subject:
            response = result.response or (
                "[질문 대상 확인]\n"
                "사람의 신체가 아닌 대상의 증상은 의료적으로 답변하기 어렵습니다. "
                "실제 사람에게 나타난 증상이라면 증상이 생긴 부위를 다시 알려주세요."
            )
            return SubjectValidityAssessment(
                is_valid_subject=False,
                reason=result.reason or result.subject_type or "non_human_subject",
                response=response,
            )
        return SubjectValidityAssessment(
            is_valid_subject=True,
            reason=result.reason or result.subject_type,
        )
    except Exception:
        if _looks_like_non_human_subject(normalized_query):
            return SubjectValidityAssessment(
                is_valid_subject=False,
                reason="non_human_subject",
                response=(
                    "[질문 대상 확인]\n"
                    "사람의 신체가 아닌 대상의 증상은 의료적으로 답변하기 어렵습니다. "
                    "실제 사람에게 나타난 증상이라면 증상이 생긴 부위를 다시 알려주세요."
                ),
            )

        return SubjectValidityAssessment(is_valid_subject=True)


def format_symptoms(parsed: dict[str, Any]) -> str:
    symptoms = ", ".join(parsed.get("symptoms", [])) or "명시되지 않음"
    return "\n".join(
        [
            "[증상 분석]",
            f"원문 질문: {parsed.get('question', '')}",
            f"추출된 증상: {symptoms}",
        ]
    )


def format_subject_validity_check(assessment: SubjectValidityAssessment) -> str:
    if assessment.is_valid_subject:
        return "[질문 대상 확인]\n사람의 신체 증상 질문으로 볼 수 있습니다."
    return assessment.response or (
        "[질문 대상 확인]\n"
        "사람의 신체가 아닌 대상의 증상은 의료적으로 답변하기 어렵습니다. "
        "실제 사람에게 나타난 증상이라면 증상이 생긴 부위를 다시 알려주세요."
    )


def assess_emergency_symptoms(question: str, symptoms: list[str]) -> EmergencySymptomAssessment:
    normalized_question = question.strip()
    lowered_question = normalized_question.lower()
    normalized_symptoms = [symptom.strip() for symptom in symptoms if symptom.strip()]

    matched_categories: list[str] = []
    clarifying_questions: list[str] = []
    safety_messages: list[str] = []
    urgency_level = "low"

    for category, rule in EMERGENCY_SYMPTOM_RULES.items():
        if _matches_emergency_rule(
            lowered_question=lowered_question,
            symptoms=normalized_symptoms,
            keywords=rule["keywords"],
        ):
            matched_categories.append(category)
            tailored_questions = _tailor_clarifying_questions(
                category=category,
                question=normalized_question,
                questions=rule["questions"],
            )
            for question_text in tailored_questions:
                if question_text not in clarifying_questions:
                    clarifying_questions.append(question_text)
            if rule["safety_message"] not in safety_messages:
                safety_messages.append(rule["safety_message"])
            urgency_level = _merge_urgency_levels(urgency_level, rule["urgency"])

    return EmergencySymptomAssessment(
        question=normalized_question,
        symptoms=normalized_symptoms,
        concerning_symptoms=matched_categories,
        urgency_level=urgency_level,
        needs_clarification=bool(matched_categories),
        clarifying_questions=clarifying_questions,
        safety_message=" ".join(safety_messages) if safety_messages else None,
    )


async def assess_emergency_symptoms_with_llm(
    question: str,
    symptoms: list[str],
) -> EmergencySymptomAssessment:
    fallback = assess_emergency_symptoms(question=question, symptoms=symptoms)

    try:
        extractor = _get_triage_llm().with_structured_output(TriageLLMOutput)
        result = await extractor.ainvoke(
            [
                (
                    "system",
                    "너는 의료 응급 신호를 선별하는 triage 보조 모델이다. "
                    "사용자 질문과 추출된 증상을 보고, 응급 가능성이 있으면 간단한 확인 질문만 만들어라. "
                    "질문에 이미 포함된 정보는 다시 묻지 마라. "
                    "clarifying_questions는 1개에서 3개 사이의 짧은 한국어 질문으로 작성하라. "
                    "urgency_level은 low, medium, high 중 하나로 반환하라. "
                    "위험 신호가 명확하지 않으면 needs_clarification을 false로 두어라.",
                ),
                (
                    "human",
                    f"사용자 질문: {question}\n"
                    f"추출된 증상: {symptoms}",
                ),
            ]
        )

        normalized_questions = [item.strip() for item in result.clarifying_questions if item.strip()][:3]
        if result.needs_clarification and not normalized_questions:
            return fallback

        return EmergencySymptomAssessment(
            question=question.strip(),
            symptoms=[symptom.strip() for symptom in symptoms if symptom.strip()],
            concerning_symptoms=result.concerning_symptoms or fallback.concerning_symptoms,
            urgency_level=result.urgency_level if result.urgency_level in {"low", "medium", "high"} else fallback.urgency_level,
            needs_clarification=result.needs_clarification,
            clarifying_questions=normalized_questions,
            safety_message=result.safety_message,
        )
    except Exception:
        return fallback


def _tailor_clarifying_questions(category: str, question: str, questions: list[str]) -> list[str]:
    tailored = list(questions)

    if category == "출혈" and _contains_bleeding_location(question):
        tailored = [item for item in tailored if "피가 나는 곳" not in item]

    return tailored


def _contains_bleeding_location(question: str) -> bool:
    lowered = question.lower()
    if "에서 피" in lowered or "부위" in lowered:
        return True
    return any(location in question for location in BLEEDING_LOCATION_HINTS)


def _matches_emergency_rule(lowered_question: str, symptoms: list[str], keywords: set[str]) -> bool:
    if any(keyword.lower() in lowered_question for keyword in keywords):
        return True

    for symptom in symptoms:
        lowered_symptom = symptom.lower()
        if any(keyword.lower() in lowered_symptom for keyword in keywords):
            return True
    return False


def _merge_urgency_levels(current: str, candidate: str) -> str:
    order = {"low": 0, "medium": 1, "high": 2}
    return candidate if order[candidate] > order[current] else current


def _looks_like_non_human_subject(query: str) -> bool:
    if any(body_part in query for body_part in BLEEDING_LOCATION_HINTS):
        return False

    lowered = query.lower()
    has_non_human_subject = any(hint.lower() in lowered for hint in NON_HUMAN_SUBJECT_HINTS)
    has_symptom_like_phrase = any(
        keyword in lowered
        for keyword in (
            "피",
            "출혈",
            "토",
            "구토",
            "설사",
            "숨",
            "호흡",
            "아프",
            "통증",
            "메스꺼",
        )
    )
    return has_non_human_subject and has_symptom_like_phrase


def _extract_subject_location(query: str) -> str | None:
    patterns = [
        r"(.+?)에서",
        r"(.+?)쪽에서",
        r"(.+?)부위에서",
    ]
    for pattern in patterns:
        match = re.search(pattern, query)
        if match:
            return match.group(1).strip()
    return None


def _looks_like_non_human_location(location: str | None) -> bool:
    if not location:
        return False
    lowered = location.lower()
    if any(body_part in location for body_part in BLEEDING_LOCATION_HINTS):
        return False
    return any(hint.lower() in lowered for hint in NON_HUMAN_SUBJECT_HINTS)


def format_emergency_symptom_triage(assessment: EmergencySymptomAssessment) -> str:
    if not assessment.needs_clarification:
        return "[응급 증상 확인]\n현재 증상만으로는 추가 응급 확인 질문이 필요하지 않습니다."

    urgency_map = {
        "high": "응급 가능성이 높으므로",
        "medium": "응급 가능성을 판단하기 위해",
        "low": "상태를 더 정확히 보기 위해",
    }
    opening = urgency_map.get(assessment.urgency_level, "상태를 더 정확히 보기 위해")

    questions = " ".join(assessment.clarifying_questions)
    return f"[응급 증상 확인]\n{opening} {questions}".strip()


def build_location_guidance(query: str, region: str = "") -> str:
    location_hint = " ".join(part for part in [region.strip(), query.strip()] if part).strip()
    lines = [
        "[위치 안내]",
        "현재는 위치 기반으로 정확한 병원 정보를 직접 제공할 수 없습니다.",
    ]

    if location_hint:
        lines.append(f"질문 내용: {location_hint}")

    lines.append("주변 병원이나 응급실은 지도 앱이나 포털 검색으로 확인해 주세요.")
    lines.append("응급 상황이 의심되면 가까운 응급실을 찾거나 119에 연락해 안내를 받는 것이 좋습니다.")
    return "\n".join(lines)


def resolve_emergency_follow_up(original_question: str, follow_up: str) -> EmergencyFollowUpResolution:
    assessment = assess_emergency_symptoms(question=original_question, symptoms=[])
    combined = f"{original_question}\n{follow_up}".lower()

    for category in assessment.concerning_symptoms:
        indicators = FOLLOW_UP_SEVERE_INDICATORS.get(category, set())
        if any(indicator in combined for indicator in indicators):
            return EmergencyFollowUpResolution(
                should_seek_emergency_care=True,
                reason=category,
                message=_build_emergency_follow_up_message(category),
            )

    return EmergencyFollowUpResolution()


def _build_emergency_follow_up_message(category: str) -> str:
    messages = {
        "출혈": "응급 가능성이 높아 보여요. 출혈이 많거나 멈추지 않으면 즉시 응급실이나 119 도움이 필요할 수 있습니다.",
        "호흡곤란": "응급 가능성이 높아 보여요. 가만히 있어도 숨이 차거나 가슴 통증이 함께 있으면 즉시 응급실이나 119 도움이 필요할 수 있습니다.",
        "흉통": "응급 가능성이 높아 보여요. 심한 흉통이나 호흡곤란이 함께 있으면 즉시 응급실이나 119 도움이 필요할 수 있습니다.",
        "의식저하": "응급 가능성이 높아 보여요. 의식 저하나 실신이 있으면 즉시 응급실이나 119 도움이 필요할 수 있습니다.",
    }
    return messages.get(category, "응급 가능성이 높아 보여요. 즉시 진료가 필요할 수 있습니다.")


async def build_medical_context(question: str, documents: list[dict[str, Any]]) -> str:
    return await _get_search_context_service().build_context(
        question=question,
        documents=documents,
        context_type="medical",
    )


async def build_hospital_context(question: str, documents: list[dict[str, Any]]) -> str:
    return await _get_search_context_service().build_context(
        question=question,
        documents=documents,
        context_type="hospital",
    )


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
