import pytest

from app.tools.medical_tools import (
    _extract_subject_location,
    assess_emergency_symptoms,
    assess_emergency_symptoms_with_llm,
    assess_subject_validity,
    build_location_guidance,
    format_emergency_symptom_triage,
    format_subject_validity_check,
    resolve_emergency_follow_up,
    SubjectValidityAssessment,
    SubjectValidityLLMOutput,
    TriageLLMOutput,
)
from app.agents.search_agent import Agent
from langchain_core.messages import ToolMessage


def test_emergency_symptom_assessment_detects_high_risk_bleeding():
    assessment = assess_emergency_symptoms(
        question="기침하다가 피가 나고 숨이 차요",
        symptoms=["객혈", "호흡곤란"],
    )

    assert assessment.needs_clarification is True
    assert assessment.urgency_level == "high"
    assert "출혈" in assessment.concerning_symptoms
    assert "호흡곤란" in assessment.concerning_symptoms
    assert len(assessment.clarifying_questions) >= 3
    assert assessment.safety_message is not None


def test_emergency_symptom_assessment_returns_low_for_non_acute_symptoms():
    assessment = assess_emergency_symptoms(
        question="이가 아파요",
        symptoms=["치통"],
    )

    assert assessment.needs_clarification is False
    assert assessment.urgency_level == "low"
    assert assessment.concerning_symptoms == []
    assert assessment.clarifying_questions == []


def test_format_emergency_symptom_triage_includes_guidance_block():
    assessment = assess_emergency_symptoms(
        question="토하고 설사가 있어요",
        symptoms=["구토", "설사"],
    )

    formatted = format_emergency_symptom_triage(assessment)

    assert "[응급 증상 확인]" in formatted
    assert "응급 가능성을 판단하기 위해" in formatted
    assert "구토를 몇 번 했는지 알려주세요." in formatted
    assert "안내:" not in formatted


def test_format_emergency_symptom_triage_is_concise_for_high_urgency():
    assessment = assess_emergency_symptoms(
        question="갑자기 숨을 쉬기 힘들어",
        symptoms=["호흡 곤란"],
    )

    formatted = format_emergency_symptom_triage(assessment)

    assert "응급 가능성이 높으므로" in formatted
    assert "원문 질문:" not in formatted
    assert "추출된 증상:" not in formatted
    assert "응급 가능성:" not in formatted


def test_bleeding_triage_skips_location_question_when_location_already_given():
    assessment = assess_emergency_symptoms(
        question="귀에서 피가 나고 있어",
        symptoms=["출혈"],
    )

    formatted = format_emergency_symptom_triage(assessment)

    assert "피가 나는 곳이 어디인지" not in formatted
    assert "출혈이 많거나 계속 나는지 알려주세요." in formatted


def test_agent_runs_guard_and_triage_only_when_symptoms_exist():
    agent = Agent()

    assert agent._should_run_subject_validity_check(
        [ToolMessage(content="[증상 분석]\n원문 질문: 숨이 차요\n추출된 증상: 호흡곤란", tool_call_id="1")],
        [],
        {},
    ) is True

    assert agent._should_run_emergency_triage(
        [ToolMessage(content="[증상 분석]\n원문 질문: 숨이 차요\n추출된 증상: 호흡곤란", tool_call_id="1")],
        ["subject_validity_check"],
        {},
    ) is True

    assert agent._should_run_emergency_triage(
        [ToolMessage(content="[증상 분석]\n원문 질문: 숨이 차요\n추출된 증상: 호흡곤란", tool_call_id="1")],
        [],
        {},
    ) is False

    assert agent._should_run_emergency_triage(
        [ToolMessage(content="[증상 분석]\n원문 질문: 안녕하세요\n추출된 증상: 명시되지 않음", tool_call_id="2")],
        [],
        {},
    ) is False

    assert agent._should_run_emergency_triage(
        [ToolMessage(content="[증상 분석]\n원문 질문: 숨이 차요\n추출된 증상: 호흡곤란", tool_call_id="3")],
        ["emergency_symptom_triage"],
        {},
    ) is False


def test_resolve_emergency_follow_up_escalates_severe_breathing_response():
    resolution = resolve_emergency_follow_up(
        original_question="갑자기 숨을 쉬기 힘들어",
        follow_up="가만히 있어도 숨이 차고 가슴 통증도 있어",
    )

    assert resolution.should_seek_emergency_care is True
    assert resolution.reason == "호흡곤란"
    assert resolution.message is not None


def test_resolve_emergency_follow_up_allows_normal_flow_when_not_severe():
    resolution = resolve_emergency_follow_up(
        original_question="갑자기 숨을 쉬기 힘들어",
        follow_up="조금 답답한데 지금은 괜찮고 말은 잘 돼",
    )

    assert resolution.should_seek_emergency_care is False


def test_build_location_guidance_recommends_search_or_119():
    guidance = build_location_guidance(query="강남 근처 응급실 알려줘", region="강남")

    assert "[위치 안내]" in guidance
    assert "정확한 병원 정보를 직접 제공할 수 없습니다." in guidance
    assert "지도 앱이나 포털 검색" in guidance
    assert "119" in guidance


def test_format_subject_validity_check_blocks_non_human_subject():
    formatted = format_subject_validity_check(
        SubjectValidityAssessment(
            is_valid_subject=False,
            reason="non_human_subject",
            response=(
                "[질문 대상 확인]\n"
                "사람의 신체가 아닌 대상의 증상은 의료적으로 답변하기 어렵습니다. "
                "실제 사람에게 나타난 증상이라면 증상이 생긴 부위를 다시 알려주세요."
            ),
        )
    )

    assert "[질문 대상 확인]" in formatted
    assert "답변하기 어렵습니다" in formatted


@pytest.mark.asyncio
async def test_assess_subject_validity_blocks_non_human_subject_by_rule():
    assessment = await assess_subject_validity("노트북에서 피가 나요")

    assert assessment.is_valid_subject is False
    assert assessment.reason == "non_human_subject"
    assert assessment.response is not None


@pytest.mark.asyncio
async def test_assess_subject_validity_blocks_trash_can_symptom_question():
    assessment = await assess_subject_validity("휴지통에서 피가 나고 있어")

    assert assessment.is_valid_subject is False
    assert assessment.reason == "non_human_subject"
    assert assessment.response is not None


@pytest.mark.asyncio
async def test_assess_subject_validity_allows_human_body_question():
    assessment = await assess_subject_validity("귀에서 피가 나요")

    assert assessment.is_valid_subject is True


@pytest.mark.asyncio
async def test_assess_subject_validity_uses_llm_for_location_context(monkeypatch):
    class DummyExtractor:
        async def ainvoke(self, _messages):
            return SubjectValidityLLMOutput(
                is_valid_subject=False,
                subject_type="non_human",
                mentioned_location="휴지통",
                reason="non_human_subject",
                response=(
                    "[질문 대상 확인]\n"
                    "사람의 신체가 아닌 대상의 증상은 의료적으로 답변하기 어렵습니다. "
                    "실제 사람에게 나타난 증상이라면 증상이 생긴 부위를 다시 알려주세요."
                ),
            )

    class DummyLLM:
        def with_structured_output(self, _schema):
            return DummyExtractor()

    monkeypatch.setattr("app.tools.medical_tools._get_triage_llm", lambda: DummyLLM())

    assessment = await assess_subject_validity("휴지통에서 피가 나고 있어")

    assert assessment.is_valid_subject is False
    assert assessment.reason == "non_human_subject"
    assert assessment.response is not None


def test_agent_extracts_invalid_subject_response():
    agent = Agent()

    response = agent._extract_invalid_subject_response_from_content(
        "[질문 대상 확인]\n사람의 신체가 아닌 대상의 증상은 의료적으로 답변하기 어렵습니다. 실제 사람에게 나타난 증상이라면 증상이 생긴 부위를 다시 알려주세요."
    )

    assert response is not None


def test_agent_ignores_valid_subject_check_response():
    agent = Agent()

    response = agent._extract_invalid_subject_response_from_content(
        "[질문 대상 확인]\n사람의 신체 증상 질문으로 볼 수 있습니다."
    )

    assert response is None


def test_extract_subject_location_from_question():
    assert _extract_subject_location("휴지통에서 피가 나고 있어") == "휴지통"
    assert _extract_subject_location("귀에서 피가 나요") == "귀"


@pytest.mark.asyncio
async def test_assess_emergency_symptoms_with_llm_uses_structured_output(monkeypatch):
    class DummyExtractor:
        async def ainvoke(self, _messages):
            return TriageLLMOutput(
                concerning_symptoms=["호흡곤란"],
                urgency_level="high",
                needs_clarification=True,
                clarifying_questions=[
                    "가만히 있어도 숨이 찬가요?",
                    "가슴 통증이 함께 있나요?",
                ],
                safety_message=None,
            )

    class DummyLLM:
        def with_structured_output(self, _schema):
            return DummyExtractor()

    monkeypatch.setattr("app.tools.medical_tools._get_triage_llm", lambda: DummyLLM())

    assessment = await assess_emergency_symptoms_with_llm(
        question="갑자기 숨을 쉬기 힘들어",
        symptoms=["호흡곤란"],
    )

    assert assessment.needs_clarification is True
    assert assessment.urgency_level == "high"
    assert assessment.clarifying_questions == [
        "가만히 있어도 숨이 찬가요?",
        "가슴 통증이 함께 있나요?",
    ]
