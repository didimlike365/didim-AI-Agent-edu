from app.evaluation.metrics import (
    ContradictoryAbstentionMetric,
    InvalidSubjectHandlingMetric,
    RequiredContentMetric,
    SearchGroundedOnlyMetric,
)


def test_search_grounded_only_metric_detects_false_abstention():
    metric = SearchGroundedOnlyMetric(name="search_grounded_only")

    result = metric.score(
        input="모세기관지염이란?",
        output="검색 결과에서 찾을 수 없었습니다.",
        context=["[who] 모세기관지염은 영아에서 흔한 하기도 감염입니다."],
        expected_behavior="answer_from_context",
    )

    assert result.value == 0.0


def test_required_content_metric_scores_partial_match():
    metric = RequiredContentMetric(name="required_content")

    result = metric.score(
        input="모세기관지염이란?",
        output="모세기관지염은 영아에서 흔한 호흡기 질환입니다.",
        must_include=["영아", "바이러스", "호흡기 질환"],
    )

    assert result.value == 2 / 3


def test_contradictory_abstention_metric_detects_mixed_answer():
    metric = ContradictoryAbstentionMetric(name="contradictory_abstention")

    result = metric.score(
        input="모세기관지염이란?",
        output=(
            "모세기관지염에 대한 정보는 검색 결과에서 찾을 수 없었습니다.\n"
            "모세기관지염은 일반적으로 소아에서 발생하는 호흡기 질환입니다."
        ),
        context=["[who] 모세기관지염은 주로 바이러스에 의해 발생합니다."],
    )

    assert result.value == 0.0


def test_invalid_subject_metric_accepts_clarifying_response():
    metric = InvalidSubjectHandlingMetric(name="invalid_subject_handling")

    result = metric.score(
        input="노트북에서 피가 나요",
        output="사람의 신체 증상을 말씀하신 건지 먼저 확인이 필요합니다. 물건에 대한 비유인지 실제 사람 상황인지 다시 알려주세요.",
        expected_behavior="reject_or_clarify",
        must_include=["사람", "확인"],
        must_not_include=["응급실", "질환"],
        category="invalid_subject",
    )

    assert result.value == 1.0


def test_invalid_subject_metric_rejects_literal_medical_advice():
    metric = InvalidSubjectHandlingMetric(name="invalid_subject_handling")

    result = metric.score(
        input="책상에서 피가 나요",
        output="출혈일 수 있으니 응급실로 가세요.",
        expected_behavior="reject_or_clarify",
        must_include=["사람"],
        must_not_include=["응급실", "출혈일 수"],
        category="invalid_subject",
    )

    assert result.value == 0.0
