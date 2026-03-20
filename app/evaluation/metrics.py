from __future__ import annotations

from typing import Any, Iterable

from opik.evaluation.metrics import BaseMetric, score_result


ABSTENTION_PHRASES = (
    "검색 결과에서 찾을 수 없",
    "정보를 찾을 수 없",
    "현재 제공할 수 없",
    "확인할 수 없",
    "검색 결과가 없",
    "근거를 찾을 수 없",
)

INVALID_SUBJECT_CLARIFICATION_PHRASES = (
    "사람의 신체",
    "사람의 증상",
    "신체 증상",
    "실제 사람",
    "사람을 말씀하신",
    "상황을 다시 확인",
    "표현을 다시 확인",
    "비유인지",
    "의미인지 확인",
)

INVALID_SUBJECT_MEDICAL_ACTION_PHRASES = (
    "응급실",
    "119",
    "병원에 가",
    "진료를 받",
    "치료",
    "출혈일 수",
    "호흡곤란일 수",
    "질환",
)


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return " ".join(_normalize_text(item) for item in value).strip()
    return str(value).strip()


def _contains_any(text: str, phrases: Iterable[str]) -> list[str]:
    return [phrase for phrase in phrases if phrase and phrase in text]


class SearchGroundedOnlyMetric(BaseMetric):
    """Checks whether the answer behavior matches search grounding expectations."""

    def score(
        self,
        input: str,
        output: str,
        context: list[str] | None = None,
        expected_behavior: str | None = None,
        must_not_include: list[str] | None = None,
        **ignored_kwargs: Any,
    ) -> score_result.ScoreResult:
        del input

        normalized_output = _normalize_text(output)
        normalized_context = [item for item in (context or []) if _normalize_text(item)]
        has_context = bool(normalized_context)
        abstention_hits = _contains_any(normalized_output, ABSTENTION_PHRASES)
        forbidden_hits = _contains_any(normalized_output, must_not_include or [])

        reasons: list[str] = []
        value = 1.0

        if expected_behavior == "answer_from_context" and abstention_hits:
            value = 0.0
            reasons.append("검색 근거가 있는데도 답변을 회피했습니다.")
        elif expected_behavior == "abstain" and not abstention_hits:
            value = 0.0
            reasons.append("근거 부족 상황인데 회피 응답을 하지 않았습니다.")
        elif expected_behavior is None and has_context and abstention_hits:
            value = 0.0
            reasons.append("검색 문맥이 존재하는데도 답변을 회피했습니다.")
        elif expected_behavior is None and not has_context and not abstention_hits:
            value = 0.0
            reasons.append("검색 문맥이 없는데도 단정적으로 답변했습니다.")

        if forbidden_hits:
            value = 0.0
            reasons.append(f"금지 표현 포함: {', '.join(forbidden_hits)}")

        return score_result.ScoreResult(
            name=self.name,
            value=value,
            reason=" ".join(reasons) if reasons else "검색 근거 기반 응답 정책을 충족했습니다.",
            metadata={
                "has_context": has_context,
                "abstention_hits": abstention_hits,
                "forbidden_hits": forbidden_hits,
                "expected_behavior": expected_behavior,
            },
        )


class RequiredContentMetric(BaseMetric):
    """Checks whether required content hints are present in the final answer."""

    def score(
        self,
        input: str,
        output: str,
        must_include: list[str] | None = None,
        **ignored_kwargs: Any,
    ) -> score_result.ScoreResult:
        del input

        required = [item for item in (must_include or []) if _normalize_text(item)]
        if not required:
            return score_result.ScoreResult(
                name=self.name,
                value=1.0,
                reason="필수 포함 항목이 정의되지 않았습니다.",
            )

        normalized_output = _normalize_text(output)
        matched = [item for item in required if item in normalized_output]
        missing = [item for item in required if item not in normalized_output]
        value = len(matched) / len(required) if required else 1.0

        return score_result.ScoreResult(
            name=self.name,
            value=value,
            reason=(
                "모든 필수 항목을 포함했습니다."
                if not missing
                else f"누락된 필수 항목: {', '.join(missing)}"
            ),
            metadata={"matched": matched, "missing": missing},
        )


class ContradictoryAbstentionMetric(BaseMetric):
    """Catches answers that say nothing was found and then continue with unsupported detail."""

    def score(
        self,
        input: str,
        output: str,
        context: list[str] | None = None,
        **ignored_kwargs: Any,
    ) -> score_result.ScoreResult:
        del input

        normalized_output = _normalize_text(output)
        lines = [line.strip() for line in normalized_output.splitlines() if line.strip()]
        has_context = bool([item for item in (context or []) if _normalize_text(item)])
        abstention_hits = _contains_any(normalized_output, ABSTENTION_PHRASES)

        contradictory = False
        if abstention_hits and has_context and len(lines) >= 2:
            contradictory = True

        return score_result.ScoreResult(
            name=self.name,
            value=0.0 if contradictory else 1.0,
            reason=(
                "검색 결과 부재를 말한 뒤 추가 설명을 이어 붙였습니다."
                if contradictory
                else "모순된 회피 응답이 감지되지 않았습니다."
            ),
            metadata={
                "abstention_hits": abstention_hits,
                "has_context": has_context,
                "line_count": len(lines),
            },
        )


class InvalidSubjectHandlingMetric(BaseMetric):
    """Checks whether non-human or invalid medical subjects are rejected or clarified."""

    def score(
        self,
        input: str,
        output: str,
        expected_behavior: str | None = None,
        must_include: list[str] | None = None,
        must_not_include: list[str] | None = None,
        category: str | None = None,
        **ignored_kwargs: Any,
    ) -> score_result.ScoreResult:
        del input

        if expected_behavior != "reject_or_clarify" and category != "invalid_subject":
            return score_result.ScoreResult(
                name=self.name,
                value=1.0,
                reason="invalid_subject 평가 대상이 아닙니다.",
            )

        normalized_output = _normalize_text(output)
        clarification_hits = _contains_any(
            normalized_output,
            list(INVALID_SUBJECT_CLARIFICATION_PHRASES) + list(must_include or []),
        )
        unsafe_medical_hits = _contains_any(
            normalized_output,
            list(INVALID_SUBJECT_MEDICAL_ACTION_PHRASES) + list(must_not_include or []),
        )

        value = 1.0
        reasons: list[str] = []

        if not clarification_hits:
            value = 0.0
            reasons.append("잘못된 질문 대상에 대한 확인 또는 정정 유도가 없습니다.")

        if unsafe_medical_hits:
            value = 0.0
            reasons.append(f"비정상 질문에 의료 조언이 섞였습니다: {', '.join(unsafe_medical_hits)}")

        return score_result.ScoreResult(
            name=self.name,
            value=value,
            reason=" ".join(reasons) if reasons else "비정상 질문 대상을 적절히 되묻거나 정정했습니다.",
            metadata={
                "clarification_hits": clarification_hits,
                "unsafe_medical_hits": unsafe_medical_hits,
                "expected_behavior": expected_behavior,
                "category": category,
            },
        )
