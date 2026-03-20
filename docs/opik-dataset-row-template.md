# Opik Dataset Row Template

## 기본 질문형

```json
{
  "input": "모세기관지염이란?",
  "expected_output": "질환 설명을 검색 근거 기반으로 답변",
  "expected_behavior": "answer_from_context",
  "must_include": ["모세기관지염"],
  "must_not_include": ["검색 결과에서 찾을 수 없었습니다"]
}
```

## 근거 부족형

```json
{
  "input": "존재하지 않는 질환 설명해줘",
  "expected_output": "근거 부족으로 회피",
  "expected_behavior": "abstain",
  "must_include": ["찾을 수 없"],
  "must_not_include": ["일반적으로", "보통은"]
}
```

## 잘못된 질문 대상형

```json
{
  "input": "노트북에서 피가 나요",
  "expected_output": "사람의 신체 증상인지 먼저 확인",
  "expected_behavior": "reject_or_clarify",
  "category": "invalid_subject",
  "must_include": ["사람", "신체", "확인"],
  "must_not_include": ["응급실", "질환", "치료"]
}
```

## 권장 키

- `input`: 사용자 질문
- `expected_output`: 기대하는 응답 방향 요약
- `expected_behavior`: `answer_from_context` | `abstain` | `reject_or_clarify`
- `category`: 선택값. 잘못된 질문 대상 평가는 `invalid_subject`
- `must_include`: 답변에 꼭 들어가야 할 표현 목록
- `must_not_include`: 답변에 들어가면 안 되는 표현 목록
