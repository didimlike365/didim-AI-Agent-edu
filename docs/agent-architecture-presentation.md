# Edu Agent 동작 구조 발표 자료

## 1. 한 줄 소개

이 Agent는 사용자 질문을 바로 답하는 구조가 아니라, 먼저 질문의 유형과 안전성을 확인하고, 필요한 도구를 실행한 뒤, 근거 기반으로 답변을 생성하는 구조다.

핵심 흐름은 다섯 단계다.

1. 질문을 해석한다.
2. 필요한 tool을 선택한다.
3. 질문 대상과 응급 가능성을 먼저 확인한다.
4. 필요한 근거를 검색하고 압축한다.
5. 최종 답변을 생성하거나, 필요한 경우 확인 질문을 먼저 반환한다.

---

## 2. 현재 구조의 설계 의도

이 프로젝트는 단순한 일반 QA Agent가 아니다.

현재 구조는 아래 문제를 해결하려는 방향으로 설계되어 있다.

- 의료 질문은 바로 답하면 안 되는 경우가 있다.
- 검색 결과 원문을 그대로 넣으면 토큰 비용이 커진다.
- 증상 질문, 설명형 질문, 위치 질문은 처리 방식이 달라야 한다.
- 사람 신체가 아닌 대상의 증상 질문은 먼저 걸러야 한다.
- 운영 중에는 어떤 단계가 실행됐는지 추적 가능해야 한다.
- 평가도 단순 정답 비교가 아니라 근거성, 회피, 잘못된 질문 방어까지 포함해야 한다.

그래서 현재 구조는 아래 축으로 나뉜다.

- tool-calling 기반 의도 분기
- 질문 대상 유효성 검사
- 증상 기반 응급 triage
- Elasticsearch 기반 근거 검색
- 검색 결과 압축
- Opik 기반 관측과 평가

---

## 3. 전체 아키텍처

```text
[User / UI]
    |
    v
POST /api/v1/chat
    |
    v
[FastAPI Router]
    |
    v
[AgentService]
    |
    |-- Opik tracer 연결
    |-- thread_id 기반 triage 상태 확인
    |-- pending triage 후속 분기
    |
    v
[Search Agent]
    |
    |-- System Prompt 주입
    |-- Tool binding
    |
    v
[Tool-Calling LLM]
    |
    |-- medical_search ?
    |-- hospital_search ?
    |-- symptom_parser ?
    |-- symptom_duration_parser ?
    |
    v
[Auto Guard / Auto Triage]
    |
    |-- subject_validity_check
    |-- emergency_symptom_triage
    |
    v
[Tools]
    |
    |-- Elasticsearch 검색
    |-- 검색 결과 압축
    |-- 증상 추출
    |-- 질문 대상 확인
    |-- 응급 확인 질문 생성
    |
    v
[Final Answer LLM or Immediate Guard/Triage Reply]
    |
    v
[SSE Streaming Response]
```

핵심은 하나의 LLM이 모든 일을 직접 처리하지 않는다는 점이다.

- 첫 번째 LLM: 어떤 tool을 쓸지 결정
- 질문 대상 검사 모델: 사람이 아닌 대상의 증상 문의인지 판별
- triage 모델: 응급 가능성이 있으면 확인 질문 생성
- 압축 모델: 검색 결과를 짧게 정리
- 최종 LLM: 정리된 근거를 바탕으로 답변 생성

---

## 4. 주요 파일별 책임

### API 진입점

- `app/main.py`
  - FastAPI 앱 생성
  - CORS 설정
  - 공통 요청 로깅
  - startup 시 Opik 상태 기록
  - `/health`에서 Opik 연결 상태 노출
- `app/api/routes/chat.py`
  - `/api/v1/chat` 엔드포인트 제공
  - SSE 방식 스트리밍 응답 반환

### 오케스트레이션 레이어

- `app/services/agent_service.py`
  - Agent 생성
  - Opik tracer 연결
  - 응답을 SSE 형식으로 변환
  - triage pending 상태 확인
  - 후속 답변을 triage 후속 턴으로 처리
  - Opik dataset 연결 상태 관리

### Agent 레이어

- `app/agents/search_agent.py`
  - tool-calling 흐름 제어
  - tool 실행 결과 조합
  - `subject_validity_check` 자동 실행
  - `emergency_symptom_triage` 자동 실행
  - 질문 대상 문제가 있으면 최종 LLM 전에 즉시 종료
  - triage가 필요하면 최종 설명 전에 확인 질문 우선 반환
- `app/agents/prompts.py`
  - tool 선택 규칙 정의
  - 질문 대상 유효성 검사 우선 규칙 정의
  - 위치 질문 응답 정책 정의

### Tool 레이어

- `app/tools/medical_tools.py`
  - `medical_search`
  - `hospital_search`
  - `symptom_parser`
  - `symptom_duration_parser`
  - `subject_validity_check`
  - `emergency_symptom_triage`

### 상태 관리 레이어

- `app/services/triage_state_service.py`
  - `thread_id` 기준 triage pending 상태 저장
  - 후속 질문이 들어왔을 때 이어서 처리할 수 있도록 지원

### 외부 시스템 연동

- `app/services/elasticsearch_service.py`
  - Elasticsearch `_search` 호출
  - 검색 결과를 문서 리스트로 정규화
- `app/services/search_context_service.py`
  - 검색 결과 1차 축소
  - 압축 LLM 호출
  - fallback 요약 생성
  - Opik span 기록

### 평가 레이어

- `app/evaluation/opik_runner.py`
  - Opik dataset 로드
  - agent 실행
  - 평가 experiment 생성
- `app/evaluation/metrics.py`
  - 검색 근거 기반 응답 평가
  - 잘못된 회피 응답 평가
  - 잘못된 질문 대상 방어 평가
- `scripts/evaluate_opik_dataset.py`
  - dataset 기반 experiment 실행 스크립트

---

## 5. 현재 Tool 구성

### `medical_search`

- 일반 의학 설명형 질문에 사용
- Elasticsearch에서 관련 문서를 검색
- 검색 결과를 압축해서 메인 LLM에 전달

### `hospital_search`

- 위치 기반 질문에 사용
- 현재는 병원 정보를 직접 추천하지 않음
- 지도 앱 검색 또는 `119` 문의를 안내하는 응답을 반환

즉 이름은 `hospital_search`지만 실제 역할은 병원 추천이 아니라 위치 안내다.

### `symptom_parser`

- 질문에서 증상 표현을 추출
- 예: `이가 아파` -> `치통`
- guard와 triage 흐름의 시작점 역할을 함

### `symptom_duration_parser`

- 질문에서 지속 기간 표현을 추출
- 예: `3주째`, `며칠째`, `몇 시간째`

### `subject_validity_check`

- 질문 대상이 실제 사람의 신체인지 확인
- 사람 신체가 아닌 대상의 증상 질문이면 즉시 차단
- 예:
  - `귀에서 피가 나요` -> 통과
  - `휴지통에서 피가 나요` -> 차단
- `~~에서`, `~~쪽에서`, `~~부위에서` 같은 위치 표현이 나오면 그 위치가 신체 부위인지 아닌지를 우선 판단
- 현재는 `TRIAGE_MODEL`을 이용해 LLM이 문맥 해석을 먼저 하고, 규칙은 fallback 역할을 한다

### `emergency_symptom_triage`

- 증상 질문에서 응급 가능성이 있는지 확인
- 출혈, 구토, 설사, 메스꺼움, 호흡곤란, 흉통, 의식저하 같은 위험 신호를 분류
- 필요한 경우 짧은 확인 질문을 먼저 반환

---

## 6. 요청이 들어왔을 때 실행 순서

### 6.1 일반 설명형 질문

예시:

```text
모세기관지염이란?
```

흐름은 아래와 같다.

1. `/api/v1/chat` 요청 수신
2. `AgentService`가 agent 생성
3. `SearchAgent`가 tool-calling LLM 호출
4. `medical_search` 실행
5. 검색 결과를 압축
6. 최종 LLM이 근거 기반 답변 생성
7. SSE로 결과 반환

### 6.2 증상 질문 처리 흐름

예시:

```text
갑자기 숨을 쉬기 힘들어
```

흐름은 아래와 같다.

1. `symptom_parser`로 증상 확인
2. 증상이 있으면 `subject_validity_check` 자동 실행
3. 질문 대상이 사람 신체가 맞으면 `emergency_symptom_triage` 자동 실행
4. 응급 가능성이 있으면 최종 설명보다 먼저 확인 질문 반환
5. 필요하면 triage 상태를 `thread_id` 기준으로 저장

즉 증상 질문은 바로 설명형 답변으로 가지 않고, 먼저 질문 대상과 위험 신호를 확인한다.

### 6.3 잘못된 질문 대상 처리 흐름

예시:

```text
휴지통에서 피가 나고 있어
```

흐름은 아래와 같다.

1. `symptom_parser`가 `피` 같은 증상 표현을 감지할 수 있음
2. `subject_validity_check`가 `휴지통에서`를 위치 문맥으로 해석
3. 질문 대상이 사람 신체가 아니라고 판단
4. 응급 triage나 검색으로 가지 않고 즉시 종료

반환 방향은 아래와 같다.

```text
[질문 대상 확인]
사람의 신체가 아닌 대상의 증상은 의료적으로 답변하기 어렵습니다. 실제 사람에게 나타난 증상이라면 증상이 생긴 부위를 다시 알려주세요.
```

### 6.4 triage 후속 흐름

예시:

- 1턴: `갑자기 숨을 쉬기 힘들어`
- Agent: `응급 가능성이 높으므로 숨찬 정도가 어느 정도인지 알려주세요...`
- 2턴: `가만히 있어도 숨이 차고 가슴 통증도 있어`

현재는 이 다음 단계까지 이어진다.

1. `AgentService`가 `thread_id` 기준 pending triage 상태 확인
2. 다음 사용자 답변을 triage 후속 답변으로 해석
3. 후속 답변이 심각하면 즉시 응급 권고 응답 반환
4. 심각하지 않으면 기존 agent 흐름으로 재진입

즉 triage는 단발 질문이 아니라 후속 턴까지 이어지는 구조다.

---

## 7. 위치 기반 질문 처리 방식

예시:

```text
구로디지털 근처 응급실 알려줘
```

현재 정책은 명확하다.

- 위치 기반 병원 정보를 정확하게 직접 제공하지 않는다.
- 병원 추천처럼 보이는 응답도 하지 않는다.
- 대신 아래처럼 안내한다.
  - 지도 앱이나 포털 검색 사용
  - 응급 상황이면 `119` 연락

이 정책을 둔 이유는 다음과 같다.

- 현재 시스템은 실시간 위치 기반 병원 추천 시스템이 아니다.
- 실제 병원 운영 여부나 거리 정보까지 보장할 수 없다.
- 의료 안전성과 정확성을 위해 과도한 안내를 피해야 한다.

---

## 8. Elasticsearch 검색 구조

### 8.1 검색 서비스 역할

`elasticsearch_service.py`는 실제 Elasticsearch `_search` API를 호출한다.

주요 특징은 아래와 같다.

- 기본 인덱스는 `edu-collection`
- `_source`에서 필요한 필드만 조회
- 결과를 Python dict 리스트로 정규화

대표 필드는 아래와 같다.

- `c_id`
- `content`
- `creation_year`
- `domain`
- `source`
- `source_spec`

### 8.2 검색 결과를 그대로 넘기지 않는 이유

검색 결과 원문을 그대로 LLM에 넘기면 다음 문제가 생긴다.

- ToolMessage가 너무 커진다.
- 최종 LLM 입력 토큰이 커진다.
- 중복 설명과 불필요한 문장이 그대로 들어간다.
- 비용과 응답 시간이 증가한다.

이 문제를 줄이기 위해 검색 결과 압축 레이어를 추가했다.

---

## 9. 검색 결과 압축 레이어

### 9.1 목적

압축 레이어는 Elasticsearch에서 받은 문서를 그대로 넘기지 않고, 메인 답변 생성에 필요한 정보만 남기기 위한 단계다.

흐름은 아래처럼 동작한다.

```text
ES 원문 문서
-> 1차 축소
-> SEARCH_SUMMARIZER_MODEL 압축
-> ToolMessage
-> 메인 LLM
```

### 9.2 1차 축소

압축 전에 규칙 기반 전처리를 먼저 수행한다.

- 문서 수 제한
- 문서당 `content` 길이 제한
- 필요한 필드만 유지
- 공백 정리

이 단계는 대폭 절감보다는, 불필요한 입력을 정리하는 기본 방어막 역할을 한다.

### 9.3 2차 압축

그 다음 `SEARCH_SUMMARIZER_MODEL`로 설정된 모델이 문서를 구조화된 요약으로 만든다.

현재 기본값은 아래와 같다.

```env
SEARCH_SUMMARIZER_MODEL=gpt-4.1-mini
```

압축 결과는 아래 구조를 가진다.

- `summary`
- `key_points`
- `cautions`

즉 메인 LLM은 긴 원문 대신, 이미 한 번 정리된 검색 결과를 받아 최종 답변을 만든다.

### 9.4 fallback 구조

압축용 LLM 호출이 실패하더라도 시스템이 멈추면 안 된다.

그래서 fallback 경로를 둔다.

- 압축 실패 시 짧은 근거 목록을 반환
- `c_id`, 연도, 출처, excerpt 중심으로 구성
- 최소한의 근거는 유지한 채 답변 가능

이 구조는 운영 안정성 측면에서 중요하다.

---

## 10. 최종 답변 생성 단계

guard나 triage 응답이 필요하지 않은 경우, `search_agent.py`는 아래 메시지 묶음을 만든다.

1. system message
2. 사용자 질문
3. LLM의 tool call 결과 메시지
4. 각 tool의 ToolMessage

그 다음 메인 LLM이 전체 문맥을 바탕으로 최종 답변을 생성한다.

중요한 원칙은 아래와 같다.

- tool 결과에 없는 내용은 추측하지 않는다.
- 복합 질문이면 여러 tool 결과를 함께 종합한다.
- 증상과 기간 정보가 있으면 답변에 반영한다.
- 위치 질문이면 직접 병원 추천 대신 안내를 제공한다.
- 질문 대상이 잘못되었으면 검색이나 triage로 가지 않는다.
- triage가 필요한 경우에는 최종 설명보다 확인 질문이 우선이다.

---

## 11. 스트리밍 응답 구조

이 프로젝트는 응답을 한 번에 보내지 않고 SSE로 나눠서 보낸다.

대표 이벤트 흐름은 아래와 같다.

```text
model -> 어떤 tool을 쓸지 알림
tools -> 각 tool 실행 결과 알림
done -> 최종 답변 전달
```

장점은 아래와 같다.

- 프론트가 중간 상태를 사용자에게 보여줄 수 있다.
- 디버깅이 쉽다.
- 어떤 tool이 실행됐는지 확인할 수 있다.

---

## 12. Opik 관측 포인트

현재 Opik에서는 크게 네 가지 레벨을 볼 수 있다.

### 12.1 tool 레벨

- `medical_search`
- `hospital_search`
- `symptom_parser`
- `symptom_duration_parser`
- `subject_validity_check`
- `emergency_symptom_triage`

즉 어떤 tool이 실제로 실행됐는지 추적 가능하다.

### 12.2 검색 압축 레벨

다음 span도 확인 가능하다.

- `search_context_preparation`
- `search_context_compression`
- `search_context_result`

여기서 아래 정보를 볼 수 있다.

- 원본 문서 개수
- 준비된 문서 개수
- 압축 전 글자 수
- 압축 모델명
- output 크기
- fallback 여부

### 12.3 triage와 guard 레벨

현재는 tool 실행 기록과 metadata로 아래를 추적할 수 있다.

- 질문 대상 유효성 검사 실행 여부
- triage pending 상태 발생 여부
- 후속 답변에서 triage 해소 여부
- 즉시 응급 권고로 끝났는지 여부

### 12.4 startup / health 레벨

현재 `/health`에서 아래 상태를 볼 수 있다.

- Opik 활성화 여부
- client 초기화 여부
- project 이름
- dataset 이름
- dataset 연결 여부
- dataset 연결 실패 이유

즉 관측 가능성은 실행 중 trace뿐 아니라 런타임 상태까지 포함한다.

---

## 13. Opik 평가 구조

현재는 `lyp-dataset-edu` dataset을 기준으로 experiment를 실행할 수 있다.

평가 흐름은 아래와 같다.

1. Opik dataset 로드
2. dataset row마다 실제 agent 실행
3. 검색 context 수집
4. metric 계산
5. experiment 업로드

실행 스크립트는 아래다.

- `scripts/evaluate_opik_dataset.py`

현재 experiment 이름은 아래 형식으로 생성된다.

```text
lyp-dataset-edu-eval-YYYYMMDD-HHMMSS
```

### 13.1 현재 평가 metric

- `search_grounded_only`
  - 검색 근거만으로 답했는지
- `required_content`
  - 꼭 들어가야 할 표현이 들어갔는지
- `contradictory_abstention`
  - 근거가 없다고 말해놓고 일반 설명을 섞었는지
- `invalid_subject_handling`
  - 사람 신체가 아닌 대상의 증상 질문을 적절히 차단했는지

필요하면 아래 judge metric도 포함할 수 있다.

- `Hallucination`
- `AnswerRelevance`

### 13.2 dataset row 권장 구조

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

즉 이 Agent는 단순 정답 비교가 아니라, 잘못된 질문을 잘 막았는지도 평가한다.

---

## 14. 현재 구조의 장점

### 14.1 역할 분리가 명확하다

- API 레이어
- 오케스트레이션 레이어
- Agent 레이어
- Tool 레이어
- 상태 관리 레이어
- 외부 서비스 연동 레이어
- 평가 레이어

수정 포인트가 비교적 명확하다.

### 14.2 질문 안전성 처리가 좋아졌다

응급 여부를 보기 전에 질문 대상 자체가 유효한지 먼저 확인한다.

즉 잘못된 질문을 정상 의료 질문처럼 처리하지 않도록 막는다.

### 14.3 복합 질문 대응이 좋다

하나의 질문 안에 여러 의도가 있어도, LLM이 여러 tool을 조합해 처리할 수 있다.

### 14.4 응급 질문 대응이 더 안전해졌다

바로 설명형 답변으로 가지 않고, 필요한 경우 먼저 확인 질문을 하도록 만들었다.

### 14.5 후속 턴까지 이어진다

triage 질문 뒤 사용자의 다음 답변도 `thread_id` 기준으로 이어서 처리할 수 있다.

### 14.6 토큰 비용을 줄이기 좋다

검색 결과 압축 레이어 덕분에 메인 LLM에 불필요한 원문을 덜 넘기게 된다.

### 14.7 관측과 평가가 가능하다

Opik을 통해 실행 흐름과 dataset 기반 성능을 함께 볼 수 있다.

---

## 15. 현재 구조의 한계

### 15.1 triage 상태 저장은 메모리 기반이다

현재 `triage_state_service`는 in-memory 저장소다.

즉 서버가 재시작되면 pending triage 상태는 사라진다.

### 15.2 위치 기반 병원 추천은 지원하지 않는다

현재는 지도 기반 병원 추천, 운영 여부 확인, 거리 계산까지 하지는 않는다.

### 15.3 검색 품질은 인덱스 품질에 의존한다

LLM이 좋아도 Elasticsearch 결과가 부정확하면 답변 품질이 떨어진다.

### 15.4 guard와 triage는 아직 더 고도화 여지가 있다

현재는 LLM 문맥 판단과 fallback 규칙을 함께 쓰고 있지만, 더 다양한 비유 표현이나 애매한 문장을 다루려면 추가 고도화가 가능하다.

### 15.5 평가셋 품질이 곧 평가 품질이다

Opik experiment는 매우 유용하지만, dataset row가 잘 설계되어야 제대로 된 평가가 가능하다.

---

## 16. 발표 순서 추천

발표는 아래 순서가 가장 이해하기 쉽다.

1. 문제 정의
   - 의료 질문은 일반 QA와 다르다
   - 바로 답하면 안 되는 경우가 있다
2. 전체 구조
   - API -> AgentService -> SearchAgent -> Tools -> LLM
3. tool-calling 구조 설명
4. 질문 대상 유효성 검사 설명
5. 응급 triage 구조 설명
6. 검색 결과 압축 구조 설명
7. 위치 질문 처리 정책 설명
8. Opik trace 설명
9. Opik dataset 평가 설명
10. 장점과 한계 설명

---

## 17. 데모 시나리오

### 시나리오 1: 일반 의학 질문

```text
모세기관지염이란?
```

확인 포인트:

- `medical_search`
- 검색 결과 압축
- 최종 설명형 답변

### 시나리오 2: 응급 triage 질문

```text
갑자기 숨을 쉬기 힘들어
```

확인 포인트:

- `symptom_parser`
- `subject_validity_check`
- `emergency_symptom_triage`
- triage 질문 우선 반환
- 다음 턴 후속 처리

### 시나리오 3: 잘못된 질문 대상

```text
휴지통에서 피가 나고 있어
```

확인 포인트:

- `symptom_parser`
- `subject_validity_check`
- 응급 triage로 가지 않음
- 질문 대상 확인 메시지로 즉시 종료

### 시나리오 4: 위치 기반 질문

```text
구로디지털 근처 응급실 알려줘
```

확인 포인트:

- `hospital_search`
- 직접 병원 추천 대신 검색/`119` 안내

---

## 18. 강조 포인트

- 이 Agent의 핵심은 답을 바로 만드는 것이 아니라, 먼저 질문의 유효성과 위험 신호를 확인하는 것이다.
- 첫 번째 LLM은 답변 모델이라기보다 의사결정자에 가깝다.
- `subject_validity_check`는 잘못된 질문을 정상 의료 질문처럼 처리하지 않도록 막는 안전장치다.
- 검색 결과 압축 레이어는 비용 최적화와 품질 안정성을 동시에 노린 설계다.
- triage 구조는 응급 가능성이 있는 질문을 바로 설명형 답변으로 흘려보내지 않도록 만든 안전장치다.
- 위치 기반 병원 안내는 정확성을 보장할 수 없는 범위까지 확장하지 않도록 의도적으로 제한했다.
- Opik은 trace 관측뿐 아니라 dataset 기반 평가까지 연결하는 역할을 한다.

---

## 19. 결론

이 Edu Agent의 현재 구조는 아래처럼 정리할 수 있다.

- FastAPI가 요청을 받는다.
- AgentService가 실행을 오케스트레이션한다.
- SearchAgent가 어떤 tool이 필요한지 결정한다.
- 증상 질문이면 먼저 질문 대상이 유효한지 확인한다.
- 유효한 질문이면 응급 triage와 검색을 이어서 수행한다.
- 검색 결과는 압축해서 메인 LLM에 전달한다.
- 최종 답변은 근거 기반으로 생성한다.
- Opik이 실행 흐름과 평가 결과를 기록한다.

즉 현재 구조의 핵심은 세 가지다.

- 바로 답하지 않는다.
- 먼저 확인하고, 필요한 만큼만 검색하고, 근거로 답한다.
- 실행과 품질을 모두 추적할 수 있다.
