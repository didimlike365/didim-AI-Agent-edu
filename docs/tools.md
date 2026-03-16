# Agent Tool Guide

현재 에이전트는 LangChain tool-calling 흐름으로 동작하며, 실제 도구 정의는 [`app/tools/medical_tools.py`](/Users/leeyongpil/working/AI-Agent/Agent-Devlop/agent/app/tools/medical_tools.py)에 있습니다.

## 전체 흐름

1. 사용자가 `/api/v1/chat`으로 질문을 보냅니다.
2. [`AgentService`](/Users/leeyongpil/working/AI-Agent/Agent-Devlop/agent/app/services/agent_service.py)가 [`Agent`](/Users/leeyongpil/working/AI-Agent/Agent-Devlop/agent/app/agents/search_agent.py)를 생성합니다.
3. 에이전트는 시스템 프롬프트를 바탕으로 어떤 tool을 쓸지 LLM이 결정합니다.
4. 선택된 tool이 [`ElasticsearchService`](/Users/leeyongpil/working/AI-Agent/Agent-Devlop/agent/app/services/elasticsearch_service.py)를 호출하거나 사용자 질문을 구조화합니다.
5. tool 결과는 `ToolMessage`로 다시 LLM에 전달됩니다.
6. 최종 응답은 `ChatResponse` 형식으로 스트리밍됩니다.

## Tool List

### 1. `medical_search`

- 목적: 일반 의학 정보, 질환 설명, 진단/치료 관련 질문에 대해 `edu-collection` 인덱스에서 본문 검색을 수행합니다.
- 구현 위치: [`medical_tools.py`](/Users/leeyongpil/working/AI-Agent/Agent-Devlop/agent/app/tools/medical_tools.py)
- 실제 조회 서비스: [`elasticsearch_service.py`](/Users/leeyongpil/working/AI-Agent/Agent-Devlop/agent/app/services/elasticsearch_service.py)

#### 입력

```json
{
  "query": "latent TB infection diagnosis에 대해 설명해줘"
}
```

#### 동작

- `ELASTICSEARCH_INDEX=edu-collection`을 사용합니다.
- `content` 필드 기준으로 `match` 검색을 수행합니다.
- 상위 `ELASTICSEARCH_TOP_K`개 문서를 가져옵니다.
- 검색 결과를 `[문서 1]`, `[문서 2]` 형식의 context 문자열로 변환합니다.

#### 출력 예시

```text
[문서 1]
c_id: 25276_3
creation_year: 2000
domain: 2
source: 5
source_spec: who
content: ...
```

#### 주로 호출되는 질문 유형

- `잠복결핵이 뭐야?`
- `결핵 진단 기준을 설명해줘`
- `LTBI 검사 방법 알려줘`

### 2. `hospital_search`

- 목적: 질문에 지역성 표현이 들어오면 같은 `edu-collection` 인덱스에서 병원/위치 관련 문맥을 보강해서 찾습니다.
- 구현 위치: [`medical_tools.py`](/Users/leeyongpil/working/AI-Agent/Agent-Devlop/agent/app/tools/medical_tools.py)

#### 입력

```json
{
  "query": "서울 근처 결핵 관련 병원 정보도 같이 알려줘",
  "region": ""
}
```

#### 동작

- 별도 병원 인덱스를 쓰지 않습니다.
- 기존 질문에 `병원`, `의료기관`, `위치`, `지역` 같은 키워드를 추가해서 `edu-collection`을 다시 검색합니다.
- 지역/병원 관련 문맥을 별도 context 블록으로 만듭니다.
- 최종 답변에서 위치 관련 정보가 있으면 함께 요약하도록 프롬프트에서 유도합니다.

#### 출력 예시

```text
[지역/병원 문맥 1]
c_id: 25276_3
creation_year: 2000
content: ...
```

#### 주로 호출되는 질문 유형

- `서울 근처 병원 정보도 같이 알려줘`
- `강남 주변 결핵 관련 병원 내용 있어?`
- `부산 인근 의료기관 정보도 같이 정리해줘`

### 3. `symptom_duration_parser`

- 목적: 사용자 질문에서 지속 기간 표현을 추출해 구조화합니다.
- 구현 위치: [`medical_tools.py`](/Users/leeyongpil/working/AI-Agent/Agent-Devlop/agent/app/tools/medical_tools.py)

#### 입력

```json
{
  "query": "기침이 3주째 계속되는데 결핵일 수 있어?"
}
```

#### 동작

- 검색을 수행하지 않습니다.
- LLM의 구조화 출력으로 질문 안의 지속 기간 표현을 추출합니다.
- `[지속기간 분석]` 블록으로 변환한 뒤 최종 답변 단계에 전달합니다.

#### 추출 대상 예시

- 기간 표현
  - `3시간`
  - `4일째`
  - `2주`
  - `3주째`
  - `몇 주`
  - `몇 달`
  - `계속`
  - `오래 지속`

#### 출력 예시

```text
[지속기간 분석]
원문 질문: 기침이 3주째 계속되는데 결핵일 수 있어?
추출된 지속기간: 3주째, 계속
```

#### 주로 호출되는 질문 유형

- `기침이 2주째야`
- `열이 며칠째 안 떨어져`
- `가래가 오래 지속돼`

### 4. `symptom_parser`

- 목적: 사용자 질문에서 증상 표현을 추출해 구조화합니다.
- 구현 위치: [`medical_tools.py`](/Users/leeyongpil/working/AI-Agent/Agent-Devlop/agent/app/tools/medical_tools.py)

#### 입력

```json
{
  "query": "신대방삼거리 근처인데 세시간째 이가 아파"
}
```

#### 동작

- 검색을 수행하지 않습니다.
- LLM의 구조화 출력으로 질문에서 증상을 추출합니다.
- 동의어는 대표 증상명으로 정규화하도록 LLM에 요청합니다.
- `[증상 분석]` 블록으로 변환한 뒤 최종 답변 단계에 전달합니다.

#### 출력 예시

```text
[증상 분석]
원문 질문: 신대방삼거리 근처인데 세시간째 이가 아파
추출된 증상: 치통
```

#### 주로 호출되는 질문 유형

- `기침이 나`
- `치통이 심해`
- `배가 아파`

## Tool Selection Rules

현재 도구 선택 규칙은 [`app/agents/prompts.py`](/Users/leeyongpil/working/AI-Agent/Agent-Devlop/agent/app/agents/prompts.py)에 정의되어 있으며, 실제 선택은 LLM이 수행합니다.

- 일반 설명형 질문: `medical_search`
- 지역/병원 관련 질문: `hospital_search`
- 증상 질문: `symptom_parser`
- 증상 지속기간 질문: `symptom_duration_parser`
- 복합 질문: 여러 tool을 함께 사용할 수 있음

예시:

- `잠복결핵이 뭐야?`
  - `medical_search`
- `서울 근처 결핵 병원 정보도 같이 알려줘`
  - `medical_search`, `hospital_search`
- `신대방삼거리 근처인데 세시간째 이가 아파`
  - `hospital_search`, `symptom_parser`, `symptom_duration_parser`
- `기침이 3주째인데 결핵일 수 있어?`
  - `medical_search`, `symptom_parser`, `symptom_duration_parser`
- `서울 근처에서 기침이 3주째 지속되는데 어디를 봐야 해?`
  - `medical_search`, `hospital_search`, `symptom_parser`, `symptom_duration_parser`

## Data Source

현재 검색은 Elasticsearch의 `edu-collection` 인덱스 하나만 사용합니다.

확인된 주요 필드:

- `c_id`
- `content`
- `creation_year`
- `domain`
- `source`
- `source_spec`

즉 `medical_search`와 `hospital_search`는 둘 다 같은 인덱스를 조회하지만, 검색 목적과 쿼리 강화 방식이 다릅니다.

## Metadata

최종 응답의 `metadata`에는 현재 다음 정보가 포함될 수 있습니다.

- `sources`
  - 검색 결과 출처 목록
- `used_tools`
  - 실제 실행된 tool 이름 목록

## Limitations

- `hospital_search`는 별도 병원 DB가 아니라 `edu-collection`의 텍스트 문맥을 다시 검색하는 방식입니다.
- 실제 거리 계산, 좌표 기반 근접 검색, 지도 검색은 지원하지 않습니다.
- `symptom_duration_parser`와 `symptom_parser`는 이제 LLM 기반 추출이므로 표현 다양성 대응은 좋아졌지만, 비용과 응답 시간이 증가할 수 있습니다.
- LLM 기반 추출 결과는 규칙 기반보다 유연하지만, 항상 동일한 결과를 보장하지는 않습니다.
- 의료 조언은 검색 문맥 기반 요약이며, 진단을 대체하지 않습니다.
