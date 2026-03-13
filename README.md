# didim-AI-Agent-edu
디딤365 AI Agent 실습

FastAPI 기반의 LangChain v1.0 에이전트 교육용 템플릿입니다.

현재 구현은 LangChain tool-calling 기반으로 동작하며, Elasticsearch의 `edu-collection` 인덱스를 조회해 의료 문맥을 검색합니다.

## 기술 스택

- FastAPI
- LangChain v1.0
- OpenAI
- Elasticsearch
- uv

## 현재 구조

- Agent orchestration: [`app/agents/search_agent.py`](/Users/leeyongpil/working/AI-Agent/Agent-Devlop/agent/app/agents/search_agent.py)
- Tool definitions: [`app/tools/medical_tools.py`](/Users/leeyongpil/working/AI-Agent/Agent-Devlop/agent/app/tools/medical_tools.py)
- Elasticsearch access: [`app/services/elasticsearch_service.py`](/Users/leeyongpil/working/AI-Agent/Agent-Devlop/agent/app/services/elasticsearch_service.py)

현재 tool 목록:
- `medical_search`
- `hospital_search`
- `symptom_parser`
- `symptom_duration_parser`

## 환경 준비 및 설치 가이드

### 1. 사전 요구사항
- Python 3.11 이상 3.13 이하
- `uv` 설치

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. 프로젝트 의존성 설치

```bash
uv sync
```

### 3. 환경 변수 설정

```bash
cp env.sample .env
```

`.env`에 OpenAI, Elasticsearch 접속 정보를 채웁니다.

### 4. 개발 서버 실행

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## 프로젝트 구조

```text
agent/
├── app/
│   ├── agents/
│   ├── api/
│   ├── core/
│   ├── models/
│   ├── services/
│   ├── tools/
│   ├── utils/
│   └── main.py
├── docs/
├── tests/
├── pyproject.toml
└── README.md
```

## API 엔드포인트

- `GET /`
- `GET /health`
- `POST /api/v1/chat`

## 문서

- API 명세: [`docs/spec.md`](/Users/leeyongpil/working/AI-Agent/Agent-Devlop/agent/docs/spec.md)
- Tool 상세: [`docs/tools.md`](/Users/leeyongpil/working/AI-Agent/Agent-Devlop/agent/docs/tools.md)
