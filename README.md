# didim-AI-Agent-edu
디딤365 AI Agent 실습

FastAPI 기반의 LangChain v1.0 에이전트 교육용 템플릿입니다.

## 기술 스택

- FastAPI
- LangChain v1.0
- OpenAI
- Elasticsearch
- uv

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
