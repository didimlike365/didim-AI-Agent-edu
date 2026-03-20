from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class OpikSettings(BaseSettings):
    """Opik configuration."""

    URL_OVERRIDE: str | None = Field(default=None, description="Opik base URL")
    # Optional if you are using Opik Cloud:
    API_KEY: str | None = Field(default=None, description="opik cloud api key here")
    WORKSPACE: str | None = Field(default=None, description="your workspace name")
    PROJECT: str | None = Field(default=None, description="your project name")
    DATASET: str | None = Field(default=None, description="your dataset name")


class Settings(BaseSettings):
    # API 설정
    API_V1_PREFIX: str

    CORS_ORIGINS: List[str] = ["*"]
    
    # IMP: LangChain 객체 및 LLM 연동에 사용되는 필수 설정값(API Key 등)
    # LangChain 설정
    OPENAI_API_KEY: str
    OPENAI_MODEL: str
    TRIAGE_MODEL: str = "gpt-4.1-nano"
    SEARCH_SUMMARIZER_MODEL: str = "gpt-4.1-mini"
    SEARCH_SUMMARIZER_ENABLED: bool = True
    SEARCH_SUMMARIZER_MAX_DOCS: int = 3
    SEARCH_SUMMARIZER_MAX_CHARS_PER_DOC: int = 800
    SEARCH_SUMMARIZER_MAX_OUTPUT_TOKENS: int = 350

    ELASTICSEARCH_URL: str
    ELASTICSEARCH_INDEX: str = "edu-collection"
    ELASTICSEARCH_USERNAME: str
    ELASTICSEARCH_PASSWORD: str
    ELASTICSEARCH_TOP_K: int = 3
    HOSPITAL_ELASTICSEARCH_INDEX: str | None = None
    HOSPITAL_SEARCH_FIELDS: List[str] = ["name", "address", "department", "description"]
    
    # 기본 설정 (추가 환경변수가 필요하면 여기에 추가하세요)

    # IMP: DeepAgents 라이브러리 실행 시 Graph 에이전트의 최대 재귀 호출 횟수(Recursion Limit) 설정
    # DeepAgents 설정
    DEEPAGENT_RECURSION_LIMIT: int = 20

    OPIK: OpikSettings | None = None
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=True,
        extra="ignore",
    )

settings = Settings()
