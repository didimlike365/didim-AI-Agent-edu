import uuid
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.agents.prompts import context_prompt, system_prompt
from app.core.config import settings
from app.services.elasticsearch_service import ElasticsearchService


class SearchMessage(BaseModel):
    tool_calls: list[dict[str, Any]]


class Agent:
    LOCATION_KEYWORDS = ("근처", "주변", "인근", "지역", "병원", "의원", "응급실")
    DURATION_KEYWORDS = ("지속", "계속", "며칠", "몇일", "몇 주", "몇주", "몇 달", "몇달", "주째", "일째", "달째", "오래")

    def __init__(self):
        self.search_service = ElasticsearchService()
        self.llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=0,
        )
        self.tool_llm = self.llm.bind_tools(
            [self.medical_search, self.hospital_search, self.symptom_duration_parser]
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", context_prompt),
            ]
        )

    async def astream(self, input_data: dict, config: dict = None, stream_mode: str = "updates"):
        messages = input_data.get("messages", [])
        question = ""
        if messages:
            question = getattr(messages[-1], "content", str(messages[-1]))

        tool_context, metadata = await self._run_tools(question)
        answer = await self._generate_answer(question, tool_context)

        message = SearchMessage(
            tool_calls=[
                {
                    "name": "ChatResponse",
                    "args": {
                        "message_id": str(uuid.uuid4()),
                        "content": answer,
                        "metadata": metadata,
                    },
                }
            ]
        )

        yield {
            "model": {
                "messages": [message],
            }
        }

    @tool
    async def medical_search(self, query: str) -> str:
        """의학 문서 인덱스에서 질병, 진단, 치료 등 설명형 정보를 검색합니다."""
        documents = await self.search_service.search(query)
        if not documents:
            return "의학 지식 검색 결과가 없습니다."
        return self._build_medical_context(documents)

    @tool
    async def hospital_search(self, query: str, region: str = "") -> str:
        """같은 edu-collection 인덱스에서 지역과 병원 관련 문맥을 우선 검색합니다."""
        augmented_query = self._build_location_query(query=query, region=region)
        documents = await self.search_service.search(augmented_query)
        if not documents:
            return "지역 또는 병원 관련 검색 결과가 없습니다."
        return self._build_hospital_context(documents)

    @tool
    async def symptom_duration_parser(self, query: str) -> str:
        """질문에서 증상과 지속 기간 표현을 추출해 구조화합니다."""
        parsed = self._parse_symptom_duration(query)
        return self._format_symptom_duration(parsed)

    async def _generate_answer(self, question: str, context: str) -> str:
        chain = self.prompt | self.llm
        response = await chain.ainvoke({"question": question, "context": context})
        return response.content if hasattr(response, "content") else str(response)

    async def _run_tools(self, question: str) -> tuple[str, dict[str, Any]]:
        initial_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=question),
        ]
        ai_message = await self.tool_llm.ainvoke(initial_messages)
        tool_calls = list(getattr(ai_message, "tool_calls", []) or [])

        if self._is_location_query(question) and not any(
            tool_call["name"] == "hospital_search" for tool_call in tool_calls
        ):
            tool_calls.append(
                {
                    "id": f"manual_hospital_search_{uuid.uuid4()}",
                    "name": "hospital_search",
                    "args": {"query": question, "region": ""},
                }
            )
        if self._is_duration_query(question) and not any(
            tool_call["name"] == "symptom_duration_parser" for tool_call in tool_calls
        ):
            tool_calls.append(
                {
                    "id": f"manual_symptom_duration_parser_{uuid.uuid4()}",
                    "name": "symptom_duration_parser",
                    "args": {"query": question},
                }
            )

        if not tool_calls:
            documents = await self.search_service.search(question)
            context = self._build_medical_context(documents) if documents else "의학 지식 검색 결과가 없습니다."
            metadata = {"sources": self._build_sources(documents), "used_tools": ["medical_search"]}
            return context, metadata

        tool_messages: list[ToolMessage] = []
        used_tools: list[str] = []
        sources: list[dict[str, Any]] = []
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            args = tool_call.get("args", {})
            used_tools.append(tool_name)

            if tool_name == "medical_search":
                documents = await self.search_service.search(args.get("query", question))
                sources.extend(self._build_sources(documents))
                tool_result = self._build_medical_context(documents) if documents else "의학 지식 검색 결과가 없습니다."
            elif tool_name == "hospital_search":
                documents = await self.search_service.search(
                    self._build_location_query(
                        query=args.get("query", question),
                        region=args.get("region", ""),
                    )
                )
                sources.extend(self._build_sources(documents))
                tool_result = (
                    self._build_hospital_context(documents)
                    if documents
                    else "지역 또는 병원 관련 검색 결과가 없습니다."
                )
            elif tool_name == "symptom_duration_parser":
                tool_result = self._format_symptom_duration(
                    self._parse_symptom_duration(args.get("query", question))
                )
            else:
                tool_result = "지원하지 않는 도구입니다."

            tool_messages.append(
                ToolMessage(
                    content=tool_result,
                    tool_call_id=tool_call["id"],
                )
            )

        context_parts = []
        for tool_call, tool_message in zip(tool_calls, tool_messages):
            context_parts.append(f"[{tool_call['name']}]\n{tool_message.content}")
        metadata = {"sources": sources, "used_tools": used_tools}
        return "\n\n".join(context_parts), metadata

    def _build_medical_context(self, documents: list[dict[str, Any]]) -> str:
        chunks = []
        for index, doc in enumerate(documents, start=1):
            chunks.append(
                "\n".join(
                    [
                        f"[문서 {index}]",
                        f"c_id: {doc.get('c_id')}",
                        f"creation_year: {doc.get('creation_year')}",
                        f"domain: {doc.get('domain')}",
                        f"source: {doc.get('source')}",
                        f"source_spec: {doc.get('source_spec')}",
                        f"content: {doc.get('content')}",
                    ]
                )
            )
        return "\n\n".join(chunks)

    def _build_hospital_context(self, documents: list[dict[str, Any]]) -> str:
        chunks = []
        for index, doc in enumerate(documents, start=1):
            chunks.append(
                "\n".join(
                    [
                        f"[지역/병원 문맥 {index}]",
                        f"c_id: {doc.get('c_id')}",
                        f"creation_year: {doc.get('creation_year')}",
                        f"content: {doc.get('content')}",
                    ]
                )
            )
        return "\n\n".join(chunks)

    def _build_sources(self, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
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

    def _is_location_query(self, question: str) -> bool:
        return any(keyword in question for keyword in self.LOCATION_KEYWORDS)

    def _is_duration_query(self, question: str) -> bool:
        return any(keyword in question for keyword in self.DURATION_KEYWORDS)

    def _build_location_query(self, query: str, region: str = "") -> str:
        parts = [query]
        if region:
            parts.append(region)
        parts.extend(["병원", "의료기관", "위치", "지역"])
        return " ".join(part for part in parts if part)

    def _parse_symptom_duration(self, question: str) -> dict[str, Any]:
        duration_patterns = [
            r"(\d+\s*시간(?:째)?)",
            r"(\d+\s*일(?:째)?)",
            r"(\d+\s*주(?:째)?)",
            r"(\d+\s*달(?:째)?)",
            r"(\d+\s*개월(?:째)?)",
            r"(며칠(?:째)?)",
            r"(몇\s*주(?:째)?)",
            r"(몇\s*달(?:째)?)",
            r"(오래\s*지속)",
            r"(계속)",
        ]
        symptom_keywords = [
            "기침",
            "발열",
            "열",
            "가래",
            "호흡곤란",
            "두통",
            "복통",
            "인후통",
            "콧물",
            "오한",
            "흉통",
            "설사",
            "구토",
            "피로",
            "체중감소",
        ]

        import re

        duration_matches: list[str] = []
        for pattern in duration_patterns:
            duration_matches.extend(match.strip() for match in re.findall(pattern, question))

        symptoms = [symptom for symptom in symptom_keywords if symptom in question]
        return {
            "question": question,
            "symptoms": symptoms,
            "durations": duration_matches,
            "has_duration_context": bool(duration_matches),
        }

    def _format_symptom_duration(self, parsed: dict[str, Any]) -> str:
        symptoms = ", ".join(parsed.get("symptoms", [])) or "명시되지 않음"
        durations = ", ".join(parsed.get("durations", [])) or "명시되지 않음"
        return "\n".join(
            [
                "[증상 지속기간 분석]",
                f"원문 질문: {parsed.get('question', '')}",
                f"추출된 증상: {symptoms}",
                f"추출된 지속기간: {durations}",
            ]
        )
