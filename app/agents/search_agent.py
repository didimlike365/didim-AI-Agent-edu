import uuid
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.agents.prompts import system_prompt
from app.core.config import settings
from app.services.elasticsearch_service import ElasticsearchService
from app.tools.medical_tools import build_sources, get_medical_tools


class SearchMessage(BaseModel):
    tool_calls: list[dict[str, Any]]


class Agent:
    def __init__(self):
        self.search_service = ElasticsearchService()
        self.llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=0,
        )
        self.tools = get_medical_tools(self.search_service)
        self.tools_by_name = {tool.name: tool for tool in self.tools}
        self.tool_llm = self.llm.bind_tools(self.tools)

    async def astream(self, input_data: dict, config: dict = None, stream_mode: str = "updates"):
        messages = input_data.get("messages", [])
        question = ""
        if messages:
            question = getattr(messages[-1], "content", str(messages[-1]))

        answer, metadata = await self._run_agent(question, config=config)
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

    async def _run_agent(self, question: str, config: dict | None = None) -> tuple[str, dict[str, Any]]:
        runtime_metadata = self._get_runtime_metadata(config)
        initial_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=question),
        ]
        invoke_config = self._build_invoke_config(config)

        agent_message = await self.tool_llm.ainvoke(initial_messages, config=invoke_config)
        tool_calls = list(getattr(agent_message, "tool_calls", []) or [])

        if not tool_calls:
            answer = agent_message.content if hasattr(agent_message, "content") else str(agent_message)
            return answer, {"sources": [], "used_tools": []}

        used_tools: list[str] = []
        sources: list[dict[str, Any]] = []
        tool_messages: list[ToolMessage] = []
        auto_subject_validity_content: str | None = None
        auto_triage_content: str | None = None

        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            if runtime_metadata.get("skip_emergency_triage") and tool_name == "emergency_symptom_triage":
                continue
            used_tools.append(tool_name)

            content = await self._execute_tool(
                tool_name=tool_name,
                args=tool_call.get("args", {}),
                question=question,
                config=invoke_config,
            )
            sources.extend(await self._collect_sources(tool_name, tool_call.get("args", {}), question))
            tool_messages.append(
                ToolMessage(
                    content=content,
                    tool_call_id=tool_call["id"],
                )
            )

        if self._should_run_subject_validity_check(tool_messages, used_tools, runtime_metadata):
            used_tools.append("subject_validity_check")
            auto_subject_validity_content = await self._execute_tool(
                tool_name="subject_validity_check",
                args={"query": question},
                question=question,
                config=invoke_config,
            )

        invalid_subject_response = self._extract_invalid_subject_response(tool_messages)
        if invalid_subject_response is None and auto_subject_validity_content is not None:
            invalid_subject_response = self._extract_invalid_subject_response_from_content(
                auto_subject_validity_content
            )
        if invalid_subject_response is not None:
            answer = self._append_used_tools(invalid_subject_response, used_tools)
            metadata = {"sources": sources, "used_tools": used_tools}
            return answer, metadata

        if self._should_run_emergency_triage(tool_messages, used_tools, runtime_metadata):
            used_tools.append("emergency_symptom_triage")
            auto_triage_content = await self._execute_tool(
                tool_name="emergency_symptom_triage",
                args={"query": question},
                question=question,
                config=invoke_config,
            )

        triage_response = self._extract_triage_response(tool_messages)
        if triage_response is None and auto_triage_content is not None:
            triage_response = self._extract_triage_response_from_content(auto_triage_content)
        if triage_response is not None:
            answer = self._append_used_tools(triage_response, used_tools)
            metadata = {
                "sources": sources,
                "used_tools": used_tools,
                "triage": {
                    "pending": True,
                    "original_question": question,
                },
            }
            return answer, metadata

        final_messages = initial_messages + [agent_message] + tool_messages
        final_response = await self.llm.ainvoke(final_messages, config=invoke_config)
        answer = final_response.content if hasattr(final_response, "content") else str(final_response)
        answer = self._append_used_tools(answer, used_tools)
        metadata = {"sources": sources, "used_tools": used_tools}
        return answer, metadata

    async def _execute_tool(
        self,
        tool_name: str,
        args: dict[str, Any],
        question: str,
        config: dict[str, Any] | None = None,
    ) -> str:
        tool = self.tools_by_name.get(tool_name)
        if tool is None:
            return "지원하지 않는 도구입니다."

        tool_args = dict(args)
        if tool_name == "medical_search":
            tool_args.setdefault("query", question)
        elif tool_name == "hospital_search":
            tool_args.setdefault("query", question)
            tool_args.setdefault("region", "")
        elif tool_name == "symptom_parser":
            tool_args.setdefault("query", question)
        elif tool_name == "symptom_duration_parser":
            tool_args.setdefault("query", question)
        elif tool_name == "subject_validity_check":
            tool_args.setdefault("query", question)
        elif tool_name == "emergency_symptom_triage":
            tool_args.setdefault("query", question)

        raw_result = await tool.ainvoke(tool_args, config=config)
        return raw_result if isinstance(raw_result, str) else str(raw_result)

    async def _collect_sources(self, tool_name: str, args: dict[str, str], question: str) -> list[dict[str, Any]]:
        if tool_name == "medical_search":
            documents = await self.search_service.search(args.get("query", question))
            return build_sources(documents)
        return []

    def _append_used_tools(self, answer: str, used_tools: list[str]) -> str:
        if not used_tools:
            return answer

        lines = ["", "[Tool Usage]"]
        for tool_name in used_tools:
            lines.append(f"- {tool_name}")
        return f"{answer}\n" + "\n".join(lines)

    def _extract_triage_response(self, tool_messages: list[ToolMessage]) -> str | None:
        for message in tool_messages:
            content = message.content if isinstance(message.content, str) else str(message.content)
            extracted = self._extract_triage_response_from_content(content)
            if extracted is not None:
                return extracted
        return None

    def _extract_invalid_subject_response(self, tool_messages: list[ToolMessage]) -> str | None:
        for message in tool_messages:
            content = message.content if isinstance(message.content, str) else str(message.content)
            extracted = self._extract_invalid_subject_response_from_content(content)
            if extracted is not None:
                return extracted
        return None

    def _extract_triage_response_from_content(self, content: str) -> str | None:
        if "[응급 증상 확인]" not in content:
            return None
        if "추가 응급 확인 질문이 필요하지 않습니다." in content:
            return None
        return content

    def _extract_invalid_subject_response_from_content(self, content: str) -> str | None:
        if "[질문 대상 확인]" not in content:
            return None
        if "사람의 신체 증상 질문으로 볼 수 있습니다." in content:
            return None
        return content

    def _should_run_subject_validity_check(
        self,
        tool_messages: list[ToolMessage],
        used_tools: list[str],
        runtime_metadata: dict[str, Any],
    ) -> bool:
        if runtime_metadata.get("skip_emergency_triage"):
            return False
        if "subject_validity_check" in used_tools:
            return False

        for message in tool_messages:
            content = message.content if isinstance(message.content, str) else str(message.content)
            if "[증상 분석]" not in content:
                continue
            if "추출된 증상: 명시되지 않음" in content:
                return False
            return True

        return False

    def _should_run_emergency_triage(
        self,
        tool_messages: list[ToolMessage],
        used_tools: list[str],
        runtime_metadata: dict[str, Any],
    ) -> bool:
        if runtime_metadata.get("skip_emergency_triage"):
            return False
        if "emergency_symptom_triage" in used_tools:
            return False
        if "subject_validity_check" not in used_tools and self._should_run_subject_validity_check(
            tool_messages,
            used_tools,
            runtime_metadata,
        ):
            return False

        for message in tool_messages:
            content = message.content if isinstance(message.content, str) else str(message.content)
            if "[증상 분석]" not in content:
                continue
            if "추출된 증상: 명시되지 않음" in content:
                return False
            return True

        return False

    def _get_runtime_metadata(self, config: dict[str, Any] | None) -> dict[str, Any]:
        if not config:
            return {}
        metadata = config.get("metadata")
        return metadata if isinstance(metadata, dict) else {}

    def _build_invoke_config(self, config: dict[str, Any] | None) -> dict[str, Any] | None:
        if not config:
            return None

        invoke_config: dict[str, Any] = {}
        callbacks = config.get("callbacks")
        tags = config.get("tags")
        metadata = config.get("metadata")
        configurable = config.get("configurable")

        if callbacks:
            invoke_config["callbacks"] = callbacks
        if tags:
            invoke_config["tags"] = tags
        if metadata:
            invoke_config["metadata"] = metadata
        if configurable:
            invoke_config["configurable"] = configurable

        return invoke_config or None
