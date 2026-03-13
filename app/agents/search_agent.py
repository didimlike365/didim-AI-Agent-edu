import uuid
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.agents.prompts import system_prompt
from app.core.config import settings
from app.services.elasticsearch_service import ElasticsearchService
from app.tools.medical_tools import build_location_query, build_sources, get_medical_tools


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

        answer, metadata = await self._run_agent(question)
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

    async def _run_agent(self, question: str) -> tuple[str, dict[str, Any]]:
        initial_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=question),
        ]

        agent_message = await self.tool_llm.ainvoke(initial_messages)
        tool_calls = list(getattr(agent_message, "tool_calls", []) or [])

        if not tool_calls:
            answer = agent_message.content if hasattr(agent_message, "content") else str(agent_message)
            return answer, {"sources": [], "used_tools": []}

        used_tools: list[str] = []
        sources: list[dict[str, Any]] = []
        tool_messages: list[ToolMessage] = []

        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            used_tools.append(tool_name)

            content = await self._execute_tool(
                tool_name=tool_name,
                args=tool_call.get("args", {}),
                question=question,
            )
            sources.extend(await self._collect_sources(tool_name, tool_call.get("args", {}), question))
            tool_messages.append(
                ToolMessage(
                    content=content,
                    tool_call_id=tool_call["id"],
                )
            )

        final_messages = initial_messages + [agent_message] + tool_messages
        final_response = await self.llm.ainvoke(final_messages)
        answer = final_response.content if hasattr(final_response, "content") else str(final_response)
        answer = self._append_used_tools(answer, used_tools)
        metadata = {"sources": sources, "used_tools": used_tools}
        return answer, metadata

    async def _execute_tool(self, tool_name: str, args: dict[str, Any], question: str) -> str:
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

        raw_result = await tool.ainvoke(tool_args)
        return raw_result if isinstance(raw_result, str) else str(raw_result)

    async def _collect_sources(self, tool_name: str, args: dict[str, str], question: str) -> list[dict[str, Any]]:
        if tool_name == "medical_search":
            documents = await self.search_service.search(args.get("query", question))
            return build_sources(documents)
        if tool_name == "hospital_search":
            documents = await self.search_service.search(
                build_location_query(
                    query=args.get("query", question),
                    region=args.get("region", ""),
                )
            )
            return build_sources(documents)
        return []

    def _append_used_tools(self, answer: str, used_tools: list[str]) -> str:
        if not used_tools:
            return answer

        lines = ["", "[Tool Usage]"]
        for tool_name in used_tools:
            lines.append(f"- {tool_name}")
        return f"{answer}\n" + "\n".join(lines)
