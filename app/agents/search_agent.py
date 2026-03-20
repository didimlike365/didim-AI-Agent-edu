import uuid
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel
from typing_extensions import TypedDict

from app.agents.prompts import system_prompt
from app.core.config import settings
from app.services.elasticsearch_service import ElasticsearchService
from app.tools.medical_tools import build_sources, get_medical_tools


class SearchMessage(BaseModel):
    tool_calls: list[dict[str, Any]]


class ToolStreamMessage(BaseModel):
    name: str
    content: str


class AgentGraphState(TypedDict, total=False):
    question: str
    runtime_metadata: dict[str, Any]
    invoke_config: dict[str, Any] | None
    initial_messages: list[Any]
    routed_messages: list[Any]
    agent_message: Any
    tool_calls: list[dict[str, Any]]
    used_tools: list[str]
    sources: list[dict[str, Any]]
    tool_messages: list[ToolMessage]
    auto_subject_validity_content: str | None
    invalid_subject_response: str | None
    auto_triage_content: str | None
    triage_response: str | None
    direct_answer: str | None
    generated_answer: str | None
    final_answer: str | None
    final_metadata: dict[str, Any]


class Agent:
    def __init__(
        self,
        search_service: ElasticsearchService | None = None,
        llm: ChatOpenAI | None = None,
        tools: list[Any] | None = None,
        routing_agent: Any | None = None,
        routing_agent_without_emergency: Any | None = None,
    ):
        self.search_service = search_service or ElasticsearchService()
        self.llm = llm or ChatOpenAI(
            model=settings.OPENAI_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=0,
        )
        self.tools = tools or get_medical_tools(self.search_service)
        self.tools_by_name = {tool.name: tool for tool in self.tools}
        self.routing_agent = routing_agent or create_agent(
            model=self.llm,
            tools=self.tools,
            interrupt_after=["tools"],
            name="search_agent_router",
        )
        self.routing_agent_without_emergency = routing_agent_without_emergency or create_agent(
            model=self.llm,
            tools=[tool for tool in self.tools if tool.name != "emergency_symptom_triage"],
            interrupt_after=["tools"],
            name="search_agent_router_without_emergency",
        )
        self.graph = self._build_graph()

    async def astream(self, input_data: dict, config: dict = None, stream_mode: str = "updates"):
        messages = input_data.get("messages", [])
        question = ""
        if messages:
            question = getattr(messages[-1], "content", str(messages[-1]))

        state = self._build_initial_state(question, config)
        final_answer = ""
        final_metadata: dict[str, Any] = {}

        async for chunk in self.graph.astream(state, stream_mode="updates"):
            for node_name, updates in chunk.items():
                if node_name == "route_tools" and updates.get("tool_calls"):
                    yield {
                        "model": {
                            "messages": [
                                SearchMessage(tool_calls=list(updates["tool_calls"])),
                            ],
                        }
                    }

                if node_name == "execute_tools":
                    for tool_message in updates.get("tool_messages", []):
                        yield {
                            "tools": {
                                "messages": [tool_message],
                            }
                        }

                if node_name == "subject_validity" and updates.get("auto_subject_validity_content"):
                    yield {
                        "tools": {
                            "messages": [
                                ToolStreamMessage(
                                    name="subject_validity_check",
                                    content=updates["auto_subject_validity_content"],
                                )
                            ],
                        }
                    }

                if node_name == "emergency_triage" and updates.get("auto_triage_content"):
                    yield {
                        "tools": {
                            "messages": [
                                ToolStreamMessage(
                                    name="emergency_symptom_triage",
                                    content=updates["auto_triage_content"],
                                )
                            ],
                        }
                    }

                if node_name.startswith("finish_"):
                    final_answer = updates.get("final_answer", final_answer)
                    final_metadata = updates.get("final_metadata", final_metadata)

        yield {
            "model": {
                "messages": [
                    SearchMessage(
                        tool_calls=[
                            {
                                "name": "ChatResponse",
                                "args": {
                                    "message_id": str(uuid.uuid4()),
                                    "content": final_answer,
                                    "metadata": final_metadata,
                                },
                            }
                        ]
                    )
                ],
            }
        }

    async def _run_agent(self, question: str, config: dict | None = None) -> tuple[str, dict[str, Any]]:
        state = self._build_initial_state(question, config)
        result = await self.graph.ainvoke(state)
        return result["final_answer"], result["final_metadata"]

    def _build_graph(self):
        graph = StateGraph(AgentGraphState)
        graph.add_node("route_tools", self._route_tools_node)
        graph.add_node("execute_tools", self._execute_tools_node)
        graph.add_node("subject_validity", self._subject_validity_node)
        graph.add_node("emergency_triage", self._emergency_triage_node)
        graph.add_node("final_answer", self._final_answer_node)
        graph.add_node("finish_direct", self._finish_direct_node)
        graph.add_node("finish_invalid_subject", self._finish_invalid_subject_node)
        graph.add_node("finish_triage", self._finish_triage_node)
        graph.add_node("finish_final", self._finish_final_node)

        graph.add_edge(START, "route_tools")
        graph.add_conditional_edges(
            "route_tools",
            self._route_after_tool_selection,
            {
                "direct": "finish_direct",
                "tools": "execute_tools",
            },
        )
        graph.add_edge("execute_tools", "subject_validity")
        graph.add_conditional_edges(
            "subject_validity",
            self._route_after_subject_validity,
            {
                "invalid_subject": "finish_invalid_subject",
                "triage": "emergency_triage",
                "final": "final_answer",
            },
        )
        graph.add_conditional_edges(
            "emergency_triage",
            self._route_after_emergency_triage,
            {
                "triage": "finish_triage",
                "final": "final_answer",
            },
        )
        graph.add_edge("final_answer", "finish_final")
        graph.add_edge("finish_direct", END)
        graph.add_edge("finish_invalid_subject", END)
        graph.add_edge("finish_triage", END)
        graph.add_edge("finish_final", END)
        return graph.compile()

    async def _route_tools_node(self, state: AgentGraphState) -> AgentGraphState:
        routing_agent = self._select_routing_agent(state.get("runtime_metadata", {}))
        result = await routing_agent.ainvoke(
            {"messages": state["initial_messages"]},
            config=state.get("invoke_config"),
        )
        routed_messages = list(result.get("messages", []))
        agent_message = self._select_agent_message(routed_messages)
        tool_calls = list(getattr(agent_message, "tool_calls", []) or []) if agent_message else []
        updates: AgentGraphState = {
            "routed_messages": routed_messages,
            "agent_message": agent_message,
            "tool_calls": tool_calls,
        }
        if not tool_calls:
            updates["direct_answer"] = (
                agent_message.content if hasattr(agent_message, "content") else str(agent_message)
            )
        return updates

    def _route_after_tool_selection(self, state: AgentGraphState) -> str:
        return "direct" if not state.get("tool_calls") else "tools"

    async def _execute_tools_node(self, state: AgentGraphState) -> AgentGraphState:
        question = state["question"]
        used_tools = list(state.get("used_tools", []))
        sources = list(state.get("sources", []))
        tool_messages = [
            message
            for message in state.get("routed_messages", [])
            if isinstance(message, ToolMessage)
        ]

        for tool_call in state.get("tool_calls", []):
            tool_name = tool_call["name"]
            used_tools.append(tool_name)
            sources.extend(await self._collect_sources(tool_name, tool_call.get("args", {}), question))

        return {
            "used_tools": used_tools,
            "sources": sources,
            "tool_messages": tool_messages,
        }

    async def _subject_validity_node(self, state: AgentGraphState) -> AgentGraphState:
        tool_messages = list(state.get("tool_messages", []))
        used_tools = list(state.get("used_tools", []))
        runtime_metadata = state.get("runtime_metadata", {})
        auto_subject_validity_content: str | None = None

        if self._should_run_subject_validity_check(tool_messages, used_tools, runtime_metadata):
            used_tools.append("subject_validity_check")
            auto_subject_validity_content = await self._execute_tool(
                tool_name="subject_validity_check",
                args={"query": state["question"]},
                question=state["question"],
                config=state.get("invoke_config"),
            )

        invalid_subject_response = self._extract_invalid_subject_response(tool_messages)
        if invalid_subject_response is None and auto_subject_validity_content is not None:
            invalid_subject_response = self._extract_invalid_subject_response_from_content(
                auto_subject_validity_content
            )

        return {
            "used_tools": used_tools,
            "auto_subject_validity_content": auto_subject_validity_content,
            "invalid_subject_response": invalid_subject_response,
        }

    def _route_after_subject_validity(self, state: AgentGraphState) -> str:
        if state.get("invalid_subject_response") is not None:
            return "invalid_subject"
        if self._should_run_emergency_triage(
            state.get("tool_messages", []),
            state.get("used_tools", []),
            state.get("runtime_metadata", {}),
        ):
            return "triage"
        return "final"

    async def _emergency_triage_node(self, state: AgentGraphState) -> AgentGraphState:
        used_tools = list(state.get("used_tools", []))
        used_tools.append("emergency_symptom_triage")
        auto_triage_content = await self._execute_tool(
            tool_name="emergency_symptom_triage",
            args={"query": state["question"]},
            question=state["question"],
            config=state.get("invoke_config"),
        )
        triage_response = self._extract_triage_response(state.get("tool_messages", []))
        if triage_response is None and auto_triage_content is not None:
            triage_response = self._extract_triage_response_from_content(auto_triage_content)
        return {
            "used_tools": used_tools,
            "auto_triage_content": auto_triage_content,
            "triage_response": triage_response,
        }

    def _route_after_emergency_triage(self, state: AgentGraphState) -> str:
        return "triage" if state.get("triage_response") is not None else "final"

    async def _final_answer_node(self, state: AgentGraphState) -> AgentGraphState:
        final_messages = state["initial_messages"] + [state["agent_message"]] + state.get("tool_messages", [])
        final_response = await self.llm.ainvoke(final_messages, config=state.get("invoke_config"))
        answer = final_response.content if hasattr(final_response, "content") else str(final_response)
        return {"generated_answer": answer}

    async def _finish_direct_node(self, state: AgentGraphState) -> AgentGraphState:
        return {
            "final_answer": state.get("direct_answer", ""),
            "final_metadata": {"sources": [], "used_tools": []},
        }

    async def _finish_invalid_subject_node(self, state: AgentGraphState) -> AgentGraphState:
        answer = self._append_used_tools(
            state.get("invalid_subject_response", ""),
            state.get("used_tools", []),
        )
        return {
            "final_answer": answer,
            "final_metadata": {
                "sources": state.get("sources", []),
                "used_tools": state.get("used_tools", []),
            },
        }

    async def _finish_triage_node(self, state: AgentGraphState) -> AgentGraphState:
        answer = self._append_used_tools(
            state.get("triage_response", ""),
            state.get("used_tools", []),
        )
        return {
            "final_answer": answer,
            "final_metadata": {
                "sources": state.get("sources", []),
                "used_tools": state.get("used_tools", []),
                "triage": {
                    "pending": True,
                    "original_question": state["question"],
                },
            },
        }

    async def _finish_final_node(self, state: AgentGraphState) -> AgentGraphState:
        answer = self._append_used_tools(
            state.get("generated_answer", ""),
            state.get("used_tools", []),
        )
        return {
            "final_answer": answer,
            "final_metadata": {
                "sources": state.get("sources", []),
                "used_tools": state.get("used_tools", []),
            },
        }

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

    def _build_initial_state(self, question: str, config: dict[str, Any] | None) -> AgentGraphState:
        runtime_metadata = self._get_runtime_metadata(config)
        return {
            "question": question,
            "runtime_metadata": runtime_metadata,
            "invoke_config": self._build_invoke_config(config),
            "initial_messages": [
                SystemMessage(content=system_prompt),
                HumanMessage(content=question),
            ],
            "used_tools": [],
            "sources": [],
            "tool_messages": [],
        }

    def _select_routing_agent(self, runtime_metadata: dict[str, Any]):
        if runtime_metadata.get("skip_emergency_triage"):
            return self.routing_agent_without_emergency
        return self.routing_agent

    def _select_agent_message(self, routed_messages: list[Any]) -> Any | None:
        ai_messages = [message for message in routed_messages if hasattr(message, "tool_calls")]
        if not ai_messages:
            return None

        for message in ai_messages:
            if getattr(message, "tool_calls", None):
                return message
        return ai_messages[-1]
