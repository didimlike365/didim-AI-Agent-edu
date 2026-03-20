from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from app.agents.search_agent import Agent
from app.core.config import settings
from app.services.elasticsearch_service import ElasticsearchService
from app.tools.medical_tools import get_medical_tools


def create_medical_agent() -> Agent:
    search_service = ElasticsearchService()
    llm = ChatOpenAI(
        model=settings.OPENAI_MODEL,
        api_key=settings.OPENAI_API_KEY,
        temperature=0,
    )
    tools = get_medical_tools(search_service)
    routing_agent = create_agent(
        model=llm,
        tools=tools,
        interrupt_after=["tools"],
        name="search_agent_router",
    )
    routing_agent_without_emergency = create_agent(
        model=llm,
        tools=[tool for tool in tools if tool.name != "emergency_symptom_triage"],
        interrupt_after=["tools"],
        name="search_agent_router_without_emergency",
    )
    return Agent(
        search_service=search_service,
        llm=llm,
        tools=tools,
        routing_agent=routing_agent,
        routing_agent_without_emergency=routing_agent_without_emergency,
    )
