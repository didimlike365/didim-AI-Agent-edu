from __future__ import annotations

import asyncio
from datetime import datetime
import os
import uuid
from typing import Any

from opik import Opik
from opik.evaluation import evaluate
from opik.evaluation.metrics import AnswerRelevance, Hallucination
from opik.api_objects import opik_client

from app.agents.search_agent import Agent
from app.core.config import settings
from app.evaluation.metrics import (
    ContradictoryAbstentionMetric,
    InvalidSubjectHandlingMetric,
    RequiredContentMetric,
    SearchGroundedOnlyMetric,
)
from app.services.elasticsearch_service import ElasticsearchService
from app.utils.logger import custom_logger


QUESTION_KEYS = ("input", "question", "message", "query", "user_input")


def build_experiment_name(dataset_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{dataset_name}-eval-{timestamp}"


def configure_opik_environment() -> None:
    if settings.OPIK is None:
        raise RuntimeError("Opik settings are not configured.")

    if settings.OPIK.URL_OVERRIDE:
        os.environ["OPIK_URL_OVERRIDE"] = settings.OPIK.URL_OVERRIDE
    if settings.OPIK.PROJECT:
        os.environ["OPIK_PROJECT_NAME"] = settings.OPIK.PROJECT
    if settings.OPIK.WORKSPACE:
        os.environ["OPIK_WORKSPACE"] = settings.OPIK.WORKSPACE
    if settings.OPIK.API_KEY:
        os.environ["OPIK_API_KEY"] = settings.OPIK.API_KEY

    opik_client.get_client_cached.cache_clear()


def get_opik_client() -> Opik:
    if settings.OPIK is None:
        raise RuntimeError("Opik settings are not configured.")

    configure_opik_environment()

    return Opik(
        project_name=settings.OPIK.PROJECT,
        workspace=settings.OPIK.WORKSPACE,
        host=settings.OPIK.URL_OVERRIDE,
        api_key=settings.OPIK.API_KEY,
        _show_misconfiguration_message=False,
    )


def get_evaluation_dataset(dataset_name: str | None = None):
    if settings.OPIK is None:
        raise RuntimeError("Opik settings are not configured.")

    resolved_name = dataset_name or settings.OPIK.DATASET
    if not resolved_name:
        raise RuntimeError("No Opik dataset name was configured.")

    client = get_opik_client()
    dataset = client.get_dataset(name=resolved_name)
    custom_logger.info("Loaded Opik dataset '%s' for evaluation", resolved_name)
    return dataset


def build_default_metrics(include_judge_metrics: bool = True):
    metrics = [
        SearchGroundedOnlyMetric(name="search_grounded_only"),
        RequiredContentMetric(name="required_content"),
        ContradictoryAbstentionMetric(name="contradictory_abstention"),
        InvalidSubjectHandlingMetric(name="invalid_subject_handling"),
    ]
    if include_judge_metrics:
        metrics.extend(
            [
                Hallucination(name="hallucination"),
                AnswerRelevance(name="answer_relevance"),
            ]
        )
    return metrics


def _extract_question(dataset_item: dict[str, Any]) -> str:
    for key in QUESTION_KEYS:
        value = dataset_item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            for nested_key in QUESTION_KEYS:
                nested_value = value.get(nested_key)
                if isinstance(nested_value, str) and nested_value.strip():
                    return nested_value.strip()

    raise ValueError(
        "Dataset item does not contain a supported question key. "
        f"Expected one of: {', '.join(QUESTION_KEYS)}"
    )


def _build_context_documents(documents: list[dict[str, Any]]) -> list[str]:
    context_documents: list[str] = []
    for doc in documents:
        content = str(doc.get("content", "")).strip()
        if not content:
            continue
        source_spec = doc.get("source_spec") or doc.get("source") or "unknown"
        context_documents.append(f"[{source_spec}] {content}")
    return context_documents


async def _run_agent_case(dataset_item: dict[str, Any]) -> dict[str, Any]:
    question = _extract_question(dataset_item)
    expected_output = dataset_item.get("expected_output")
    agent = Agent()
    answer, metadata = await agent._run_agent(
        question,
        config={
            "configurable": {"thread_id": str(uuid.uuid4())},
            "metadata": {"evaluation": True},
        },
    )

    documents = await ElasticsearchService().search(question)
    context = _build_context_documents(documents)

    return {
        "source": "agent-evaluation",
        "input": question,
        "output": answer,
        "dataset_input": question,
        "dataset_expected_output": expected_output,
        "expected_output": expected_output,
        "context": context,
        "used_tools": metadata.get("used_tools", []),
        "sources": metadata.get("sources", []),
        "expected_behavior": dataset_item.get("expected_behavior"),
        "must_include": dataset_item.get("must_include") or [],
        "must_not_include": dataset_item.get("must_not_include") or [],
        "category": dataset_item.get("category"),
    }


def evaluation_task(dataset_item: dict[str, Any]) -> dict[str, Any]:
    return asyncio.run(_run_agent_case(dataset_item))


def run_dataset_experiment(
    dataset_name: str | None = None,
    experiment_name: str | None = None,
    include_judge_metrics: bool = True,
    nb_samples: int | None = None,
):
    configure_opik_environment()
    resolved_dataset_name = dataset_name or settings.OPIK.DATASET
    dataset = get_evaluation_dataset(dataset_name)
    metrics = build_default_metrics(include_judge_metrics=include_judge_metrics)

    experiment_config = {
        "openai_model": settings.OPENAI_MODEL,
        "triage_model": settings.TRIAGE_MODEL,
        "search_summarizer_model": settings.SEARCH_SUMMARIZER_MODEL,
        "dataset_name": resolved_dataset_name,
        "search_grounded_policy": "search_only",
    }

    resolved_experiment_name = experiment_name or build_experiment_name(
        resolved_dataset_name or "opik-dataset"
    )

    return evaluate(
        dataset=dataset,
        task=evaluation_task,
        scoring_metrics=metrics,
        experiment_name=resolved_experiment_name,
        experiment_name_prefix=None,
        project_name=settings.OPIK.PROJECT if settings.OPIK is not None else None,
        experiment_config=experiment_config,
        nb_samples=nb_samples,
        task_threads=1,
    )
