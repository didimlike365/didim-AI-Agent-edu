from typing import Any

import httpx

from app.core.config import settings
from app.utils.logger import custom_logger, log_execution


class ElasticsearchService:
    def __init__(self):
        self.base_url = settings.ELASTICSEARCH_URL.rstrip("/")
        self.index = settings.ELASTICSEARCH_INDEX
        self.top_k = settings.ELASTICSEARCH_TOP_K
        self.auth = (settings.ELASTICSEARCH_USERNAME, settings.ELASTICSEARCH_PASSWORD)

    @log_execution
    async def search(self, query: str) -> list[dict[str, Any]]:
        payload = {
            "size": self.top_k,
            "_source": [
                "c_id",
                "content",
                "creation_year",
                "domain",
                "source",
                "source_spec",
            ],
            "query": {
                "match": {
                    "content": {
                        "query": query,
                    }
                }
            },
        }
        return await self._execute_search(index=self.index, payload=payload)

    @log_execution
    async def search_hospitals(
        self,
        query: str,
        region: str | None = None,
    ) -> list[dict[str, Any]]:
        if not settings.HOSPITAL_ELASTICSEARCH_INDEX:
            return []

        should_queries: list[dict[str, Any]] = [
            {
                "multi_match": {
                    "query": query,
                    "fields": settings.HOSPITAL_SEARCH_FIELDS,
                }
            }
        ]
        if region:
            should_queries.append(
                {
                    "multi_match": {
                        "query": region,
                        "fields": settings.HOSPITAL_SEARCH_FIELDS,
                    }
                }
            )

        payload = {
            "size": self.top_k,
            "query": {
                "bool": {
                    "should": should_queries,
                    "minimum_should_match": 1,
                }
            },
        }
        return await self._execute_search(index=settings.HOSPITAL_ELASTICSEARCH_INDEX, payload=payload)

    async def _execute_search(self, index: str, payload: dict[str, Any]) -> list[dict[str, Any]]:

        async with httpx.AsyncClient(timeout=10.0, verify=True) as client:
            response = await client.post(
                f"{self.base_url}/{index}/_search",
                auth=self.auth,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()

        body = response.json()
        hits = body.get("hits", {}).get("hits", [])
        documents = []
        for hit in hits:
            source = hit.get("_source", {})
            documents.append(
                {
                    "id": hit.get("_id"),
                    "score": hit.get("_score"),
                    "c_id": source.get("c_id"),
                    "content": source.get("content", ""),
                    "creation_year": source.get("creation_year"),
                    "domain": source.get("domain"),
                    "source": source.get("source"),
                    "source_spec": source.get("source_spec"),
                    "raw": source,
                }
            )

        custom_logger.info("Elasticsearch search returned %s documents", len(documents))
        return documents
