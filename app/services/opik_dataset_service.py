from dataclasses import dataclass
from typing import Any

from app.utils.logger import custom_logger


@dataclass(slots=True)
class OpikDatasetBinding:
    name: str
    connected: bool = False
    dataset_id: str | None = None
    error: str | None = None


class OpikDatasetService:
    """Resolve an already-registered Opik dataset for later evaluation use."""

    def __init__(self, client: Any | None, dataset_name: str | None):
        self.client = client
        self.dataset_name = dataset_name

    def connect(self) -> OpikDatasetBinding | None:
        if self.client is None or not self.dataset_name:
            return None

        try:
            dataset = self.client.get_dataset(name=self.dataset_name)
        except Exception as exc:
            message = str(exc)
            custom_logger.warning(
                "Failed to connect Opik dataset '%s': %s",
                self.dataset_name,
                exc,
            )
            return OpikDatasetBinding(
                name=self.dataset_name,
                connected=False,
                error=message,
            )

        dataset_id = getattr(dataset, "id", None)
        custom_logger.info(
            "Connected Opik dataset '%s'%s",
            self.dataset_name,
            f" (id={dataset_id})" if dataset_id else "",
        )
        return OpikDatasetBinding(
            name=self.dataset_name,
            connected=True,
            dataset_id=dataset_id,
        )
