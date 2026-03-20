from app.services.opik_dataset_service import OpikDatasetBinding, OpikDatasetService


class FakeDataset:
    def __init__(self, dataset_id: str):
        self.id = dataset_id


class FakeOpikClient:
    def __init__(self):
        self.requested_name = None

    def get_dataset(self, name: str):
        self.requested_name = name
        return FakeDataset(dataset_id="dataset-123")


def test_opik_dataset_service_connects_registered_dataset():
    client = FakeOpikClient()

    binding = OpikDatasetService(client=client, dataset_name="lyp-dataset-edu").connect()

    assert binding == OpikDatasetBinding(
        name="lyp-dataset-edu",
        connected=True,
        dataset_id="dataset-123",
    )
    assert client.requested_name == "lyp-dataset-edu"


def test_opik_dataset_service_skips_when_dataset_name_missing():
    client = FakeOpikClient()

    binding = OpikDatasetService(client=client, dataset_name=None).connect()

    assert binding is None
    assert client.requested_name is None
