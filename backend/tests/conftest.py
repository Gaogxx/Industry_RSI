import os
import tempfile

import pytest

_tmp = tempfile.mkdtemp(prefix="kaizen-test-")
os.environ.update({
    "KAIZEN_DATABASE_URL": f"sqlite:///{_tmp}/test.db",
    "KAIZEN_DBOS_DATABASE_URL": f"sqlite:///{_tmp}/dbos.sqlite",
    "KAIZEN_NIGHTLY_CRON": "",
    "KAIZEN_LLM_API_KEY": "",
    "KAIZEN_API_KEY": "",
    "KAIZEN_SEED_DEMO": "true",
    "KAIZEN_SEED_BACKGROUND": "false",
    "KAIZEN_EMBEDDING_MODEL": "",
    "PYDANTIC_AI_NO_BANNER": "1",
})


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture
def settings():
    from app.config import get_settings

    s = get_settings()
    snapshot = s.model_dump()
    yield s
    for k, v in snapshot.items():
        setattr(s, k, v)


def open_case(client, symptom: str) -> dict:
    return next(c for c in client.get("/api/cases", params={"status": "open"}).json() if c["symptom"] == symptom)
