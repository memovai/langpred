"""Shared pytest fixtures — TestClient against the FastAPI app, fresh store."""
from __future__ import annotations

import os
import sys

# Make the in-tree packages importable without installation.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "server"))
sys.path.insert(0, os.path.join(ROOT, "sdk-python"))

import pytest


@pytest.fixture(autouse=True)
def fresh_store_and_service():
    from langpred_server import db, predict

    db.reset_store_for_tests()
    predict.reset_service_for_tests()
    yield


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from langpred_server.main import app

    with TestClient(app) as c:
        yield c
