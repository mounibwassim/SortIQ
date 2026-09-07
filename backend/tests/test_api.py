import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import httpx
from main import app

@pytest.mark.anyio
async def test_health_check_endpoint():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

@pytest.mark.anyio
async def test_stats_endpoint():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
        response = await client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_scans" in data

@pytest.mark.anyio
async def test_history_endpoint():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
        response = await client.get("/history?limit=5")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
