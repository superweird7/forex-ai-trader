import json

def test_status_endpoint(authed_client):
    resp = authed_client.get("/api/status")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert "mt5_connected" in data
    assert "model_loaded" in data
    assert "daily_stats" in data

def test_signals_endpoint_empty(authed_client):
    resp = authed_client.get("/api/signals")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert "signals" in data
    assert "stats" in data

def test_signals_count(authed_client):
    resp = authed_client.get("/api/signals/count")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert "total_all_time" in data

def test_symbols_endpoint(authed_client):
    resp = authed_client.get("/api/symbols")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert isinstance(data, list)
    assert "XAUUSD.m" in data

def test_history_page_loads(authed_client):
    resp = authed_client.get("/history")
    assert resp.status_code == 200

def test_simulate_page_loads(authed_client):
    resp = authed_client.get("/simulate")
    assert resp.status_code == 200

def test_admin_page_loads(authed_client):
    resp = authed_client.get("/admin/models")
    assert resp.status_code == 200
