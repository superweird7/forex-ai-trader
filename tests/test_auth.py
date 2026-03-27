def test_index_redirects_to_login(client):
    resp = client.get("/", follow_redirects=False)
    assert resp.status_code == 302
    assert "/login" in resp.headers["Location"]

def test_login_page_loads(client):
    resp = client.get("/login")
    assert resp.status_code == 200
    assert b"Sign In" in resp.data

def test_login_success(client):
    resp = client.post("/login", data={"username": "testuser", "password": "testpass"}, follow_redirects=False)
    assert resp.status_code == 302

def test_login_failure(client):
    resp = client.post("/login", data={"username": "wrong", "password": "wrong"})
    assert resp.status_code == 200
    assert b"Invalid" in resp.data

def test_api_key_auth(client):
    resp = client.get("/api/status", headers={"X-API-Key": "test-api-key"})
    assert resp.status_code == 200

def test_api_no_auth_401(client):
    resp = client.get("/api/status")
    assert resp.status_code == 401

def test_api_wrong_key_401(client):
    resp = client.get("/api/status", headers={"X-API-Key": "wrong-key"})
    assert resp.status_code == 401

def test_authenticated_access(authed_client):
    resp = authed_client.get("/")
    assert resp.status_code == 200

def test_logout(authed_client):
    resp = authed_client.get("/logout", follow_redirects=False)
    assert resp.status_code == 302
    # After logout, should be redirected to login
    resp2 = authed_client.get("/", follow_redirects=False)
    assert resp2.status_code == 302
