import pytest
import os
import sys
import tempfile

# Add dashboard to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

@pytest.fixture
def app_env():
    """Set up environment variables for testing."""
    os.environ["FLASK_SECRET_KEY"] = "test-secret"
    os.environ["DASHBOARD_API_KEY"] = "test-api-key"
    os.environ["DASHBOARD_USER"] = "testuser"
    os.environ["DASHBOARD_PASS"] = "testpass"
    os.environ["TELEGRAM_BOT_TOKEN"] = ""
    os.environ["TELEGRAM_CHAT_ID"] = ""
    yield
    # Cleanup handled by test teardown

@pytest.fixture
def test_app(app_env):
    """Create test app with temp database."""
    import app as dashboard_app
    dashboard_app.app.config["TESTING"] = True

    # Use temp DB
    original_db = dashboard_app.DB_PATH
    tmp_db = tempfile.mktemp(suffix=".db")
    dashboard_app.DB_PATH = tmp_db
    dashboard_app.init_db()

    yield dashboard_app

    # Restore and cleanup
    dashboard_app.DB_PATH = original_db
    try:
        os.unlink(tmp_db)
    except Exception:
        pass

@pytest.fixture
def client(test_app):
    """Unauthenticated test client."""
    return test_app.app.test_client()

@pytest.fixture
def authed_client(test_app):
    """Authenticated test client (logged in)."""
    c = test_app.app.test_client()
    c.post("/login", data={"username": "testuser", "password": "testpass"})
    return c

@pytest.fixture
def sample_signal():
    """Sample signal data for testing."""
    return {
        "time": "2026-03-27 12:00:00",
        "signal": "BUY",
        "confidence": 95.0,
        "price": 4430.0,
        "bid": 4429.7,
        "score": 10,
        "buy_prob": 95.0,
        "sell_prob": 2.0,
        "no_trade_prob": 3.0,
        "rsi": 62.0,
        "stoch_k": 88.0,
        "atr": 24.0,
        "atr_vs_avg": 0.94,
        "adx": 0,
        "bb_position": 0.98,
        "ema_trend": "BEARISH",
        "macd": -5.0,
        "body_ratio": 0.05,
        "is_bullish": False,
        "reasons": ["Test reason 1", "Test reason 2"],
    }
