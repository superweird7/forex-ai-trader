from unittest.mock import patch

def test_alert_skipped_when_disabled(test_app):
    test_app.CONFIG["alerts"]["enabled"] = False
    # Should return without error
    test_app.send_telegram_alert({"signal": "BUY", "confidence": 99})

def test_alert_skipped_below_threshold(test_app):
    test_app.CONFIG["alerts"]["enabled"] = True
    test_app.CONFIG["alerts"]["min_confidence"] = 90
    with patch("requests.post") as mock_post:
        test_app.send_telegram_alert({"signal": "BUY", "confidence": 80})
        mock_post.assert_not_called()

def test_alert_skipped_no_token(test_app):
    import os
    test_app.CONFIG["alerts"]["enabled"] = True
    test_app.CONFIG["alerts"]["min_confidence"] = 90
    os.environ["TELEGRAM_BOT_TOKEN"] = ""
    os.environ["TELEGRAM_CHAT_ID"] = ""
    with patch("requests.post") as mock_post:
        test_app.send_telegram_alert({"signal": "BUY", "confidence": 95})
        mock_post.assert_not_called()

def test_alert_sent_when_enabled(test_app):
    import os
    test_app.CONFIG["alerts"]["enabled"] = True
    test_app.CONFIG["alerts"]["min_confidence"] = 90
    os.environ["TELEGRAM_BOT_TOKEN"] = "fake-token"
    os.environ["TELEGRAM_CHAT_ID"] = "12345"
    with patch("requests.post") as mock_post:
        test_app.send_telegram_alert({
            "signal": "BUY", "confidence": 95, "price": 4430,
            "score": 10, "rsi": 62, "atr": 24, "time": "2026-03-27 12:00:00"
        })
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "fake-token" in call_args[0][0]
        assert "BUY" in call_args[1]["json"]["text"]
