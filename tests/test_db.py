def test_insert_and_query(test_app, sample_signal):
    test_app.db_insert_signal(sample_signal)
    results = test_app.db_get_signals(limit=10)
    assert len(results) == 1
    assert results[0]["signal"] == "BUY"
    assert results[0]["confidence"] == 95.0

def test_insert_multiple(test_app, sample_signal):
    test_app.db_insert_signal(sample_signal)
    sell_signal = sample_signal.copy()
    sell_signal["signal"] = "SELL"
    sell_signal["time"] = "2026-03-27 12:30:00"
    test_app.db_insert_signal(sell_signal)
    results = test_app.db_get_signals(limit=10)
    assert len(results) == 2

def test_filter_by_signal(test_app, sample_signal):
    test_app.db_insert_signal(sample_signal)
    sell_signal = sample_signal.copy()
    sell_signal["signal"] = "SELL"
    test_app.db_insert_signal(sell_signal)
    buys = test_app.db_get_signals(signal_filter="BUY")
    assert len(buys) == 1
    assert buys[0]["signal"] == "BUY"

def test_stats_empty(test_app):
    stats = test_app.db_get_stats()
    assert stats["total_all_time"] == 0

def test_stats_with_data(test_app, sample_signal):
    test_app.db_insert_signal(sample_signal)
    stats = test_app.db_get_stats()
    assert stats["total_all_time"] == 1

def test_reasons_stored_as_string(test_app, sample_signal):
    test_app.db_insert_signal(sample_signal)
    results = test_app.db_get_signals(limit=1)
    assert "Test reason 1" in results[0]["reasons"]
    assert "|" in results[0]["reasons"]
