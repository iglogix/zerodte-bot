"""Tests for web.log_streamer — log filtering and tailing utilities."""

import pytest

from web.log_streamer import LogFilter, tail_lines


@pytest.fixture
def f():
    return LogFilter()


class TestLogFilter:
    def test_filter_accepts_error(self, f):
        assert f.accept("[ERROR] Something went wrong") is True

    def test_filter_accepts_warning(self, f):
        assert f.accept("[WARNING] Connection lost") is True

    def test_filter_accepts_meaningful_info(self, f):
        assert f.accept("[INFO] Cycle complete — outcome=win") is True
        assert f.accept("[INFO] Trade filled for SPX") is True
        assert f.accept("[INFO] Bot initialized successfully") is True

    def test_filter_rejects_debug(self, f):
        assert f.accept("[DEBUG] Entering function xyz") is False

    def test_filter_rejects_routine_info(self, f):
        assert f.accept("[INFO] src.cache: Polling market data") is False

    def test_classify_level(self, f):
        assert f.classify("[ERROR] disk full") == "error"
        assert f.classify("[WARNING] slow response") == "warning"
        assert f.classify("[INFO] Cycle complete") == "info"


class TestTailLines:
    def test_tail_lines(self, tmp_path):
        p = tmp_path / "test.log"
        p.write_text("line1\nline2\nline3\nline4\nline5\n")
        result = tail_lines(str(p), n=3)
        assert result == ["line3", "line4", "line5"]

    def test_tail_lines_missing_file(self):
        result = tail_lines("/nonexistent/path/to/file.log")
        assert result == []
