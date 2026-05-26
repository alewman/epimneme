"""Tests for structured JSON logging formatter."""

from __future__ import annotations

import json
import logging


from epimneme.logging import JSONFormatter


class TestJSONFormatter:
    def setup_method(self):
        self.formatter = JSONFormatter()
        self.handler = logging.StreamHandler()
        self.handler.setFormatter(self.formatter)
        self.logger = logging.getLogger("test.json")
        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.DEBUG)

    def teardown_method(self):
        self.logger.removeHandler(self.handler)

    def test_basic_format(self):
        record = logging.LogRecord(
            name="test.json",
            level=logging.INFO,
            pathname="test_logging.py",
            lineno=42,
            msg="hello world",
            args=(),
            exc_info=None,
        )
        output = self.formatter.format(record)
        data = json.loads(output)
        assert data["level"] == "INFO"
        assert data["logger"] == "test.json"
        assert data["msg"] == "hello world"
        assert data["line"] == 42
        assert "ts" in data
        assert "exc" not in data

    def test_format_with_args(self):
        record = logging.LogRecord(
            name="test.json",
            level=logging.WARNING,
            pathname="test_logging.py",
            lineno=10,
            msg="count=%d",
            args=(5,),
            exc_info=None,
        )
        output = self.formatter.format(record)
        data = json.loads(output)
        assert data["msg"] == "count=5"
        assert data["level"] == "WARNING"

    def test_format_with_exception(self):
        try:
            raise ValueError("boom")
        except ValueError:
            import sys
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test.json",
            level=logging.ERROR,
            pathname="test_logging.py",
            lineno=99,
            msg="something failed",
            args=(),
            exc_info=exc_info,
        )
        output = self.formatter.format(record)
        data = json.loads(output)
        assert data["level"] == "ERROR"
        assert "exc" in data
        assert "ValueError: boom" in data["exc"]

    def test_output_is_single_line(self):
        record = logging.LogRecord(
            name="test.json",
            level=logging.DEBUG,
            pathname="test_logging.py",
            lineno=1,
            msg="multi\nline\nmessage",
            args=(),
            exc_info=None,
        )
        output = self.formatter.format(record)
        # json.dumps ensure_ascii=False, but newlines within strings are escaped
        assert "\n" not in output
        data = json.loads(output)
        assert "multi\nline\nmessage" == data["msg"]

    def test_all_log_levels(self):
        for level in (logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL):
            record = logging.LogRecord(
                name="test.json",
                level=level,
                pathname="test_logging.py",
                lineno=1,
                msg="test",
                args=(),
                exc_info=None,
            )
            output = self.formatter.format(record)
            data = json.loads(output)
            assert data["level"] == logging.getLevelName(level)
