"""Tests for engram.bulk_import — file ingestion and chat normalization."""

import json
import tempfile
from pathlib import Path

from epimneme.bulk_import import (
    ImportChunk,
    import_project_files,
    import_chat_file,
    import_chat_directory,
    normalize_chat,
    _chunk_text,
    _detect_room,
    scan_project_files,
)


class TestChunkText:
    def test_basic_chunking(self):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = _chunk_text(text, source="test.py")
        assert len(chunks) >= 1
        assert all(isinstance(c, ImportChunk) for c in chunks)
        assert all(c.source_file == "test.py" for c in chunks)

    def test_empty_text(self):
        chunks = _chunk_text("", source="test.py")
        assert chunks == []

    def test_whitespace_only(self):
        chunks = _chunk_text("   \n\n  ", source="test.py")
        assert chunks == []

    def test_long_text_splits(self):
        # Create text longer than CHUNK_SIZE (800)
        text = "\n\n".join(f"Paragraph {i} with enough content to matter." for i in range(50))
        chunks = _chunk_text(text, source="big.py")
        assert len(chunks) > 1
        # All chunks should be non-empty
        assert all(c.content.strip() for c in chunks)

    def test_tags_propagated(self):
        text = "Hello world content with enough text to pass the minimum size filter. " * 5
        chunks = _chunk_text(text, source="test.py", tags=["python", "source"])
        assert chunks
        assert "python" in chunks[0].tags
        assert "source" in chunks[0].tags


class TestDetectRoom:
    def test_src_directory(self):
        room = _detect_room("project/src/utils/helper.py")
        assert room  # should detect something

    def test_test_directory(self):
        room = _detect_room("project/tests/test_main.py")
        assert "test" in room.lower()

    def test_docs_directory(self):
        room = _detect_room("project/docs/guide.md")
        assert "doc" in room.lower()


class TestNormalizeChat:
    def test_claude_json(self):
        data = [
            {"role": "human", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        exchanges = normalize_chat(json.dumps(data))
        assert len(exchanges) == 2
        assert exchanges[0] == ("human", "Hello")
        assert exchanges[1] == ("assistant", "Hi there!")

    def test_chatgpt_json(self):
        data = [
            {
                "mapping": {
                    "a": {"message": {"author": {"role": "user"}, "content": {"parts": ["Hello"]}}},
                    "b": {"message": {"author": {"role": "assistant"}, "content": {"parts": ["Hi!"]}}},
                }
            }
        ]
        exchanges = normalize_chat(json.dumps(data))
        assert len(exchanges) >= 1

    def test_plain_text(self):
        text = "> What is Python?\n\nPython is a programming language."
        exchanges = normalize_chat(text)
        assert len(exchanges) >= 1

    def test_empty_input(self):
        exchanges = normalize_chat("")
        assert exchanges == [] or len(exchanges) == 0


class TestImportProjectFiles:
    def test_import_python_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files with enough content to pass chunk minimum
            py_content = "def hello():\n    '''A greeting function.'''\n    print('hello world')\n    return True\n\ndef goodbye():\n    print('goodbye')\n"
            (Path(tmpdir) / "main.py").write_text(py_content * 3)
            (Path(tmpdir) / "utils.py").write_text(py_content * 2)
            (Path(tmpdir) / "README.md").write_text("# My Project\n\nA test project with enough content.\n\n" * 5)
            # Create an ignored file type
            (Path(tmpdir) / "image.png").write_bytes(b"\x89PNG")

            chunks, result = import_project_files(tmpdir, wing="test-project")

            assert result.files_processed >= 2
            assert len(chunks) >= 2
            assert all(isinstance(c, ImportChunk) for c in chunks)

    def test_skips_node_modules(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nm = Path(tmpdir) / "node_modules" / "pkg"
            nm.mkdir(parents=True)
            (nm / "index.js").write_text("module.exports = { value: true };\n" * 5)
            (Path(tmpdir) / "app.js").write_text("const x = require('./module');\nconsole.log(x);\n" * 5)

            chunks, result = import_project_files(tmpdir)
            # node_modules should be skipped
            sources = [c.source_file for c in chunks]
            assert not any("node_modules" in s for s in sources)

    def test_limit_parameter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            content = "x = 42\nprint(x)\n# some content\n" * 5
            for i in range(10):
                (Path(tmpdir) / f"file{i}.py").write_text(content)

            chunks, result = import_project_files(tmpdir, limit=3)
            assert result.files_processed <= 3

    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chunks, result = import_project_files(tmpdir)
            assert result.files_processed == 0
            assert len(chunks) == 0


class TestImportChatFile:
    def test_import_claude_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data = [
                {"role": "human", "content": "Can you explain what Python is and why it is so popular in programming?"},
                {"role": "assistant", "content": "Python is a high-level, interpreted programming language known for its readability and versatility."},
            ]
            fpath = Path(tmpdir) / "chat.json"
            fpath.write_text(json.dumps(data))

            chunks, result = import_chat_file(str(fpath), wing="test")
            assert result.files_processed == 1
            assert len(chunks) >= 1

    def test_import_plain_text(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            text = "> How do I use git for version control in a software project?\n\nGit is a distributed version control system. Use git init to initialize a repository and git commit to save changes."
            fpath = Path(tmpdir) / "chat.txt"
            fpath.write_text(text)

            chunks, result = import_chat_file(str(fpath), wing="test")
            assert result.files_processed == 1


class TestImportChatDirectory:
    def test_import_multiple_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(3):
                data = [
                    {"role": "human", "content": f"Please explain concept number {i} in detail with examples"},
                    {"role": "assistant", "content": f"Here is a detailed answer about concept {i} with multiple examples and explanations"},
                ]
                (Path(tmpdir) / f"chat{i}.json").write_text(json.dumps(data))

            chunks, result = import_chat_directory(tmpdir, wing="test")
            assert result.files_processed == 3
            assert len(chunks) >= 3

    def test_ignores_non_chat_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "image.png").write_bytes(b"\x89PNG")
            (Path(tmpdir) / "data.csv").write_text("a,b,c")

            chunks, result = import_chat_directory(tmpdir, wing="test")
            assert result.files_processed == 0


class TestScanProjectFiles:
    def test_scan_returns_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.py").write_text("print('hi')\n# a Python script\n" * 3)
            files = list(scan_project_files(tmpdir))
            assert len(files) == 1
            path, content = files[0]
            assert "print" in content
