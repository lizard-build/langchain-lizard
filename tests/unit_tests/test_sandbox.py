"""Test conversions, file errors and async calls without network access."""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from lizard import Sandbox
from lizard.errors import AuthenticationError, LizardError, NotFoundError

from langchain_lizard import LizardSandbox


@pytest.fixture
def native() -> Mock:
    sandbox = Mock(spec=Sandbox)
    sandbox.sandbox_id = "test-sandbox"
    sandbox.process = Mock()
    sandbox.fs = Mock()
    sandbox.process.exec_.return_value = SimpleNamespace(
        stdout="out", stderr="err", exit_code=7, truncated=True
    )
    return sandbox


def test_execute_preserves_results(native: Mock) -> None:
    backend = LizardSandbox(native, timeout=25)
    result = backend.execute("echo out; echo err >&2; exit 7")
    assert backend.id == "test-sandbox"
    assert (result.output, result.exit_code, result.truncated) == ("out\nerr", 7, True)
    native.process.exec_.assert_called_once_with(
        "echo out; echo err >&2; exit 7", timeout_ms=25_000
    )
    native.kill.assert_not_called()


@pytest.mark.parametrize("timeout", [0, -1, 601, True, 1.5])
def test_bad_timeout_never_reaches_sdk(native: Mock, timeout: int) -> None:
    with pytest.raises(ValueError, match="1 through 600"):
        LizardSandbox(native).execute("echo ok", timeout=timeout)
    native.process.exec_.assert_not_called()


def test_timeout_override(native: Mock) -> None:
    LizardSandbox(native, timeout=25).execute("sleep 1", timeout=3)
    native.process.exec_.assert_called_once_with("sleep 1", timeout_ms=3000)


def test_exact_bytes_and_order(native: Mock) -> None:
    backend = LizardSandbox(native)
    files = [("/tmp/empty", b""), ("/tmp/binary", b"\x00\xff\n")]
    assert [r.error for r in backend.upload_files(files)] == [None, None]
    assert [c.args for c in native.fs.write.call_args_list] == files
    native.fs.read_bytes.side_effect = [b"", b"\x00\xff\n"]
    assert [r.content for r in backend.download_files([p for p, _ in files])] == [
        b"",
        b"\x00\xff\n",
    ]


@pytest.mark.parametrize("path", ["relative", "", "/tmp/\x00file"])
def test_bad_paths_are_per_file_errors(native: Mock, path: str) -> None:
    backend = LizardSandbox(native)
    assert backend.upload_files([(path, b"x")])[0].error == "invalid_path"
    result = backend.download_files([path])[0]
    assert result.error == "invalid_path"
    assert result.content is None
    native.fs.write.assert_not_called()
    native.fs.read_bytes.assert_not_called()


def test_missing_file_checks_sandbox_before_mapping_error(native: Mock) -> None:
    native.fs.read_bytes.side_effect = NotFoundError("missing file")
    result = LizardSandbox(native).download_files(["/tmp/missing"])[0]
    assert result.error == "file_not_found"
    native.get_info.assert_called_once()


def test_expired_sandbox_is_not_a_missing_file(native: Mock) -> None:
    native.fs.read_bytes.side_effect = NotFoundError("gone")
    native.get_info.side_effect = NotFoundError("sandbox expired")
    with pytest.raises(NotFoundError, match="sandbox expired"):
        LizardSandbox(native).download_files(["/tmp/file"])


def test_directory_error_uses_stat(native: Mock) -> None:
    error = LizardError("directory read rejected")
    error.status_code = 400
    native.fs.read_bytes.side_effect = error
    native.fs.stat.return_value = SimpleNamespace(type="dir")
    assert LizardSandbox(native).download_files(["/tmp/dir"])[0].error == "is_directory"
    native.fs.stat.assert_called_once_with("/tmp/dir")


def test_unrelated_400_is_not_a_file_error(native: Mock) -> None:
    error = LizardError("bad request")
    error.status_code = 400
    native.fs.read_bytes.side_effect = error
    native.fs.stat.return_value = SimpleNamespace(type="file")
    with pytest.raises(LizardError, match="bad request"):
        LizardSandbox(native).download_files(["/tmp/file"])


@pytest.mark.parametrize("error", [AuthenticationError("denied"), LizardError("server")])
def test_api_failures_reach_caller(native: Mock, error: LizardError) -> None:
    native.fs.write.side_effect = error
    with pytest.raises(type(error)):
        LizardSandbox(native).upload_files([("/tmp/file", b"x")])


@pytest.mark.parametrize(
    "code,expected",
    [
        ("EISDIR", "is_directory"),
        ("EACCES", "permission_denied"),
        ("ENOENT", "file_not_found"),
        ("EINVAL", "invalid_path"),
    ],
)
def test_known_file_errors(native: Mock, code: str, expected: str) -> None:
    error = LizardError("file operation failed")
    error.code = code
    native.fs.read_bytes.side_effect = error
    assert LizardSandbox(native).download_files(["/tmp/file"])[0].error == expected


async def test_async_uses_worker_thread(native: Mock) -> None:
    main_thread = threading.get_ident()
    calls = []

    def execute(*args: object, **kwargs: object) -> SimpleNamespace:
        calls.append(threading.get_ident())
        return SimpleNamespace(stdout="ok", stderr="", exit_code=0, truncated=False)

    native.process.exec_.side_effect = execute
    backend = LizardSandbox(native)
    result = await backend.aexecute("echo ok", timeout=2)
    assert result.output == "ok"
    assert calls and calls[0] != main_thread
    native.fs.read_bytes.return_value = b"data"
    uploaded, downloaded = await asyncio.gather(
        backend.aupload_files([("/tmp/file", b"data")]),
        backend.adownload_files(["/tmp/file"]),
    )
    assert uploaded[0].error is None
    assert downloaded[0].content == b"data"
