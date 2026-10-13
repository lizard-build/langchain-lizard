"""Run Deep Agents commands and file tools in Lizard Sandboxes."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from functools import partial
from typing import TypeVar

from deepagents.backends.protocol import ExecuteResponse, FileDownloadResponse, FileUploadResponse
from deepagents.backends.sandbox import BaseSandbox
from lizard import Sandbox
from lizard.errors import AuthenticationError, LizardError, NotFoundError

_T = TypeVar("_T")


def _valid_path(path: str) -> bool:
    return bool(path) and path.startswith("/") and "\x00" not in path


def _file_error(error: LizardError) -> str:
    # Account/auth failures must reach the caller; an agent cannot repair them.
    if isinstance(error, AuthenticationError):
        raise error
    if isinstance(error, NotFoundError):
        return "file_not_found"
    code = (error.code or "").upper()
    if code in {"EISDIR", "IS_DIRECTORY"}:
        return "is_directory"
    if code in {"ENOENT", "FILE_NOT_FOUND", "ENOTDIR"}:
        return "file_not_found"
    if code in {"EACCES", "EPERM", "PERMISSION_DENIED"}:
        return "permission_denied"
    if code in {"EINVAL", "INVALID_PATH"}:
        return "invalid_path"
    raise error


class LizardSandbox(BaseSandbox):
    """Wrap an existing Lizard SDK sandbox for Deep Agents.

    The caller owns the sandbox's lifetime. Use the SDK's context manager or
    ``try/finally`` to kill it after the agent finishes. Async methods run the
    synchronous SDK in worker threads so they do not block the event loop.

    Args:
        sandbox: A sandbox created or connected through Lizard SDK.
        timeout: Default command timeout in seconds, from 1 through 600.
    """

    def __init__(self, sandbox: Sandbox, *, timeout: int = 60) -> None:
        self._timeout_ms(timeout)
        self._sandbox = sandbox
        self._timeout = timeout

    @property
    def id(self) -> str:
        """The ID of the wrapped sandbox."""
        return str(self._sandbox.sandbox_id)

    @staticmethod
    def _timeout_ms(timeout: int) -> int:
        if isinstance(timeout, bool) or not isinstance(timeout, int) or not 1 <= timeout <= 600:
            raise ValueError("Command timeout must be an integer from 1 through 600 seconds.")
        return timeout * 1000

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        """Execute a shell command, preserving its exit code and truncation flag."""
        result = self._sandbox.process.exec_(
            command,
            timeout_ms=self._timeout_ms(self._timeout if timeout is None else timeout),
        )
        output = result.stdout
        if result.stderr:
            output += ("\n" if output and not output.endswith("\n") else "") + result.stderr
        return ExecuteResponse(
            output=output,
            exit_code=result.exit_code,
            truncated=result.truncated,
        )

    async def aexecute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        """Execute without blocking the event loop."""
        return await asyncio.to_thread(self.execute, command, timeout=timeout)

    def _transfer(self, path: str, operation: Callable[[], _T]) -> tuple[_T | None, str | None]:
        if not _valid_path(path):
            return None, "invalid_path"
        try:
            return operation(), None
        except LizardError as error:
            # The file API reports directory reads as HTTP 400 without a code.
            # Check the path's type instead of parsing a human error message.
            if error.status_code == 400 and not error.code:
                try:
                    info = self._sandbox.fs.stat(path)
                except LizardError:
                    raise error from None
                if info.type == "dir":
                    return None, "is_directory"
            # A 404 can mean the sandbox expired, rather than a missing file.
            # Verify the sandbox before turning it into a per-file result.
            if isinstance(error, NotFoundError):
                self._sandbox.get_info()
            return None, _file_error(error)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload bytes in order; report file errors without hiding API failures."""
        responses = []
        for path, content in files:
            _, error = self._transfer(path, partial(self._sandbox.fs.write, path, content))
            responses.append(FileUploadResponse(path=path, error=error))
        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download exact bytes, including empty and non-UTF-8 files."""
        responses = []
        for path in paths:
            content, error = self._transfer(path, partial(self._sandbox.fs.read_bytes, path))
            responses.append(FileDownloadResponse(path=path, content=content, error=error))
        return responses

    async def aupload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload without blocking the event loop."""
        return await asyncio.to_thread(self.upload_files, files)

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download without blocking the event loop."""
        return await asyncio.to_thread(self.download_files, paths)
