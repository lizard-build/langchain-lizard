"""Run LangChain's standard suite against a temporary Lizard sandbox."""

import os
from collections.abc import Iterator

import pytest
from deepagents import create_deep_agent
from deepagents.backends.protocol import SandboxBackendProtocol
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_tests.integration_tests import SandboxIntegrationTests
from lizard import Sandbox

from langchain_lizard import LizardSandbox

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("LIZARD_LIVE_TESTS") != "1",
        reason="Set LIZARD_LIVE_TESTS=1 to run tests that create a sandbox.",
    ),
]

_legacy_glob = pytest.mark.xfail(
    strict=True,
    reason=(
        "langchain-tests 1.1.9 expects relative glob paths; "
        "Deep Agents 0.7.24 returns absolute paths."
    ),
)


class TestLizardSandbox(SandboxIntegrationTests):
    # Keep these upstream cases visible. Tests below check the current contract.
    test_glob = _legacy_glob(SandboxIntegrationTests.test_glob)
    test_glob_basic_pattern = _legacy_glob(SandboxIntegrationTests.test_glob_basic_pattern)
    test_glob_hidden_files_explicitly = _legacy_glob(
        SandboxIntegrationTests.test_glob_hidden_files_explicitly
    )
    test_glob_with_character_class = _legacy_glob(
        SandboxIntegrationTests.test_glob_with_character_class
    )
    test_glob_with_question_mark = _legacy_glob(
        SandboxIntegrationTests.test_glob_with_question_mark
    )
    test_glob_with_directories = pytest.mark.xfail(
        strict=True,
        reason=(
            "Deep Agents 0.7.24 glob returns files only; the standard suite expects directories."
        ),
    )(SandboxIntegrationTests.test_glob_with_directories)
    test_write_existing_file_fails = pytest.mark.xfail(
        strict=True,
        reason=(
            "Deep Agents 0.7.24 write replaces existing files; the standard suite expects an error."
        ),
    )(SandboxIntegrationTests.test_write_existing_file_fails)

    @pytest.fixture(scope="class")
    def sandbox(self) -> Iterator[SandboxBackendProtocol]:
        with Sandbox.create(
            "interpreter",
            project_id=os.environ["LIZARD_PROJECT_ID"],
            timeout_ms=1_800_000,
            size="small",
            metadata={"purpose": "langchain-standard-tests"},
        ) as native:
            yield LizardSandbox(native)

    def test_current_glob_contract(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        sandbox_backend.execute(f"mkdir -p {sandbox_test_root}/directory")
        path = f"{sandbox_test_root}/file.txt"
        sandbox_backend.write(path, "data")
        result = sandbox_backend.glob("*", path=sandbox_test_root)
        assert result.error is None
        assert result.matches is not None
        assert [match["path"] for match in result.matches] == [path]

    def test_current_write_contract(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        path = f"{sandbox_test_root}/existing.txt"
        assert sandbox_backend.write(path, "before").error is None
        assert sandbox_backend.write(path, "after").error is None
        assert sandbox_backend.download_files([path])[0].content == b"after"

    def test_deep_agent_executes_tool(self, sandbox_backend: SandboxBackendProtocol) -> None:
        class ToolModel(FakeMessagesListChatModel):
            def bind_tools(self, tools, **kwargs):
                return self

        model = ToolModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "execute",
                            "args": {"command": "printf 42"},
                            "id": "execute-test",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="done"),
            ]
        )
        agent = create_deep_agent(model=model, backend=sandbox_backend)
        result = agent.invoke({"messages": [{"role": "user", "content": "Run the command."}]})
        outputs = [message for message in result["messages"] if isinstance(message, ToolMessage)]
        assert len(outputs) == 1
        assert "42" in outputs[0].content
