# langchain-lizard

Run [LangChain Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview)
commands and file tools in [Lizard (lizard.build)](https://lizard.build/sandboxes).
`LizardSandbox` wraps a sandbox from the Lizard SDK and implements Deep Agents'
`BaseSandbox` interface.

## Install

Requires Python 3.11 or later.

```bash
pip install langchain-lizard
```

Create a project at [lizard.build](https://lizard.build), then create a key scoped
to that project:

```bash
lizard keys create langchain --project <project-id>
export LIZARD_API_KEY="your-key"
export LIZARD_PROJECT_ID="your-project-id"
```

The SDK also accepts `LIZARD_API_URL` for a custom API endpoint. Keep API keys out
of source control.

## Run a command

```python
import os

from lizard import Sandbox
from langchain_lizard import LizardSandbox

with Sandbox.create(
    "interpreter",
    project_id=os.environ["LIZARD_PROJECT_ID"],
    timeout_ms=300_000,
    size="small",
) as sandbox:
    backend = LizardSandbox(sandbox)
    result = backend.execute("python -c 'print(2 ** 10)'")
    print(result.output)
    assert result.exit_code == 0

    backend.upload_files([("/tmp/input.txt", b"hello")])
    downloaded = backend.download_files(["/tmp/input.txt"])
    assert downloaded[0].content == b"hello"
```

The SDK context manager ends the sandbox even if the code raises an error.
The adapter does not own or end the sandbox. You can also pass a sandbox returned
by `Sandbox.connect()` and manage its lifetime yourself.

## Use with Deep Agents

Install a model integration and set its credentials. For example, with OpenAI:

```bash
pip install langchain-openai
export OPENAI_API_KEY="your-model-key"
python examples/deep_agent.py
```

```python
import os

from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI
from lizard import Sandbox
from langchain_lizard import LizardSandbox

with Sandbox.create(
    "interpreter", project_id=os.environ["LIZARD_PROJECT_ID"], timeout_ms=300_000
) as sandbox:
    agent = create_deep_agent(
        model=ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini")),
        backend=LizardSandbox(sandbox),
    )
    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": "Run Python to print the first ten Fibonacci numbers.",
        }]
    })
    print(result["messages"][-1].content)
```

Set `OPENAI_BASE_URL` and `OPENAI_MODEL` to use an OpenAI-compatible endpoint.
Only the agent's shell and file tools run inside the sandbox; the model request
runs in your Python process.

## Behavior and limits

- `execute()` keeps the command's exit code. It joins stdout and stderr into the
  `ExecuteResponse.output` field; it does not preserve their interleaving.
- Command timeouts use seconds: 1–600, with a 60-second default. Pass
  `LizardSandbox(sandbox, timeout=120)` or `execute(command, timeout=120)`.
- The SDK caps each output stream at 512 KiB. The adapter preserves its
  `truncated` flag. Write large output to a file and use `download_files()`.
- File transfers accept absolute paths and preserve bytes and input order.
  Known file errors appear on the matching response. Authentication failures,
  expired sandboxes and server errors reach the caller as exceptions.
- Async methods use worker threads for the synchronous SDK. Cancelling the
  Python task does not stop a command that has already reached the sandbox.
- BaseSandbox supplies read, write, edit, ls, glob and grep through Python 3.
  Use a template that includes Python 3; the example uses `interpreter`.
- The examples set a five-minute sandbox lifetime. Commands do not extend it.
  Use the SDK to set a longer lifetime when needed.

## Development

```bash
python -m pip install -e '.[dev]'
ruff check .
mypy langchain_lizard
pytest tests/unit_tests --disable-socket --allow-unix-socket
python -m build
```

The standard tests create a temporary sandbox and end it after the suite. They
use your Lizard account and may incur usage charges:

```bash
LIZARD_LIVE_TESTS=1 pytest tests/integration_tests
```

The suite pins `langchain-tests==1.1.9`. Seven inherited tests still expect older
Deep Agents behavior: relative glob paths, directories in glob results, and an
error when writing an existing file. Deep Agents 0.7.24 uses absolute paths,
returns files from glob, and replaces existing files on write. Those seven cases
carry strict `xfail` marks, and extra live tests check the current behavior.
An unexpected pass fails the suite so that we review these marks on upgrades.

See [CONTRIBUTING.md](CONTRIBUTING.md) for release steps.

## License

Apache-2.0. Lizard's hosted service has its own terms.
