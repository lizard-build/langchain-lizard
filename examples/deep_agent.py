"""Run a Deep Agent with shell and file tools in a temporary sandbox."""

import os

from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI
from lizard import Sandbox

from langchain_lizard import LizardSandbox


def main() -> None:
    with Sandbox.create(
        "interpreter",
        project_id=os.environ["LIZARD_PROJECT_ID"],
        timeout_ms=300_000,
        size="small",
    ) as sandbox:
        agent = create_deep_agent(
            model=ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini")),
            backend=LizardSandbox(sandbox),
        )
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Run Python to print the first ten Fibonacci numbers.",
                    }
                ]
            }
        )
        print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
