# Contributing

Use Python 3.11 or later and install `.[dev]`. Run the checks in the README before
opening a pull request. Unit tests need no credentials. Live tests need a Lizard
API key and a project reserved for tests.

## Publish

The release workflow uses PyPI Trusted Publishing. It does not need an API token.
For the first release, an account owner must add a pending publisher at
<https://pypi.org/manage/account/publishing/>:

| Field | Value |
| --- | --- |
| PyPI project name | `langchain-lizard` |
| GitHub owner | `lizard-build` |
| Repository | `langchain-lizard` |
| Workflow filename | `publish.yml` |
| Environment name | Leave blank |

Then run **Publish to PyPI** in GitHub Actions and enter the version from
`pyproject.toml`. The workflow runs tests, builds both distributions, checks their
metadata, and publishes only from `main`. PyPI does not let you replace a release
file, so bump the version before each later release.

After PyPI accepts the first release, submit the package through LangChain's
[integration form](https://github.com/langchain-ai/docs/issues/new?template=06-integration-submission.yml).
The package's release and its catalog listing are separate steps.
