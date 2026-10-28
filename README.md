# HeatWave AI Search
Related article: [Build an AI powered search engine with MySQL HeatWave and OCI GenAI](https://dasini.net/blog/2026/10/27/build-an-ai-powered-search-engine-with-mysql-heatwave-and-oci-genai/)

A Python and Flask application for semantic, keyword, and hybrid search over
WordPress posts. OCI Generative AI generates embeddings, summaries, and keywords;
MySQL HeatWave stores vectors and performs vector, full-text, and JSON searches.

## Requirements

- Python 3.11 or newer and [uv](https://docs.astral.sh/uv/getting-started/installation/).
- A MySQL HeatWave service supporting `VECTOR(1024)`, `STRING_TO_VECTOR()`,
  `DISTANCE(..., 'COSINE')`, full-text indexes, and JSON multi-value indexes.
- An existing `wp_posts` table with published WordPress posts in your own database.
  No content or database exports are included in this repository.
- OCI SDK credentials configured outside the repository, with permission to use
  the chosen compartment, Generative AI endpoint, and models. See the
  [OCI SDK configuration guide](https://docs.oracle.com/en-us/iaas/tools/python/latest/configuration.html).
- A trusted CA certificate for the database server, stored outside the repository.
  Use a database hostname matching its TLS certificate.

## Installation

Download or clone this repository and run these commands from its root:

```bash
uv sync --locked --no-dev --python 3.12
cp .env.example .env
```

On Windows, copy `.env.example` to `.env` with `Copy-Item .env.example .env`.
Fill in the blank OCI and MySQL settings. `MYSQL_SSL_CA` points to the database
CA certificate. MySQL connections verify both the certificate and hostname.
Keep the OCI SDK configuration and private signing key outside this repository.

The application loads only the `.env` beside `config.py`; existing environment
variables take precedence. `.env.example` contains configuration names and
non-secret defaults only. Never commit a filled-in `.env` or credentials.

Dependencies are constrained in `pyproject.toml` and resolved in `uv.lock`.
This is a script application; no Python package build or installation is needed.

## Database setup

Connect to your database using a trusted MySQL client and run `schema.sql` in the
same database as `wp_posts`. The schema creates the destination table and the
full-text and JSON indexes used by keyword search. It contains no user data or
credentials. It does not alter an existing table; check its columns, vector
dimensions, and indexes if you already have one.

The embedding model must match the model that generated the stored vectors,
even when two models produce vectors of the same size. The default is
`cohere.embed-multilingual-v3.0`. For `cohere.embed-v4.0`, the helper requests
1,024 dimensions. Other model IDs must support the request format and produce
1,024-dimensional embeddings, or require corresponding schema and SDK changes.
Select an available summary model compatible with the OCI Cohere chat API.
Rebuild all embeddings in a separate, empty table when changing models; avoid
serving search against a partially migrated collection.

## Quickstart

Generate summaries, keywords, and embeddings for your published posts:

```bash
uv run --locked --no-dev python generate_embeddings.py --commit-every 25
```

This command sends post text to OCI Generative AI and makes billable API calls.
It upserts rows by WordPress post ID. Completed batches remain committed if a
later batch fails. Ensure you have permission to process the content and use a
separate ingestion account with source `SELECT` and destination `INSERT`/`UPDATE`
permissions. Use a destination `SELECT` account for the web application.

Start the local application:

```bash
uv run --locked --no-dev python app.py
```

Open <http://127.0.0.1:5000>. To check OCI embedding access without writing to
MySQL, run:

```bash
uv run --locked --no-dev python runhelper.py "Generative artificial intelligence"
```

Search options:

- **Semantic:** cosine distance weighted by title (10%), excerpt (20%), and
  summary (70%); lower scores indicate closer matches.
- **Keyword:** full-text relevance (70%) and JSON keyword overlap (30%);
  use comma-separated keyword phrases such as `vector search, databases`.
- **Hybrid:** reciprocal-rank fusion of semantic and keyword rankings.

The model field allows a compatible OCI model ID for each query. It must match
the stored embeddings. Keyword search does not call OCI.

## Deployment and security

The bundled Flask server listens on loopback for local use. Keep
`FLASK_DEBUG=false`; enable debugging only on a trusted development machine.
Before exposing the application, use a production WSGI server behind HTTPS,
authentication, request throttling, and appropriate network access controls.
The demo has no user authentication or API quota enforcement; semantic and
hybrid requests can incur OCI charges. Restrict access to trusted users.

The web app limits request and input sizes, escapes rendered text, restricts
result links to HTTP(S), and returns generic backend errors. Logs record the
exception class without exception messages, queries, or credential values.
Browser protections follow the
[Flask security guidance](https://flask.palletsprojects.com/en/stable/web-security/).
Database TLS uses the connector's documented
[certificate and identity verification options](https://dev.mysql.com/doc/connector-python/en/connector-python-connectargs.html).

If a credential has ever been shared or committed, revoke and replace it.
Deleting a file does not remove it from existing Git history or remote copies.

## Development

```bash
uv sync --locked --python 3.12
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked python -m unittest discover -s tests -v
uv run --locked pip-audit
```

Tests mock OCI and MySQL and do not need credentials, network access, or raw data.
Live database SQL and OCI model access must be checked in your own environment.
GitHub Actions runs lint, formatting, tests, and dependency auditing on Python
3.11 and 3.12 for pushes and pull requests, with read-only repository permissions.
To refresh dependencies, run `uv lock --upgrade`, repeat these checks, and commit
the updated lockfile. Report vulnerabilities privately through the repository's
GitHub Security Advisories when enabled; omit credentials and private content.

## License

MIT. See [LICENSE](LICENSE). Dependencies retain their respective licenses.
