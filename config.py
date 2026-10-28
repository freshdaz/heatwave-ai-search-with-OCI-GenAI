"""Configuration shared by the demo application and ingestion script."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

TABLE_NAME = "`wp_posts_embeddings_cohere`"


def required_environment(name: str) -> str:
    """Return an environment variable or raise a clear setup error."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} must be set. Copy .env.example to .env and update it.")
    return value


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment variables or a local .env file."""

    oci_compartment_id: str
    oci_genai_endpoint: str
    oci_config_file: str
    oci_config_profile: str
    embedding_model_id: str
    summary_model_id: str
    mysql_host: str
    mysql_port: int
    mysql_user: str
    mysql_password: str = field(repr=False)
    mysql_database: str
    mysql_ssl_ca: str
    port: int
    debug: bool

    @classmethod
    def from_environment(cls) -> "Settings":
        load_dotenv(Path(__file__).with_name(".env"))
        try:
            mysql_port = int(os.getenv("MYSQL_PORT", "3306"))
            port = int(os.getenv("PORT", "5000"))
        except ValueError as exc:
            raise RuntimeError("MYSQL_PORT and PORT must be integers.") from exc
        if not 1 <= mysql_port <= 65535 or not 1 <= port <= 65535:
            raise RuntimeError("MYSQL_PORT and PORT must be between 1 and 65535.")

        return cls(
            oci_compartment_id=required_environment("OCI_COMPARTMENT_ID"),
            oci_genai_endpoint=required_environment("OCI_GENAI_ENDPOINT"),
            oci_config_file=os.getenv("OCI_CONFIG_FILE", "~/.oci/config"),
            oci_config_profile=os.getenv("OCI_CONFIG_PROFILE", "DEFAULT"),
            embedding_model_id=os.getenv(
                "OCI_EMBEDDING_MODEL_ID", "cohere.embed-multilingual-v3.0"
            ),
            summary_model_id=os.getenv("OCI_SUMMARY_MODEL_ID", "cohere.command-a-03-2025"),
            mysql_host=required_environment("MYSQL_HOST"),
            mysql_port=mysql_port,
            mysql_user=required_environment("MYSQL_USER"),
            mysql_password=required_environment("MYSQL_PASSWORD"),
            mysql_database=os.getenv("MYSQL_DATABASE", "WP_embeddings"),
            mysql_ssl_ca=str(Path(required_environment("MYSQL_SSL_CA")).expanduser()),
            port=port,
            debug=os.getenv("FLASK_DEBUG", "false").lower() in {"1", "true", "yes"},
        )
