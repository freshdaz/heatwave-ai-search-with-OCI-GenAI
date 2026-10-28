"""Populate the HeatWave demo table with OCI summaries, keywords, and embeddings."""

import argparse
import json
import re
from html.parser import HTMLParser

import mysql.connector
from mysql.connector.abstracts import MySQLConnectionAbstract
from oci.generative_ai_inference.models import (
    ChatDetails,
    CohereChatRequest,
    CohereResponseJsonFormat,
    OnDemandServingMode,
)

from config import TABLE_NAME, Settings
from oci_embedding_helper import OCIEmbeddingHelper

SOURCE_POSTS_SQL = """
    SELECT ID, post_title, post_excerpt, post_content, guid
    FROM wp_posts
    WHERE post_status = 'publish' AND post_type = 'post'
"""

INSERT_EMBEDDING_SQL = f"""
    INSERT INTO {TABLE_NAME} (
        ID, post_title, post_excerpt, guid,
        post_title_embedding, post_excerpt_embedding,
        post_summary, post_summary_embedding, post_keywords
    ) VALUES (
        %s, %s, %s, %s,
        STRING_TO_VECTOR(%s), STRING_TO_VECTOR(%s),
        %s, STRING_TO_VECTOR(%s), %s
    )
    ON DUPLICATE KEY UPDATE
        post_title = VALUES(post_title),
        post_excerpt = VALUES(post_excerpt),
        guid = VALUES(guid),
        post_title_embedding = VALUES(post_title_embedding),
        post_excerpt_embedding = VALUES(post_excerpt_embedding),
        post_summary = VALUES(post_summary),
        post_summary_embedding = VALUES(post_summary_embedding),
        post_keywords = VALUES(post_keywords)
"""


class PostTextExtractor(HTMLParser):
    """Extract visible text while excluding non-content HTML elements."""

    ignored_tags = {"script", "style", "noscript", "template"}

    def __init__(self) -> None:
        super().__init__()
        self.ignored_depth = 0
        self.parts = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.ignored_tags:
            self.ignored_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self.ignored_tags and self.ignored_depth:
            self.ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self.ignored_depth:
            self.parts.append(data)


def sanitize_post_content(content: str | None) -> str:
    """Remove HTML and normalize whitespace before sending content to the LLM."""
    parser = PostTextExtractor()
    parser.feed(content or "")
    parser.close()
    return re.sub(r"\s+", " ", " ".join(parser.parts)).strip()


def generate_summary_and_keywords(
    helper: OCIEmbeddingHelper, settings: Settings, content: str, title: str
) -> tuple[str, str]:
    """Generate a summary and a JSON-serializable keyword list for one post."""
    request = CohereChatRequest(
        message=(
            "Summarize the following sanitized blog post. Return a concise, factual "
            "summary and up to five distinct, relevant keywords.\n\n"
            f"Post:\n{content}"
        ),
        response_format=CohereResponseJsonFormat(
            schema={
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "keywords": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["summary", "keywords"],
                "additionalProperties": False,
            }
        ),
        max_tokens=400,
        temperature=0,
    )
    response = helper.client.chat(
        ChatDetails(
            serving_mode=OnDemandServingMode(model_id=settings.summary_model_id),
            compartment_id=helper.compartment_id,
            chat_request=request,
        )
    )
    payload = json.loads(response.data.chat_response.text)
    if not isinstance(payload, dict):
        raise ValueError("The summary model must return a JSON object.")
    summary = payload.get("summary")
    summary = summary.strip() if isinstance(summary, str) else ""
    summary = summary or title
    keywords = payload.get("keywords", [])
    if not isinstance(keywords, list):
        keywords = []
    keywords = [item.strip() for item in keywords if isinstance(item, str) and item.strip()][:5]
    return summary, json.dumps(keywords)


def mysql_connection(settings: Settings) -> MySQLConnectionAbstract:
    """Open a verified TLS connection for ingestion."""
    return mysql.connector.connect(
        host=settings.mysql_host,
        port=settings.mysql_port,
        user=settings.mysql_user,
        password=settings.mysql_password,
        database=settings.mysql_database,
        ssl_ca=settings.mysql_ssl_ca,
        ssl_verify_cert=True,
        ssl_verify_identity=True,
        connection_timeout=10,
    )


def populate_embeddings(settings: Settings, commit_every: int) -> None:
    """Generate and upsert embeddings for all published WordPress posts."""
    if commit_every < 1:
        raise ValueError("commit_every must be at least 1.")
    helper = OCIEmbeddingHelper(
        compartment_id=settings.oci_compartment_id,
        endpoint=settings.oci_genai_endpoint,
        config_file=settings.oci_config_file,
        profile=settings.oci_config_profile,
    )
    count = 0
    with (
        mysql_connection(settings) as connection,
        connection.cursor(dictionary=True, buffered=True) as reader,
        connection.cursor() as writer,
    ):
        reader.execute(SOURCE_POSTS_SQL)
        for count, row in enumerate(reader, start=1):
            title = (row["post_title"] or "").strip() or "Untitled post"
            excerpt = (row["post_excerpt"] or "").strip() or title
            content = sanitize_post_content(row["post_content"]) or excerpt

            summary, keywords = generate_summary_and_keywords(helper, settings, content, title)
            _, title_vector = helper.embed(title, settings.embedding_model_id, "SEARCH_DOCUMENT")
            _, excerpt_vector = helper.embed(
                excerpt, settings.embedding_model_id, "SEARCH_DOCUMENT"
            )
            _, summary_vector = helper.embed(
                summary, settings.embedding_model_id, "SEARCH_DOCUMENT"
            )
            writer.execute(
                INSERT_EMBEDDING_SQL,
                (
                    row["ID"],
                    title,
                    excerpt,
                    row["guid"],
                    title_vector,
                    excerpt_vector,
                    summary,
                    summary_vector,
                    keywords,
                ),
            )

            if count % commit_every == 0:
                connection.commit()
                print(f"{count} rows processed")

        connection.commit()
        print(f"{count} rows processed")


def parse_arguments() -> argparse.Namespace:
    """Parse ingestion options and require a positive transaction batch size."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--commit-every", type=int, default=100, help="Commit every N posts (default: 100)."
    )
    arguments = parser.parse_args()
    if arguments.commit_every < 1:
        parser.error("--commit-every must be at least 1")
    return arguments


if __name__ == "__main__":
    arguments = parse_arguments()
    populate_embeddings(Settings.from_environment(), arguments.commit_every)
