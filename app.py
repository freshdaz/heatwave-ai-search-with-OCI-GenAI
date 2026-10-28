"""Flask demo for semantic, keyword, and hybrid HeatWave searches."""

import json
from typing import Any
from urllib.parse import urlsplit

import mysql.connector
from flask import Flask, Response, render_template, request
from mysql.connector.abstracts import MySQLConnectionAbstract

from config import TABLE_NAME, Settings
from oci_embedding_helper import OCIEmbeddingHelper

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024
app.config["MAX_FORM_MEMORY_SIZE"] = 16 * 1024
app.config["MAX_FORM_PARTS"] = 10


def embedding_helper(settings: Settings) -> OCIEmbeddingHelper:
    """Build an OCI client from the current demo settings."""
    return OCIEmbeddingHelper(
        compartment_id=settings.oci_compartment_id,
        endpoint=settings.oci_genai_endpoint,
        config_file=settings.oci_config_file,
        profile=settings.oci_config_profile,
    )


def mysql_connection(settings: Settings) -> MySQLConnectionAbstract:
    """Open a MySQL connection for one search operation."""
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


def semantic_search(
    settings: Settings, query: str, model_id: str, limit: int = 10
) -> list[dict[str, Any]]:
    """Search title, excerpt, and summary embeddings with weighted distance."""
    _, query_vector = embedding_helper(settings).embed(query, model_id, "SEARCH_QUERY")

    # STRING_TO_VECTOR and DISTANCE run in MySQL; OCI only generated query_vector.
    statement = f"""
        WITH query_vector AS (
            SELECT STRING_TO_VECTOR(%s) AS embedding
        ), distances AS (
            SELECT ID, post_title, post_excerpt, guid,
                   DISTANCE(query_vector.embedding, post_title_embedding, 'COSINE')
                       AS title_distance,
                   DISTANCE(query_vector.embedding, post_excerpt_embedding, 'COSINE')
                       AS excerpt_distance,
                   DISTANCE(query_vector.embedding, post_summary_embedding, 'COSINE')
                       AS summary_distance
            FROM {TABLE_NAME}
            CROSS JOIN query_vector
        )
        SELECT ID, post_title, post_excerpt, guid,
               (0.1 * title_distance + 0.2 * excerpt_distance + 0.7 * summary_distance)
                   AS combined_distance
        FROM distances
        ORDER BY combined_distance ASC, ID ASC
        LIMIT %s
    """
    with mysql_connection(settings) as connection, connection.cursor(dictionary=True) as cursor:
        cursor.execute(statement, (query_vector, limit))
        results = cursor.fetchall()
        for result in results:
            result["result_type"] = "semantic"
        return results


def keyword_terms(query: str) -> list[str]:
    """Convert comma-separated keyword phrases to a JSON-array search input."""
    terms = [term.strip() for term in query.split(",") if term.strip()]
    return terms or [query]


def keyword_search(settings: Settings, query: str, limit: int = 10) -> list[dict[str, Any]]:
    """Combine full-text and JSON multi-value-index keyword matches."""
    terms_json = json.dumps(keyword_terms(query))
    statement = f"""
        SELECT ID, post_title, post_excerpt, guid,
               MATCH(post_title, post_excerpt, post_summary)
                   AGAINST(%s IN NATURAL LANGUAGE MODE) AS fulltext_score,
               COALESCE(JSON_OVERLAPS(post_keywords, %s), 0) AS keywords_match,
               (
                   0.7 * MATCH(post_title, post_excerpt, post_summary)
                       AGAINST(%s IN NATURAL LANGUAGE MODE) +
                   0.3 * COALESCE(JSON_OVERLAPS(post_keywords, %s), 0)
               ) AS combined_score
        FROM {TABLE_NAME}
        WHERE MATCH(post_title, post_excerpt, post_summary)
                  AGAINST(%s IN NATURAL LANGUAGE MODE)
           OR JSON_OVERLAPS(post_keywords, %s)
        ORDER BY combined_score DESC, ID ASC
        LIMIT %s
    """
    with mysql_connection(settings) as connection, connection.cursor(dictionary=True) as cursor:
        cursor.execute(
            statement,
            (
                query,
                terms_json,
                query,
                terms_json,
                query,
                terms_json,
                limit,
            ),
        )
        results = cursor.fetchall()
        for result in results:
            result["result_type"] = "keyword"
        return results


def hybrid_search(
    settings: Settings, query: str, model_id: str, limit: int = 10
) -> list[dict[str, Any]]:
    """Merge semantic and keyword rankings with reciprocal-rank fusion."""
    semantic_results = semantic_search(settings, query, model_id, limit)
    keyword_results = keyword_search(settings, query, limit)
    merged = {}

    for rank, result in enumerate(semantic_results, start=1):
        item = merged.setdefault(result["ID"], {**result, "fusion_score": 0, "sources": []})
        item["fusion_score"] += 1 / (60 + rank)
        item["sources"].append("semantic")

    for rank, result in enumerate(keyword_results, start=1):
        item = merged.setdefault(result["ID"], {**result, "fusion_score": 0, "sources": []})
        item["fusion_score"] += 1 / (60 + rank)
        item["sources"].append("keyword")

    results = sorted(merged.values(), key=lambda result: (-result["fusion_score"], result["ID"]))[
        :limit
    ]
    for result in results:
        result["result_type"] = "hybrid"
        result["sources"] = " + ".join(result["sources"])
    return results


def search(
    settings: Settings, query: str, model_id: str, method: str, limit: int = 10
) -> list[dict[str, Any]]:
    """Run the selected search method."""
    if method == "keyword":
        return keyword_search(settings, query, limit)
    if method == "hybrid":
        return hybrid_search(settings, query, model_id, limit)
    return semantic_search(settings, query, model_id, limit)


@app.template_filter("safe_url")
def safe_url(value: Any) -> str:
    """Allow only absolute HTTP(S) links from database content."""
    if not isinstance(value, str) or "\\" in value:
        return "#"
    if any(ord(character) <= 32 or ord(character) == 127 for character in value):
        return "#"
    try:
        parsed = urlsplit(value)
        if parsed.scheme in {"http", "https"} and parsed.hostname:
            return value
    except ValueError:
        pass
    return "#"


@app.after_request
def security_headers(response: Response) -> Response:
    """Constrain browser content and prevent framing and MIME sniffing."""
    response.headers["Content-Security-Policy"] = (
        "default-src 'none'; style-src 'self' 'unsafe-inline'; "
        "form-action 'self'; base-uri 'none'; frame-ancestors 'none'"
    )
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"
    return response


@app.route("/", methods=["GET", "POST"])
def index() -> str:
    """Render the search form and display results without backend error details."""
    query = ""
    settings = Settings.from_environment()
    model_id = settings.embedding_model_id
    method = "semantic"
    results = []
    error = None

    if request.method == "POST":
        query = request.form.get("query", "").strip()
        model_id = (
            request.form.get("model_id", settings.embedding_model_id).strip()
            or settings.embedding_model_id
        )
        method = request.form.get("method", "semantic")
        if method not in {"semantic", "keyword", "hybrid"}:
            method = "semantic"
        if not query:
            error = "Enter a search sentence or keywords."
        elif len(query) > 2000 or len(model_id) > 512:
            error = "Use at most 2,000 query characters and 512 model ID characters."
        else:
            try:
                results = search(settings, query, model_id, method)
            except Exception as exc:
                app.logger.error("Search failed (%s)", type(exc).__name__)
                error = "Search is unavailable. Check the service configuration and try again."

    return render_template(
        "index.html",
        query=query,
        model_id=model_id,
        method=method,
        results=results,
        error=error,
        searched=request.method == "POST",
    )


if __name__ == "__main__":
    settings = Settings.from_environment()
    app.run(host="127.0.0.1", port=settings.port, debug=settings.debug)
