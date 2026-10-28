"""Offline security and search regression tests with synthetic inputs."""

import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import app
import config
import generate_embeddings
from config import Settings
from oci_embedding_helper import OCIEmbeddingHelper


class SettingsTests(unittest.TestCase):
    """Keep credentials out of representations and validate configuration."""

    def setUp(self):
        self.environment = {
            "OCI_COMPARTMENT_ID": "test-compartment",
            "OCI_GENAI_ENDPOINT": "https://example.invalid",
            "MYSQL_HOST": "database.example.invalid",
            "MYSQL_USER": "test-user",
            "MYSQL_PASSWORD": "<test-only-placeholder>",
            "MYSQL_SSL_CA": "ca.crt",
        }

    def settings(self):
        with patch.dict(os.environ, self.environment, clear=True), patch("config.load_dotenv"):
            return Settings.from_environment()

    def test_password_is_excluded_from_representation(self):
        self.assertNotIn(self.environment["MYSQL_PASSWORD"], repr(self.settings()))

    def test_debug_is_off_by_default(self):
        self.assertFalse(self.settings().debug)

    def test_tls_ca_is_required(self):
        del self.environment["MYSQL_SSL_CA"]
        with self.assertRaisesRegex(RuntimeError, "MYSQL_SSL_CA"):
            self.settings()

    def test_ports_are_validated(self):
        for value in ("invalid", "0", "65536", "-1"):
            with self.subTest(value=value):
                self.environment["PORT"] = value
                with self.assertRaises(RuntimeError):
                    self.settings()

    def test_dotenv_is_loaded_only_from_the_project(self):
        with patch.dict(os.environ, self.environment, clear=True):
            with patch("config.load_dotenv") as loader:
                Settings.from_environment()
        loader.assert_called_once_with(Path(config.__file__).with_name(".env"))

    def test_both_database_clients_verify_tls(self):
        settings = self.settings()
        for module in (app, generate_embeddings):
            with self.subTest(module=module.__name__):
                with patch.object(module.mysql.connector, "connect") as connect:
                    module.mysql_connection(settings)
                self.assertTrue(connect.call_args.kwargs["ssl_verify_cert"])
                self.assertTrue(connect.call_args.kwargs["ssl_verify_identity"])
                self.assertEqual(connect.call_args.kwargs["ssl_ca"], settings.mysql_ssl_ca)


class WebSecurityTests(unittest.TestCase):
    """Exercise rendered responses without accessing cloud services."""

    def setUp(self):
        self.settings = SimpleNamespace(embedding_model_id="test-model")
        self.loader = patch("app.Settings.from_environment", return_value=self.settings)
        self.loader.start()
        self.addCleanup(self.loader.stop)
        self.client = app.app.test_client()

    def test_backend_errors_do_not_expose_sensitive_details(self):
        detail = "<private-backend-error-detail>"
        with patch("app.search", side_effect=RuntimeError(detail)):
            with self.assertLogs(app.app.logger, level="ERROR") as logs:
                response = self.client.post("/", data={"query": "test"})
        self.assertEqual(response.status_code, 200)
        self.assertNotIn(detail, response.get_data(as_text=True))
        self.assertNotIn(detail, " ".join(logs.output))
        self.assertIn("Search is unavailable", response.get_data(as_text=True))

    def test_result_links_and_text_are_sanitized(self):
        result = {
            "guid": "javascript:alert(1)",
            "post_title": "<script>alert(1)</script>",
            "post_excerpt": "<img src=x onerror=alert(1)>",
            "result_type": "semantic",
            "combined_distance": 0.1,
        }
        with patch("app.search", return_value=[result]):
            response = self.client.post("/", data={"query": "test"})
        html = response.get_data(as_text=True)
        self.assertIn('href="#"', html)
        self.assertNotIn("javascript:", html)
        self.assertNotIn("<script>", html)
        self.assertNotIn("<img", html)
        self.assertIn("&lt;script&gt;", html)

    def test_unsafe_url_variants_are_rejected(self):
        for url in (
            None,
            "//example.invalid/path",
            "data:text/html,<script>alert(1)</script>",
            "java\nscript:alert(1)",
            "https://[invalid",
            "https://example.invalid/\\unsafe",
        ):
            with self.subTest(url=url):
                self.assertEqual(app.safe_url(url), "#")
        self.assertEqual(app.safe_url("https://example.org/post"), "https://example.org/post")

    def test_long_inputs_and_empty_queries_do_not_call_services(self):
        for data in ({"query": " "}, {"query": "x" * 2001}, {"query": "x", "model_id": "x" * 513}):
            with self.subTest(data_length=len(str(data))):
                with patch("app.search") as search:
                    response = self.client.post("/", data=data)
                search.assert_not_called()
                self.assertEqual(response.status_code, 200)

    def test_oversized_request_is_rejected(self):
        with patch("app.search") as search:
            response = self.client.post("/", data={"query": "x" * 20000})
        self.assertEqual(response.status_code, 413)
        search.assert_not_called()

    def test_browser_protection_headers(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")
        self.assertIn("frame-ancestors 'none'", response.headers["Content-Security-Policy"])


class SearchTests(unittest.TestCase):
    """Verify query parameter binding and ranking behavior."""

    def test_keyword_search_keeps_input_out_of_sql(self):
        query = "' OR 1=1; --"
        connection = MagicMock()
        cursor = connection.cursor.return_value.__enter__.return_value
        cursor.fetchall.return_value = []
        with patch("app.mysql_connection") as connect:
            connect.return_value.__enter__.return_value = connection
            app.keyword_search(SimpleNamespace(), query)
        statement, parameters = cursor.execute.call_args.args
        self.assertNotIn(query, statement)
        self.assertEqual(parameters[0], query)
        self.assertEqual(json.loads(parameters[1]), [query])

    def test_hybrid_fusion_rewards_shared_results_and_breaks_ties_by_id(self):
        with (
            patch("app.semantic_search", return_value=[{"ID": 1}, {"ID": 3}]),
            patch("app.keyword_search", return_value=[{"ID": 2}, {"ID": 3}]),
        ):
            results = app.hybrid_search(SimpleNamespace(), "query", "model")
        self.assertEqual([item["ID"] for item in results], [3, 1, 2])
        self.assertEqual(results[0]["sources"], "semantic + keyword")

    def test_ingestion_removes_non_content_elements(self):
        content = "<p>Hello &amp; welcome</p><script>hidden</script><style>hidden</style> world"
        self.assertEqual(
            generate_embeddings.sanitize_post_content(content), "Hello & welcome world"
        )

    def test_summary_response_handles_unexpected_field_types(self):
        helper = MagicMock()
        helper.client.chat.return_value.data.chat_response.text = json.dumps(
            {"summary": None, "keywords": [None, "  databases  "]}
        )
        summary, keywords = generate_embeddings.generate_summary_and_keywords(
            helper, SimpleNamespace(summary_model_id="test-model"), "content", "Title"
        )
        self.assertEqual(summary, "Title")
        self.assertEqual(json.loads(keywords), ["databases"])

    def test_embeddings_are_serialized_and_v4_dimensions_are_explicit(self):
        with (
            patch("oci_embedding_helper.oci.config.from_file", return_value={}),
            patch("oci_embedding_helper.GenerativeAiInferenceClient") as client,
        ):
            client.return_value.embed_text.return_value.data.embeddings = [[0.1, 0.2]]
            helper = OCIEmbeddingHelper("test-compartment", "https://example.invalid")
        vector, encoded = helper.embed("query", "cohere.embed-v4.0", "SEARCH_QUERY")
        self.assertEqual(json.loads(encoded), vector)
        details = client.return_value.embed_text.call_args.args[0]
        self.assertEqual(details.output_dimensions, 1024)
        self.assertEqual(details.input_type, "SEARCH_QUERY")
        with self.assertRaises(ValueError):
            helper.embed(" ")


if __name__ == "__main__":
    unittest.main()
