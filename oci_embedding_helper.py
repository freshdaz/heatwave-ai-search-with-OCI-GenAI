"""Small wrapper around OCI Generative AI embedding requests."""

import json

import oci
from oci.generative_ai_inference import GenerativeAiInferenceClient
from oci.generative_ai_inference.models import EmbedTextDetails, OnDemandServingMode


class OCIEmbeddingHelper:
    """Create OCI embeddings and return them in MySQL-ready JSON format."""

    def __init__(
        self,
        compartment_id: str,
        endpoint: str,
        config_file: str = "~/.oci/config",
        profile: str = "DEFAULT",
    ) -> None:
        config = oci.config.from_file(config_file, profile)
        self.client = GenerativeAiInferenceClient(
            config=config,
            service_endpoint=endpoint,
            retry_strategy=oci.retry.DEFAULT_RETRY_STRATEGY,
        )
        self.compartment_id = compartment_id

    def embed(
        self,
        text: str,
        model_id: str = "cohere.embed-multilingual-v3.0",
        input_type: str | None = None,
    ) -> tuple[list[float], str]:
        """Return an embedding and its JSON representation for STRING_TO_VECTOR()."""
        text = (text or "").strip()
        if not text:
            raise ValueError("Embedding input must not be empty.")

        request = {
            "serving_mode": OnDemandServingMode(model_id=model_id),
            "compartment_id": self.compartment_id,
            "inputs": [text],
            "truncate": "END",
        }
        if input_type:
            request["input_type"] = input_type
        if model_id == "cohere.embed-v4.0":
            request["output_dimensions"] = 1024

        response = self.client.embed_text(EmbedTextDetails(**request))
        vector = response.data.embeddings[0]
        return vector, json.dumps(vector)
