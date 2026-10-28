"""Generate one OCI embedding to verify the local demo configuration."""

import argparse

from config import Settings
from oci_embedding_helper import OCIEmbeddingHelper


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="?", default="Generative artificial intelligence")
    arguments = parser.parse_args()
    settings = Settings.from_environment()
    helper = OCIEmbeddingHelper(
        settings.oci_compartment_id,
        settings.oci_genai_endpoint,
        settings.oci_config_file,
        settings.oci_config_profile,
    )
    _, vector = helper.embed(arguments.text, settings.embedding_model_id, "SEARCH_QUERY")
    print(vector)


if __name__ == "__main__":
    main()
