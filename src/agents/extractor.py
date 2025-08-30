"""
The Extractor Agent iterates through included documents and extracts
pre-defined features from them based on `features.yaml`.
"""
import logging
import json
import ollama
from typing import Dict, List
from pydantic import ValidationError

from src.schemas.output_schemas import AppState, DocumentExtractionResult, ExtractedDataPoint
from src.schemas.config_schemas import FeatureConfig, Feature
from src.prompts.extraction_prompts import create_extraction_prompt

def _extract_feature(client: ollama.Client, model: str, text: str, feature: Feature) -> ExtractedDataPoint:
    """Helper function to perform a single feature extraction LLM call."""
    prompt = create_extraction_prompt(document_text=text, feature=feature)

    response = client.chat(
        model=model,
        messages=[{'role': 'user', 'content': prompt}],
        options={'temperature': 0.0},
        format='json'
    )

    response_content = response['message']['content']
    result_data = json.loads(response_content)

    # The LLM doesn't know the chunk_id, so we can't add it here.
    # This could be improved by processing chunk by chunk.
    # For now, we validate what we can get from the LLM.
    return ExtractedDataPoint(**result_data)


def extract_features(state: Dict) -> Dict:
    """
    The entry point for the Extraction agent.

    This agent performs the following steps:
    1. Filters for documents that were marked "Include" by the Screener.
    2. For each included document, it performs a feature-by-feature extraction.
    3. For each feature in `features.yaml`, it generates a prompt and calls the LLM.
    4. It parses, validates, and collects the `ExtractedDataPoint` from each call.
    5. The collected data points for a document are stored in a `DocumentExtractionResult`.
    6. The final results are appended to the `extraction_results` list in the state.

    Args:
        state: The current application state dictionary.

    Returns:
        The updated application state dictionary.
    """
    app_state = AppState(**state)

    if not app_state.screening_results:
        logging.warning("Skipping Extraction Stage: No documents were screened.")
        return app_state.model_dump()

    try:
        feature_config = FeatureConfig(**app_state.feature_config)
    except ValidationError as e:
        logging.error(f"Failed to parse feature_config from app state: {e}", exc_info=True)
        app_state.processing_errors["Extractor"] = "Could not parse feature configuration."
        return app_state.model_dump()

    logging.info("--- Starting Extraction Stage ---")

    try:
        client = ollama.Client()
        client.show(app_state.model_name)
    except Exception as e:
        logging.error(f"Ollama client failed: {e}", exc_info=True)
        app_state.processing_errors["Extractor"] = f"Ollama client failed for model '{app_state.model_name}': {e}"
        return app_state.model_dump()

    included_doc_ids = {res.document_id for res in app_state.screening_results if res.status == "Include"}
    if not included_doc_ids:
        logging.info("No documents were marked for inclusion. Skipping feature extraction.")
        return app_state.model_dump()

    for doc_id in included_doc_ids:
        if doc_id not in app_state.document_chunks:
            logging.warning(f"Cannot extract from '{doc_id}': No chunks found in state.")
            continue

        logging.info(f"Extracting features from document: {doc_id}")
        full_text = "\n\n".join([chunk.text for chunk in app_state.document_chunks[doc_id]])

        extracted_features_for_doc: List[ExtractedDataPoint] = []
        for feature in feature_config.features:
            logging.info(f"  - Extracting feature: '{feature.name}'")
            try:
                data_point = _extract_feature(client, app_state.model_name, full_text, feature)
                extracted_features_for_doc.append(data_point)
            except json.JSONDecodeError as e:
                logging.error(f"Failed to decode LLM JSON for feature '{feature.name}' in '{doc_id}': {e}")
                app_state.processing_errors[f"{doc_id}_{feature.name}"] = "LLM returned malformed JSON."
            except ValidationError as e:
                logging.error(f"Failed to validate extracted data for feature '{feature.name}' in '{doc_id}': {e}")
                app_state.processing_errors[f"{doc_id}_{feature.name}"] = "LLM response did not match schema."
            except Exception as e:
                logging.error(f"Unexpected error extracting feature '{feature.name}' in '{doc_id}': {e}", exc_info=True)
                app_state.processing_errors[f"{doc_id}_{feature.name}"] = "An unexpected error occurred."

        app_state.extraction_results.append(
            DocumentExtractionResult(
                document_id=doc_id,
                extracted_features=extracted_features_for_doc
            )
        )

    logging.info(f"--- Extraction Stage Complete. Extracted features from {len(included_doc_ids)} documents. ---")
    return app_state.model_dump()
