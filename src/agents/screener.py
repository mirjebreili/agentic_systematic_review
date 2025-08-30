"""
The Screener Agent evaluates each document against the inclusion/exclusion
criteria defined in `inclusion.yaml`.
"""
import logging
import json
import ollama
from typing import Dict
from pydantic import ValidationError

from src.schemas.output_schemas import AppState, ScreeningResult
from src.schemas.config_schemas import InclusionConfig
from src.prompts.screening_prompts import create_screening_prompt

def screen_documents(state: Dict) -> Dict:
    """
    The entry point for the Screening agent.

    This agent performs the following steps:
    1. Loads the `inclusion_config` from the application state.
    2. Initializes an Ollama client to communicate with the LLM.
    3. For each document, it concatenates the text from all its chunks.
    4. It generates a detailed prompt using the `create_screening_prompt` function.
    5. It calls the Ollama LLM, requesting a JSON response.
    6. It parses and validates the response into a `ScreeningResult` object.
    7. It handles any errors during the process (e.g., LLM errors, JSON decoding, validation).
    8. Appends the result to the `screening_results` list in the state and returns the updated state.

    Args:
        state: The current application state dictionary.

    Returns:
        The updated application state dictionary.
    """
    app_state = AppState(**state)

    # Exit early if there's nothing to screen
    if not app_state.document_chunks:
        logging.warning("Skipping Screening Stage: No documents were successfully ingested.")
        return app_state.model_dump()

    try:
        inclusion_config = InclusionConfig(**app_state.inclusion_config)
    except ValidationError as e:
        logging.error(f"Failed to parse inclusion_config from app state: {e}", exc_info=True)
        app_state.processing_errors["Screener"] = "Could not parse inclusion configuration."
        return app_state.model_dump()

    logging.info("--- Starting Screening Stage ---")

    try:
        client = ollama.Client()
        client.show(app_state.model_name)
        logging.info(f"Successfully connected to Ollama with model '{app_state.model_name}'.")
    except Exception as e:
        logging.error(f"Ollama client failed: {e}", exc_info=True)
        app_state.processing_errors["Screener"] = f"Ollama client failed for model '{app_state.model_name}': {e}"
        return app_state.model_dump()

    for doc_id, chunks in app_state.document_chunks.items():
        logging.info(f"Screening document: {doc_id}")

        # TODO: Implement intelligent context window management for very large documents.
        full_text = "\n\n".join([chunk.text for chunk in chunks])

        try:
            prompt = create_screening_prompt(document_text=full_text, config=inclusion_config)

            response = client.chat(
                model=app_state.model_name,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.0},
                format='json'
            )

            response_content = response['message']['content']
            result_data = json.loads(response_content)
            result_data['document_id'] = doc_id # Inject the doc_id

            screening_result = ScreeningResult(**result_data)
            app_state.screening_results.append(screening_result)

            logging.info(f"Successfully screened '{doc_id}'. Status: {screening_result.status}")

        except json.JSONDecodeError as e:
            logging.error(f"Failed to decode LLM JSON response for '{doc_id}': {e}\nRaw response: {response_content}")
            app_state.processing_errors[doc_id] = f"LLM returned malformed JSON during screening."
        except ValidationError as e:
            logging.error(f"Failed to validate screening result for '{doc_id}': {e}\nParsed data: {result_data}")
            app_state.processing_errors[doc_id] = f"LLM response for screening did not match required schema."
        except Exception as e:
            logging.error(f"An unexpected error occurred while screening '{doc_id}': {e}", exc_info=True)
            app_state.processing_errors[doc_id] = f"An unexpected error occurred during screening: {e}"

    logging.info(f"--- Screening Stage Complete. Screened {len(app_state.document_chunks)} documents. ---")
    return app_state.model_dump()
