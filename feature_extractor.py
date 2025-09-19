"""
Core extraction logic combining semantic search and LLM calls.
"""

from typing import Dict, List, Any
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import logging

from embedding_manager import EmbeddingManager
from llm_client import LLMClient

logger = logging.getLogger(__name__)

class FeatureExtractor:
    def __init__(self, embedding_manager: EmbeddingManager, llm_client: LLMClient, top_k_chunks: int):
        self.embedding_manager = embedding_manager
        self.llm_client = llm_client
        self.top_k_chunks = top_k_chunks

    def extract_all_features(self, paper_collection: Chroma, fields: List[Dict[str, Any]]) -> Dict[str, Dict]:
        """
        Extract all features defined in the fields list for a single paper.
        """
        if not paper_collection:
            logger.error("Cannot extract features, the paper collection is null.")
            return {}

        all_extracted_data = {}
        total_features = len(fields)
        logger.info(f"Starting extraction for {total_features} features.")

        for index, field in enumerate(fields):
            feature_name = field["field_name"]
            feature_description = field["description"]
            logger.info(f"--- Extracting feature {index + 1}/{total_features}: {feature_name} ---")

            extracted_data = self.extract_single_feature(
                collection=paper_collection,
                feature_name=feature_name,
                feature_description=feature_description
            )
            all_extracted_data[feature_name] = extracted_data

        logger.info("Completed extraction for all features.")
        return all_extracted_data

    def extract_single_feature(self, collection: Chroma, feature_name: str, feature_description: str) -> Dict:
        """
        Extract a single feature from the paper's collection.
        """
        # 1. Build an enhanced search query
        search_query = self.build_search_query(feature_name, feature_description)

        # 2. Semantic search for relevant chunks
        relevant_chunks = self.embedding_manager.semantic_search(
            collection,
            query=search_query,
            k=self.top_k_chunks
        )

        if not relevant_chunks:
            logger.warning(f"No relevant chunks found for feature: {feature_name}")
            return {"value": "NOT_FOUND", "confidence": 0.0, "found": False, "explanation": "No relevant text found in the document."}

        # 3. Create context from the top-k chunks
        context = self._prepare_context_from_chunks(relevant_chunks)

        # 4. Call LLM with the focused prompt
        extracted_result = self.llm_client.extract_feature(
            feature_name=feature_name,
            feature_description=feature_description,
            context=context
        )

        return extracted_result

    def build_search_query(self, feature_name: str, description: str) -> str:
        """
        Create an enhanced query for semantic search by combining feature name and description.
        """
        # Example enhancement: could add keywords or rephrase as a question
        query = f"Regarding '{feature_name}': {description}"
        logger.debug(f"Built search query: {query}")
        return query

    def _prepare_context_from_chunks(self, chunks: List[Document]) -> str:
        """
        Prepare a single string context from a list of document chunks.
        """
        context = []
        for chunk in chunks:
            page_num = chunk.metadata.get('page', 'N/A')
            content = chunk.page_content
            context.append(f"--- Chunk from Page {page_num} ---\n{content}\n")

        return "\n".join(context)
