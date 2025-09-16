"""
Unified interface for Ollama and vLLM providers using LangChain.
"""

from langchain_community.llms import Ollama, VLLM
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging
from utils import retry_with_backoff

logger = logging.getLogger(__name__)

# Pydantic model for structured output
class FeatureExtractionOutput(BaseModel):
    value: Any = Field(description="The extracted value of the feature.")
    confidence: float = Field(description="A confidence score between 0.0 and 1.0.", ge=0.0, le=1.0)
    found: bool = Field(description="A boolean indicating if the feature was found in the text.")
    explanation: str = Field(description="A brief explanation of how the value was derived from the context.")

class LLMClient:
    def __init__(self, provider: str, base_url: str, model_name: str, temperature: float, max_tokens: int):
        """Initialize LLM based on provider."""
        self.provider = provider
        self.output_parser = JsonOutputParser(pydantic_object=FeatureExtractionOutput)

        common_params = {
            "model": model_name,
            "temperature": temperature,
            "top_p": 0.9,
            "stop": ["<|eot_id|>", "```"], # Stop tokens for cleaner output
        }

        if provider == "ollama":
            self.llm = Ollama(
                base_url=base_url,
                **common_params
            )
        elif provider == "vllm":
            self.llm = VLLM(
                model=model_name,
                temperature=temperature,
                max_new_tokens=max_tokens,
                vllm_kwargs={"stop": ["<|eot_id|>", "```"]},
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        logger.info(f"Initialized LLM client for provider: {provider} with model: {model_name}")

        self.prompt_template = self._create_prompt_template()

    def _create_prompt_template(self) -> PromptTemplate:
        """Creates the prompt template for feature extraction."""
        template = """
        You are an expert academic researcher tasked with extracting specific information from a scientific paper.
        Your goal is to extract the value for a single feature based on the provided context.

        **Feature to Extract:**
        - **Name:** {feature_name}
        - **Description:** {feature_description}

        **Instructions:**
        1.  Carefully read the **Context** provided below, which consists of relevant text chunks from the paper.
        2.  Identify the value for the feature based *only* on the given context.
        3.  If you find the information, extract it accurately.
        4.  If the information is not present in the context, explicitly state that it was not found.
        5.  Provide a confidence score (0.0 to 1.0) indicating your certainty. 1.0 means you are absolutely certain.
        6.  Provide a brief explanation for your decision.
        7.  You MUST respond in the following JSON format. Do not include any other text, just the JSON object.

        **Context from the paper:**
        ---
        {context}
        ---

        **Output Format (JSON only):**
        {format_instructions}
        """
        return PromptTemplate(
            template=template,
            input_variables=["feature_name", "feature_description", "context"],
            partial_variables={"format_instructions": self.output_parser.get_format_instructions()}
        )

    @retry_with_backoff(retries=3, backoff_in_seconds=2.0)
    def extract_feature(self, feature_name: str, feature_description: str, context: str) -> Dict:
        """
        Call LLM with a formatted prompt and parse the structured response.
        """
        prompt = self.prompt_template.format(
            feature_name=feature_name,
            feature_description=feature_description,
            context=context
        )

        logger.debug(f"Invoking LLM for feature '{feature_name}'")

        try:
            chain = self.prompt_template | self.llm | self.output_parser
            response = chain.invoke({
                "feature_name": feature_name,
                "feature_description": feature_description,
                "context": context
            })
            logger.info(f"Successfully extracted feature '{feature_name}': {response.get('value')}")
            return response
        except Exception as e:
            logger.error(f"Failed to extract feature '{feature_name}' after retries. Error: {e}")
            # Return a default error structure
            return {
                "value": "EXTRACTION_ERROR",
                "confidence": 0.0,
                "found": False,
                "explanation": str(e)
            }
