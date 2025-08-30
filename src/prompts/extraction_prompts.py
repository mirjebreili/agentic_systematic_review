"""
Contains functions for generating prompts for the Extractor Agent.
"""
from src.schemas.config_schemas import Feature
from src.schemas.output_schemas import ExtractedDataPoint

def create_extraction_prompt(document_text: str, feature: Feature) -> str:
    """
    Generates a focused prompt for the LLM to extract a single feature
    from a document.

    The prompt is designed to be highly specific to maximize accuracy, asking
    the LLM to perform only one task at a time.

    Args:
        document_text: The full text of the document to be analyzed.
        feature: The specific feature to extract, as defined in features.yaml.

    Returns:
        A string containing the complete prompt for the LLM.
    """
    # Create a dynamic JSON example for the LLM to follow.
    example = ExtractedDataPoint(
        feature_name=feature.name,
        value="[example value, should be of type '{feature.type}']",
        confidence_score=0.9,
        evidence_quote="A direct quote from the text that contains or supports the extracted value.",
        page_number=1, # Example page number, if determinable
        chunk_id=f"example_doc.pdf_p1_c0", # Example chunk id
        reasoning="The value was found in this quote from the methods section of the paper."
    )
    # Using model_dump_json to ensure it's a JSON string
    json_example = example.model_dump_json(indent=2)

    prompt = f"""You are a highly specialized data extraction AI. Your sole task is to find and extract a single, specific piece of information (a "feature") from the provided text of a scientific paper.

**FEATURE TO EXTRACT:**
- **Name:** {feature.name}
- **Required Type:** {feature.type}
- **Description:** {feature.description}

**INSTRUCTIONS:**
1.  Thoroughly read the provided "DOCUMENT TEXT TO ANALYZE" to find the information that precisely matches the feature description.
2.  Extract the value. The value you provide MUST be of the type `{feature.type}`.
3.  If you cannot find the information or are not confident, you MUST return `null` for the "value" field. Do not guess or make up information.
4.  Provide a confidence score between 0.0 (no confidence) and 1.0 (complete confidence).
5.  You MUST provide a direct, verbatim quote from the text that contains the evidence for your extracted value. If the value is `null`, the quote can be `null` as well.
6.  Provide a brief reasoning for your extraction. If the value is `null`, explain why (e.g., "The document does not mention the number of participants.").
7.  Your entire response MUST be a single, valid JSON object. Do not include any explanatory text, markdown, or any characters outside of the JSON structure.

**JSON OUTPUT FORMAT EXAMPLE:**
```json
{json_example}
```

**DOCUMENT TEXT TO ANALYZE:**
---
{document_text}
---

Now, provide the extracted data for ONLY the feature "{feature.name}" in a single JSON object.
"""
    return prompt
