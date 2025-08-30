"""
Contains functions for generating prompts for the Screener Agent.
"""
from src.schemas.config_schemas import InclusionConfig
from src.schemas.output_schemas import ScreeningResult, CriterionEvidence

def create_screening_prompt(document_text: str, config: InclusionConfig) -> str:
    """
    Generates a detailed prompt for the LLM to screen a document against
    inclusion and exclusion criteria.

    The prompt includes:
    - A clear role and set of instructions.
    - The specific criteria from the user's configuration.
    - A JSON object example to guide the output format.

    Args:
        document_text: The full text of the document to be screened.
        config: The parsed inclusion.yaml configuration.

    Returns:
        A string containing the complete prompt for the LLM.
    """
    # Build the criteria section of the prompt for clear instructions
    require_all_str = "\n".join(
        [f"- ID: {c.id}\n  Description: {c.description}" for c in config.logic.require_all]
    ) if config.logic.require_all else "N/A"

    exclude_if_any_str = "\n".join(
        [f"- ID: {c.id}\n  Description: {c.description}" for c in config.logic.exclude_if_any]
    ) if config.logic.exclude_if_any else "N/A"

    # Create a dynamic JSON example for the LLM to follow, based on the Pydantic models.
    # This ensures the LLM's output is more likely to match the expected schema.
    example = ScreeningResult(
        document_id="example_doc.pdf",
        status="Include",
        reasoning="The document meets all inclusion criteria and does not meet any exclusion criteria.",
        evidence=[
            CriterionEvidence(
                criterion_id="is_rct",
                decision="Met",
                reasoning="The abstract explicitly states 'this was a randomized controlled trial'.",
                quote="this was a randomized controlled trial",
                page_number=1
            )
        ]
    )
    json_example = example.model_dump_json(indent=2)

    prompt = f"""You are a meticulous researcher conducting a systematic review. Your task is to analyze the provided document text and decide whether it meets the specified criteria for inclusion in the study.

**CRITERIA:**

**Inclusion Criteria (ALL of these MUST be met):**
{require_all_str}

**Exclusion Criteria (ANY of these will exclude the document):**
{exclude_if_any_str}

**INSTRUCTIONS:**
1.  Carefully evaluate the document against each criterion.
2.  For each criterion, provide a decision: "Met", "Not Met", or "Unsure".
3.  Provide a concise reasoning for each decision, referencing the text.
4.  If a criterion requires a quote, you MUST provide a verbatim quote from the text as evidence. If no quote is found, you cannot mark the criterion as "Met".
5.  Determine the final status of the document: "Include", "Exclude", or "Unsure".
    - "Include": All 'require_all' criteria are "Met" AND no 'exclude_if_any' criteria are "Met".
    - "Exclude": Any 'require_all' criterion is "Not Met" OR any 'exclude_if_any' criterion is "Met".
    - "Unsure": If you cannot confidently make a decision for any criterion.
6.  You MUST provide your response in a single, valid JSON object. Do not add any text, markdown, or commentary before or after the JSON object.

**JSON OUTPUT FORMAT EXAMPLE:**
```json
{json_example}
```

**DOCUMENT TEXT TO ANALYZE:**
---
{document_text}
---

Now, provide your screening decision for the document text above in a single JSON object.
"""
    return prompt
