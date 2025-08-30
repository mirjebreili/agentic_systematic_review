"""
Pydantic models for structuring the outputs of the screening, extraction,
and verification agents. These models ensure that the data passed between
agents and written to the final reports is consistent and valid.
"""
import datetime
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Any, Dict

# --- Schemas for Screening Agent Output ---

class CriterionEvidence(BaseModel):
    """Evidence for a single screening criterion decision."""
    criterion_id: str = Field(..., description="The ID of the criterion being evaluated.")
    decision: Literal["Met", "Not Met", "Unsure"] = Field(..., description="The agent's decision for this criterion.")
    reasoning: str = Field(..., description="The reasoning behind the agent's decision.")
    quote: Optional[str] = Field(None, description="The verbatim quote from the document supporting the decision.")
    page_number: Optional[int] = Field(None, description="The page number where the evidence was found.")

class ScreeningResult(BaseModel):
    """The complete screening result for a single document."""
    document_id: str = Field(..., description="A unique identifier for the document, typically its filename.")
    status: Literal["Include", "Exclude", "Unsure"] = Field(..., description="The final screening status of the document.")
    reasoning: str = Field(..., description="A summary of the reasoning for the final status.")
    evidence: List[CriterionEvidence] = Field(..., description="A list of evidence for each evaluated criterion.")

# --- Schemas for Extraction Agent Output ---

class ExtractedDataPoint(BaseModel):
    """A single piece of data extracted for one feature."""
    feature_name: str = Field(..., description="The name of the feature this data point corresponds to.")
    value: Optional[Any] = Field(None, description="The extracted value. Can be of any type defined in the feature config.")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="The agent's confidence in the accuracy of the extracted value (0.0 to 1.0).")
    evidence_quote: Optional[str] = Field(None, description="The verbatim quote from the document supporting the extraction.")
    page_number: Optional[int] = Field(None, description="The page number where the evidence was found.")
    chunk_id: Optional[str] = Field(None, description="The ID of the text chunk from which the data was extracted.")
    reasoning: str = Field(..., description="The reasoning behind the extraction, including why a value might be missing.")

class DocumentExtractionResult(BaseModel):
    """All extracted data for a single document."""
    document_id: str = Field(..., description="A unique identifier for the document.")
    extracted_features: List[ExtractedDataPoint] = Field(..., description="A list of all data points extracted from the document.")


# --- Schema for Chunked Document Content ---
class Chunk(BaseModel):
    """Represents a single chunk of text from a document."""
    document_id: str = Field(..., description="The ID of the document this chunk belongs to.")
    chunk_id: str = Field(..., description="A unique identifier for this chunk.")
    page_number: Optional[int] = Field(None, description="The page number from which this chunk was derived.")
    text: str = Field(..., description="The text content of the chunk.")


# --- Schema for the main application state graph ---

class AppState(BaseModel):
    """
    The state object that is passed between agents in the LangGraph workflow.
    It accumulates results from each step of the pipeline.
    """
    root_path: str
    model_name: str
    output_path: str

    # Configuration loaded at the start
    inclusion_config: Optional[Any] = None # Will hold InclusionConfig
    feature_config: Optional[Any] = None # Will hold FeatureConfig

    # List of files to process
    documents_to_process: List[str] = Field(default_factory=list)

    # Results from Ingest Agent
    document_chunks: Dict[str, List[Chunk]] = Field(default_factory=dict)

    # Results from agents
    screening_results: List[ScreeningResult] = Field(default_factory=list)
    extraction_results: List[DocumentExtractionResult] = Field(default_factory=list)

    # Files to be included in the final output
    included_files: List[str] = Field(default_factory=list)

    # Error logging
    processing_errors: Dict[str, str] = Field(default_factory=dict)

    # Timing
    start_time: Optional[datetime.datetime] = None
    end_time: Optional[datetime.datetime] = None
