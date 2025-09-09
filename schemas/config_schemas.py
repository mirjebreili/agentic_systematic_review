"""
Pydantic models for parsing and validating the user-provided YAML
configuration files (`inclusion.yaml` and `features.yaml`).
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Literal

# --- Schemas for inclusion.yaml ---

class InclusionCriterion(BaseModel):
    """Defines a single inclusion or exclusion criterion."""
    id: str = Field(..., description="A unique identifier for the criterion.")
    description: str = Field(..., description="A detailed description of the criterion for the LLM to evaluate.")
    evidence_must_be_quoted: bool = Field(default=False, description="Whether the LLM must provide a direct quote as evidence.")

class InclusionLogic(BaseModel):
    """Defines the logic for applying the criteria."""
    require_all: List[InclusionCriterion] = Field(default_factory=list, description="A list of criteria that must all be met for inclusion.")
    exclude_if_any: List[InclusionCriterion] = Field(default_factory=list, description="A list of criteria where meeting any one will lead to exclusion.")

class InclusionConfig(BaseModel):
    """The root model for the inclusion.yaml configuration file."""
    version: float = Field(..., description="The version of the configuration schema.")
    logic: InclusionLogic
    notes: Optional[str] = Field(None, description="General notes or instructions for the screening agent.")

# --- Schemas for features.yaml ---

class Feature(BaseModel):
    """Defines a single data feature to be extracted from the documents."""
    name: str = Field(..., description="The name of the feature, which will become a column in the output Excel file.")
    type: Literal["integer", "string", "float", "boolean"] = Field(..., description="The expected data type of the feature's value.")
    description: str = Field(..., description="A detailed description of the feature for the LLM to understand what to extract.")
    regex_hints: List[str] = Field(default_factory=list, description="Optional list of regex patterns to help locate the feature.")

class FeatureConfig(BaseModel):
    """The root model for the features.yaml configuration file."""
    version: float = Field(..., description="The version of the configuration schema.")
    features: List[Feature]
