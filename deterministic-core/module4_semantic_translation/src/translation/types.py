"""Data types for Module 4 Semantic Translation.

ZERO ML DEPENDENCIES.
"""
from enum import Enum
from typing import List
from pydantic import BaseModel, Field


class Generator(str, Enum):
    """Generator type used for causal log synthesis."""
    GENERATOR_UNSPECIFIED = "GENERATOR_UNSPECIFIED"
    GENERATOR_TEMPLATE = "GENERATOR_TEMPLATE"
    GENERATOR_CONSTRAINED = "GENERATOR_CONSTRAINED"


class CausalLog(BaseModel):
    """Operator-facing causal explanation, grounded in physical violation evidence."""
    action_id: str
    text: str
    grounded_entities: List[str] = Field(
        default_factory=list,
        description="Elements cited in the text (must be a subset of violation element_ids)",
    )
    generator: Generator = Generator.GENERATOR_TEMPLATE
