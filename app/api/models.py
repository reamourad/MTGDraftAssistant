"""
Pydantic request models for API validation.
"""

from pydantic import BaseModel
from typing import List


class PredictRequest(BaseModel):
    """Request model for card pick prediction."""
    set: str
    deck: List[str]
    pack: List[str]