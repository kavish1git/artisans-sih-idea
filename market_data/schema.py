"""Normalized Market-Product Schema (Phase 1).
Defines standardized artisan market product models adhering to SIH26090 requirements.
Provides strict validation, no hallucinated/invented fields, and database compatibility.
"""

from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Dict, List, Optional, Union
import uuid
from pydantic import BaseModel, Field, field_validator, model_validator


class MarketProduct(BaseModel):
    """
    Normalized schema for artisan market products.
    Mandatory SIH Contract:
    {
      "product_id": "",
      "source": "itokri",
      "source_product_id": "",
      "product_name": "",
      "category": "",
      "subcategory": "",
      "craft_type": "",
      "material": "",
      "region": "",
      "color": [],
      "size": "",
      "handmade": null,
      "description": "",
      "price": 0,
      "currency": "INR",
      "unit": "",
      "source_url": "",
      "image_url": "",
      "collected_at": ""
    }
    Rules:
    - Never invent missing fields (default unprovided fields to None/null).
    - Maintain strict type validation and zero hallucination.
    """

    product_id: str = Field(
        default="",
        description="Unique system identifier for the listing (UUID or SHA256)",
    )
    source: str = Field(
        default="itokri",
        description="Data origin source identifier (e.g. itokri, ccic, tribes_india)",
    )
    source_product_id: Optional[str] = Field(
        default=None,
        description="Source platform original SKU / listing ID",
    )
    product_name: str = Field(
        ...,
        min_length=1,
        description="Clean title / name of the handicraft item",
    )
    category: str = Field(
        ...,
        min_length=1,
        description="Standardized craft category (pottery, textile, wood_bamboo, metal_craft, etc.)",
    )
    subcategory: Optional[str] = Field(
        default=None,
        description="Specific craft subcategory or taxonomy term",
    )
    craft_type: Optional[str] = Field(
        default=None,
        description="Artisanal craft technique (e.g. Blue Pottery, Phulkari, Dhokra)",
    )
    material: Optional[str] = Field(
        default=None,
        description="Primary material used (e.g. Clay, Silk, Brass, Teak Wood)",
    )
    region: Optional[str] = Field(
        default=None,
        description="Geographical origin / GI region (e.g. Jaipur, Punjab, Bastar)",
    )
    color: List[str] = Field(
        default_factory=list,
        description="List of dominant colors identified for the product",
    )
    size: Optional[str] = Field(
        default=None,
        description="Size or dimensions (e.g. Medium, 25cm, Free Size)",
    )
    handmade: Optional[bool] = Field(
        default=None,
        description="Flag indicating if product is verified handmade (null if unknown)",
    )
    description: Optional[str] = Field(
        default=None,
        description="Detailed product description from source",
    )
    price: float = Field(
        ...,
        ge=0.0,
        description="Current listing price in target currency",
    )
    currency: str = Field(
        default="INR",
        description="ISO 4217 Currency code (default: INR)",
    )
    unit: Optional[str] = Field(
        default=None,
        description="Unit of measurement (e.g. piece, set, meter, pair)",
    )
    source_url: Optional[str] = Field(
        default=None,
        description="Direct URL to authorized listing",
    )
    image_url: Optional[str] = Field(
        default=None,
        description="Primary product photo URL",
    )
    collected_at: Optional[str] = Field(
        default=None,
        description="ISO 8601 UTC timestamp of data ingestion",
    )
    rating: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=5.0,
        description="Product rating if available on source",
    )
    review_count: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of reviews if available",
    )

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, v: str) -> str:
        return (v or "INR").upper().strip()

    @field_validator("color", mode="before")
    @classmethod
    def parse_colors(cls, v: Any) -> List[str]:
        if v is None:
            return []
        if isinstance(v, list):
            return [str(c).strip().lower() for c in v if str(c).strip()]
        if isinstance(v, str):
            # Check if JSON encoded list
            trimmed = v.strip()
            if trimmed.startswith("[") and trimmed.endswith("]"):
                try:
                    parsed = json.loads(trimmed)
                    if isinstance(parsed, list):
                        return [str(c).strip().lower() for c in parsed if str(c).strip()]
                except Exception:
                    pass
            # Split comma separated
            return [c.strip().lower() for c in v.split(",") if c.strip()]
        return []

    @model_validator(mode="after")
    def populate_defaults(self) -> "MarketProduct":
        # 1. Assign deterministic or unique product_id if not present
        if not self.product_id:
            if self.source and self.source_product_id:
                seed = f"{self.source}:{self.source_product_id}".encode("utf-8")
                self.product_id = hashlib.sha256(seed).hexdigest()[:16]
            else:
                self.product_id = uuid.uuid4().hex[:16]

        # 2. Assign timestamp if not present
        if not self.collected_at:
            self.collected_at = datetime.now(timezone.utc).isoformat()

        return self

    def to_dict(self) -> Dict[str, Any]:
        """Serializes to clean python dictionary matching SIH specification."""
        return self.model_dump()

    def to_db_row(self) -> Dict[str, Any]:
        """Serializes fields for SQL database insertion."""
        d = self.model_dump()
        d["color"] = json.dumps(d["color"])
        if d["handmade"] is not None:
            d["handmade"] = 1 if d["handmade"] else 0
        return d

    @classmethod
    def from_db_row(cls, row: Union[Dict[str, Any], tuple], columns: Optional[List[str]] = None) -> "MarketProduct":
        """Constructs MarketProduct from SQL row (dict or tuple)."""
        if isinstance(row, dict):
            data = dict(row)
        elif columns and isinstance(row, (tuple, list)):
            data = dict(zip(columns, row))
        else:
            raise ValueError("Row must be a dictionary or a tuple with column names provided.")

        # Decode JSON color
        if "color" in data and isinstance(data["color"], str):
            try:
                data["color"] = json.loads(data["color"])
            except Exception:
                data["color"] = [c.strip() for c in data["color"].split(",") if c.strip()]

        # Decode handmade int to bool
        if "handmade" in data and data["handmade"] is not None:
            data["handmade"] = bool(data["handmade"])

        return cls(**data)
