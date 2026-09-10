"""Tests for Phase 1: Normalized Market-Product Schema & Database.
Verifies:
- Adherence to SIH26090 normalized schema contract
- Zero hallucination (unprovided fields default strictly to None/null)
- Field validation (non-negative price, normalized currency, robust color parsing)
- Database table creation and indexing
- CRUD operations (upsert, batch upsert, get, list, search)
"""

import tempfile
from pathlib import Path
import pytest
from pydantic import ValidationError

from market_data.schema import MarketProduct
from market_data.database import DatabaseManager


class TestMarketProductSchema:
    """Verifies MarketProduct Pydantic schema contracts and constraints."""

    def test_complete_valid_market_product(self):
        sample = {
            "product_id": "itokri_101",
            "source": "itokri",
            "source_product_id": "SKU-9921",
            "product_name": "Traditional Hand-Painted Terracotta Pot",
            "category": "pottery",
            "subcategory": "terracotta_pottery",
            "craft_type": "Terracotta Craft",
            "material": "Clay",
            "region": "Rajasthan",
            "color": ["terracotta", "black", "white"],
            "size": "Medium (20cm)",
            "handmade": True,
            "description": "Handcrafted terracotta decorative pot painted with traditional folk motifs.",
            "price": 450.0,
            "currency": "INR",
            "unit": "piece",
            "source_url": "https://www.itokri.com/products/terracotta-pot-101",
            "image_url": "https://www.itokri.com/images/pot-101.jpg",
            "collected_at": "2026-09-10T12:00:00Z",
        }
        product = MarketProduct(**sample)

        assert product.product_id == "itokri_101"
        assert product.source == "itokri"
        assert product.product_name == "Traditional Hand-Painted Terracotta Pot"
        assert product.category == "pottery"
        assert product.price == 450.0
        assert product.currency == "INR"
        assert product.color == ["terracotta", "black", "white"]
        assert product.handmade is True

    def test_missing_fields_default_to_none_zero_hallucination(self):
        """Rule: Do not invent missing fields; unprovided fields must be None."""
        minimal = {
            "product_name": "Khurja Ceramic Mug",
            "category": "pottery",
            "price": 280.0,
        }
        product = MarketProduct(**minimal)

        # Mandatory defaults
        assert product.source == "itokri"
        assert product.currency == "INR"
        assert product.price == 280.0
        assert len(product.product_id) > 0  # auto-generated
        assert product.collected_at is not None

        # Optional fields MUST NOT be hallucinated
        assert product.source_product_id is None
        assert product.subcategory is None
        assert product.craft_type is None
        assert product.material is None
        assert product.region is None
        assert product.size is None
        assert product.handmade is None
        assert product.description is None
        assert product.unit is None
        assert product.source_url is None
        assert product.image_url is None
        assert product.rating is None
        assert product.review_count is None
        assert product.color == []

    def test_negative_price_rejected(self):
        with pytest.raises(ValidationError):
            MarketProduct(
                product_name="Defective Item",
                category="pottery",
                price=-150.0,
            )

    def test_currency_normalized_to_uppercase(self):
        product = MarketProduct(
            product_name="Brass Diya",
            category="metal_craft",
            price=320.0,
            currency="inr",
        )
        assert product.currency == "INR"

    def test_color_parsing_formats(self):
        # Comma-separated string
        p1 = MarketProduct(
            product_name="Pot", category="pottery", price=100.0,
            color="Red, Blue, Ochre"
        )
        assert p1.color == ["red", "blue", "ochre"]

        # JSON-encoded string
        p2 = MarketProduct(
            product_name="Pot", category="pottery", price=100.0,
            color='["Indigo", "White"]'
        )
        assert p2.color == ["indigo", "white"]

        # None input
        p3 = MarketProduct(
            product_name="Pot", category="pottery", price=100.0,
            color=None
        )
        assert p3.color == []


class TestDatabaseManager:
    """Verifies SQLite/PostgreSQL-compatible storage, tables, and CRUD operations."""

    @pytest.fixture
    def temp_db(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            temp_path = f.name
        db = DatabaseManager(db_path=temp_path)
        yield db
        try:
            Path(temp_path).unlink(missing_ok=True)
        except Exception:
            pass

    def test_database_tables_initialization(self, temp_db):
        """Verifies that all 5 required tables exist in the schema."""
        with temp_db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [r[0] for r in cursor.fetchall()]

        required_tables = [
            "sources",
            "market_products",
            "product_embeddings",
            "pricing_predictions",
            "comparable_products",
        ]
        for tbl in required_tables:
            assert tbl in tables, f"Table {tbl} must exist in the database"

    def test_product_upsert_and_retrieval(self, temp_db):
        product = MarketProduct(
            product_id="prod_001",
            source="itokri",
            source_product_id="ITK-001",
            product_name="Handmade Clay Vase",
            category="pottery",
            craft_type="Terracotta",
            material="Natural Clay",
            region="Khurja",
            color=["brown", "beige"],
            price=399.0,
            handmade=True,
            description="Authentic Khurja clay vase",
        )

        # Insert
        pid = temp_db.upsert_product(product)
        assert pid == "prod_001"

        # Fetch
        fetched = temp_db.get_product("prod_001")
        assert fetched is not None
        assert fetched.product_id == "prod_001"
        assert fetched.product_name == "Handmade Clay Vase"
        assert fetched.category == "pottery"
        assert fetched.craft_type == "Terracotta"
        assert fetched.material == "Natural Clay"
        assert fetched.price == 399.0
        assert fetched.color == ["brown", "beige"]
        assert fetched.handmade is True

    def test_batch_upsert_and_filtering(self, temp_db):
        products = [
            MarketProduct(
                product_id=f"pot_{i}",
                source="itokri",
                product_name=f"Decorative Pot {i}",
                category="pottery",
                craft_type="Terracotta Pottery",
                material="Clay",
                price=200.0 + (i * 50),
            )
            for i in range(5)
        ]
        textile_product = MarketProduct(
            product_id="textile_1",
            source="itokri",
            product_name="Phulkari Embroidered Dupatta",
            category="textile",
            craft_type="Phulkari",
            material="Chiffon Silk",
            price=1200.0,
        )
        products.append(textile_product)

        count = temp_db.upsert_products_batch(products)
        assert count == 6

        # Total count
        assert temp_db.count_products() == 6

        # Category filter
        pottery_items = temp_db.list_products(category="pottery")
        assert len(pottery_items) == 5
        assert all(p.category == "pottery" for p in pottery_items)

        textile_items = temp_db.list_products(category="textile")
        assert len(textile_items) == 1
        assert textile_items[0].product_name == "Phulkari Embroidered Dupatta"

    def test_search_products(self, temp_db):
        p1 = MarketProduct(
            product_id="p1",
            source="itokri",
            product_name="Blue Art Pottery Decorative Bowl",
            category="pottery",
            craft_type="Jaipur Blue Pottery",
            material="Quartz / Ceramic",
            price=650.0,
        )
        p2 = MarketProduct(
            product_id="p2",
            source="itokri",
            product_name="Brass Peacock Diya",
            category="metal_craft",
            craft_type="Dhokra Brass",
            material="Brass",
            price=850.0,
        )
        temp_db.upsert_products_batch([p1, p2])

        # Search by craft name keyword
        results = temp_db.search_products(query="Blue Pottery")
        assert len(results) == 1
        assert results[0].product_id == "p1"

        # Search by material keyword
        results_brass = temp_db.search_products(query="Brass")
        assert len(results_brass) == 1
        assert results_brass[0].product_id == "p2"

    def test_source_management(self, temp_db):
        sources = temp_db.get_sources()
        assert any(s["source_id"] == "itokri" for s in sources)

        # Add additional authorized source
        temp_db.upsert_source(
            source_id="tribes_india",
            name="Tribes India (TRIFED)",
            base_url="https://www.tribesindia.com",
            ingestion_type="csv",
        )

        updated_sources = temp_db.get_sources()
        assert len(updated_sources) >= 2
        assert any(s["source_id"] == "tribes_india" for s in updated_sources)
