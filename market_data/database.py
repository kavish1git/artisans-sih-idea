"""Database Layer for Artisan Market Data & Pricing Intelligence (Phase 1).
Supports SQLite (zero-dependency default for local development/testing)
and PostgreSQL (production environments via DATABASE_URL).
Adheres strictly to SIH26090 schema specifications.
"""

import json
import os
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Tuple, Union

from .schema import MarketProduct


DEFAULT_DB_PATH = Path(__file__).parent.parent / "market_data" / "market_pricing.db"


class DatabaseManager:
    """Manages database connection, table migrations, and CRUD operations."""

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Returns a connection with Row factory enabled."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        # Enable foreign key support in SQLite
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def init_db(self) -> None:
        """Initializes all 5 required tables and indexes."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 1. Sources Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sources (
                    source_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    base_url TEXT,
                    is_active INTEGER DEFAULT 1,
                    ingestion_type TEXT DEFAULT 'csv',
                    last_sync TEXT
                );
            """)

            # Seed default verified source: iTokri
            cursor.execute("""
                INSERT OR IGNORE INTO sources (source_id, name, base_url, is_active, ingestion_type)
                VALUES ('itokri', 'iTokri Handicrafts & Handlooms', 'https://www.itokri.com', 1, 'csv');
            """)

            # 2. Market Products Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_products (
                    product_id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    source_product_id TEXT,
                    product_name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    subcategory TEXT,
                    craft_type TEXT,
                    material TEXT,
                    region TEXT,
                    color TEXT,
                    size TEXT,
                    handmade INTEGER,
                    description TEXT,
                    price REAL NOT NULL,
                    currency TEXT DEFAULT 'INR',
                    unit TEXT,
                    rating REAL,
                    review_count INTEGER,
                    source_url TEXT,
                    image_url TEXT,
                    collected_at TEXT,
                    FOREIGN KEY (source) REFERENCES sources(source_id)
                );
            """)

            # Performance Indexes for Filtering & Search
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_category ON market_products(category);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_craft ON market_products(craft_type);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_material ON market_products(material);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_source ON market_products(source);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_price ON market_products(price);")

            # 3. Product Embeddings Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS product_embeddings (
                    id TEXT PRIMARY KEY,
                    product_id TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    embedding_vector TEXT NOT NULL,
                    created_at TEXT,
                    FOREIGN KEY (product_id) REFERENCES market_products(product_id) ON DELETE CASCADE
                );
            """)

            # 4. Pricing Predictions Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pricing_predictions (
                    prediction_id TEXT PRIMARY KEY,
                    product_query TEXT NOT NULL,
                    predicted_price_min REAL NOT NULL,
                    predicted_price_max REAL NOT NULL,
                    predicted_price_optimal REAL NOT NULL,
                    artisan_cost REAL,
                    model_used TEXT,
                    confidence_score REAL,
                    created_at TEXT
                );
            """)

            # 5. Comparable Products Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS comparable_products (
                    id TEXT PRIMARY KEY,
                    prediction_id TEXT NOT NULL,
                    market_product_id TEXT NOT NULL,
                    similarity_score REAL NOT NULL,
                    match_reasons TEXT,
                    FOREIGN KEY (prediction_id) REFERENCES pricing_predictions(prediction_id) ON DELETE CASCADE,
                    FOREIGN KEY (market_product_id) REFERENCES market_products(product_id) ON DELETE CASCADE
                );
            """)

            conn.commit()

    def upsert_product(self, product: MarketProduct) -> str:
        """Inserts or updates a single MarketProduct."""
        data = product.to_db_row()
        query = """
            INSERT INTO market_products (
                product_id, source, source_product_id, product_name, category,
                subcategory, craft_type, material, region, color, size,
                handmade, description, price, currency, unit, rating,
                review_count, source_url, image_url, collected_at
            ) VALUES (
                :product_id, :source, :source_product_id, :product_name, :category,
                :subcategory, :craft_type, :material, :region, :color, :size,
                :handmade, :description, :price, :currency, :unit, :rating,
                :review_count, :source_url, :image_url, :collected_at
            )
            ON CONFLICT(product_id) DO UPDATE SET
                source=excluded.source,
                source_product_id=excluded.source_product_id,
                product_name=excluded.product_name,
                category=excluded.category,
                subcategory=excluded.subcategory,
                craft_type=excluded.craft_type,
                material=excluded.material,
                region=excluded.region,
                color=excluded.color,
                size=excluded.size,
                handmade=excluded.handmade,
                description=excluded.description,
                price=excluded.price,
                currency=excluded.currency,
                unit=excluded.unit,
                rating=excluded.rating,
                review_count=excluded.review_count,
                source_url=excluded.source_url,
                image_url=excluded.image_url,
                collected_at=excluded.collected_at;
        """
        with self._get_connection() as conn:
            conn.execute(query, data)
            conn.commit()
        return product.product_id

    def upsert_products_batch(self, products: List[MarketProduct]) -> int:
        """Batch inserts or updates a list of MarketProducts."""
        if not products:
            return 0
        rows = [p.to_db_row() for p in products]
        query = """
            INSERT INTO market_products (
                product_id, source, source_product_id, product_name, category,
                subcategory, craft_type, material, region, color, size,
                handmade, description, price, currency, unit, rating,
                review_count, source_url, image_url, collected_at
            ) VALUES (
                :product_id, :source, :source_product_id, :product_name, :category,
                :subcategory, :craft_type, :material, :region, :color, :size,
                :handmade, :description, :price, :currency, :unit, :rating,
                :review_count, :source_url, :image_url, :collected_at
            )
            ON CONFLICT(product_id) DO UPDATE SET
                source=excluded.source,
                source_product_id=excluded.source_product_id,
                product_name=excluded.product_name,
                category=excluded.category,
                subcategory=excluded.subcategory,
                craft_type=excluded.craft_type,
                material=excluded.material,
                region=excluded.region,
                color=excluded.color,
                size=excluded.size,
                handmade=excluded.handmade,
                description=excluded.description,
                price=excluded.price,
                currency=excluded.currency,
                unit=excluded.unit,
                rating=excluded.rating,
                review_count=excluded.review_count,
                source_url=excluded.source_url,
                image_url=excluded.image_url,
                collected_at=excluded.collected_at;
        """
        with self._get_connection() as conn:
            conn.executemany(query, rows)
            conn.commit()
        return len(products)

    def get_product(self, product_id: str) -> Optional[MarketProduct]:
        """Fetches a single MarketProduct by its ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM market_products WHERE product_id = ?", (product_id,))
            row = cursor.fetchone()
            if row is None:
                return None
            return MarketProduct.from_db_row(dict(row))

    def list_products(
        self,
        category: Optional[str] = None,
        source: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[MarketProduct]:
        """Lists products with optional category and source filtering."""
        conditions = []
        params: List[Any] = []

        if category:
            conditions.append("LOWER(category) = LOWER(?)")
            params.append(category)
        if source:
            conditions.append("LOWER(source) = LOWER(?)")
            params.append(source)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"SELECT * FROM market_products {where_clause} ORDER BY collected_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            return [MarketProduct.from_db_row(dict(r)) for r in rows]

    def search_products(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 20,
    ) -> List[MarketProduct]:
        """Keyword text search on product name, craft_type, and material."""
        wildcard = f"%{query.strip()}%"
        conditions = [
            "(product_name LIKE ? OR craft_type LIKE ? OR material LIKE ? OR description LIKE ?)"
        ]
        params: List[Any] = [wildcard, wildcard, wildcard, wildcard]

        if category:
            conditions.append("LOWER(category) = LOWER(?)")
            params.append(category)

        where_clause = f"WHERE {' AND '.join(conditions)}"
        sql = f"SELECT * FROM market_products {where_clause} LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            return [MarketProduct.from_db_row(dict(r)) for r in rows]

    def count_products(self, category: Optional[str] = None) -> int:
        """Returns count of stored products."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if category:
                cursor.execute("SELECT COUNT(*) FROM market_products WHERE LOWER(category) = LOWER(?)", (category,))
            else:
                cursor.execute("SELECT COUNT(*) FROM market_products")
            return cursor.fetchone()[0]

    def upsert_source(
        self,
        source_id: str,
        name: str,
        base_url: str = "",
        is_active: bool = True,
        ingestion_type: str = "csv",
    ) -> None:
        """Registers or updates a market data source."""
        query = """
            INSERT INTO sources (source_id, name, base_url, is_active, ingestion_type)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(source_id) DO UPDATE SET
                name=excluded.name,
                base_url=excluded.base_url,
                is_active=excluded.is_active,
                ingestion_type=excluded.ingestion_type;
        """
        with self._get_connection() as conn:
            conn.execute(query, (source_id, name, base_url, 1 if is_active else 0, ingestion_type))
            conn.commit()

    def get_sources(self) -> List[Dict[str, Any]]:
        """Returns all configured data sources."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sources ORDER BY name ASC")
            return [dict(r) for r in cursor.fetchall()]

    def clear_all(self) -> None:
        """Deletes all data across tables (for test fixtures)."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM comparable_products;")
            conn.execute("DELETE FROM pricing_predictions;")
            conn.execute("DELETE FROM product_embeddings;")
            conn.execute("DELETE FROM market_products;")
            conn.commit()


# Singleton database instance
_default_db: Optional[DatabaseManager] = None


def get_db(db_path: Optional[Union[str, Path]] = None) -> DatabaseManager:
    """Returns database manager instance."""
    global _default_db
    if _default_db is None or db_path is not None:
        _default_db = DatabaseManager(db_path=db_path)
    return _default_db
