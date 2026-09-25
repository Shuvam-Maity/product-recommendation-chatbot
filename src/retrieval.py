"""
retrieval.py
Builds a FAISS index over the product catalog and provides hybrid retrieval:
metadata filtering (category, price) + semantic similarity search.
"""

import pandas as pd
import numpy as np
import json
import faiss
from sentence_transformers import SentenceTransformer

CATALOG_PATH = "data/processed/catalog.csv"
INDEX_PATH = "data/processed/catalog.index"
MODEL_NAME = "all-MiniLM-L6-v2"


class ProductRetriever:
    def __init__(self, catalog_path=CATALOG_PATH):
        self.df = pd.read_csv(catalog_path)
        self.df["tags"] = self.df["tags"].apply(json.loads)
        self.df["specs"] = self.df["specs"].apply(json.loads)

        self.model = SentenceTransformer(MODEL_NAME)
        self.index = None
        self._build_index()

    def _build_index(self):
        # Combine description + tags into one text blob per product for embedding
        texts = (
            self.df["title"] + ". " +
            self.df["description"] + " Tags: " +
            self.df["tags"].apply(lambda t: ", ".join(t))
        ).tolist()

        embeddings = self.model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype="float32")

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors = cosine similarity
        self.index.add(embeddings)
        self.embeddings = embeddings

    def _apply_filters(self, category=None, max_price=None, min_price=None, brand=None):
        mask = pd.Series(True, index=self.df.index)
        if category:
            mask &= self.df["category"] == category
        if max_price is not None:
            mask &= self.df["price"] <= max_price
        if min_price is not None:
            mask &= self.df["price"] >= min_price
        if brand:
            mask &= self.df["brand"].str.contains(brand, case=False, na=False)
        return mask

    def search(self, query, category=None, max_price=None, min_price=None,
               brand=None, top_k=5):
        """
        Hybrid retrieval: filter by metadata first, then rank the filtered
        subset by semantic similarity to the query.
        """
        mask = self._apply_filters(category, max_price, min_price, brand)
        candidate_idx = self.df.index[mask].tolist()

        if not candidate_idx:
            return pd.DataFrame(columns=self.df.columns)

        query_vec = self.model.encode([query], normalize_embeddings=True).astype("float32")

        candidate_embeddings = self.embeddings[candidate_idx]
        scores = candidate_embeddings @ query_vec[0]  # cosine similarity (normalized vectors)

        ranked = sorted(zip(candidate_idx, scores), key=lambda x: x[1], reverse=True)
        top_idx = [idx for idx, _ in ranked[:top_k]]

        result = self.df.loc[top_idx].copy()
        result["similarity"] = [score for _, score in ranked[:top_k]]
        return result


if __name__ == "__main__":
    retriever = ProductRetriever()

    print("\n--- Test 1: Black shirts for Goa trip ---")
    results = retriever.search(
        query="black casual shirt for a warm beach vacation",
        category="clothing",
        top_k=5
    )
    print(results[["title", "brand", "price", "similarity"]])

    print("\n--- Test 2: Camera phone under 40000 ---")
    results = retriever.search(
        query="best camera phone for photography",
        category="smartphone",
        max_price=40000,
        top_k=5
    )
    print(results[["title", "brand", "price", "similarity"]])

    print("\n--- Test 3: Laptop for engineering ---")
    results = retriever.search(
        query="laptop for programming and computer engineering studies",
        category="laptop",
        top_k=5
    )
    print(results[["title", "brand", "price", "similarity"]])