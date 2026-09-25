"""
data_prep.py
Cleans and unifies phone, laptop, and fashion (clothing + footwear) datasets
into a single product catalog with a common schema:
    product_id, category, title, brand, price, description, tags, specs
"""

import pandas as pd
import numpy as np
import json
import random

random.seed(42)
np.random.seed(42)

RAW_DIR = "data/raw"
OUT_PATH = "data/processed/catalog.csv"

# Target sample size per category
SAMPLE_SIZE = 120


# ---------------------------------------------------------------------------
# 1. LAPTOPS
# ---------------------------------------------------------------------------
def load_laptops():
    df = pd.read_csv(f"{RAW_DIR}/laptop1.csv")
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])

    # Drop rows with missing essentials
    df = df.dropna(subset=["brand", "name", "price"])

    # Sample across brands for diversity
    df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42).reset_index(drop=True)

    records = []
    for i, row in df.iterrows():
        specs = {
            "processor": row.get("processor"),
            "cpu": row.get("CPU"),
            "ram": row.get("Ram"),
            "ram_type": row.get("Ram_type"),
            "storage": row.get("ROM"),
            "storage_type": row.get("ROM_type"),
            "gpu": row.get("GPU"),
            "display_size": row.get("display_size"),
            "resolution": f"{row.get('resolution_width')}x{row.get('resolution_height')}",
            "os": row.get("OS"),
            "warranty_years": row.get("warranty"),
        }

        name_lower = str(row["name"]).lower()
        tags = []
        if "gaming" in name_lower:
            tags.append("gaming")
        if "ultrabook" in name_lower or "thin" in name_lower:
            tags.append("ultrabook")
        if "business" in name_lower:
            tags.append("business")
        tags.append("laptop")

        description = (
            f"{row['brand']} {row['name']} with {row.get('processor', 'a processor')}, "
            f"{row.get('Ram', '?')} RAM, {row.get('ROM', '?')} storage, "
            f"{row.get('GPU', 'integrated graphics')}, running {row.get('OS', 'an OS')}."
        )

        records.append({
            "product_id": f"laptop_{i:04d}",
            "category": "laptop",
            "title": row["name"],
            "brand": row["brand"],
            "price": row["price"],
            "description": description,
            "tags": json.dumps(tags),
            "specs": json.dumps(specs, default=str),
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 2. SMARTPHONES
# ---------------------------------------------------------------------------
def load_phones():
    df = pd.read_csv(f"{RAW_DIR}/phone2.csv")
    df = df.dropna(subset=["brand_name", "model", "price"])

    df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42).reset_index(drop=True)

    records = []
    for i, row in df.iterrows():
        specs = {
            "processor_brand": row.get("processor_brand"),
            "num_cores": row.get("num_cores"),
            "processor_speed": row.get("processor_speed"),
            "battery_capacity": row.get("battery_capacity"),
            "fast_charging": row.get("fast_charging"),
            "ram": row.get("ram_capacity"),
            "internal_memory": row.get("internal_memory"),
            "screen_size": row.get("screen_size"),
            "refresh_rate": row.get("refresh_rate"),
            "num_rear_cameras": row.get("num_rear_cameras"),
            "os": row.get("os"),
            "primary_camera_rear": row.get("primary_camera_rear"),
            "primary_camera_front": row.get("primary_camera_front"),
            "5g": bool(row.get("5G_or_not")),
            "rating": row.get("avg_rating"),
        }

        tags = []
        if row.get("5G_or_not"):
            tags.append("5g")
        rear_cam = row.get("primary_camera_rear")
        if pd.notna(rear_cam) and rear_cam >= 48:
            tags.append("camera-focused")
        if pd.notna(row.get("refresh_rate")) and row.get("refresh_rate") >= 90:
            tags.append("high-refresh-display")
        if pd.notna(row.get("battery_capacity")) and row.get("battery_capacity") >= 5000:
            tags.append("long-battery")
        tags.append("smartphone")

        description = (
            f"{row['brand_name'].title()} {row['model']} with {row.get('ram_capacity', '?')}GB RAM, "
            f"{row.get('primary_camera_rear', '?')}MP rear camera, "
            f"{row.get('battery_capacity', '?')}mAh battery, "
            f"{'5G' if row.get('5G_or_not') else '4G'} support, running {row.get('os', 'Android')}."
        )

        records.append({
            "product_id": f"phone_{i:04d}",
            "category": "smartphone",
            "title": row["model"],
            "brand": row["brand_name"].title(),
            "price": row["price"],
            "description": description,
            "tags": json.dumps(tags),
            "specs": json.dumps(specs, default=str),
        })

    return pd.DataFrame(records)

# ---------------------------------------------------------------------------
# 3 & 4. CLOTHING (Shirts/Tshirts) + FOOTWEAR — from fashion dataset
# ---------------------------------------------------------------------------
def synth_price(category, usage):
    """Rough synthetic price ranges since this dataset has no price column."""
    ranges = {
        ("clothing", "Casual"): (400, 1500),
        ("clothing", "Formal"): (800, 2500),
        ("clothing", "Sports"): (500, 2000),
        ("footwear", "Casual"): (700, 3000),
        ("footwear", "Formal"): (1200, 4000),
        ("footwear", "Sports"): (1000, 5000),
    }
    lo, hi = ranges.get((category, usage), (500, 2000))
    return random.randint(lo, hi)


def guess_brand(name):
    """Very rough heuristic: take leading words before a gender token as 'brand'."""
    for token in ["Men", "Women", "Boys", "Girls", "Unisex"]:
        if token in name:
            prefix = name.split(token)[0].strip()
            return prefix if prefix else "Generic"
    return "Generic"


def load_fashion():
    df = pd.read_csv(f"{RAW_DIR}/styles.csv", on_bad_lines="skip", engine="python")
    df = df.dropna(subset=["productDisplayName", "articleType", "baseColour"])

    clothing_df = df[
        (df["subCategory"] == "Topwear") &
        (df["articleType"].isin(["Shirts", "Tshirts"]))
    ].sample(n=min(SAMPLE_SIZE, len(df)), random_state=42).reset_index(drop=True)

    footwear_df = df[
        df["masterCategory"] == "Footwear"
    ].sample(n=min(SAMPLE_SIZE, len(df)), random_state=43).reset_index(drop=True)

    records = []

    for i, row in clothing_df.iterrows():
        brand = guess_brand(row["productDisplayName"])
        price = synth_price("clothing", row.get("usage", "Casual"))
        tags = [row["gender"], row["usage"], row["season"], row["baseColour"], row["articleType"]]
        tags = [str(t) for t in tags if pd.notna(t)]

        description = (
            f"{row['gender']} {row['usage']} {row['articleType']} in {row['baseColour']}, "
            f"suitable for {row['season']} season."
        )

        specs = {
            "gender": row["gender"],
            "article_type": row["articleType"],
            "color": row["baseColour"],
            "season": row["season"],
            "usage": row["usage"],
        }

        records.append({
            "product_id": f"clothing_{i:04d}",
            "category": "clothing",
            "title": row["productDisplayName"],
            "brand": brand,
            "price": price,
            "description": description,
            "tags": json.dumps(tags),
            "specs": json.dumps(specs, default=str),
        })

    for i, row in footwear_df.iterrows():
        brand = guess_brand(row["productDisplayName"])
        price = synth_price("footwear", row.get("usage", "Casual"))
        tags = [row["gender"], row["usage"], row["season"], row["baseColour"], row["articleType"]]
        tags = [str(t) for t in tags if pd.notna(t)]

        description = (
            f"{row['gender']} {row['usage']} {row['articleType']} in {row['baseColour']}, "
            f"suitable for {row['season']} season."
        )

        specs = {
            "gender": row["gender"],
            "article_type": row["articleType"],
            "color": row["baseColour"],
            "season": row["season"],
            "usage": row["usage"],
        }

        records.append({
            "product_id": f"footwear_{i:04d}",
            "category": "footwear",
            "title": row["productDisplayName"],
            "brand": brand,
            "price": price,
            "description": description,
            "tags": json.dumps(tags),
            "specs": json.dumps(specs, default=str),
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("Loading laptops...")
    laptops = load_laptops()
    print(f"  -> {len(laptops)} rows")

    print("Loading phones...")
    phones = load_phones()
    print(f"  -> {len(phones)} rows")

    print("Loading fashion (clothing + footwear)...")
    fashion = load_fashion()
    print(f"  -> {len(fashion)} rows")

    catalog = pd.concat([laptops, phones, fashion], ignore_index=True)
    catalog.to_csv(OUT_PATH, index=False)

    print(f"\nSaved unified catalog: {len(catalog)} rows -> {OUT_PATH}")
    print(catalog["category"].value_counts())


if __name__ == "__main__":
    main()