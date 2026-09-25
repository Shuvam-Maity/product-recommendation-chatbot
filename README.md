# 🛍️ Virtual Sales Associate — Intelligent Product Recommendation Chatbot

An AI-powered conversational chatbot that acts as a virtual sales associate for e-commerce, helping customers discover, compare, and select products through natural language — instead of relying on keyword search and filters.

Ask it things like:
- *"Suggest some shirts for my upcoming Goa trip"*
- *"Recommend the best camera phone under ₹40,000"*
- *"Compare iPhone 16 Pro and Samsung S25 Ultra"*
- *"Help me buy a laptop for Computer Engineering studies"*
- *"I am joining a gym next week. What shoes and clothes should I buy?"*

The bot understands intent, asks follow-up questions when details are missing, retrieves relevant products from a catalog, and explains *why* each recommendation fits — rather than just listing search results.

---

## How it works

```
User message
     │
     ▼
Intent & Slot Extraction (Gemini)
  → category, budget, use-case, comparison targets, missing info
     │
     ▼
Missing info? ──yes──► Ask a follow-up question, wait for reply, merge into intent
     │ no
     ▼
Hybrid Retrieval (FAISS + metadata filters)
  → semantic search over product embeddings + price/category filtering
     │
     ▼
LLM Reasoning & Generation (Gemini)
  → ranks candidates by actual fit, explains why, or builds a comparison table
     │
     ▼
Response shown in a Streamlit chat interface
```

**Core components:**
| Module | Responsibility |
|---|---|
| `src/data_prep.py` | Cleans and unifies phone, laptop, and fashion datasets into one product catalog |
| `src/retrieval.py` | Builds sentence-embeddings + FAISS index; hybrid semantic + metadata search |
| `src/intent_extraction.py` | Uses Gemini to parse raw messages into structured shopping intent |
| `src/chatbot.py` | Generates recommendations/comparisons grounded in retrieved candidates |
| `src/dialogue.py` | Manages multi-turn conversation state and follow-up slot-filling |
| `app/main.py` | Streamlit chat UI tying everything together |

---

## Tech stack

- **LLM**: Google Gemini API (`gemini-3.8-flash` / `gemini-3.5-flash-lite`, with automatic fallback between them)
- **Embeddings**: `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Vector search**: FAISS
- **UI**: Streamlit
- **Data**: pandas / numpy

---

## Product catalog

A unified catalog of **480 products** across 4 categories (120 each), built from real public datasets:

| Category | Source |
|---|---|
| Smartphones | [Smartphone Dataset](https://www.kaggle.com/datasets/muzammilbaloch/smartphone_dataset) (Kaggle) — real INR pricing |
| Laptops | Laptop specifications/pricing dataset (Kaggle) |
| Clothing | [Fashion Product Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset) — `styles.csv` metadata only (no images used) |
| Footwear | Same fashion dataset, filtered to `Footwear` |

**Disclosed assumptions** (for transparency):
- Clothing/footwear prices are **synthetically generated** within realistic ranges per category and usage type, since the source dataset has no price column.
- Clothing/footwear "brand" is inferred with a simple heuristic from product titles (may be noisy in places).
- Product comparisons for items *not* in the catalog (e.g. very recent flagship phones) fall back to the LLM's general knowledge, with an explicit disclosure in the response.

---

## Setup

### 1. Clone and create a virtual environment
```bash
git clone https://github.com/Shuvam-Maity/product-recommendation-chatbot.git
cd product-recommendation-chatbot
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get a free Gemini API key
- Go to [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
- Sign in and create a key

### 4. Add your key
Create a `.env` file in the project root:
```
GEMINI_API_KEY=your_key_here
```

### 5. (Optional) Rebuild the catalog from raw data
The processed catalog is already included in `data/processed/catalog.csv`, so this step is optional. To rebuild it from scratch:
```bash
python src/data_prep.py
```

### 6. Run the app
```bash
streamlit run app/main.py
```
Opens at `http://localhost:8501`.

---

## Project structure

```
product-recommendation-chatbot/
├── data/
│   ├── raw/            # original downloaded datasets (not tracked in git)
│   └── processed/      # unified catalog.csv (tracked)
├── src/
│   ├── data_prep.py
│   ├── retrieval.py
│   ├── intent_extraction.py
│   ├── chatbot.py
│   └── dialogue.py
├── app/
│   └── main.py
├── requirements.txt
└── README.md
```

---

## Known limitations

- **Free-tier Gemini rate limits** (5 requests/min on the primary model) mean occasional short delays; the app automatically falls back to a secondary model when rate-limited.
- **Catalog size is intentionally small** (480 products) for a fast, demoable portfolio project — not representative of a production-scale catalog.
- **Semantic retrieval alone doesn't rank by fine-grained spec fitness** (e.g. RAM, camera MP) — this is handled by the LLM reasoning layer on top of retrieved candidates, not by the retrieval step itself.

---

## Author

**Shuvam Maity**
M.Sc. Data Science, St. Xavier's College, Kolkata
[LinkedIn](https://www.linkedin.com/in/shuvam-maity) · [GitHub](https://github.com/Shuvam-Maity)
