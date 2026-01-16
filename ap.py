from fastapi import FastAPI
from pydantic import BaseModel
import os
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from fastapi.middleware.cors import CORSMiddleware


# -------------------- APP SETUP --------------------

app = FastAPI(title="Government Scheme Recommendation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------- LOAD MODEL & DATA --------------------

metadata_path = os.path.join(BASE_DIR, "metadata.csv")
index_path = os.path.join(BASE_DIR, "scheme.index")

# Load FAISS index
index = faiss.read_index(index_path)

# Load metadata CSV
metadata = pd.read_csv(metadata_path)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------- REQUEST SCHEMA --------------------

class SearchRequest(BaseModel):
    query: str

# -------------------- SEARCH ENDPOINT --------------------

@app.post("/search")
def search_schemes(req: SearchRequest):
    """
    Accepts a natural-language query and returns top recommended schemes
    using FAISS + Sentence Transformers.
    """

    try:
        # Encode query
        query_embedding = model.encode([req.query])

        # Search FAISS index
        distances, indices = index.search(query_embedding, 5)

        # Validate FAISS indices
        valid_indices = [
            int(i) for i in indices[0]
            if i >= 0 and i < len(metadata)
        ]

        # No valid results
        if not valid_indices:
            return {
                "schemes": []
            }

        # Fetch metadata safely
        results = metadata.iloc[valid_indices].to_dict(orient="records")

        return {
            "schemes": results
        }

    except Exception as e:
        # Safe error handling (prevents 500 crash)
        return {
            "schemes": [],
            "error": str(e)
        }

# -------------------- ROOT (OPTIONAL) --------------------

@app.get("/")
def root():
    return {
        "message": "Government Scheme Recommendation API is running"
    }
