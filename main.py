import os
import uuid

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

load_dotenv()

EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "http://localhost:8080/v1")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8081/v1")
API_KEY = os.getenv("API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

COLLECTION_NAME = "documents"

app = FastAPI()

qdrant = QdrantClient(url=QDRANT_URL)


def ensure_collection(vector_size: int):
    collections = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME not in collections:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


async def get_embeddings(texts: list[str]) -> list[list[float]]:
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{EMBEDDING_BASE_URL}/embeddings",
            json={"input": texts, "model": EMBEDDING_MODEL},
            headers={"Authorization": f"Bearer {API_KEY}"},
        )
        resp.raise_for_status()
        data = resp.json()
        return [item["embedding"] for item in data["data"]]


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return [c for c in chunks if c.strip()]


@app.post("/api/upload")
async def upload_document(
    file: UploadFile = File(...),
    chunk_size: int = Form(500),
    chunk_overlap: int = Form(50),
):
    content = await file.read()
    text = content.decode("utf-8")
    filename = file.filename or "unknown.md"

    chunks = chunk_text(text, chunk_size, chunk_overlap)
    if not chunks:
        return JSONResponse({"error": "No content to index"}, status_code=400)

    embeddings = await get_embeddings(chunks)

    ensure_collection(len(embeddings[0]))

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=emb,
            payload={"text": chunk, "filename": filename, "chunk_index": i},
        )
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
    ]

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

    return {"message": f"Indexed {len(chunks)} chunks from {filename}"}


@app.post("/api/search")
async def search(query: str = Form(...), top_k: int = Form(5)):
    embeddings = await get_embeddings([query])
    query_vector = embeddings[0]

    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )

    return {
        "results": [
            {
                "text": point.payload.get("text", ""),
                "filename": point.payload.get("filename", ""),
                "chunk_index": point.payload.get("chunk_index", 0),
                "score": point.score,
            }
            for point in results.points
        ]
    }


@app.get("/api/settings")
async def get_settings():
    return {
        "embedding_base_url": EMBEDDING_BASE_URL,
        "llm_base_url": LLM_BASE_URL,
        "embedding_model": EMBEDDING_MODEL,
        "qdrant_url": QDRANT_URL,
    }


app.mount("/", StaticFiles(directory="static", html=True), name="static")
