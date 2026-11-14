"""
Embeddings that live in ANOTHER container.

Instead of loading the model inside this app (HuggingFaceEmbeddings), we call the
embedding-service over HTTP. Same model, same vectors, different place.
"""
import os

import httpx
from langchain_core.embeddings import Embeddings

EMBEDDING_URL = os.getenv("EMBEDDING_URL", "http://localhost:8001")


class RemoteEmbeddings(Embeddings):
    def __init__(self, url: str = EMBEDDING_URL, batch_size: int = 64):
        self.url = url.rstrip("/")
        self.batch_size = batch_size

    def _embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            try:
                resp = httpx.post(
                    f"{self.url}/embed",
                    json={"texts": texts[i : i + self.batch_size]},
                    timeout=60,
                )
                resp.raise_for_status()
                vectors.extend(resp.json()["embeddings"])
            except httpx.ConnectError as e:
                raise RuntimeError(
                    f"[EMBEDDINGS CONNECTION ERROR] Could not reach embedding-service at '{self.url}/embed'. "
                    f"Ensure the 'embeddings' container is running (check 'docker compose ps'). Detail: {e}"
                ) from e
            except httpx.HTTPStatusError as e:
                raise RuntimeError(
                    f"[EMBEDDINGS HTTP ERROR] Embedding service at '{self.url}/embed' returned status {e.response.status_code}: {e.response.text}"
                ) from e
            except Exception as e:
                raise RuntimeError(
                    f"[EMBEDDINGS ERROR] Unexpected error while communicating with '{self.url}/embed': {e}"
                ) from e
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]
