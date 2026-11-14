import os
import warnings

# In corporate/proxy networks, bypass TLS verification and inject browser headers for Hugging Face
os.environ.setdefault("HF_HUB_DISABLE_SSL_VERIFICATION", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

BROWSER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"

try:
    import huggingface_hub.constants as constants
    constants.USER_AGENT = BROWSER_UA
except Exception:
    pass

try:
    import huggingface_hub.utils._headers as _h
    _orig_build_hf_headers = _h.build_hf_headers
    def _custom_build_hf_headers(*args, **kwargs):
        h = _orig_build_hf_headers(*args, **kwargs)
        h["user-agent"] = BROWSER_UA
        return h
    _h.build_hf_headers = _custom_build_hf_headers
except Exception:
    pass

try:
    from huggingface_hub.utils import httpx, set_client_factory
    def force_browser_headers_hook(request):
        request.headers["user-agent"] = BROWSER_UA
        request.headers["accept"] = "application/json, text/plain, */*"
        request.headers["sec-ch-ua"] = '"Chromium";v="130", "Google Chrome";v="130", "Not?A_Brand";v="99"'
        request.headers["sec-ch-ua-mobile"] = "?0"
        request.headers["sec-ch-ua-platform"] = '"Windows"'
    try:
        from huggingface_hub.utils._http import hf_request_event_hook
        hooks = {"request": [hf_request_event_hook, force_browser_headers_hook]}
    except Exception:
        hooks = {"request": [force_browser_headers_hook]}
    kwargs = {"verify": False, "follow_redirects": True, "timeout": None, "event_hooks": hooks}
    set_client_factory(lambda: httpx.Client(**kwargs))
except Exception:
    pass

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DEVICE = os.getenv("DEVICE", "cpu")  # "cpu" or "cuda"

app = FastAPI(title="embedding-service")

# Loaded ONCE at startup, then reused for every request.
model = SentenceTransformer(MODEL_NAME, device=DEVICE)


class EmbedRequest(BaseModel):
    texts: list[str]


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "device": str(model.device)}


@app.post("/embed")
def embed(req: EmbedRequest):
    vectors = model.encode(req.texts, convert_to_numpy=True)
    return {"embeddings": vectors.tolist(), "dim": vectors.shape[1]}
