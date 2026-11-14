import os
import ssl
import warnings

warnings.filterwarnings("ignore")

# Corporate proxy bypass for Hugging Face downloads
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"
ssl._create_default_https_context = ssl._create_unverified_context

BROWSER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"

# 1. Monkeypatch huggingface_hub header builder so it doesn't send python/huggingface User-Agent
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

from huggingface_hub.utils import httpx, set_client_factory

# 2. Forcibly inject browser headers on every request right before it hits the wire
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

client_kwargs = {
    "verify": False,
    "follow_redirects": True,
    "timeout": None,
    "event_hooks": hooks,
}

set_client_factory(lambda: httpx.Client(**client_kwargs))

try:
    from huggingface_hub.utils import set_async_client_factory
    set_async_client_factory(lambda: httpx.AsyncClient(**client_kwargs))
except Exception:
    pass

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
IGNORE_PATTERNS = ["*.onnx", "*.h5", "*.ot", "*.msgpack", "onnx/*", "openvino/*"]

# 3. Test probe
probe_url = f"https://huggingface.co/api/models/{MODEL_ID}/revision/main"
print(f"Testing connectivity to {probe_url} with browser identity...")
try:
    with httpx.Client(**client_kwargs) as client:
        r = client.get(probe_url)
        print(f"Probe response status: {r.status_code}")
        if r.status_code != 200:
            print(f"Probe server/headers: {dict(r.headers)}")
            print(f"Probe body snippet: {r.text[:300]}")
except Exception as e:
    print(f"Probe error: {e}")

from huggingface_hub import snapshot_download

try:
    print(f"Downloading {MODEL_ID} from https://huggingface.co...")
    snapshot_download(
        MODEL_ID,
        headers={"User-Agent": BROWSER_UA},
        user_agent=BROWSER_UA,
        ignore_patterns=IGNORE_PATTERNS,
    )
    print("Model download complete.")
except Exception as e:
    print(f"\n[WARNING] Could not pre-download model during Docker build: {e}")
    print("[INFO] Build will continue. The model will load from volume or at runtime.\n")
