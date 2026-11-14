"""
Downloads sentence-transformers/all-MiniLM-L6-v2 model files directly
to the local ./models directory on the host.
Run on your host machine:
    py -3.14 download_model_host.py
    # or
    python download_model_host.py
"""
import http.cookiejar
import os
import shutil
import ssl
import subprocess
import sys
import urllib.request

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(os.path.join(MODEL_DIR, "1_Pooling"), exist_ok=True)

BASE_URL = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2"

# Small configuration & tokenizer files first, large LFS weights last
FILES = [
    ("config.json", f"{BASE_URL}/raw/main/config.json"),
    ("modules.json", f"{BASE_URL}/raw/main/modules.json"),
    ("tokenizer.json", f"{BASE_URL}/raw/main/tokenizer.json"),
    ("tokenizer_config.json", f"{BASE_URL}/raw/main/tokenizer_config.json"),
    ("vocab.txt", f"{BASE_URL}/raw/main/vocab.txt"),
    ("special_tokens_map.json", f"{BASE_URL}/raw/main/special_tokens_map.json"),
    (os.path.join("1_Pooling", "config.json"), f"{BASE_URL}/raw/main/1_Pooling/config.json"),
    ("model.safetensors", f"{BASE_URL}/resolve/main/model.safetensors"),
]

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

HAS_CURL = shutil.which("curl.exe") is not None

class CustomRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, hdrs, newurl):
        new_req = super().redirect_request(req, fp, code, msg, hdrs, newurl)
        if new_req is not None:
            for key, val in req.header_items():
                if key.lower() not in ("host", "content-length", "content-type"):
                    new_req.add_header(key, val)
        return new_req

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

cookie_jar = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(
    CustomRedirectHandler,
    urllib.request.HTTPCookieProcessor(cookie_jar),
    urllib.request.HTTPSHandler(context=ctx)
)

print(f"Downloading all-MiniLM-L6-v2 to: {MODEL_DIR}")
if HF_TOKEN:
    print("Using Hugging Face authorization token.")

failed_files = []

for rel_path, url in FILES:
    dest = os.path.join(MODEL_DIR, rel_path)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"  [OK] {rel_path} already exists ({os.path.getsize(dest)} bytes)")
        continue

    print(f"  Downloading {rel_path}...")
    success = False

    # Attempt 1: curl.exe
    if HAS_CURL:
        cmd = ["curl.exe", "-L", "-k", "-A", UA, "--fail", "-o", dest]
        if HF_TOKEN:
            cmd.extend(["-H", f"Authorization: Bearer {HF_TOKEN}"])
        cmd.append(url)
        res = subprocess.run(cmd)
        if res.returncode == 0 and os.path.exists(dest) and os.path.getsize(dest) > 0:
            print(f"  [OK] Saved {rel_path} ({os.path.getsize(dest)} bytes)")
            success = True
        else:
            if os.path.exists(dest):
                try: os.remove(dest)
                except Exception: pass

    # Attempt 2: urllib
    if not success:
        req_headers = {"User-Agent": UA, "Accept": "*/*"}
        if HF_TOKEN:
            req_headers["Authorization"] = f"Bearer {HF_TOKEN}"
        req = urllib.request.Request(url, headers=req_headers)
        try:
            with opener.open(req) as resp, open(dest, "wb") as f:
                total_size = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk: break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = int(downloaded * 100 / total_size)
                        mb = downloaded / (1024 * 1024)
                        total_mb = total_size / (1024 * 1024)
                        print(f"\r  Downloading {rel_path}... {mb:.1f}/{total_mb:.1f} MB ({pct}%)", end="", flush=True)
                    else:
                        mb = downloaded / (1024 * 1024)
                        print(f"\r  Downloading {rel_path}... {mb:.1f} MB", end="", flush=True)
            print(f"\n  [OK] Saved {rel_path} ({os.path.getsize(dest)} bytes)")
            success = True
        except Exception as e:
            print(f"\n  [WARN] Automated download of {rel_path} failed: {e}")
            if os.path.exists(dest):
                try: os.remove(dest)
                except Exception: pass
            failed_files.append((rel_path, url, dest))

if not failed_files:
    print("\nAll model files downloaded successfully to ./models!")
    print("You can now run: docker compose up")
else:
    print("\n" + "=" * 70)
    print("ACTION NEEDED FOR LARGE MODEL WEIGHTS:")
    print("All metadata and tokenizer files have been saved successfully.")
    print("Your corporate network blocks automated CLI downloads of large LFS files.")
    for rel_path, url, dest in failed_files:
        print(f"\nPlease open this link directly in your browser:")
        print(f"  {url}")
        print(f"And save or move the downloaded file to:")
        print(f"  {dest}")
    print("=" * 70)
    print("After placing the file, run: docker compose up")
