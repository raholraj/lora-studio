# ============================================================================
#  LoRA Studio server  (Colab pe chalao: GPU aur Google Drive yahin se use hota hai)
#
#  Kya karta hai:
#   1. Model + Drive ki LoRA files load karta hai
#   2. Ek public link banata hai (cloudflared) -> phone app us link se judta hai
#   3. Phone se prompt aata hai -> yahan GPU par image banti hai -> phone ko wapas
#
#  Sirf Python ki apni library use hoti hai (FastAPI/Gradio nahi), isliye install kam.
# ============================================================================
import os, io, re, sys, json, time, uuid, base64, random, secrets, threading, subprocess
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from collections import OrderedDict

PORT = int(os.environ.get("STUDIO_PORT", "8000"))
MOCK = os.environ.get("STUDIO_MOCK") == "1"          # test mode: bina GPU ke
MODEL_ID = "SG161222/Realistic_Vision_V5.1_noVAE"
VAE_ID = "stabilityai/sd-vae-ft-mse"
LLM_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_QUEUE = 6                                         # ek saath itne hi log line me
JOB_TTL = 60 * 30

EMBEDDED_ASSETS = None  # BUILD_INJECT (build.py yaha app ki files daal deta hai)

# ----------------------------------------------------------------------------
# Safety: yeh app sirf safe images ke liye hai. Prompt filter + negative prompt
# + diffusers ka safety checker, teeno on rehte hain.
# ----------------------------------------------------------------------------
BLOCK_RE = re.compile(
    r"\b(nude|nudes|nudity|naked|nsfw|porn\w*|explicit|topless|bottomless|genitals?|erotic\w*|fetish\w*|"
    r"hentai|loli\w*|shota|underage|nanga|nangi|nagn|chudai|xxx)\b", re.I)

QUALITY = ("RAW photo, 8k uhd, dslr, soft natural lighting, high quality, film grain, Fujifilm XT3, "
           "sharp focus, natural skin texture, visible skin pores, realistic, photorealistic")
SAFE_NEG = "nsfw, nude, naked, nudity, explicit"
BASE_NEG = ("(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime)1.4, "
            "(worst quality, low quality)1.4, plastic skin, airbrushed, oversmooth, waxy skin, over-sharpened, "
            "oversaturated, blurry, deformed, bad anatomy, bad proportions, bad hands, extra fingers, missing fingers, "
            "fused fingers, mutated hands, poorly drawn face, asymmetric eyes, cross-eyed, cloned face, disfigured, "
            "extra limbs, long neck, cropped, out of frame, duplicate, watermark, text, signature, jpeg artifacts")
SHAPES = {"portrait": (512, 768), "square": (512, 512), "landscape": (768, 512)}

SMART_SYSTEM = (
    "You write prompts for Stable Diffusion photo models. The user's idea may be in English, Hindi or Hinglish. "
    "Rewrite it as ONE English prompt of comma-separated phrases, 40-70 words: subject, age, clothing, pose, "
    "expression, hair, setting/background, lighting, camera/lens, photo style. Describe it like a real candid "
    "photograph, with natural imperfect skin and realistic lighting, not a painting or CGI. Be specific and "
    "consistent, no contradictions. Always keep the person fully clothed and the content safe for work. "
    "Keep the trigger word exactly as given. Output only the prompt, nothing else.")


class UserError(Exception):
    pass


# ----------------------------------------------------------------------------
# LoRA folder helpers (Drive: lora_studio/<naam>/pytorch_lora_weights.safetensors)
# ----------------------------------------------------------------------------
def list_loras(drive_dir):
    out = []
    if os.path.isdir(drive_dir):
        for n in sorted(os.listdir(drive_dir)):
            f = os.path.join(drive_dir, n, "pytorch_lora_weights.safetensors")
            if os.path.exists(f):
                meta = {}
                try:
                    with open(os.path.join(drive_dir, n, "meta.json")) as fh:
                        meta = json.load(fh)
                except Exception:
                    pass
                out.append({"name": n, "trigger": meta.get("trigger", ""), "step": meta.get("step")})
    return out


def a1111_to_compel(t):
    return re.sub(r"\(([^()]+?):\s*([0-9.]+)\)", r"(\1)\2", t)


def clean_params(d):
    """Phone se aayi settings ko check karke saaf karta hai."""
    def num(k, default, lo, hi, cast):
        try:
            v = cast(d.get(k, default))
        except Exception:
            v = default
        return max(lo, min(hi, v))
    p = {
        "prompt": str(d.get("prompt", "")).strip()[:1000],
        "negative": str(d.get("negative", "")).strip()[:500],
        "lora": (d.get("lora") or None),
        "trigger": str(d.get("trigger", "")).strip()[:60],
        "lora_scale": num("lora_scale", 0.5, 0.0, 1.5, float),
        "steps": num("steps", 30, 15, 50, int),
        "guidance": num("guidance", 6.0, 3.0, 12.0, float),
        "seed": num("seed", -1, -1, 2**32 - 1, int),
        "hires": bool(d.get("hires", True)),
        "smart": bool(d.get("smart", False)),
        "shape": d.get("shape") if d.get("shape") in SHAPES else "portrait",
    }
    if not p["prompt"]:
        raise UserError("The prompt is empty.")
    # Sirf prompt aur trigger check hote hain. "Avoid" field me aise words dalna safe hai (wo to hum khud bhi lagate hain).
    m = BLOCK_RE.search(p["prompt"]) or BLOCK_RE.search(p["trigger"])
    if m:
        raise UserError(f'The word "{m.group(0)}" is not allowed in this app. Remove it and try again.')
    return p


# ----------------------------------------------------------------------------
# Engine (asli): Stable Diffusion + LoRA
# ----------------------------------------------------------------------------
class Engine:
    def __init__(self, drive_dir):
        self.drive_dir = drive_dir
        self.lora_key = None
        self.llm = None
        self.gpu = "?"

    def load(self):
        import torch
        from diffusers import (StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, AutoencoderKL,
                               DPMSolverMultistepScheduler)
        from compel import Compel, ReturnedEmbeddingsType
        self.torch = torch
        self.gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU (bahut slow)"
        print("GPU:", self.gpu)
        vae = AutoencoderKL.from_pretrained(VAE_ID, torch_dtype=torch.float16)
        pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, vae=vae, torch_dtype=torch.float16)  # safety ON
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, algorithm_type="dpmsolver++", use_karras_sigmas=True)
        pipe = pipe.to("cuda")
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
        self.pipe = pipe
        self.img2img = StableDiffusionImg2ImgPipeline.from_pipe(pipe)
        self.compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder,
                             returned_embeddings_type=ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED,
                             truncate_long_prompts=False)
        print("Model loaded")

    def loras(self):
        return list_loras(self.drive_dir)

    def _smart(self, idea, trigger):
        try:
            torch = self.torch
            if self.llm is None:
                from transformers import AutoTokenizer, AutoModelForCausalLM
                tok = AutoTokenizer.from_pretrained(LLM_ID)
                model = AutoModelForCausalLM.from_pretrained(LLM_ID, torch_dtype=torch.float16).to("cuda")
                self.llm = (tok, model)
            tok, model = self.llm
            msgs = [{"role": "system", "content": SMART_SYSTEM},
                    {"role": "user", "content": f"Trigger word: {trigger}\nIdea: {idea}"}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inp = tok([text], return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=140, do_sample=True, temperature=0.6, top_p=0.9)
            res = tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip().split("\n")[0]
            if BLOCK_RE.search(res):
                return idea
            return res if len(res) > 10 else idea
        except Exception as e:
            print("Smart prompt fail:", e)
            return idea

    def generate(self, p, progress):
        from PIL import Image
        torch = self.torch
        pipe = self.pipe
        seed = p["seed"] if p["seed"] >= 0 else random.randint(0, 2**32 - 1)

        trigger = p["trigger"]
        prompt = self._smart(p["prompt"], trigger) if p["smart"] else p["prompt"]
        if trigger and trigger.lower() not in prompt.lower():
            prompt = f"{trigger}, {prompt}"
        final_prompt = f"{prompt}, {QUALITY}"
        final_neg = ", ".join(x for x in [p["negative"], BASE_NEG, SAFE_NEG] if x)

        cross = None
        lora = p["lora"]
        wfile = os.path.join(self.drive_dir, lora or "", "pytorch_lora_weights.safetensors")
        if lora and re.fullmatch(r"[A-Za-z0-9_\-]+", lora) and os.path.exists(wfile):
            key = (wfile, os.path.getmtime(wfile))
            if self.lora_key != key:
                try:
                    pipe.unload_lora_weights()
                except Exception:
                    pass
                pipe.load_lora_weights(os.path.join(self.drive_dir, lora))
                self.lora_key = key
            cross = {"scale": p["lora_scale"]}
        elif self.lora_key is not None:
            pipe.unload_lora_weights()
            self.lora_key = None

        cond = self.compel(a1111_to_compel(final_prompt))
        neg = self.compel(a1111_to_compel(final_neg))
        cond, neg = self.compel.pad_conditioning_tensors_to_same_length([cond, neg])

        w, h = SHAPES[p["shape"]]
        steps = p["steps"]
        total = steps * 2 if p["hires"] else steps
        done = [0]

        def cb(_pipe, i, t, kw):
            done[0] += 1
            progress(done[0] / total)
            return kw

        gen = torch.Generator("cuda").manual_seed(seed)
        out = pipe(prompt_embeds=cond, negative_prompt_embeds=neg, num_inference_steps=steps,
                   guidance_scale=p["guidance"], generator=gen, height=h, width=w,
                   cross_attention_kwargs=cross, callback_on_step_end=cb)
        if out.nsfw_content_detected and out.nsfw_content_detected[0]:
            raise UserError("The safety filter blocked this image. Change the prompt and try again.")
        image = out.images[0]

        if p["hires"]:
            up = image.resize((int(w * 1.5), int(h * 1.5)), Image.LANCZOS)
            hs = 0.3
            gen2 = torch.Generator("cuda").manual_seed(seed)
            out = self.img2img(prompt_embeds=cond, negative_prompt_embeds=neg, image=up, strength=hs,
                               num_inference_steps=int(steps / hs), guidance_scale=p["guidance"],
                               generator=gen2, cross_attention_kwargs=cross, callback_on_step_end=cb)
            if out.nsfw_content_detected and out.nsfw_content_detected[0]:
                raise UserError("The safety filter blocked this image. Change the prompt and try again.")
            image = out.images[0]
        progress(1.0)
        return image, seed, final_prompt


# ----------------------------------------------------------------------------
# MockEngine: GPU ke bina test ke liye
# ----------------------------------------------------------------------------
class MockEngine:
    def __init__(self, drive_dir):
        self.drive_dir = drive_dir
        self.gpu = "Mock GPU (test)"

    def load(self):
        pass

    def loras(self):
        return list_loras(self.drive_dir)

    def generate(self, p, progress):
        from PIL import Image, ImageDraw
        seed = p["seed"] if p["seed"] >= 0 else random.randint(0, 2**32 - 1)
        rnd = random.Random(seed)
        w, h = SHAPES[p["shape"]]
        steps = p["steps"]
        for i in range(steps):
            time.sleep(float(os.environ.get("STUDIO_MOCK_DELAY", "0.05")))
            progress((i + 1) / steps)
        c1 = tuple(rnd.randint(30, 200) for _ in range(3))
        c2 = tuple(rnd.randint(30, 200) for _ in range(3))
        img = Image.new("RGB", (w, h))
        d = ImageDraw.Draw(img)
        for y in range(h):
            t = y / h
            d.line([(0, y), (w, y)], fill=tuple(int(c1[k] * (1 - t) + c2[k] * t) for k in range(3)))
        d.text((16, 16), f"MOCK seed {seed}\n{p['prompt'][:60]}\nlora={p['lora']} x{p['lora_scale']}", fill=(255, 255, 255))
        return img, seed, p["prompt"]


# ----------------------------------------------------------------------------
# Job queue: phone job bhejta hai, baad me status poochta hai (tunnel timeout se bachne ke liye)
# ----------------------------------------------------------------------------
class Jobs:
    def __init__(self, engine):
        self.engine = engine
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)
        self.jobs = OrderedDict()
        self.waiting = []
        threading.Thread(target=self._worker, daemon=True).start()

    def submit(self, params):
        with self.lock:
            self._gc()
            if len(self.waiting) >= MAX_QUEUE:
                raise UserError("The queue is full. Try again in a minute.")
            jid = uuid.uuid4().hex[:12]
            self.jobs[jid] = {"id": jid, "status": "queued", "progress": 0.0, "params": params,
                              "error": None, "jpeg": None, "seed": None, "prompt": None, "t": time.time(),
                              "cancel": False}
            self.waiting.append(jid)
            self.cv.notify()
            return jid

    def view(self, jid):
        with self.lock:
            j = self.jobs.get(jid)
            if not j:
                return None
            pos = self.waiting.index(jid) + 1 if jid in self.waiting else 0
            return {"id": jid, "status": j["status"], "progress": round(j["progress"], 3), "queue": pos,
                    "error": j["error"], "seed": j["seed"], "prompt": j["prompt"]}

    def image(self, jid):
        with self.lock:
            j = self.jobs.get(jid)
            return j["jpeg"] if j and j["status"] == "done" else None

    def cancel(self, jid):
        with self.lock:
            j = self.jobs.get(jid)
            if not j:
                return False
            j["cancel"] = True
            if jid in self.waiting:
                self.waiting.remove(jid)
                j["status"] = "cancelled"
            return True

    def _gc(self):
        now = time.time()
        for k in [k for k, j in self.jobs.items() if now - j["t"] > JOB_TTL and j["status"] not in ("queued", "running")]:
            del self.jobs[k]

    def _worker(self):
        while True:
            with self.lock:
                while not self.waiting:
                    self.cv.wait()
                jid = self.waiting.pop(0)
                j = self.jobs[jid]
                j["status"] = "running"
            def progress(f, j=j):
                if j["cancel"]:
                    raise UserError("Cancelled")
                j["progress"] = f
            try:
                img, seed, final_prompt = self.engine.generate(j["params"], progress)
                buf = io.BytesIO()
                img.convert("RGB").save(buf, "JPEG", quality=93)
                with self.lock:
                    j.update(status="done", progress=1.0, jpeg=buf.getvalue(), seed=seed, prompt=final_prompt)
            except UserError as e:
                with self.lock:
                    j.update(status="cancelled" if str(e) == "Cancelled" else "error", error=str(e))
            except Exception as e:  # GPU memory etc.
                print("Generate error:", repr(e))
                with self.lock:
                    j.update(status="error", error="Could not make the image. Try again with fewer steps or with the sharper-result option off.")
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass


# ----------------------------------------------------------------------------
# App ki files (HTML/JS) : build ke baad is file ke andar hi rehti hain
# ----------------------------------------------------------------------------
def load_assets():
    if EMBEDDED_ASSETS:
        return {k: (base64.b64decode(v[0]), v[1]) for k, v in EMBEDDED_ASSETS.items()}
    here = os.path.dirname(os.path.abspath(__file__))
    web = os.path.join(os.path.dirname(here), "webapp")
    if not os.path.isdir(web):
        web = here
    types = {".html": "text/html; charset=utf-8", ".js": "text/javascript", ".webmanifest": "application/manifest+json",
             ".png": "image/png", ".svg": "image/svg+xml"}
    out = {}
    for f in os.listdir(web):
        ext = os.path.splitext(f)[1]
        if ext in types:
            with open(os.path.join(web, f), "rb") as fh:
                out["/" + f] = (fh.read(), types[ext])
    return out


def make_handler(state):
    assets = state["assets"]

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "LoRAStudio"

        def log_message(self, *a):
            pass

        def _send(self, code, body=b"", ctype="application/json", extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "X-Key, Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
            self.send_header("Cache-Control", "no-store" if ctype == "application/json" else "no-cache")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, code, obj):
            self._send(code, json.dumps(obj).encode())

        def _authed(self):
            q = parse_qs(urlparse(self.path).query)
            k = self.headers.get("X-Key") or (q.get("key") or [""])[0]
            return secrets.compare_digest(k.encode(), state["key"].encode())

        def do_OPTIONS(self):
            self._send(204)

        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/api/health":
                return self._json(200, {"ok": True, "name": "lora-studio"})
            if path.startswith("/api/"):
                if not self._authed():
                    return self._json(401, {"error": "Wrong key."})
                if path == "/api/info":
                    return self._json(200, {"gpu": state["engine"].gpu, "loras": state["engine"].loras(),
                                            "shapes": list(SHAPES), "mock": MOCK})
                m = re.fullmatch(r"/api/jobs/([a-f0-9]{12})(/image)?", path)
                if m:
                    if m.group(2):
                        data = state["jobs"].image(m.group(1))
                        if data is None:
                            return self._json(404, {"error": "The image is not ready."})
                        return self._send(200, data, "image/jpeg")
                    v = state["jobs"].view(m.group(1))
                    return self._json(200, v) if v else self._json(404, {"error": "Job not found."})
                return self._json(404, {"error": "Not found"})
            a = assets.get("/index.html" if path in ("/", "") else path)
            if not a:
                return self._json(404, {"error": "Not found"})
            self._send(200, a[0], a[1])

        def do_POST(self):
            path = urlparse(self.path).path
            try:
                n = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                n = 0
            if n > 20000:                      # bahut badi body: padhe bina connection band
                self.close_connection = True
                return self._json(400, {"error": "Request too large."})
            raw = self.rfile.read(n) if n else b""   # body hamesha padho, warna keep-alive connection bigad jata hai
            if path != "/api/jobs":
                return self._json(404, {"error": "Not found"})
            if not self._authed():
                return self._json(401, {"error": "Wrong key."})
            try:
                data = json.loads(raw or b"{}")
                params = clean_params(data)
                jid = state["jobs"].submit(params)
            except UserError as e:
                return self._json(400, {"error": str(e)})
            except Exception:
                return self._json(400, {"error": "Could not read the request."})
            self._json(202, {"id": jid})

        def do_DELETE(self):
            m = re.fullmatch(r"/api/jobs/([a-f0-9]{12})", urlparse(self.path).path)
            if not m:
                return self._json(404, {"error": "Not found"})
            if not self._authed():
                return self._json(401, {"error": "Wrong key."})
            self._json(200, {"ok": state["jobs"].cancel(m.group(1))})

    return H


# ----------------------------------------------------------------------------
# Colab: install, Drive, tunnel
# ----------------------------------------------------------------------------
def install_packages():
    pkgs = ["huggingface_hub>=0.25.0,<1.0", "diffusers==0.31.0", "transformers==4.45.2",
            "accelerate==0.34.2", "peft==0.13.2", "compel==2.0.3", "qrcode"]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)


def start_tunnel(port):
    """cloudflared se free public https link (account/token nahi chahiye)."""
    exe = "/content/cloudflared" if os.path.isdir("/content") else "/tmp/cloudflared"
    if not os.path.exists(exe):
        url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
        subprocess.run(["curl", "-sSL", "-o", exe, url], check=True)
        os.chmod(exe, 0o755)
    proc = subprocess.Popen([exe, "tunnel", "--url", f"http://localhost:{port}", "--no-autoupdate"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    found = {}

    def reader():
        for line in proc.stdout:
            m = re.search(r"https://[a-z0-9\-]+\.trycloudflare\.com", line)
            if m and "url" not in found and not m.group(0).startswith("https://api."):
                found["url"] = m.group(0)
    threading.Thread(target=reader, daemon=True).start()
    for _ in range(60):
        if "url" in found:
            return found["url"]
        time.sleep(1)
    return None


def show_qr(link):
    try:
        import qrcode
        img = qrcode.make(link)
        try:
            from IPython.display import display
            display(img)
        except Exception:
            pass
    except Exception:
        pass


def main():
    in_colab = "google.colab" in sys.modules or os.path.isdir("/content")
    if not MOCK:
        install_packages()
    drive_dir = os.environ.get("STUDIO_DRIVE_DIR")
    if not drive_dir:
        drive_dir = "/content/drive/MyDrive/lora_studio"
        try:
            from google.colab import drive
            drive.mount("/content/drive")
        except Exception as e:
            print("Drive mount nahi hua:", e)
            drive_dir = "/content/lora_studio"
    os.makedirs(drive_dir, exist_ok=True)

    engine = (MockEngine if MOCK else Engine)(drive_dir)
    engine.load()
    key = os.environ.get("STUDIO_KEY") or secrets.token_urlsafe(9)
    state = {"engine": engine, "jobs": Jobs(engine), "key": key, "assets": load_assets()}
    httpd = ThreadingHTTPServer(("0.0.0.0", PORT), make_handler(state))
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    print(f"\nServer chal raha hai: http://localhost:{PORT}")
    print("LoRA folder:", drive_dir, "| LoRA mili:", [l['name'] for l in engine.loras()])

    base = None
    if os.environ.get("STUDIO_TUNNEL") == "1" or (in_colab and not MOCK):
        base = start_tunnel(PORT)
    if base:
        link = f"{base}/#key={key}"
        print("\n" + "=" * 60)
        print("PHONE ME YE LINK KHOLO (Chrome):")
        print(link)
        print("Phir Chrome menu -> 'Add to Home screen' / 'Install app'")
        print("=" * 60)
        show_qr(link)
    else:
        print("Public link nahi bana. Key:", key)
    print("\nYe cell chalta rehna chahiye. Band karoge to app band ho jayegi.")
    while True:
        time.sleep(60)


if __name__ == "__main__":
    main()
