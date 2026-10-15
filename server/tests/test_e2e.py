"""Server (mock model) + phone app ko end-to-end test karta hai. Run: python3 tests/test_e2e.py"""
import os, sys, json, time, subprocess, tempfile, urllib.request, urllib.error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # = server/
PORT, KEY = 8765, "testkey123"
BASE = f"http://localhost:{PORT}"
SHOTS = os.path.join(ROOT, "tests", "shots")
os.makedirs(SHOTS, exist_ok=True)

drive = tempfile.mkdtemp()
os.makedirs(os.path.join(drive, "my_lora"))
open(os.path.join(drive, "my_lora", "pytorch_lora_weights.safetensors"), "wb").write(b"x")
json.dump({"trigger": "ohwx person", "step": 800}, open(os.path.join(drive, "my_lora", "meta.json"), "w"))

env = dict(os.environ, STUDIO_MOCK="1", STUDIO_PORT=str(PORT), STUDIO_KEY=KEY, STUDIO_DRIVE_DIR=drive,
           STUDIO_MOCK_DELAY="0.08", PYTHONUNBUFFERED="1")
srv = subprocess.Popen([sys.executable, os.path.join(ROOT, "colab_server.py")], env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
ok_all = True


def check(name, cond, extra=""):
    global ok_all
    print(("PASS " if cond else "FAIL ") + name + (f"  [{extra}]" if extra and not cond else ""))
    ok_all &= bool(cond)


def call(method, path, body=None, key=KEY, raw=False):
    req = urllib.request.Request(BASE + path, method=method, data=json.dumps(body).encode() if body is not None else None)
    if key:
        req.add_header("X-Key", key)
    if body is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            data = r.read()
            return r.status, (data if raw else json.loads(data)), r.headers
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}"), e.headers


try:
    for _ in range(50):
        try:
            urllib.request.urlopen(BASE + "/api/health", timeout=1)
            break
        except Exception:
            time.sleep(0.2)

    # ---------------- API ----------------
    c, j, h = call("GET", "/api/health", key=None)
    check("health open + CORS", c == 200 and j["name"] == "lora-studio" and h["Access-Control-Allow-Origin"] == "*")
    c, j, _ = call("GET", "/api/info", key="wrong")
    check("info rejects wrong key", c == 401)
    c, j, _ = call("GET", "/api/info")
    check("info lists LoRA + trigger", c == 200 and j["loras"] and j["loras"][0]["trigger"] == "ohwx person", str(j))
    for bad in ["nude woman on beach", "A NAKED person", "xxx", "loli", "underage girl"]:
        c, j, _ = call("POST", "/api/jobs", {"prompt": bad})
        check(f"blocks prompt '{bad}'", c == 400, str(j))
    c, j, _ = call("POST", "/api/jobs", {"prompt": "  "})
    check("rejects empty prompt", c == 400)
    c, j, _ = call("POST", "/api/jobs", {"prompt": "a man in a park", "negative": "nsfw"})
    check("negative field may contain these words (protective)", c == 202, str(j))
    if c == 202:
        call("DELETE", "/api/jobs/" + j["id"])
    c, j, _ = call("POST", "/api/jobs", {"prompt": "naked person"})
    check("message names the blocked word", c == 400 and '"naked"' in j["error"], str(j))
    c, j, _ = call("POST", "/api/jobs", {"prompt": "a calm lake", "steps": 9999, "lora_scale": 99, "shape": "evil"}, key="bad")
    check("job needs key", c == 401)
    c, j, _ = call("POST", "/api/jobs", {"prompt": "a calm lake", "steps": 9999, "lora_scale": 99, "shape": "evil", "seed": 42})
    check("job accepted (values clamped)", c == 202 and "id" in j, str(j))
    jid = j["id"]
    for _ in range(100):
        c, v, _ = call("GET", f"/api/jobs/{jid}")
        if v["status"] in ("done", "error"):
            break
        time.sleep(0.3)
    check("job finishes", v["status"] == "done" and v["seed"] == 42, str(v))
    c, img, h = call("GET", f"/api/jobs/{jid}/image", raw=True)
    check("image is JPEG 512x768", c == 200 and img[:2] == b"\xff\xd8" and h["Content-Type"] == "image/jpeg")
    from PIL import Image
    import io
    im = Image.open(io.BytesIO(img))
    check("clamped shape -> portrait size", im.size == (512, 768), str(im.size))
    # queue + cancel
    ids = []
    for i in range(3):
        c, j, _ = call("POST", "/api/jobs", {"prompt": f"scene {i}", "steps": 30})
        ids.append(j["id"])
    time.sleep(0.4)
    c, v, _ = call("GET", f"/api/jobs/{ids[2]}")
    check("third job waits in queue", v["status"] == "queued" and v["queue"] >= 1, str(v))
    call("DELETE", f"/api/jobs/{ids[2]}")
    c, v, _ = call("GET", f"/api/jobs/{ids[2]}")
    check("queued job can be cancelled", v["status"] == "cancelled", str(v))
    call("DELETE", f"/api/jobs/{ids[0]}")
    time.sleep(1)
    c, v, _ = call("GET", f"/api/jobs/{ids[0]}")
    check("running job can be cancelled", v["status"] == "cancelled", str(v))
    import http.client
    hc = http.client.HTTPConnection("localhost", PORT, timeout=10)
    hc.request("POST", "/api/jobs", body=json.dumps({"prompt": "x" * 500}), headers={"X-Key": "wrong", "Content-Type": "application/json"})
    r1 = hc.getresponse(); r1.read()
    hc.request("GET", "/api/health")
    r2 = hc.getresponse(); body2 = r2.read()
    check("keep-alive stays healthy after rejected POST", r1.status == 401 and r2.status == 200 and b"lora-studio" in body2, f"{r1.status} {r2.status} {body2[:40]}")
    hc.close()
    c, j, _ = call("GET", "/api/jobs/../../etc/passwd")
    check("path traversal not served", c == 404)
    # queue limit
    res = [call("POST", "/api/jobs", {"prompt": f"q{i}", "steps": 50}) for i in range(10)]
    codes = [r[0] for r in res]
    check("queue limit returns 400", 400 in codes, str(codes))
    for r in res:
        if r[0] == 202:
            call("DELETE", "/api/jobs/" + r[1]["id"])
    for jid_ in ids[1:2]:
        call("DELETE", "/api/jobs/" + jid_)
    time.sleep(1.5)

    # ---------------- Phone UI ----------------
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        errors = []

        def ctx(scheme="light"):
            c = b.new_context(viewport={"width": 390, "height": 844}, device_scale_factor=2, color_scheme=scheme,
                              is_mobile=True, has_touch=True, accept_downloads=True)
            return c

        c1 = ctx()
        p = c1.new_page()
        p.on("pageerror", lambda e: errors.append("pageerror: " + str(e)))
        p.on("console", lambda m: errors.append("console: " + m.text) if m.type == "error" and "net::ERR" not in m.text and "Failed to load resource" not in m.text else None)
        # wait for queue to drain from API tests
        time.sleep(1)
        p.goto(f"{BASE}/#key={KEY}")
        p.wait_for_function("document.getElementById('chipText').textContent.startsWith('Connected')", timeout=10000)
        check("auto-connect from link", True)
        check("key removed from address bar", "#" not in p.url, p.url)
        opts = p.eval_on_selector_all("#lora option", "els => els.map(e => e.value)")
        check("LoRA dropdown filled", "my_lora" in opts, str(opts))
        check("trigger shown", "ohwx person" in p.inner_text("#trig"))
        p.screenshot(path=f"{SHOTS}/1-create-light.png")
        p.fill("#prompt", "park me muskurati hui, casual kapde")
        p.click("#go")
        p.wait_for_selector("#status:has-text('Creating')", timeout=8000)
        p.screenshot(path=f"{SHOTS}/2-generating.png")
        p.wait_for_selector("#status:has-text('Done')", timeout=30000)
        check("generation completes in UI", p.is_visible("#shot"))
        w = p.eval_on_selector("#shot", "i => i.naturalWidth")
        check("result image decoded", w == 512, str(w))
        p.screenshot(path=f"{SHOTS}/3-result.png", full_page=True)
        with p.expect_download() as d:
            p.click("#save")
        check("save downloads a jpg", d.value.suggested_filename.endswith(".jpg"), d.value.suggested_filename)
        p.click("#again")
        p.wait_for_selector("#status:has-text('Done')", timeout=30000)
        p.click("button.tab[data-v=gallery]")
        p.wait_for_selector(".frame")
        check("gallery has 2 photos", p.locator(".frame").count() == 2, str(p.locator(".frame").count()))
        check("empty card hidden when photos exist", not p.is_visible("#empty"))
        p.screenshot(path=f"{SHOTS}/4-gallery.png")
        p.locator(".frame").first.click()
        p.wait_for_selector("#viewer[open]")
        p.screenshot(path=f"{SHOTS}/5-viewer.png")
        p.click("#vuse")
        check("use settings returns to Create", p.is_visible("#go"))
        p.click("button.tab[data-v=gallery]")
        p.locator(".frame").first.click()
        p.click("#vdel")
        p.wait_for_timeout(500)
        check("delete removes a photo", p.locator(".frame").count() == 1)
        # reload keeps gallery + settings
        p.reload()
        p.wait_for_function("document.getElementById('chipText').textContent.startsWith('Connected')", timeout=10000)
        check("prompt remembered after reload", "muskurati" in p.input_value("#prompt"))
        p.click("button.tab[data-v=gallery]")
        p.wait_for_selector(".frame")
        check("gallery survives reload", p.locator(".frame").count() == 1)
        # blocked prompt message in UI
        p.click("button.tab[data-v=create]")
        p.fill("#prompt", "naked person")
        p.click("#go")
        p.wait_for_selector("#status.bad", timeout=8000)
        check("blocked prompt shows friendly error", "not allowed" in p.inner_text("#status"), p.inner_text("#status"))
        p.fill("#prompt", "a lighthouse at dusk")
        # horizontal overflow check on every tab
        for t in ("create", "gallery", "connect"):
            p.click(f"button.tab[data-v={t}]")
            ov = p.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
            check(f"no sideways scroll on {t}", ov <= 0, str(ov))
        p.screenshot(path=f"{SHOTS}/6-connect.png", full_page=True)
        # manifest + sw + icons reachable
        for path in ("/manifest.webmanifest", "/sw.js", "/icon-192.png", "/icon-512.png"):
            r = p.request.get(BASE + path)
            check(f"served {path}", r.status == 200)
        sw = p.evaluate("navigator.serviceWorker.getRegistration().then(r => !!r)")
        check("service worker registered", sw is True)
        c1.close()

        # wrong key -> offline chip + message
        c2 = ctx("dark")
        q = c2.new_page()
        q.goto(f"{BASE}/#key=nope")
        q.wait_for_function("document.getElementById('chipText').textContent == 'Offline'", timeout=10000)
        check("wrong key shows Offline + message", "Wrong key" in q.inner_text("#cmsg"), q.inner_text("#cmsg"))
        q.screenshot(path=f"{SHOTS}/7-wrongkey-dark.png")
        # now fix via pasted link
        q.fill("#link", f"{BASE}/#key={KEY}")
        q.click("#connect")
        q.wait_for_function("document.getElementById('chipText').textContent.startsWith('Connected')", timeout=10000)
        check("paste-link connect works", True)
        q.wait_for_selector("#go")
        q.screenshot(path=f"{SHOTS}/8-create-dark.png")
        c2.close()

        # Colab band ho jaye: installed app (service worker cache) phir bhi khulni chahiye, saaf message dikhaye
        c3 = ctx()
        r = c3.new_page()
        r.goto(f"{BASE}/#key={KEY}")
        r.wait_for_function("document.getElementById('chipText').textContent.startsWith('Connected')", timeout=10000)
        r.evaluate("navigator.serviceWorker.ready.then(() => true)")
        r.reload()   # ab SW page ko control karta hai
        r.wait_for_function("document.getElementById('chipText').textContent.startsWith('Connected')", timeout=10000)
        srv.terminate(); srv.wait(timeout=5)
        r.reload()
        r.wait_for_function("document.getElementById('chipText').textContent === 'Offline'", timeout=15000)
        check("app opens from cache when Colab is off", r.is_visible("header.top"))
        check("offline message is clear", "Cannot reach the server" in r.inner_text("#cmsg"), r.inner_text("#cmsg"))
        r.screenshot(path=f"{SHOTS}/9-offline.png")
        c3.close()
        b.close()
        check("no JS errors during run", not errors, "; ".join(errors[:5]))
finally:
    if srv.poll() is None:
        srv.terminate()
    out = srv.stdout.read() if srv.stdout else ""
    if "Traceback" in out:
        print("SERVER LOG:\n", out[-2000:])
        ok_all = False

print("\nALL PASS" if ok_all else "\nSOME FAILED")
sys.exit(0 if ok_all else 1)
