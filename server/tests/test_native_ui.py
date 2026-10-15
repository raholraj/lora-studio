"""Android app ke andar wali web UI ko test karta hai (NativeStudio bridge ko nakli bana kar). Run: python3 tests/test_native_ui.py"""
import os, sys, io, base64, threading, functools, http.server, socketserver
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(os.path.dirname(ROOT), "webapp")
SHOTS = os.path.join(ROOT, "tests", "shots")
os.makedirs(SHOTS, exist_ok=True)
PORT = 8766

buf = io.BytesIO()
Image.new("RGB", (64, 64), (200, 80, 60)).save(buf, "JPEG")
TINY = base64.b64encode(buf.getvalue()).decode()

MOCK = """
(() => {
  const st = { modelReady: false, trigger: "", modelMB: 0, calls: [], onDevice: %ONDEVICE% };
  window.__calls = st.calls;
  let iv = null, genId = 0;
  const emit = o => setTimeout(() => window.__studioNative && window.__studioNative(JSON.stringify(o)), 20);
  window.NativeStudio = {
    info() { return JSON.stringify({ native: true, onDevice: st.onDevice, modelReady: st.modelReady, trigger: st.trigger, modelMB: st.modelMB, ramMB: 8000 }); },
    importModel(t) { st.calls.push(["import", t]); st.trigger = t; emit({ type: "model", state: "working", p: 0.4, message: "Copying files..." });
      setTimeout(() => { st.modelReady = true; st.modelMB = 1900; emit({ type: "model", state: "done", p: 1, message: "Model ready." }); }, 250); },
    downloadModel(u, t) { st.calls.push(["download", u, t]); },
    deleteModel() { st.modelReady = false; st.modelMB = 0; return "ok"; },
    generate(j) { const p = JSON.parse(j); st.calls.push(["gen", p]); const id = "g" + (++genId); let i = 0;
      emit({ type: "loading", id });
      iv = setInterval(() => { i++; emit({ type: "progress", id, p: i / 6 });
        if (i === 6) { clearInterval(iv); emit({ type: "done", id, seed: p.seed < 0 ? 4242 : p.seed, prompt: p.prompt, image: "%TINY%" }); } }, %DELAY%);
      return JSON.stringify({ id }); },
    cancel() { st.calls.push(["cancel"]); if (iv) { clearInterval(iv); iv = null; emit({ type: "cancelled", id: "g" + genId }); } },
    saveImage(b64, n) { st.calls.push(["save", n, b64.length]); return "ok"; },
    shareImage(b64, n) { st.calls.push(["share", n]); return "ok"; }
  };
})();
"""

ok_all = True
def check(name, cond, extra=""):
    global ok_all
    print(("PASS " if cond else "FAIL ") + name + (f"  [{extra}]" if extra and not cond else ""))
    ok_all &= bool(cond)

handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=WEB)
handler.log_message = lambda *a, **k: None
socketserver.TCPServer.allow_reuse_address = True
httpd = socketserver.ThreadingTCPServer(("127.0.0.1", PORT), handler)
threading.Thread(target=httpd.serve_forever, daemon=True).start()
URL = f"http://localhost:{PORT}/index.html"

from playwright.sync_api import sync_playwright
with sync_playwright() as pw:
    b = pw.chromium.launch()
    errors = []

    def page(ondevice=True, delay=80):
        ctx = b.new_context(viewport={"width": 390, "height": 844}, device_scale_factor=2, is_mobile=True, has_touch=True)
        ctx.add_init_script(MOCK.replace("%ONDEVICE%", "true" if ondevice else "false").replace("%TINY%", TINY).replace("%DELAY%", str(delay)))
        p = ctx.new_page()
        p.on("pageerror", lambda e: errors.append("pageerror: " + str(e)))
        p.on("console", lambda m: errors.append("console: " + m.text) if m.type == "error" and "net::ERR" not in m.text and "Failed to load resource" not in m.text else None)
        return ctx, p

    # ---------------- Full app ----------------
    ctx, p = page(True)
    p.goto(URL)
    p.wait_for_selector("#modeCard:not([hidden])")
    check("mode switch visible in Full app", True)
    check("starts in Colab mode, Connect tab", p.is_visible("#v-connect.on"))
    p.click("label[for=m2]")
    check("phone mode hides Colab-only controls", p.get_attribute("body", "data-mode") == "phone" and not p.is_visible("#smart") and not p.is_visible("#hires"))
    check("chip says No model", p.inner_text("#chipText") == "No model")
    p.fill("#ptrig", "ohwx person")
    p.click("#pimport")
    p.wait_for_selector("#pstate:has-text('Model installed (1900 MB)')", timeout=5000)
    check("import flow completes", True)
    check("chip says On this phone", p.inner_text("#chipText") == "On this phone")
    p.screenshot(path=f"{SHOTS}/n1-phone-connect.png", full_page=True)
    p.click("button.tab[data-v=create]")
    check("phone hint visible", p.is_visible("#phoneHint"))
    p.fill("#prompt", "a girl reading in a library")
    p.click("#go")
    p.wait_for_selector("#status:has-text('Creating on this phone')", timeout=5000)
    p.screenshot(path=f"{SHOTS}/n2-phone-generating.png")
    p.wait_for_selector("#status:has-text('Done')", timeout=8000)
    check("on-phone generation shows result", p.is_visible("#shot") and p.eval_on_selector("#shot", "i => i.naturalWidth") == 64)
    calls = p.evaluate("window.__calls")
    gen = [c for c in calls if c[0] == "gen"][0][1]
    check("bridge got prompt, steps, seed -1", gen["prompt"] == "a girl reading in a library" and gen["seed"] == -1 and 10 <= gen["steps"] <= 50, str(gen))
    p.click("#save")
    p.wait_for_selector("#toast")
    check("Save goes through the app bridge", any(c[0] == "save" and c[1].endswith(".jpg") and c[2] > 100 for c in p.evaluate("window.__calls")))
    check("toast confirms save", "Saved" in p.inner_text("#toast"))
    p.click("#share")
    p.wait_for_function("window.__calls.some(c => c[0] === 'share')", timeout=3000)
    check("Share goes through the app bridge", True)
    p.click("button.tab[data-v=gallery]")
    p.wait_for_selector(".frame")
    check("photo saved to gallery", p.locator(".frame").count() == 1)
    p.click("button.tab[data-v=create]")
    p.click("#again")
    p.wait_for_selector("#status:has-text('Done')", timeout=8000)
    check("New seed works in phone mode", p.evaluate("window.__calls").count(["cancel"]) == 0 and len([c for c in p.evaluate("window.__calls") if c[0] == "gen"]) == 2)
    # cancel
    p.click("#go")
    p.wait_for_selector("#status:has-text('Creating')", timeout=5000)
    p.click("#cancel")
    p.wait_for_selector("#status:has-text('Cancelled')", timeout=5000)
    check("cancel reaches the app and UI recovers", ["cancel"] in p.evaluate("window.__calls") and not p.is_disabled("#go"))
    # empty prompt does nothing
    p.fill("#prompt", "")
    n0 = len(p.evaluate("window.__calls"))
    p.click("#go")
    check("empty prompt does not start a job", len(p.evaluate("window.__calls")) == n0)
    # mode survives reload
    p.fill("#prompt", "x")
    p.reload()
    p.wait_for_selector("#modeCard:not([hidden])")
    check("phone mode remembered after reload", p.get_attribute("body", "data-mode") == "phone")
    # (the fake app forgets its model on reload, so import again, then delete)
    p.click("button.tab[data-v=connect]")
    p.fill("#ptrig", "ohwx person")
    p.click("#pimport")
    p.wait_for_selector("#pstate:has-text('Model installed')", timeout=5000)
    p.click("#pdel")
    p.wait_for_selector("#pstate:has-text('No model on this phone yet')")
    check("delete model updates UI", p.inner_text("#chipText") == "No model" and not p.is_visible("#pdel"))
    p.fill("#purl", "http://insecure.example/m.zip")
    p.click("#pdl")
    check("http download link is refused", "https://" in p.inner_text("#pmsg") and not any(c[0] == "download" for c in p.evaluate("window.__calls")))
    p.fill("#purl", "https://example.com/model.zip")
    p.click("#pdl")
    check("https link reaches the app", any(c[0] == "download" and c[1].startswith("https://") for c in p.evaluate("window.__calls")))
    ov = p.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
    check("no sideways scroll on Connect (phone mode)", ov <= 0, str(ov))
    # back to Colab mode
    p.click("label[for=m1]")
    check("switching back shows Colab controls", p.is_visible("button.tab[data-v=create]") and p.get_attribute("body", "data-mode") == "colab")
    ctx.close()

    # ---------------- Lite app (no on-device) ----------------
    ctx, p = page(False)
    p.goto(URL)
    p.wait_for_selector("#phoneNote")
    check("Lite app hides mode switch", not p.is_visible("#modeCard") and not p.is_visible("#phoneBox"))
    check("Lite app explains itself", "Lite app" in p.inner_text("#phoneNote"))
    ctx.close()

    # ---------------- Plain browser (no bridge) still fine ----------------
    ctx = b.new_context(viewport={"width": 390, "height": 844})
    p = ctx.new_page()
    p.on("pageerror", lambda e: errors.append("pageerror: " + str(e)))
    p.goto(URL)
    p.wait_for_selector("#phoneNote")
    check("browser: no mode switch, normal note", not p.is_visible("#modeCard") and "Full Android app" in p.inner_text("#phoneNote"))
    ctx.close()
    b.close()
    check("no JS errors", not errors, "; ".join(errors[:5]))

httpd.shutdown()
print("\nALL PASS" if ok_all else "\nSOME FAILED")
sys.exit(0 if ok_all else 1)
