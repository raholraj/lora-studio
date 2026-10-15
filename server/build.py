"""../webapp ki files ko server.py ke andar jodkar colab_server.py banata hai.  Run: python3 build.py"""
import base64, os, textwrap
HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(os.path.dirname(HERE), "webapp")
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript", ".webmanifest": "application/manifest+json",
         ".png": "image/png", ".svg": "image/svg+xml"}
files = [f for f in sorted(os.listdir(WEB)) if os.path.splitext(f)[1] in TYPES]
lines = ["EMBEDDED_ASSETS = {"]
for f in files:
    b64 = base64.b64encode(open(os.path.join(WEB, f), "rb").read()).decode()
    lines.append(f'    "/{f}": ("".join([')
    lines += [f'        "{c}",' for c in textwrap.wrap(b64, 100)]
    lines.append(f'    ]), "{TYPES[os.path.splitext(f)[1]]}"),')
lines.append("}")
code = open(os.path.join(HERE, "server.py")).read()
marker = [l for l in code.splitlines() if l.startswith("EMBEDDED_ASSETS = None")]
assert marker, "marker line not found in server.py"
code = code.replace(marker[0], "\n".join(lines))
open(os.path.join(HERE, "colab_server.py"), "w").write(code)
print("built colab_server.py:", os.path.getsize(os.path.join(HERE, "colab_server.py")) // 1024, "KB;", len(files), "app files")
