# ============================================================================
#  Project ko GitHub par daalna (Colab me chalao, phone se bhi ho jata hai)
#
#  Isse GitHub apne aap APK bana deta hai (Actions) aur Releases page par rakh deta hai.
#
#  Pehle ye karo:
#   1. github.com par free account banao
#   2. github.com/settings/tokens -> "Generate new token (classic)"
#      scopes me ye dono tick karo:  repo  aur  workflow   -> token copy karo
#   3. lora-studio.zip ko Drive me MyDrive/lora_studio/ me rakho
#  Kaam ke baad token ko github.com/settings/tokens se delete kar dena.
# ============================================================================
import os, re, json, getpass, subprocess, zipfile, shutil, urllib.request, urllib.error

try:
    from google.colab import drive
    drive.mount("/content/drive")
except Exception as e:
    print("Drive mount skip:", e)

ZIP = "/content/drive/MyDrive/lora_studio/lora-studio.zip"
if not os.path.exists(ZIP):
    from google.colab import files
    print("Drive me lora-studio.zip nahi mili. Ab upload karo:")
    up = files.upload()
    ZIP = "/content/" + next(iter(up))

user = input("GitHub username: ").strip()
repo = (input("Repo ka naam [lora-studio]: ").strip() or "lora-studio")
token = getpass.getpass("GitHub token (dikhega nahi): ").strip()
if not user or not token:
    raise SystemExit("Username aur token zaroori hain.")


def mask(t):
    return t.replace(token, "***")


def git(*args, cwd):
    r = subprocess.run(["git", *args], cwd=cwd, text=True, capture_output=True)
    out = mask((r.stdout or "") + (r.stderr or ""))
    if r.returncode != 0:
        print(out)
        raise SystemExit("git step fail hua: git " + args[0])
    return out


work = "/content/lora-studio"
shutil.rmtree(work, ignore_errors=True)
with zipfile.ZipFile(ZIP) as z:
    z.extractall("/content/_unzip")
# zip ke andar ek folder ho to usi ko project maano
entries = os.listdir("/content/_unzip")
src = os.path.join("/content/_unzip", entries[0]) if len(entries) == 1 and os.path.isdir(os.path.join("/content/_unzip", entries[0])) else "/content/_unzip"
shutil.move(src, work)
if not os.path.exists(os.path.join(work, ".github", "workflows", "build-apk.yml")):
    raise SystemExit(".github/workflows/build-apk.yml zip me nahi mili. Zip dobara banao/upload karo.")

# repo banana (pehle se ho to chalega)
req = urllib.request.Request(
    "https://api.github.com/user/repos", method="POST",
    data=json.dumps({"name": repo, "private": False, "description": "LoRA Studio app"}).encode(),
    headers={"Authorization": f"token {token}", "Accept": "application/vnd.github+json", "Content-Type": "application/json"})
try:
    urllib.request.urlopen(req, timeout=30)
    print("Repo ban gaya:", f"https://github.com/{user}/{repo}")
except urllib.error.HTTPError as e:
    body = e.read().decode(errors="ignore")
    if e.code == 422:
        print("Repo pehle se hai, usi me daal rahe hain.")
    elif e.code in (401, 403):
        raise SystemExit("Token galat hai ya uske scopes kam hain (repo + workflow chahiye).")
    else:
        raise SystemExit(f"Repo nahi bana ({e.code}): {mask(body)[:300]}")

git("init", "-b", "main", cwd=work)
git("config", "user.name", user, cwd=work)
git("config", "user.email", f"{user}@users.noreply.github.com", cwd=work)
git("add", "-A", cwd=work)
git("commit", "-m", "LoRA Studio", cwd=work)
remote = f"https://x-access-token:{token}@github.com/{user}/{repo}.git"
git("remote", "add", "origin", remote, cwd=work)
git("push", "-f", "origin", "main", cwd=work)
tag = input("Release ka naam (APK Releases page par aayega) [v1.0]: ").strip() or "v1.0"
if not re.fullmatch(r"v[0-9][0-9A-Za-z._-]*", tag):
    raise SystemExit("Naam 'v' se shuru hona chahiye, jaise v1.0")
git("tag", "-f", tag, cwd=work)
git("push", "-f", "origin", tag, cwd=work)

print("\nHO GAYA.")
print(f"1. Build dekho  : https://github.com/{user}/{repo}/actions   (5-10 minute lagte hain)")
print(f"2. APK download : https://github.com/{user}/{repo}/releases  (build ke baad)")
print("   LoRA-Studio-Lite.apk = sirf Colab mode | LoRA-Studio-Full.apk = Colab + phone par")
print("3. Token ko github.com/settings/tokens se delete kar do.")
