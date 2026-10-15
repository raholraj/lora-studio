# ============================================================================
#  Phone model banana (Colab me chalao) -- GPU ki zarurat nahi, CPU runtime bhi chalega
#
#  Kya karta hai:
#   1. Base model (Realistic Vision) + aapki LoRA ko jodkar ek hi model banata hai
#      (LoRA "bake" ho jati hai, isliye trigger word prompt me likhna zaroori hai)
#   2. Usko Google MediaPipe ke phone format me badalta hai
#   3. Ek zip banakar Drive me rakhta hai:  MyDrive/lora_studio/phone_model.zip
#   4. Us zip ko phone me LoRA Studio (Full app) me Import karo
#
#  DHYAN: Ye script maine test nahi ki hai (Colab/GitHub yahan se chalane ka tareeka nahi hai).
#  Kisi step me error aaye to poora error text bhej do.
#  MediaPipe ka image generator Google ne "no longer actively maintained" bataya hai,
#  aur ye sirf Stable Diffusion 1.5 jaise models (jaise Realistic Vision) ke liye hai.
# ============================================================================
import os, sys, json, shutil, subprocess, zipfile

DRIVE      = "/content/drive/MyDrive/lora_studio"
LORA_NAME  = "my_lora"        # Drive/lora_studio me jo LoRA folder hai. LoRA nahi chahiye to "" likho
LORA_POWER = 0.8              # LoRA ko kitna mila-na hai (0.5 - 1.0)
OUT_NAME   = "phone_model"    # Drive me phone_model.zip banega
BASE_MODEL = "SG161222/Realistic_Vision_V5.1_noVAE"
VAE        = "stabilityai/sd-vae-ft-mse"

WORK = "/content/phone_build"
DIFF_DIR = f"{WORK}/diffusers_model"
CKPT = f"{WORK}/model.ckpt"
BINS = f"{WORK}/bins"


def sh(cmd, **kw):
    print("\n$", cmd if isinstance(cmd, str) else " ".join(cmd))
    r = subprocess.run(cmd, shell=isinstance(cmd, str), text=True, capture_output=True, **kw)
    if r.stdout.strip():
        print(r.stdout[-3000:])
    if r.returncode != 0:
        print(r.stderr[-4000:])
        raise SystemExit(f"STEP FAILED (exit {r.returncode}). Upar ka error text bhej do.")
    return r


try:
    from google.colab import drive
    drive.mount("/content/drive")
except Exception as e:
    print("Drive mount skip:", e)

shutil.rmtree(WORK, ignore_errors=True)
os.makedirs(WORK)

print("== 1/5 Packages")
sh([sys.executable, "-m", "pip", "install", "-q", "diffusers==0.31.0", "transformers==4.45.2", "accelerate==0.34.2",
    "peft==0.13.2", "huggingface_hub>=0.25.0,<1.0", "safetensors", "typing_extensions", "numpy", "Pillow",
    "requests", "pytorch_lightning", "absl-py", "omegaconf"])

print("== 2/5 Base model + LoRA ko jodna")
import torch
from diffusers import StableDiffusionPipeline, AutoencoderKL
vae = AutoencoderKL.from_pretrained(VAE, torch_dtype=torch.float32)
pipe = StableDiffusionPipeline.from_pretrained(BASE_MODEL, vae=vae, torch_dtype=torch.float32, safety_checker=None)
trigger = ""
if LORA_NAME:
    lora_dir = os.path.join(DRIVE, LORA_NAME)
    if not os.path.exists(os.path.join(lora_dir, "pytorch_lora_weights.safetensors")):
        raise SystemExit(f"LoRA nahi mili: {lora_dir}")
    try:
        trigger = json.load(open(os.path.join(lora_dir, "meta.json"))).get("trigger", "")
    except Exception:
        pass
    pipe.load_lora_weights(lora_dir)
    pipe.fuse_lora(lora_scale=LORA_POWER)
    pipe.unload_lora_weights()          # fused weights base me hi rehte hain
    print("LoRA jod di:", LORA_NAME, "| power", LORA_POWER, "| trigger:", trigger or "(meta.json me nahi mila)")
pipe.save_pretrained(DIFF_DIR, safe_serialization=True)
del pipe, vae

print("== 3/5 Diffusers -> original Stable Diffusion checkpoint (.ckpt)")
conv = f"{WORK}/diffusers_to_ckpt.py"
sh(["curl", "-sSL", "-o", conv,
    "https://raw.githubusercontent.com/huggingface/diffusers/main/scripts/convert_diffusers_to_original_stable_diffusion.py"])
if os.path.getsize(conv) < 2000:
    raise SystemExit("diffusers ki convert script nahi mili (link badal gaya ho sakta hai). Mujhe batao.")
sh([sys.executable, conv, "--model_path", DIFF_DIR, "--checkpoint_path", CKPT])
shutil.rmtree(DIFF_DIR, ignore_errors=True)

print("== 4/5 MediaPipe phone format")
sh(["git", "clone", "--depth", "1", "https://github.com/googlesamples/mediapipe", f"{WORK}/mp"])
tool = f"{WORK}/mp/tools/image_generator_converter"
req = os.path.join(tool, "requirements.txt")
if os.path.exists(req):
    sh([sys.executable, "-m", "pip", "install", "-q", "-r", req])
sh([sys.executable, os.path.join(tool, "convert.py"), "--ckpt_path", CKPT, "--output_path", BINS])
os.remove(CKPT)

print("== 5/5 Zip banakar Drive me rakhna")
os.makedirs(DRIVE, exist_ok=True)
zip_path = os.path.join(DRIVE, OUT_NAME + ".zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as z:     # bins compress nahi hote, STORED tez hai
    for root, _, files in os.walk(BINS):
        for f in files:
            full = os.path.join(root, f)
            z.write(full, os.path.relpath(full, BINS))
print(f"\nTAIYAR: {zip_path}  ({os.path.getsize(zip_path) / 1e9:.2f} GB)")
print("Phone me kya karna hai:")
print("  1. Drive app me ye zip phone me download karo")
print("  2. LoRA Studio (Full app) -> Connect -> 'This phone' -> Import model file (.zip)")
print(f"  3. Trigger word box me likho: {trigger or '(jo LoRA training me use kiya tha)'}")
