# LoRA Studio

Photos from your own LoRA model. Two ways to make them:

* **Colab mode**: your Colab GPU makes the photo, the phone shows it.
* **This phone mode** (Full app only): the phone makes the photo itself. Works offline.

## Folders

| Folder | What is inside |
|---|---|
| `webapp/` | The app screens (one HTML file). The Android app and the Colab server both use it. |
| `server/` | `server.py` runs in Colab. `build.py` packs the app into one file, `colab_server.py`. `tests/` has the tests. |
| `android/` | The Android project. GitHub builds it into APK files. |
| `tools/` | `convert_for_phone.py` makes the phone model. `push_to_github.py` uploads this project to GitHub. |
| `.github/workflows/build-apk.yml` | Tells GitHub how to build the APKs. |

## Two APKs

* **LoRA-Studio-Lite.apk**: Colab mode only. Small. Works on almost every phone.
* **LoRA-Studio-Full.apk**: Colab mode and This phone mode. Needs an arm64 phone with 8 GB RAM or more for This phone mode.

## Get the APK (no computer needed)

1. Create a free account on github.com.
2. Open github.com/settings/tokens, choose **Generate new token (classic)**, tick **repo** and **workflow**, copy the token.
3. Put `lora-studio.zip` in Google Drive, folder `MyDrive/lora_studio/`.
4. In Colab run `tools/push_to_github.py` (paste it in a cell).
5. After 5 to 10 minutes open `github.com/<you>/lora-studio/releases`. The APK files are there. Send that link to anyone.
6. Delete the token on github.com/settings/tokens.

On the phone: download the APK, open it, allow "Install unknown apps" when asked.

## Make the phone model (for This phone mode)

Run `tools/convert_for_phone.py` in Colab (CPU runtime is fine). It joins the base model and your LoRA, converts it, and writes `phone_model.zip` to Drive. On the phone: Full app, Connect, This phone, Import model file. Type the trigger word.

## Run the tests

```
cd server
python3 build.py
python3 tests/test_e2e.py        # Colab mode, mock model
python3 tests/test_native_ui.py  # phone mode screens, fake Android bridge
```

## Limits to know

* The phone model has the LoRA joined into it. It has no strength slider, no negative prompt, no image safety check (the prompt word filter still applies).
* Google's MediaPipe image generator is marked "no longer actively maintained" and works with Stable Diffusion 1.5 style models.
* Use a face LoRA only for people who agreed to it.
* The signing key in `android/app/lorastudio.keystore` is a hobby key so updates install over each other. It is for sharing the APK directly, not for the Play Store.
