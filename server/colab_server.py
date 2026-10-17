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

EMBEDDED_ASSETS = {
    "/icon-192.png": ("".join([
        "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",
    ]), "image/png"),
    "/icon-512.png": ("".join([
        "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",
    ]), "image/png"),
    "/index.html": ("".join([
        "PCFkb2N0eXBlIGh0bWw+CjxodG1sIGxhbmc9ImVuIj4KPGhlYWQ+CjxtZXRhIGNoYXJzZXQ9InV0Zi04Ij4KPG1ldGEgbmFtZT0i",
        "dmlld3BvcnQiIGNvbnRlbnQ9IndpZHRoPWRldmljZS13aWR0aCwgaW5pdGlhbC1zY2FsZT0xLCB2aWV3cG9ydC1maXQ9Y292ZXIi",
        "Pgo8dGl0bGU+TG9SQSBTdHVkaW88L3RpdGxlPgo8bWV0YSBuYW1lPSJkZXNjcmlwdGlvbiIgY29udGVudD0iTWFrZSBwaG90b3Mg",
        "ZnJvbSB5b3VyIG93biBMb1JBIG1vZGVscyB1c2luZyB5b3VyIENvbGFiIEdQVS4iPgo8bWV0YSBuYW1lPSJ0aGVtZS1jb2xvciIg",
        "Y29udGVudD0iIzBjMTExNyI+CjxsaW5rIHJlbD0ibWFuaWZlc3QiIGhyZWY9Im1hbmlmZXN0LndlYm1hbmlmZXN0Ij4KPGxpbmsg",
        "cmVsPSJpY29uIiBocmVmPSJpY29uLTE5Mi5wbmciPgo8bGluayByZWw9ImFwcGxlLXRvdWNoLWljb24iIGhyZWY9Imljb24tMTky",
        "LnBuZyI+CjxsaW5rIHJlbD0icHJlY29ubmVjdCIgaHJlZj0iaHR0cHM6Ly9mb250cy5nb29nbGVhcGlzLmNvbSI+CjxsaW5rIHJl",
        "bD0icHJlY29ubmVjdCIgaHJlZj0iaHR0cHM6Ly9mb250cy5nc3RhdGljLmNvbSIgY3Jvc3NvcmlnaW4+CjxsaW5rIGhyZWY9Imh0",
        "dHBzOi8vZm9udHMuZ29vZ2xlYXBpcy5jb20vY3NzMj9mYW1pbHk9QnJpY29sYWdlK0dyb3Rlc3F1ZTpvcHN6LHdnaHRAMTIuLjk2",
        "LDYwMDsxMi4uOTYsNzAwJmZhbWlseT1JbnN0cnVtZW50K1NhbnM6d2dodEA0MDA7NTAwOzYwMCZmYW1pbHk9SUJNK1BsZXgrTW9u",
        "bzp3Z2h0QDQwMDs1MDAmZGlzcGxheT1zd2FwIiByZWw9InN0eWxlc2hlZXQiPgo8c3R5bGU+Ci8qIExheW91dDogb25lIGNvbHVt",
        "biBwaG9uZSBhcHAsIHRocmVlIHRhYnMgKENyZWF0ZSAvIEdhbGxlcnkgLyBDb25uZWN0KSwgY29udGFjdC1zaGVldCBnYWxsZXJ5",
        "LgogICBQYWxldHRlOiBjeWFub3R5cGUgYmx1ZSBvbiBjb29sIHBhcGVyIChsaWdodCkgb3IgY29vbCBpbmsgKGRhcmspLiAqLwo6",
        "cm9vdCB7CiAgLS1iZzogI2VlZjFmNDsgLS1zdXJmYWNlOiAjZmZmZmZmOyAtLXJhaXNlOiAjZjRmNmY4OyAtLWxpbmU6ICNkNWRj",
        "ZTM7CiAgLS1mZzogIzExMTgyMDsgLS1tdXRlOiAjNTY2MTcwOyAtLWFjY2VudDogIzFkNGVkODsgLS1vbi1hY2NlbnQ6ICNmZmZm",
        "ZmY7CiAgLS1vazogIzEyODA0YTsgLS1iYWQ6ICNiNDIzMTg7IC0td2FybjogIzlhNjcwMDsKICAtLWZvbnQtZGlzcGxheTogIkJy",
        "aWNvbGFnZSBHcm90ZXNxdWUiLCAiU2Vnb2UgVUkiLCBzeXN0ZW0tdWksIHNhbnMtc2VyaWY7CiAgLS1mb250LWJvZHk6ICJJbnN0",
        "cnVtZW50IFNhbnMiLCBzeXN0ZW0tdWksIC1hcHBsZS1zeXN0ZW0sICJTZWdvZSBVSSIsIFJvYm90bywgc2Fucy1zZXJpZjsKICAt",
        "LWZvbnQtbW9ubzogIklCTSBQbGV4IE1vbm8iLCB1aS1tb25vc3BhY2UsICJTRiBNb25vIiwgTWVubG8sIG1vbm9zcGFjZTsKICAt",
        "LXI6IDE0cHg7CiAgY29sb3Itc2NoZW1lOiBsaWdodDsKfQpAbWVkaWEgKHByZWZlcnMtY29sb3Itc2NoZW1lOiBkYXJrKSB7CiAg",
        "OnJvb3Q6bm90KFtkYXRhLXRoZW1lPSJsaWdodCJdKSB7CiAgICAtLWJnOiAjMGMxMTE3OyAtLXN1cmZhY2U6ICMxNDFiMjM7IC0t",
        "cmFpc2U6ICMxYTIzMmQ7IC0tbGluZTogIzI2MzIzZTsKICAgIC0tZmc6ICNlN2VjZjE7IC0tbXV0ZTogIzg3OTRhMzsgLS1hY2Nl",
        "bnQ6ICM2YzliZmY7IC0tb24tYWNjZW50OiAjMGExMjI2OwogICAgLS1vazogIzQzZDE4YTsgLS1iYWQ6ICNmZjdiNzI7IC0td2Fy",
        "bjogI2UzYjM0MTsgY29sb3Itc2NoZW1lOiBkYXJrOwogIH0KfQo6cm9vdFtkYXRhLXRoZW1lPSJkYXJrIl0gewogIC0tYmc6ICMw",
        "YzExMTc7IC0tc3VyZmFjZTogIzE0MWIyMzsgLS1yYWlzZTogIzFhMjMyZDsgLS1saW5lOiAjMjYzMjNlOwogIC0tZmc6ICNlN2Vj",
        "ZjE7IC0tbXV0ZTogIzg3OTRhMzsgLS1hY2NlbnQ6ICM2YzliZmY7IC0tb24tYWNjZW50OiAjMGExMjI2OwogIC0tb2s6ICM0M2Qx",
        "OGE7IC0tYmFkOiAjZmY3YjcyOyAtLXdhcm46ICNlM2IzNDE7IGNvbG9yLXNjaGVtZTogZGFyazsKfQoqIHsgYm94LXNpemluZzog",
        "Ym9yZGVyLWJveDsgLXdlYmtpdC10YXAtaGlnaGxpZ2h0LWNvbG9yOiB0cmFuc3BhcmVudDsgfQpbaGlkZGVuXSB7IGRpc3BsYXk6",
        "IG5vbmUgIWltcG9ydGFudDsgfQpodG1sLCBib2R5IHsgbWFyZ2luOiAwOyB9CmJvZHkgewogIGJhY2tncm91bmQ6IHZhcigtLWJn",
        "KTsgY29sb3I6IHZhcigtLWZnKTsgZm9udDogNDAwIDE2cHgvMS40NSB2YXIoLS1mb250LWJvZHkpOwogIG1pbi1oZWlnaHQ6IDEw",
        "MGR2aDsgcGFkZGluZy1pbmxpbmU6IDE2cHg7CiAgcGFkZGluZy1ibG9jazogY2FsYyhlbnYoc2FmZS1hcmVhLWluc2V0LXRvcCwg",
        "MHB4KSArIDBweCkgY2FsYyhlbnYoc2FmZS1hcmVhLWluc2V0LWJvdHRvbSwgMHB4KSArIDg0cHgpOwp9Ci53cmFwIHsgbWF4LXdp",
        "ZHRoOiA1NjBweDsgbWFyZ2luLWlubGluZTogYXV0bzsgfQpoMSwgaDIsIGgzIHsgZm9udC1mYW1pbHk6IHZhcigtLWZvbnQtZGlz",
        "cGxheSk7IG1hcmdpbjogMDsgdGV4dC13cmFwOiBiYWxhbmNlOyBsZXR0ZXItc3BhY2luZzogLTAuMDFlbTsgfQpidXR0b24sIGlu",
        "cHV0LCBzZWxlY3QsIHRleHRhcmVhIHsgZm9udDogaW5oZXJpdDsgY29sb3I6IGluaGVyaXQ7IH0KYnV0dG9uIHsgY3Vyc29yOiBw",
        "b2ludGVyOyB9Cjpmb2N1cy12aXNpYmxlIHsgb3V0bGluZTogMnB4IHNvbGlkIHZhcigtLWFjY2VudCk7IG91dGxpbmUtb2Zmc2V0",
        "OiAycHg7IH0KLm1vbm8geyBmb250LWZhbWlseTogdmFyKC0tZm9udC1tb25vKTsgZm9udC1zaXplOiAxMnB4OyBsZXR0ZXItc3Bh",
        "Y2luZzogMC4wMmVtOyB9Ci5tdXRlIHsgY29sb3I6IHZhcigtLW11dGUpOyB9Ci5zbWFsbCB7IGZvbnQtc2l6ZTogMTNweDsgfQoK",
        "LyogaGVhZGVyICovCi50b3AgeyBkaXNwbGF5OiBmbGV4OyBhbGlnbi1pdGVtczogY2VudGVyOyBqdXN0aWZ5LWNvbnRlbnQ6IHNw",
        "YWNlLWJldHdlZW47IGdhcDogMTJweDsgcGFkZGluZy1ibG9jazogMTZweCAxMnB4OyB9Ci5icmFuZCB7IGRpc3BsYXk6IGZsZXg7",
        "IGFsaWduLWl0ZW1zOiBjZW50ZXI7IGdhcDogMTBweDsgbWluLXdpZHRoOiAwOyB9Ci5icmFuZCBpbWcgeyB3aWR0aDogMzBweDsg",
        "aGVpZ2h0OiAzMHB4OyBib3JkZXItcmFkaXVzOiA4cHg7IH0KLmJyYW5kIGgxIHsgZm9udC1zaXplOiAyMXB4OyB3aGl0ZS1zcGFj",
        "ZTogbm93cmFwOyB9Ci5jaGlwIHsKICBkaXNwbGF5OiBpbmxpbmUtZmxleDsgYWxpZ24taXRlbXM6IGNlbnRlcjsgZ2FwOiA4cHg7",
        "IG1pbi1oZWlnaHQ6IDM2cHg7IG1heC13aWR0aDogNTglOwogIHBhZGRpbmc6IDAgMTJweDsgYm9yZGVyLXJhZGl1czogOTk5cHg7",
        "IGJvcmRlcjogMXB4IHNvbGlkIHZhcigtLWxpbmUpOyBiYWNrZ3JvdW5kOiB2YXIoLS1zdXJmYWNlKTsKICBmb250LXNpemU6IDEz",
        "cHg7IHdoaXRlLXNwYWNlOiBub3dyYXA7IG92ZXJmbG93OiBoaWRkZW47IHRleHQtb3ZlcmZsb3c6IGVsbGlwc2lzOwp9Ci5jaGlw",
        "IGkgeyB3aWR0aDogOHB4OyBoZWlnaHQ6IDhweDsgYm9yZGVyLXJhZGl1czogNTAlOyBiYWNrZ3JvdW5kOiB2YXIoLS1tdXRlKTsg",
        "ZmxleDogbm9uZTsgfQouY2hpcFtkYXRhLXM9Im9rIl0gaSB7IGJhY2tncm91bmQ6IHZhcigtLW9rKTsgfQouY2hpcFtkYXRhLXM9",
        "ImJhZCJdIGkgeyBiYWNrZ3JvdW5kOiB2YXIoLS1iYWQpOyB9Ci5jaGlwW2RhdGEtcz0iYnVzeSJdIGkgeyBiYWNrZ3JvdW5kOiB2",
        "YXIoLS13YXJuKTsgfQoKLyogc2VjdGlvbnMgKi8KLnZpZXcgeyBkaXNwbGF5OiBub25lOyBmbGV4LWRpcmVjdGlvbjogY29sdW1u",
        "OyBnYXA6IDE2cHg7IHBhZGRpbmctYm90dG9tOiA4cHg7IH0KLnZpZXcub24geyBkaXNwbGF5OiBmbGV4OyB9Ci5jYXJkIHsgYmFj",
        "a2dyb3VuZDogdmFyKC0tc3VyZmFjZSk7IGJvcmRlcjogMXB4IHNvbGlkIHZhcigtLWxpbmUpOyBib3JkZXItcmFkaXVzOiB2YXIo",
        "LS1yKTsgcGFkZGluZzogMTZweDsgZGlzcGxheTogZmxleDsgZmxleC1kaXJlY3Rpb246IGNvbHVtbjsgZ2FwOiAxNHB4OyB9Ci5m",
        "aWVsZCB7IGRpc3BsYXk6IGZsZXg7IGZsZXgtZGlyZWN0aW9uOiBjb2x1bW47IGdhcDogNnB4OyBtaW4td2lkdGg6IDA7IH0KLmZp",
        "ZWxkID4gbGFiZWwsIC5sYmwgeyBmb250LXNpemU6IDEycHg7IGZvbnQtd2VpZ2h0OiA2MDA7IHRleHQtdHJhbnNmb3JtOiB1cHBl",
        "cmNhc2U7IGxldHRlci1zcGFjaW5nOiAwLjA2ZW07IGNvbG9yOiB2YXIoLS1tdXRlKTsgfQp0ZXh0YXJlYSwgaW5wdXRbdHlwZT10",
        "ZXh0XSwgaW5wdXRbdHlwZT1wYXNzd29yZF0sIGlucHV0W3R5cGU9bnVtYmVyXSwgc2VsZWN0IHsKICB3aWR0aDogMTAwJTsgbWlu",
        "LWhlaWdodDogNDZweDsgcGFkZGluZzogMTFweCAxMnB4OyBib3JkZXItcmFkaXVzOiAxMHB4OwogIGJvcmRlcjogMXB4IHNvbGlk",
        "IHZhcigtLWxpbmUpOyBiYWNrZ3JvdW5kOiB2YXIoLS1yYWlzZSk7Cn0KdGV4dGFyZWEgeyByZXNpemU6IHZlcnRpY2FsOyBtaW4t",
        "aGVpZ2h0OiAxMDRweDsgbGluZS1oZWlnaHQ6IDEuNDsgfQpzZWxlY3QgeyBhcHBlYXJhbmNlOiBub25lOyBiYWNrZ3JvdW5kLWlt",
        "YWdlOiBsaW5lYXItZ3JhZGllbnQoNDVkZWcsIHRyYW5zcGFyZW50IDUwJSwgdmFyKC0tbXV0ZSkgNTAlKSwgbGluZWFyLWdyYWRp",
        "ZW50KDEzNWRlZywgdmFyKC0tbXV0ZSkgNTAlLCB0cmFuc3BhcmVudCA1MCUpOyBiYWNrZ3JvdW5kLXBvc2l0aW9uOiBjYWxjKDEw",
        "MCUgLSAxOHB4KSAyMHB4LCBjYWxjKDEwMCUgLSAxM3B4KSAyMHB4OyBiYWNrZ3JvdW5kLXNpemU6IDVweCA1cHg7IGJhY2tncm91",
        "bmQtcmVwZWF0OiBuby1yZXBlYXQ7IHBhZGRpbmctcmlnaHQ6IDM0cHg7IH0KaW5wdXRbdHlwZT1yYW5nZV0geyB3aWR0aDogMTAw",
        "JTsgYWNjZW50LWNvbG9yOiB2YXIoLS1hY2NlbnQpOyBtaW4taGVpZ2h0OiAzMnB4OyB9Ci5yb3cgeyBkaXNwbGF5OiBmbGV4OyBn",
        "YXA6IDEycHg7IGFsaWduLWl0ZW1zOiBjZW50ZXI7IH0KLnJvdy53cmFwMiB7IGZsZXgtd3JhcDogd3JhcDsgfQouZ3JvdyB7IGZs",
        "ZXg6IDE7IG1pbi13aWR0aDogMDsgfQoudmFsIHsgZm9udC1mYW1pbHk6IHZhcigtLWZvbnQtbW9ubyk7IGZvbnQtc2l6ZTogMTNw",
        "eDsgbWluLXdpZHRoOiAzOHB4OyB0ZXh0LWFsaWduOiByaWdodDsgfQoKLnNlZyB7IGRpc3BsYXk6IGdyaWQ7IGdyaWQtdGVtcGxh",
        "dGUtY29sdW1uczogcmVwZWF0KDMsIDFmcik7IGJvcmRlcjogMXB4IHNvbGlkIHZhcigtLWxpbmUpOyBib3JkZXItcmFkaXVzOiAx",
        "MHB4OyBvdmVyZmxvdzogaGlkZGVuOyBiYWNrZ3JvdW5kOiB2YXIoLS1yYWlzZSk7IH0KLnNlZyBpbnB1dCB7IHBvc2l0aW9uOiBh",
        "YnNvbHV0ZTsgb3BhY2l0eTogMDsgcG9pbnRlci1ldmVudHM6IG5vbmU7IH0KLnNlZyBsYWJlbCB7IHRleHQtYWxpZ246IGNlbnRl",
        "cjsgcGFkZGluZzogMTFweCA2cHg7IGZvbnQtc2l6ZTogMTRweDsgY3Vyc29yOiBwb2ludGVyOyBib3JkZXItcmlnaHQ6IDFweCBz",
        "b2xpZCB2YXIoLS1saW5lKTsgfQouc2VnIGxhYmVsOmxhc3Qtb2YtdHlwZSB7IGJvcmRlci1yaWdodDogMDsgfQouc2VnIGlucHV0",
        "OmNoZWNrZWQgKyBsYWJlbCB7IGJhY2tncm91bmQ6IHZhcigtLWFjY2VudCk7IGNvbG9yOiB2YXIoLS1vbi1hY2NlbnQpOyBmb250",
        "LXdlaWdodDogNjAwOyB9Ci5zZWcgaW5wdXQ6Zm9jdXMtdmlzaWJsZSArIGxhYmVsIHsgb3V0bGluZTogMnB4IHNvbGlkIHZhcigt",
        "LWFjY2VudCk7IG91dGxpbmUtb2Zmc2V0OiAtMnB4OyB9Cgouc3cgeyBkaXNwbGF5OiBmbGV4OyBhbGlnbi1pdGVtczogZmxleC1z",
        "dGFydDsgZ2FwOiAxMnB4OyB9Ci5zdyBpbnB1dCB7IHdpZHRoOiAyMnB4OyBoZWlnaHQ6IDIycHg7IG1hcmdpbjogMnB4IDAgMDsg",
        "YWNjZW50LWNvbG9yOiB2YXIoLS1hY2NlbnQpOyBmbGV4OiBub25lOyB9Ci5zdyBiIHsgZm9udC13ZWlnaHQ6IDYwMDsgZGlzcGxh",
        "eTogYmxvY2s7IH0KCmRldGFpbHMgeyBib3JkZXItdG9wOiAxcHggc29saWQgdmFyKC0tbGluZSk7IHBhZGRpbmctdG9wOiAxMnB4",
        "OyB9CnN1bW1hcnkgeyBjdXJzb3I6IHBvaW50ZXI7IGZvbnQtd2VpZ2h0OiA2MDA7IG1pbi1oZWlnaHQ6IDMycHg7IGRpc3BsYXk6",
        "IGZsZXg7IGFsaWduLWl0ZW1zOiBjZW50ZXI7IH0KZGV0YWlsc1tvcGVuXSBzdW1tYXJ5IHsgbWFyZ2luLWJvdHRvbTogMTBweDsg",
        "fQouc3RhY2sgeyBkaXNwbGF5OiBmbGV4OyBmbGV4LWRpcmVjdGlvbjogY29sdW1uOyBnYXA6IDE0cHg7IH0KCi5idG4geyBtaW4t",
        "aGVpZ2h0OiA0OHB4OyBwYWRkaW5nOiAwIDE4cHg7IGJvcmRlci1yYWRpdXM6IDEycHg7IGJvcmRlcjogMXB4IHNvbGlkIHZhcigt",
        "LWxpbmUpOyBiYWNrZ3JvdW5kOiB2YXIoLS1zdXJmYWNlKTsgZm9udC13ZWlnaHQ6IDYwMDsgfQouYnRuLnByaW1hcnkgeyBiYWNr",
        "Z3JvdW5kOiB2YXIoLS1hY2NlbnQpOyBjb2xvcjogdmFyKC0tb24tYWNjZW50KTsgYm9yZGVyLWNvbG9yOiB0cmFuc3BhcmVudDsg",
        "fQouYnRuLmRhbmdlciB7IGNvbG9yOiB2YXIoLS1iYWQpOyB9Ci5idG46ZGlzYWJsZWQgeyBvcGFjaXR5OiAwLjU1OyBjdXJzb3I6",
        "IG5vdC1hbGxvd2VkOyB9Ci5idG4uYmxvY2sgeyB3aWR0aDogMTAwJTsgfQouYnRucyB7IGRpc3BsYXk6IGZsZXg7IGdhcDogMTBw",
        "eDsgZmxleC13cmFwOiB3cmFwOyB9Ci5idG5zIC5idG4geyBmbGV4OiAxOyBtaW4td2lkdGg6IDEyMHB4OyB9CgovKiByZXN1bHQg",
        "Ki8KLmJhciB7IGhlaWdodDogOHB4OyBib3JkZXItcmFkaXVzOiA5OXB4OyBiYWNrZ3JvdW5kOiB2YXIoLS1yYWlzZSk7IGJvcmRl",
        "cjogMXB4IHNvbGlkIHZhcigtLWxpbmUpOyBvdmVyZmxvdzogaGlkZGVuOyB9Ci5iYXIgPiBkaXYgeyBoZWlnaHQ6IDEwMCU7IHdp",
        "ZHRoOiAwOyBiYWNrZ3JvdW5kOiB2YXIoLS1hY2NlbnQpOyB0cmFuc2l0aW9uOiB3aWR0aCAuMzVzIGVhc2U7IH0KLnNob3QgeyB3",
        "aWR0aDogMTAwJTsgYm9yZGVyLXJhZGl1czogMTJweDsgZGlzcGxheTogYmxvY2s7IGJhY2tncm91bmQ6IHZhcigtLXJhaXNlKTsg",
        "fQoubXNnIHsgZm9udC1zaXplOiAxNHB4OyB9Ci5tc2cuYmFkIHsgY29sb3I6IHZhcigtLWJhZCk7IH0KCi8qIGdhbGxlcnk6IGNv",
        "bnRhY3Qgc2hlZXQgKi8KLnNoZWV0IHsgZGlzcGxheTogZ3JpZDsgZ3JpZC10ZW1wbGF0ZS1jb2x1bW5zOiByZXBlYXQoMiwgMWZy",
        "KTsgZ2FwOiAxMHB4OyB9Ci5mcmFtZSB7IGJhY2tncm91bmQ6ICMwYjBmMTQ7IGJvcmRlci1yYWRpdXM6IDZweDsgcGFkZGluZzog",
        "N3B4IDdweCA2cHg7IGJvcmRlcjogMDsgdGV4dC1hbGlnbjogbGVmdDsgY29sb3I6ICNjZmQ4ZTM7IGRpc3BsYXk6IGZsZXg7IGZs",
        "ZXgtZGlyZWN0aW9uOiBjb2x1bW47IGdhcDogNnB4OyBtaW4td2lkdGg6IDA7IH0KLmZyYW1lIGltZyB7IHdpZHRoOiAxMDAlOyBh",
        "c3BlY3QtcmF0aW86IDIvMzsgb2JqZWN0LWZpdDogY292ZXI7IGJvcmRlci1yYWRpdXM6IDJweDsgZGlzcGxheTogYmxvY2s7IGJh",
        "Y2tncm91bmQ6ICMxYjI0MzA7IH0KLmZyYW1lIC5tZXRhIHsgZGlzcGxheTogZmxleDsganVzdGlmeS1jb250ZW50OiBzcGFjZS1i",
        "ZXR3ZWVuOyBnYXA6IDZweDsgZm9udC1mYW1pbHk6IHZhcigtLWZvbnQtbW9ubyk7IGZvbnQtc2l6ZTogMTFweDsgY29sb3I6ICM5",
        "ZmIwYzM7IH0KLmVtcHR5IHsgdGV4dC1hbGlnbjogY2VudGVyOyBwYWRkaW5nOiAzNnB4IDEycHg7IH0KLmVtcHR5IGgyIHsgZm9u",
        "dC1zaXplOiAyMHB4OyBtYXJnaW4tYm90dG9tOiA4cHg7IH0KCmRpYWxvZyB7IHdpZHRoOiBtaW4oNTYwcHgsIGNhbGMoMTAwJSAt",
        "IDI0cHgpKTsgbWF4LWhlaWdodDogY2FsYygxMDBkdmggLSAyNHB4KTsgb3ZlcmZsb3c6IGF1dG87IGJvcmRlcjogMXB4IHNvbGlk",
        "IHZhcigtLWxpbmUpOyBib3JkZXItcmFkaXVzOiAxNnB4OyBwYWRkaW5nOiAxNHB4OyBiYWNrZ3JvdW5kOiB2YXIoLS1zdXJmYWNl",
        "KTsgY29sb3I6IHZhcigtLWZnKTsgfQpkaWFsb2c6OmJhY2tkcm9wIHsgYmFja2dyb3VuZDogcmdiYSgwLCAwLCAwLCAuNik7IH0K",
        "ZGlhbG9nIC5zaG90IHsgbWF4LWhlaWdodDogNjBkdmg7IG9iamVjdC1maXQ6IGNvbnRhaW47IH0KCi8qIHRhYnMgKi8KLnRhYnMg",
        "eyBwb3NpdGlvbjogZml4ZWQ7IGxlZnQ6IDA7IHJpZ2h0OiAwOyBib3R0b206IDA7IGJhY2tncm91bmQ6IHZhcigtLXN1cmZhY2Up",
        "OyBib3JkZXItdG9wOiAxcHggc29saWQgdmFyKC0tbGluZSk7IHBhZGRpbmc6IDZweCAxMnB4IGNhbGMoZW52KHNhZmUtYXJlYS1p",
        "bnNldC1ib3R0b20sIDBweCkgKyA2cHgpOyB6LWluZGV4OiA1OyB9Ci50YWJzIC5pbiB7IG1heC13aWR0aDogNTYwcHg7IG1hcmdp",
        "bi1pbmxpbmU6IGF1dG87IGRpc3BsYXk6IGdyaWQ7IGdyaWQtdGVtcGxhdGUtY29sdW1uczogcmVwZWF0KDMsIDFmcik7IGdhcDog",
        "NnB4OyB9Ci50YWIgeyBtaW4taGVpZ2h0OiA1MnB4OyBib3JkZXI6IDA7IGJhY2tncm91bmQ6IHRyYW5zcGFyZW50OyBib3JkZXIt",
        "cmFkaXVzOiAxMnB4OyBkaXNwbGF5OiBmbGV4OyBmbGV4LWRpcmVjdGlvbjogY29sdW1uOyBhbGlnbi1pdGVtczogY2VudGVyOyBq",
        "dXN0aWZ5LWNvbnRlbnQ6IGNlbnRlcjsgZ2FwOiAycHg7IGZvbnQtc2l6ZTogMTJweDsgZm9udC13ZWlnaHQ6IDYwMDsgY29sb3I6",
        "IHZhcigtLW11dGUpOyB9Ci50YWIgc3ZnIHsgd2lkdGg6IDIycHg7IGhlaWdodDogMjJweDsgc3Ryb2tlOiBjdXJyZW50Q29sb3I7",
        "IGZpbGw6IG5vbmU7IHN0cm9rZS13aWR0aDogMS44OyBzdHJva2UtbGluZWNhcDogcm91bmQ7IHN0cm9rZS1saW5lam9pbjogcm91",
        "bmQ7IH0KLnRhYlthcmlhLXNlbGVjdGVkPSJ0cnVlIl0geyBjb2xvcjogdmFyKC0tYWNjZW50KTsgYmFja2dyb3VuZDogdmFyKC0t",
        "cmFpc2UpOyB9Cm9sLnN0ZXBzIHsgbWFyZ2luOiAwOyBwYWRkaW5nLWxlZnQ6IDIwcHg7IGRpc3BsYXk6IGZsZXg7IGZsZXgtZGly",
        "ZWN0aW9uOiBjb2x1bW47IGdhcDogNnB4OyB9Ci5zZWcudHdvIHsgZ3JpZC10ZW1wbGF0ZS1jb2x1bW5zOiByZXBlYXQoMiwgMWZy",
        "KTsgfQpib2R5W2RhdGEtbW9kZT0icGhvbmUiXSAuY29sYWItb25seSB7IGRpc3BsYXk6IG5vbmUgIWltcG9ydGFudDsgfQoudG9h",
        "c3QgeyBwb3NpdGlvbjogZml4ZWQ7IGxlZnQ6IDUwJTsgdHJhbnNmb3JtOiB0cmFuc2xhdGVYKC01MCUpOyBib3R0b206IGNhbGMo",
        "ZW52KHNhZmUtYXJlYS1pbnNldC1ib3R0b20sIDBweCkgKyA3OHB4KTsgbWF4LXdpZHRoOiBtaW4oNTIwcHgsIGNhbGMoMTAwJSAt",
        "IDMycHgpKTsgYmFja2dyb3VuZDogdmFyKC0tZmcpOyBjb2xvcjogdmFyKC0tYmcpOyBwYWRkaW5nOiAxMHB4IDE2cHg7IGJvcmRl",
        "ci1yYWRpdXM6IDk5OXB4OyBmb250LXNpemU6IDE0cHg7IHotaW5kZXg6IDIwOyB9Ci5ub3RlIHsgYmFja2dyb3VuZDogdmFyKC0t",
        "cmFpc2UpOyBib3JkZXI6IDFweCBzb2xpZCB2YXIoLS1saW5lKTsgYm9yZGVyLXJhZGl1czogMTBweDsgcGFkZGluZzogMTJweDsg",
        "Zm9udC1zaXplOiAxNHB4OyB9CkBtZWRpYSAocHJlZmVycy1yZWR1Y2VkLW1vdGlvbjogcmVkdWNlKSB7IC5iYXIgPiBkaXYgeyB0",
        "cmFuc2l0aW9uOiBub25lOyB9IH0KPC9zdHlsZT4KPC9oZWFkPgo8Ym9keT4KPGRpdiBjbGFzcz0id3JhcCI+CiAgPGhlYWRlciBj",
        "bGFzcz0idG9wIj4KICAgIDxkaXYgY2xhc3M9ImJyYW5kIj48aW1nIHNyYz0iaWNvbi0xOTIucG5nIiBhbHQ9IiI+PGgxPkxvUkEg",
        "U3R1ZGlvPC9oMT48L2Rpdj4KICAgIDxidXR0b24gY2xhc3M9ImNoaXAiIGlkPSJjaGlwIiBkYXRhLXM9IiIgdHlwZT0iYnV0dG9u",
        "IiBhcmlhLWxhYmVsPSJDb25uZWN0aW9uIHN0YXR1cyI+PGk+PC9pPjxzcGFuIGlkPSJjaGlwVGV4dCI+Tm90IGNvbm5lY3RlZDwv",
        "c3Bhbj48L2J1dHRvbj4KICA8L2hlYWRlcj4KCiAgPCEtLSBDUkVBVEUgLS0+CiAgPHNlY3Rpb24gY2xhc3M9InZpZXcgb24iIGlk",
        "PSJ2LWNyZWF0ZSI+CiAgICA8ZGl2IGNsYXNzPSJjYXJkIj4KICAgICAgPGRpdiBjbGFzcz0ibm90ZSBzbWFsbCIgaWQ9InBob25l",
        "SGludCIgaGlkZGVuPlBob3RvcyBhcmUgbWFkZSBvbiB0aGlzIHBob25lIChzcXVhcmUsIDUxMiBweCkuIEl0IGlzIHNsb3dlciB0",
        "aGFuIENvbGFiLiBZb3VyIG1vZGVsJ3MgdHJpZ2dlciB3b3JkIGlzIGFkZGVkIGF1dG9tYXRpY2FsbHkuPC9kaXY+CiAgICAgIDxk",
        "aXYgY2xhc3M9ImZpZWxkIj4KICAgICAgICA8bGFiZWwgZm9yPSJwcm9tcHQiPkRlc2NyaWJlIHRoZSBwaG90bzwvbGFiZWw+CiAg",
        "ICAgICAgPHRleHRhcmVhIGlkPSJwcm9tcHQiIHBsYWNlaG9sZGVyPSJlLmcuIHNtaWxpbmcgaW4gYSBwYXJrLCBjYXN1YWwgY2xv",
        "dGhlcywgZXZlbmluZyBzdW5saWdodCYjMTA7KEhpbmRpIC8gSGluZ2xpc2ggd29ya3MgdG9vKSI+PC90ZXh0YXJlYT4KICAgICAg",
        "PC9kaXY+CiAgICAgIDxkaXYgY2xhc3M9ImZpZWxkIGNvbGFiLW9ubHkiPgogICAgICAgIDxsYWJlbCBmb3I9ImxvcmEiPkxvUkEg",
        "bW9kZWw8L2xhYmVsPgogICAgICAgIDxzZWxlY3QgaWQ9ImxvcmEiPjxvcHRpb24gdmFsdWU9IiI+Tm9uZSAoYmFzZSBtb2RlbCk8",
        "L29wdGlvbj48L3NlbGVjdD4KICAgICAgICA8ZGl2IGNsYXNzPSJzbWFsbCBtdXRlIiBpZD0idHJpZyIgaGlkZGVuPjwvZGl2Pgog",
        "ICAgICA8L2Rpdj4KICAgICAgPGRpdiBjbGFzcz0iZmllbGQiIGlkPSJzY2FsZUJveCIgaGlkZGVuPgogICAgICAgIDxsYWJlbCBm",
        "b3I9InNjYWxlIj5Mb1JBIHN0cmVuZ3RoPC9sYWJlbD4KICAgICAgICA8ZGl2IGNsYXNzPSJyb3ciPjxpbnB1dCBpZD0ic2NhbGUi",
        "IHR5cGU9InJhbmdlIiBtaW49IjAiIG1heD0iMS41IiBzdGVwPSIwLjA1IiB2YWx1ZT0iMC41IiBjbGFzcz0iZ3JvdyI+PHNwYW4g",
        "Y2xhc3M9InZhbCIgaWQ9InNjYWxlViI+MC41MDwvc3Bhbj48L2Rpdj4KICAgICAgPC9kaXY+CiAgICAgIDxkaXYgY2xhc3M9ImZp",
        "ZWxkIGNvbGFiLW9ubHkiPgogICAgICAgIDxzcGFuIGNsYXNzPSJsYmwiIGlkPSJzaGFwZUwiPlNoYXBlPC9zcGFuPgogICAgICAg",
        "IDxkaXYgY2xhc3M9InNlZyIgcm9sZT0icmFkaW9ncm91cCIgYXJpYS1sYWJlbGxlZGJ5PSJzaGFwZUwiPgogICAgICAgICAgPGlu",
        "cHV0IHR5cGU9InJhZGlvIiBuYW1lPSJzaGFwZSIgaWQ9InMxIiB2YWx1ZT0icG9ydHJhaXQiIGNoZWNrZWQ+PGxhYmVsIGZvcj0i",
        "czEiPlBvcnRyYWl0PC9sYWJlbD4KICAgICAgICAgIDxpbnB1dCB0eXBlPSJyYWRpbyIgbmFtZT0ic2hhcGUiIGlkPSJzMiIgdmFs",
        "dWU9InNxdWFyZSI+PGxhYmVsIGZvcj0iczIiPlNxdWFyZTwvbGFiZWw+CiAgICAgICAgICA8aW5wdXQgdHlwZT0icmFkaW8iIG5h",
        "bWU9InNoYXBlIiBpZD0iczMiIHZhbHVlPSJsYW5kc2NhcGUiPjxsYWJlbCBmb3I9InMzIj5XaWRlPC9sYWJlbD4KICAgICAgICA8",
        "L2Rpdj4KICAgICAgPC9kaXY+CiAgICAgIDxsYWJlbCBjbGFzcz0ic3cgY29sYWItb25seSI+PGlucHV0IHR5cGU9ImNoZWNrYm94",
        "IiBpZD0ic21hcnQiIGNoZWNrZWQ+PHNwYW4+PGI+U21hcnQgcHJvbXB0PC9iPjxzcGFuIGNsYXNzPSJzbWFsbCBtdXRlIj5BSSBl",
        "eHBhbmRzIHlvdXIgaWRlYSBpbnRvIGEgZGV0YWlsZWQgcGhvdG8gcHJvbXB0Ljwvc3Bhbj48L3NwYW4+PC9sYWJlbD4KICAgICAg",
        "PGxhYmVsIGNsYXNzPSJzdyBjb2xhYi1vbmx5Ij48aW5wdXQgdHlwZT0iY2hlY2tib3giIGlkPSJoaXJlcyIgY2hlY2tlZD48c3Bh",
        "bj48Yj5TaGFycGVyIHJlc3VsdDwvYj48c3BhbiBjbGFzcz0ic21hbGwgbXV0ZSI+RXh0cmEgcGFzcyBmb3IgZGV0YWlsLiBUYWtl",
        "cyBhIGJpdCBsb25nZXIuPC9zcGFuPjwvc3Bhbj48L2xhYmVsPgogICAgICA8ZGV0YWlscz4KICAgICAgICA8c3VtbWFyeT5Nb3Jl",
        "IHNldHRpbmdzPC9zdW1tYXJ5PgogICAgICAgIDxkaXYgY2xhc3M9InN0YWNrIj4KICAgICAgICAgIDxkaXYgY2xhc3M9ImZpZWxk",
        "Ij48bGFiZWwgZm9yPSJzdGVwcyI+U3RlcHM8L2xhYmVsPjxkaXYgY2xhc3M9InJvdyI+PGlucHV0IGlkPSJzdGVwcyIgdHlwZT0i",
        "cmFuZ2UiIG1pbj0iMjAiIG1heD0iNDUiIHN0ZXA9IjEiIHZhbHVlPSIzMCIgY2xhc3M9Imdyb3ciPjxzcGFuIGNsYXNzPSJ2YWwi",
        "IGlkPSJzdGVwc1YiPjMwPC9zcGFuPjwvZGl2PjwvZGl2PgogICAgICAgICAgPGRpdiBjbGFzcz0iZmllbGQgY29sYWItb25seSI+",
        "PGxhYmVsIGZvcj0iZ3VpZCI+UHJvbXB0IHN0cmljdG5lc3M8L2xhYmVsPjxkaXYgY2xhc3M9InJvdyI+PGlucHV0IGlkPSJndWlk",
        "IiB0eXBlPSJyYW5nZSIgbWluPSI0IiBtYXg9IjEwIiBzdGVwPSIwLjUiIHZhbHVlPSI2IiBjbGFzcz0iZ3JvdyI+PHNwYW4gY2xh",
        "c3M9InZhbCIgaWQ9Imd1aWRWIj42LjA8L3NwYW4+PC9kaXY+PC9kaXY+CiAgICAgICAgICA8ZGl2IGNsYXNzPSJmaWVsZCI+PGxh",
        "YmVsIGZvcj0ic2VlZCI+U2VlZCAoYmxhbmsgPSByYW5kb20pPC9sYWJlbD48aW5wdXQgaWQ9InNlZWQiIHR5cGU9Im51bWJlciIg",
        "aW5wdXRtb2RlPSJudW1lcmljIiBtaW49IjAiIHBsYWNlaG9sZGVyPSJyYW5kb20iPjwvZGl2PgogICAgICAgICAgPGRpdiBjbGFz",
        "cz0iZmllbGQgY29sYWItb25seSI+PGxhYmVsIGZvcj0ibmVnIj5Bdm9pZCBpbiB0aGUgcGhvdG88L2xhYmVsPjxpbnB1dCBpZD0i",
        "bmVnIiB0eXBlPSJ0ZXh0IiBwbGFjZWhvbGRlcj0iZS5nLiBoYXQsIGdsYXNzZXMiPjwvZGl2PgogICAgICAgIDwvZGl2PgogICAg",
        "ICA8L2RldGFpbHM+CiAgICAgIDxidXR0b24gY2xhc3M9ImJ0biBwcmltYXJ5IGJsb2NrIiBpZD0iZ28iIHR5cGU9ImJ1dHRvbiI+",
        "R2VuZXJhdGU8L2J1dHRvbj4KICAgIDwvZGl2PgoKICAgIDxkaXYgY2xhc3M9ImNhcmQiIGlkPSJyZXN1bHQiIGhpZGRlbj4KICAg",
        "ICAgPGRpdiBjbGFzcz0icm93Ij48c3BhbiBjbGFzcz0ibXNnIGdyb3ciIGlkPSJzdGF0dXMiIHJvbGU9InN0YXR1cyIgYXJpYS1s",
        "aXZlPSJwb2xpdGUiPjwvc3Bhbj48c3BhbiBjbGFzcz0ibW9ubyBtdXRlIiBpZD0idGltZXIiPjwvc3Bhbj48L2Rpdj4KICAgICAg",
        "PGRpdiBjbGFzcz0iYmFyIiBpZD0iYmFyIj48ZGl2PjwvZGl2PjwvZGl2PgogICAgICA8ZGl2IGNsYXNzPSJidG5zIiBpZD0iY2Fu",
        "Y2VsUm93Ij48YnV0dG9uIGNsYXNzPSJidG4gZGFuZ2VyIiBpZD0iY2FuY2VsIiB0eXBlPSJidXR0b24iPkNhbmNlbDwvYnV0dG9u",
        "PjwvZGl2PgogICAgICA8aW1nIGNsYXNzPSJzaG90IiBpZD0ic2hvdCIgYWx0PSJHZW5lcmF0ZWQgcGhvdG8iIGhpZGRlbj4KICAg",
        "ICAgPGRpdiBjbGFzcz0iYnRucyIgaWQ9ImFmdGVyIiBoaWRkZW4+CiAgICAgICAgPGJ1dHRvbiBjbGFzcz0iYnRuIiBpZD0ic2F2",
        "ZSIgdHlwZT0iYnV0dG9uIj5TYXZlPC9idXR0b24+CiAgICAgICAgPGJ1dHRvbiBjbGFzcz0iYnRuIiBpZD0ic2hhcmUiIHR5cGU9",
        "ImJ1dHRvbiI+U2hhcmU8L2J1dHRvbj4KICAgICAgICA8YnV0dG9uIGNsYXNzPSJidG4iIGlkPSJhZ2FpbiIgdHlwZT0iYnV0dG9u",
        "Ij5OZXcgc2VlZDwvYnV0dG9uPgogICAgICA8L2Rpdj4KICAgICAgPGRpdiBjbGFzcz0ibW9ubyBtdXRlIiBpZD0iaW5mbyIgaGlk",
        "ZGVuPjwvZGl2PgogICAgPC9kaXY+CiAgPC9zZWN0aW9uPgoKICA8IS0tIEdBTExFUlkgLS0+CiAgPHNlY3Rpb24gY2xhc3M9InZp",
        "ZXciIGlkPSJ2LWdhbGxlcnkiPgogICAgPGRpdiBjbGFzcz0icm93Ij48aDIgY2xhc3M9Imdyb3ciPkdhbGxlcnk8L2gyPjxzcGFu",
        "IGNsYXNzPSJtb25vIG11dGUiIGlkPSJjb3VudCI+PC9zcGFuPjwvZGl2PgogICAgPGRpdiBjbGFzcz0ic2hlZXQiIGlkPSJzaGVl",
        "dCI+PC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJjYXJkIGVtcHR5IiBpZD0iZW1wdHkiIGhpZGRlbj4KICAgICAgPGgyPk5vIHBob3Rv",
        "cyB5ZXQ8L2gyPgogICAgICA8cCBjbGFzcz0ibXV0ZSIgc3R5bGU9Im1hcmdpbjowIj5QaG90b3MgeW91IG1ha2UgYXBwZWFyIGhl",
        "cmUuIFRoZXkgc3RheSBvbiB0aGlzIHBob25lIG9ubHkuPC9wPgogICAgPC9kaXY+CiAgPC9zZWN0aW9uPgoKICA8IS0tIENPTk5F",
        "Q1QgLS0+CiAgPHNlY3Rpb24gY2xhc3M9InZpZXciIGlkPSJ2LWNvbm5lY3QiPgogICAgPGgyPkNvbm5lY3Q8L2gyPgogICAgPGRp",
        "diBjbGFzcz0iY2FyZCIgaWQ9Im1vZGVDYXJkIiBoaWRkZW4+CiAgICAgIDxoMyBpZD0ibW9kZUwiPldoZXJlIHRvIG1ha2UgcGhv",
        "dG9zPC9oMz4KICAgICAgPGRpdiBjbGFzcz0ic2VnIHR3byIgcm9sZT0icmFkaW9ncm91cCIgYXJpYS1sYWJlbGxlZGJ5PSJtb2Rl",
        "TCI+CiAgICAgICAgPGlucHV0IHR5cGU9InJhZGlvIiBuYW1lPSJtb2RlIiBpZD0ibTEiIHZhbHVlPSJjb2xhYiIgY2hlY2tlZD48",
        "bGFiZWwgZm9yPSJtMSI+Q29sYWIgR1BVPC9sYWJlbD4KICAgICAgICA8aW5wdXQgdHlwZT0icmFkaW8iIG5hbWU9Im1vZGUiIGlk",
        "PSJtMiIgdmFsdWU9InBob25lIj48bGFiZWwgZm9yPSJtMiI+VGhpcyBwaG9uZTwvbGFiZWw+CiAgICAgIDwvZGl2PgogICAgPC9k",
        "aXY+CiAgICA8ZGl2IGNsYXNzPSJjYXJkIj4KICAgICAgPGRpdiBjbGFzcz0iZmllbGQiPgogICAgICAgIDxsYWJlbCBmb3I9Imxp",
        "bmsiPkNvbGFiIGxpbms8L2xhYmVsPgogICAgICAgIDxpbnB1dCBpZD0ibGluayIgdHlwZT0idGV4dCIgaW5wdXRtb2RlPSJ1cmwi",
        "IGF1dG9jb21wbGV0ZT0ib2ZmIiBhdXRvY2FwaXRhbGl6ZT0ib2ZmIiBzcGVsbGNoZWNrPSJmYWxzZSIgcGxhY2Vob2xkZXI9IlBh",
        "c3RlIHRoZSBmdWxsIGxpbmsgZnJvbSBDb2xhYiI+CiAgICAgIDwvZGl2PgogICAgICA8ZGl2IGNsYXNzPSJmaWVsZCI+CiAgICAg",
        "ICAgPGxhYmVsIGZvcj0ia2V5Ij5LZXkgKGZpbGxlZCBhdXRvbWF0aWNhbGx5IGlmIHRoZSBsaW5rIGhhcyBpdCk8L2xhYmVsPgog",
        "ICAgICAgIDxpbnB1dCBpZD0ia2V5IiB0eXBlPSJwYXNzd29yZCIgYXV0b2NvbXBsZXRlPSJvZmYiIHBsYWNlaG9sZGVyPSJrZXki",
        "PgogICAgICA8L2Rpdj4KICAgICAgPGJ1dHRvbiBjbGFzcz0iYnRuIHByaW1hcnkgYmxvY2siIGlkPSJjb25uZWN0IiB0eXBlPSJi",
        "dXR0b24iPkNvbm5lY3Q8L2J1dHRvbj4KICAgICAgPGRpdiBjbGFzcz0ibXNnIiBpZD0iY21zZyIgcm9sZT0ic3RhdHVzIiBhcmlh",
        "LWxpdmU9InBvbGl0ZSI+PC9kaXY+CiAgICA8L2Rpdj4KICAgIDxkaXYgY2xhc3M9ImNhcmQiPgogICAgICA8aDM+SG93IHRvIGdl",
        "dCB0aGUgbGluazwvaDM+CiAgICAgIDxvbCBjbGFzcz0ic3RlcHMgc21hbGwiPgogICAgICAgIDxsaT5PcGVuIHRoZSBDb2xhYiBu",
        "b3RlYm9vayBhbmQgcnVuIHRoZSA8Yj5zZXJ2ZXIgY2VsbDwvYj4uPC9saT4KICAgICAgICA8bGk+V2FpdCB1bnRpbCBpdCBwcmlu",
        "dHMgPGI+UEhPTkUgTUUgWUUgTElOSyBLSE9MTzwvYj4gYW5kIGEgbGluay48L2xpPgogICAgICAgIDxsaT5PcGVuIHRoYXQgbGlu",
        "ayBoZXJlIChvciBwYXN0ZSBpdCBhYm92ZSkuIENvbGFiIGdpdmVzIGEgbmV3IGxpbmsgZXZlcnkgdGltZSB5b3UgcmVzdGFydCBp",
        "dC48L2xpPgogICAgICA8L29sPgogICAgICA8YnV0dG9uIGNsYXNzPSJidG4gYmxvY2siIGlkPSJpbnN0YWxsIiB0eXBlPSJidXR0",
        "b24iIGhpZGRlbj5JbnN0YWxsIHRoaXMgYXBwPC9idXR0b24+CiAgICAgIDxkaXYgY2xhc3M9Im5vdGUgc21hbGwiIGlkPSJpbnN0",
        "YWxsSGludCI+VG8gaW5zdGFsbDogaW4gQ2hyb21lIG9wZW4gdGhlIG1lbnUgYW5kIGNob29zZSA8Yj5BZGQgdG8gSG9tZSBzY3Jl",
        "ZW48L2I+IG9yIDxiPkluc3RhbGwgYXBwPC9iPi48L2Rpdj4KICAgIDwvZGl2PgogICAgPGRpdiBjbGFzcz0iY2FyZCIgaWQ9InBo",
        "b25lQ2FyZCI+CiAgICAgIDxoMz5SdW4gb24gdGhpcyBwaG9uZTwvaDM+CiAgICAgIDxwIGNsYXNzPSJzbWFsbCBtdXRlIiBpZD0i",
        "cGhvbmVOb3RlIiBzdHlsZT0ibWFyZ2luOjAiPlBob3RvcyBhcmUgbWFkZSBvbiB0aGUgQ29sYWIgR1BVLCBzbyB0aGlzIG5lZWRz",
        "IGludGVybmV0IGFuZCBhIHJ1bm5pbmcgQ29sYWIuIE1ha2luZyBwaG90b3Mgb24gdGhlIHBob25lIGl0c2VsZiBpcyBvbmx5IGlu",
        "IHRoZSBGdWxsIEFuZHJvaWQgYXBwLjwvcD4KICAgICAgPGRpdiBjbGFzcz0ic3RhY2siIGlkPSJwaG9uZUJveCIgaGlkZGVuPgog",
        "ICAgICAgIDxkaXYgY2xhc3M9Im1zZyIgaWQ9InBzdGF0ZSI+PC9kaXY+CiAgICAgICAgPGRpdiBjbGFzcz0iZmllbGQiPjxsYWJl",
        "bCBmb3I9InB0cmlnIj5UcmlnZ2VyIHdvcmQgb2YgdGhlIG1vZGVsPC9sYWJlbD48aW5wdXQgaWQ9InB0cmlnIiB0eXBlPSJ0ZXh0",
        "IiBhdXRvY29tcGxldGU9Im9mZiIgYXV0b2NhcGl0YWxpemU9Im9mZiIgc3BlbGxjaGVjaz0iZmFsc2UiIHBsYWNlaG9sZGVyPSJl",
        "LmcuIG9od3ggcGVyc29uIj48L2Rpdj4KICAgICAgICA8YnV0dG9uIGNsYXNzPSJidG4gYmxvY2siIGlkPSJwaW1wb3J0IiB0eXBl",
        "PSJidXR0b24iPkltcG9ydCBtb2RlbCBmaWxlICguemlwKTwvYnV0dG9uPgogICAgICAgIDxkaXYgY2xhc3M9ImZpZWxkIj48bGFi",
        "ZWwgZm9yPSJwdXJsIj5PciBkb3dubG9hZCBmcm9tIGEgbGluayAoaHR0cHMpPC9sYWJlbD48aW5wdXQgaWQ9InB1cmwiIHR5cGU9",
        "InRleHQiIGlucHV0bW9kZT0idXJsIiBhdXRvY29tcGxldGU9Im9mZiIgYXV0b2NhcGl0YWxpemU9Im9mZiIgc3BlbGxjaGVjaz0i",
        "ZmFsc2UiIHBsYWNlaG9sZGVyPSJodHRwczovL+KApi9tb2RlbC56aXAiPjwvZGl2PgogICAgICAgIDxidXR0b24gY2xhc3M9ImJ0",
        "biBibG9jayIgaWQ9InBkbCIgdHlwZT0iYnV0dG9uIj5Eb3dubG9hZCBtb2RlbDwvYnV0dG9uPgogICAgICAgIDxkaXYgY2xhc3M9",
        "ImJhciIgaWQ9InBiYXIiIGhpZGRlbj48ZGl2PjwvZGl2PjwvZGl2PgogICAgICAgIDxkaXYgY2xhc3M9Im1zZyIgaWQ9InBtc2ci",
        "IHJvbGU9InN0YXR1cyIgYXJpYS1saXZlPSJwb2xpdGUiPjwvZGl2PgogICAgICAgIDxidXR0b24gY2xhc3M9ImJ0biBkYW5nZXIg",
        "YmxvY2siIGlkPSJwZGVsIiB0eXBlPSJidXR0b24iIGhpZGRlbj5EZWxldGUgbW9kZWw8L2J1dHRvbj4KICAgICAgICA8ZGl2IGNs",
        "YXNzPSJub3RlIHNtYWxsIj5OZWVkcyBhIHBob25lIHdpdGggOCBHQiBSQU0gb3IgbW9yZS4gT25lIHBob3RvIHRha2VzIGZyb20g",
        "YWJvdXQgMTUgc2Vjb25kcyB0byBhIGZldyBtaW51dGVzLiBPbiB0aGUgcGhvbmUgdGhlcmUgaXMgbm8gbmVnYXRpdmUgcHJvbXB0",
        "IGFuZCBubyBpbWFnZSBzYWZldHkgY2hlY2ssIHNvIG9ubHkgdGhlIHByb21wdCBmaWx0ZXIgYXBwbGllcy48L2Rpdj4KICAgICAg",
        "PC9kaXY+CiAgICA8L2Rpdj4KICAgIDxkaXYgY2xhc3M9Im5vdGUgc21hbGwiPjxiPlNhZmUgcGhvdG9zIG9ubHkuPC9iPiBUaGUg",
        "c2VydmVyIGZpbHRlcnMgcHJvbXB0cyBhbmQgY2hlY2tzIGV2ZXJ5IGltYWdlLiBUaGlzIGNhbm5vdCBiZSBzd2l0Y2hlZCBvZmYg",
        "ZnJvbSB0aGUgYXBwLjwvZGl2PgogICAgPGJ1dHRvbiBjbGFzcz0iYnRuIGRhbmdlciIgaWQ9ImZvcmdldCIgdHlwZT0iYnV0dG9u",
        "Ij5Gb3JnZXQgdGhpcyBzZXJ2ZXI8L2J1dHRvbj4KICA8L3NlY3Rpb24+CjwvZGl2PgoKPGRpYWxvZyBpZD0idmlld2VyIj4KICA8",
        "ZGl2IGNsYXNzPSJzdGFjayI+CiAgICA8aW1nIGNsYXNzPSJzaG90IiBpZD0idmltZyIgYWx0PSIiPgogICAgPGRpdiBjbGFzcz0i",
        "bW9ubyBtdXRlIiBpZD0idmluZm8iPjwvZGl2PgogICAgPGRpdiBjbGFzcz0ic21hbGwiIGlkPSJ2cHJvbXB0IiBzdHlsZT0ib3Zl",
        "cmZsb3ctd3JhcDphbnl3aGVyZSI+PC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJidG5zIj4KICAgICAgPGJ1dHRvbiBjbGFzcz0iYnRu",
        "IiBpZD0idnNhdmUiIHR5cGU9ImJ1dHRvbiI+U2F2ZTwvYnV0dG9uPgogICAgICA8YnV0dG9uIGNsYXNzPSJidG4iIGlkPSJ2c2hh",
        "cmUiIHR5cGU9ImJ1dHRvbiI+U2hhcmU8L2J1dHRvbj4KICAgICAgPGJ1dHRvbiBjbGFzcz0iYnRuIiBpZD0idnVzZSIgdHlwZT0i",
        "YnV0dG9uIj5Vc2Ugc2V0dGluZ3M8L2J1dHRvbj4KICAgIDwvZGl2PgogICAgPGRpdiBjbGFzcz0iYnRucyI+CiAgICAgIDxidXR0",
        "b24gY2xhc3M9ImJ0biBkYW5nZXIiIGlkPSJ2ZGVsIiB0eXBlPSJidXR0b24iPkRlbGV0ZTwvYnV0dG9uPgogICAgICA8YnV0dG9u",
        "IGNsYXNzPSJidG4iIGlkPSJ2Y2xvc2UiIHR5cGU9ImJ1dHRvbiI+Q2xvc2U8L2J1dHRvbj4KICAgIDwvZGl2PgogIDwvZGl2Pgo8",
        "L2RpYWxvZz4KCjxuYXYgY2xhc3M9InRhYnMiIGFyaWEtbGFiZWw9Ik1haW4iPjxkaXYgY2xhc3M9ImluIiByb2xlPSJ0YWJsaXN0",
        "Ij4KICA8YnV0dG9uIGNsYXNzPSJ0YWIiIHJvbGU9InRhYiIgZGF0YS12PSJjcmVhdGUiIGFyaWEtc2VsZWN0ZWQ9InRydWUiIHR5",
        "cGU9ImJ1dHRvbiI+PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiPjxjaXJjbGUgY3g9IjEyIiBjeT0iMTIiIHI9IjkiLz48cGF0aCBk",
        "PSJNMTIgOHY4TTggMTJoOCIvPjwvc3ZnPkNyZWF0ZTwvYnV0dG9uPgogIDxidXR0b24gY2xhc3M9InRhYiIgcm9sZT0idGFiIiBk",
        "YXRhLXY9ImdhbGxlcnkiIGFyaWEtc2VsZWN0ZWQ9ImZhbHNlIiB0eXBlPSJidXR0b24iPjxzdmcgdmlld0JveD0iMCAwIDI0IDI0",
        "Ij48cmVjdCB4PSIzIiB5PSI0IiB3aWR0aD0iMTgiIGhlaWdodD0iMTYiIHJ4PSIyIi8+PGNpcmNsZSBjeD0iOSIgY3k9IjEwIiBy",
        "PSIyIi8+PHBhdGggZD0iTTIxIDE2bC01LTUtOCA5Ii8+PC9zdmc+R2FsbGVyeTwvYnV0dG9uPgogIDxidXR0b24gY2xhc3M9InRh",
        "YiIgcm9sZT0idGFiIiBkYXRhLXY9ImNvbm5lY3QiIGFyaWEtc2VsZWN0ZWQ9ImZhbHNlIiB0eXBlPSJidXR0b24iPjxzdmcgdmll",
        "d0JveD0iMCAwIDI0IDI0Ij48cGF0aCBkPSJNMTAgMTRhNCA0IDAgMCAwIDUuNyAwbDMtM2E0IDQgMCAwIDAtNS43LTUuN2wtMSAx",
        "Ii8+PHBhdGggZD0iTTE0IDEwYTQgNCAwIDAgMC01LjcgMGwtMyAzQTQgNCAwIDAgMCAxMSAxOC43bDEtMSIvPjwvc3ZnPkNvbm5l",
        "Y3Q8L2J1dHRvbj4KPC9kaXY+PC9uYXY+Cgo8c2NyaXB0PgooKCkgPT4gewogICJ1c2Ugc3RyaWN0IjsKICBjb25zdCAkID0gaWQg",
        "PT4gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoaWQpOwogIGNvbnN0IExTID0gewogICAgZ2V0KGssIGQpIHsgdHJ5IHsgY29uc3Qg",
        "diA9IGxvY2FsU3RvcmFnZS5nZXRJdGVtKCJsc18iICsgayk7IHJldHVybiB2ID09IG51bGwgPyBkIDogSlNPTi5wYXJzZSh2KTsg",
        "fSBjYXRjaCAoZSkgeyByZXR1cm4gZDsgfSB9LAogICAgc2V0KGssIHYpIHsgdHJ5IHsgbG9jYWxTdG9yYWdlLnNldEl0ZW0oImxz",
        "XyIgKyBrLCBKU09OLnN0cmluZ2lmeSh2KSk7IH0gY2F0Y2ggKGUpIHt9IH0sCiAgICBkZWwoaykgeyB0cnkgeyBsb2NhbFN0b3Jh",
        "Z2UucmVtb3ZlSXRlbSgibHNfIiArIGspOyB9IGNhdGNoIChlKSB7fSB9CiAgfTsKICBjb25zdCBTID0geyB1cmw6IExTLmdldCgi",
        "dXJsIiwgIiIpLCBrZXk6IExTLmdldCgia2V5IiwgIiIpLCBpbmZvOiBudWxsLCBidXN5OiBmYWxzZSwgam9iSWQ6IG51bGwsIGxh",
        "c3Q6IG51bGwsIG1vZGU6ICJjb2xhYiIgfTsKCiAgLyogLS0tLS0tLS0tLSBBbmRyb2lkIGFwcCBicmlkZ2UgKG9ubHkgZXhpc3Rz",
        "IGluc2lkZSB0aGUgQVBLKSAtLS0tLS0tLS0tICovCiAgY29uc3QgTkFUSVZFID0gKHR5cGVvZiB3aW5kb3cuTmF0aXZlU3R1ZGlv",
        "ICE9PSAidW5kZWZpbmVkIikgPyB3aW5kb3cuTmF0aXZlU3R1ZGlvIDogbnVsbDsKICBsZXQgTkkgPSBudWxsOwogIGZ1bmN0aW9u",
        "IHJlYWROYXRpdmUoKSB7IGlmICghTkFUSVZFKSByZXR1cm47IHRyeSB7IE5JID0gSlNPTi5wYXJzZShOQVRJVkUuaW5mbygpKTsg",
        "fSBjYXRjaCAoZSkgeyBOSSA9IG51bGw7IH0gfQogIHJlYWROYXRpdmUoKTsKICBpZiAoTkkgJiYgTkkub25EZXZpY2UgJiYgTFMu",
        "Z2V0KCJtb2RlIiwgImNvbGFiIikgPT09ICJwaG9uZSIpIFMubW9kZSA9ICJwaG9uZSI7CgogIC8qIC0tLS0tLS0tLS0gdGFicyAt",
        "LS0tLS0tLS0tICovCiAgZnVuY3Rpb24gc2hvdyh2KSB7CiAgICBkb2N1bWVudC5xdWVyeVNlbGVjdG9yQWxsKCIudmlldyIpLmZv",
        "ckVhY2goZWwgPT4gZWwuY2xhc3NMaXN0LnRvZ2dsZSgib24iLCBlbC5pZCA9PT0gInYtIiArIHYpKTsKICAgIGRvY3VtZW50LnF1",
        "ZXJ5U2VsZWN0b3JBbGwoIi50YWIiKS5mb3JFYWNoKHQgPT4gdC5zZXRBdHRyaWJ1dGUoImFyaWEtc2VsZWN0ZWQiLCBTdHJpbmco",
        "dC5kYXRhc2V0LnYgPT09IHYpKSk7CiAgICBpZiAodiA9PT0gImdhbGxlcnkiKSByZW5kZXJHYWxsZXJ5KCk7CiAgICB3aW5kb3cu",
        "c2Nyb2xsVG8oMCwgMCk7CiAgfQogIGRvY3VtZW50LnF1ZXJ5U2VsZWN0b3JBbGwoIi50YWIiKS5mb3JFYWNoKHQgPT4gdC5hZGRF",
        "dmVudExpc3RlbmVyKCJjbGljayIsICgpID0+IHNob3codC5kYXRhc2V0LnYpKSk7CiAgJCgiY2hpcCIpLmFkZEV2ZW50TGlzdGVu",
        "ZXIoImNsaWNrIiwgKCkgPT4gc2hvdygiY29ubmVjdCIpKTsKCiAgLyogLS0tLS0tLS0tLSBjb25uZWN0aW9uIC0tLS0tLS0tLS0g",
        "Ki8KICBmdW5jdGlvbiBwYXJzZUxpbmsodCkgewogICAgdCA9ICh0IHx8ICIiKS50cmltKCk7CiAgICBsZXQgdXJsID0gdCwga2V5",
        "ID0gIiI7CiAgICBjb25zdCBpID0gdC5pbmRleE9mKCIjIik7CiAgICBpZiAoaSA+PSAwKSB7CiAgICAgIHVybCA9IHQuc2xpY2Uo",
        "MCwgaSk7CiAgICAgIGNvbnN0IG0gPSB0LnNsaWNlKGkpLm1hdGNoKC9rZXk9KFteJlxzXSspLyk7CiAgICAgIGlmIChtKSB7IHRy",
        "eSB7IGtleSA9IGRlY29kZVVSSUNvbXBvbmVudChtWzFdKTsgfSBjYXRjaCAoZSkgeyBrZXkgPSBtWzFdOyB9IH0KICAgIH0KICAg",
        "IHVybCA9IHVybC5yZXBsYWNlKC9cLyskLywgIiIpOwogICAgaWYgKHVybCAmJiAhL15odHRwcz86XC9cLy9pLnRlc3QodXJsKSkg",
        "dXJsID0gImh0dHBzOi8vIiArIHVybDsKICAgIHJldHVybiB7IHVybCwga2V5IH07CiAgfQogIGZ1bmN0aW9uIHRvYXN0KHRleHQp",
        "IHsKICAgIGNvbnN0IG9sZCA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCJ0b2FzdCIpOyBpZiAob2xkKSBvbGQucmVtb3ZlKCk7",
        "CiAgICBjb25zdCB0ID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudCgiZGl2Iik7IHQuaWQgPSAidG9hc3QiOyB0LmNsYXNzTmFtZSA9",
        "ICJ0b2FzdCI7IHQuc2V0QXR0cmlidXRlKCJyb2xlIiwgInN0YXR1cyIpOyB0LnRleHRDb250ZW50ID0gdGV4dDsKICAgIGRvY3Vt",
        "ZW50LmJvZHkuYXBwZW5kQ2hpbGQodCk7IHNldFRpbWVvdXQoKCkgPT4gdC5yZW1vdmUoKSwgMjYwMCk7CiAgfQogIGZ1bmN0aW9u",
        "IGNoaXAoc3RhdGUsIHRleHQpIHsgJCgiY2hpcCIpLmRhdGFzZXQucyA9IHN0YXRlOyAkKCJjaGlwVGV4dCIpLnRleHRDb250ZW50",
        "ID0gdGV4dDsgfQogIGZ1bmN0aW9uIGNtc2codGV4dCwgYmFkKSB7IGNvbnN0IG0gPSAkKCJjbXNnIik7IG0udGV4dENvbnRlbnQg",
        "PSB0ZXh0IHx8ICIiOyBtLmNsYXNzTmFtZSA9ICJtc2ciICsgKGJhZCA/ICIgYmFkIiA6ICIiKTsgfQoKICBmdW5jdGlvbiBwYWlu",
        "dFBob25lKCkgewogICAgY29uc3QgZnVsbCA9ICEhKE5BVElWRSAmJiBOSSAmJiBOSS5vbkRldmljZSk7CiAgICAkKCJtb2RlQ2Fy",
        "ZCIpLmhpZGRlbiA9ICFmdWxsOwogICAgJCgicGhvbmVCb3giKS5oaWRkZW4gPSAhZnVsbDsKICAgICQoInBob25lTm90ZSIpLmhp",
        "ZGRlbiA9IGZ1bGw7CiAgICBpZiAoTkFUSVZFICYmICFmdWxsKSB7ICQoInBob25lTm90ZSIpLmhpZGRlbiA9IGZhbHNlOyAkKCJw",
        "aG9uZU5vdGUiKS50ZXh0Q29udGVudCA9ICJUaGlzIGlzIHRoZSBMaXRlIGFwcC4gSXQgbWFrZXMgcGhvdG9zIG9uIENvbGFiLiBJ",
        "bnN0YWxsIHRoZSBGdWxsIGFwcCB0byBtYWtlIHBob3RvcyBvbiB0aGUgcGhvbmUgaXRzZWxmLiI7IH0KICAgIGlmICghZnVsbCkg",
        "cmV0dXJuOwogICAgJCgicHN0YXRlIikudGV4dENvbnRlbnQgPSBOSS5tb2RlbFJlYWR5ID8gIk1vZGVsIGluc3RhbGxlZCAoIiAr",
        "IE5JLm1vZGVsTUIgKyAiIE1CKS4gVHJpZ2dlciB3b3JkOiAiICsgKE5JLnRyaWdnZXIgfHwgIm5vbmUiKSA6ICJObyBtb2RlbCBv",
        "biB0aGlzIHBob25lIHlldC4iOwogICAgJCgicGRlbCIpLmhpZGRlbiA9ICFOSS5tb2RlbFJlYWR5OwogICAgaWYgKE5JLnRyaWdn",
        "ZXIgJiYgISQoInB0cmlnIikudmFsdWUpICQoInB0cmlnIikudmFsdWUgPSBOSS50cmlnZ2VyOwogIH0KICBmdW5jdGlvbiBhcHBs",
        "eU1vZGUoKSB7CiAgICBkb2N1bWVudC5ib2R5LmRhdGFzZXQubW9kZSA9IFMubW9kZTsKICAgIGRvY3VtZW50LnF1ZXJ5U2VsZWN0",
        "b3JBbGwoImlucHV0W25hbWU9bW9kZV0iKS5mb3JFYWNoKHIgPT4geyByLmNoZWNrZWQgPSByLnZhbHVlID09PSBTLm1vZGU7IH0p",
        "OwogICAgJCgicGhvbmVIaW50IikuaGlkZGVuID0gUy5tb2RlICE9PSAicGhvbmUiOwogICAgaWYgKFMubW9kZSA9PT0gInBob25l",
        "IikgY2hpcChOSSAmJiBOSS5tb2RlbFJlYWR5ID8gIm9rIiA6ICJiYWQiLCBOSSAmJiBOSS5tb2RlbFJlYWR5ID8gIk9uIHRoaXMg",
        "cGhvbmUiIDogIk5vIG1vZGVsIik7CiAgICBlbHNlIGNoaXAoUy5pbmZvID8gIm9rIiA6IChTLnVybCA/ICJiYWQiIDogIiIpLCBT",
        "LmluZm8gPyAiQ29ubmVjdGVkIiA6IChTLnVybCA/ICJPZmZsaW5lIiA6ICJOb3QgY29ubmVjdGVkIikpOwogICAgcGFpbnRQaG9u",
        "ZSgpOwogIH0KCiAgYXN5bmMgZnVuY3Rpb24gYXBpKHBhdGgsIG9wdCA9IHt9LCB0aW1lb3V0ID0gMjAwMDApIHsKICAgIGlmICgh",
        "Uy51cmwpIHRocm93IG5ldyBFcnJvcigiT3BlbiB0aGUgQ29ubmVjdCB0YWIgYW5kIHBhc3RlIHlvdXIgQ29sYWIgbGluayBmaXJz",
        "dC4iKTsKICAgIGNvbnN0IGN0bCA9IG5ldyBBYm9ydENvbnRyb2xsZXIoKTsKICAgIGNvbnN0IHRvID0gc2V0VGltZW91dCgoKSA9",
        "PiBjdGwuYWJvcnQoKSwgdGltZW91dCk7CiAgICB0cnkgewogICAgICBjb25zdCBoZWFkZXJzID0geyAiWC1LZXkiOiBTLmtleSB9",
        "OwogICAgICBpZiAob3B0LmJvZHkpIGhlYWRlcnNbIkNvbnRlbnQtVHlwZSJdID0gImFwcGxpY2F0aW9uL2pzb24iOwogICAgICBj",
        "b25zdCByID0gYXdhaXQgZmV0Y2goUy51cmwgKyBwYXRoLCB7IG1ldGhvZDogb3B0Lm1ldGhvZCB8fCAiR0VUIiwgYm9keTogb3B0",
        "LmJvZHksIGhlYWRlcnMsIHNpZ25hbDogY3RsLnNpZ25hbCB9KTsKICAgICAgaWYgKG9wdC5ibG9iKSB7IGlmICghci5vaykgdGhy",
        "b3cgbmV3IEVycm9yKCJDb3VsZCBub3QgZG93bmxvYWQgdGhlIHBob3RvLiIpOyByZXR1cm4gYXdhaXQgci5ibG9iKCk7IH0KICAg",
        "ICAgbGV0IGogPSBudWxsOyB0cnkgeyBqID0gYXdhaXQgci5qc29uKCk7IH0gY2F0Y2ggKGUpIHt9CiAgICAgIGlmICghci5vaykg",
        "dGhyb3cgbmV3IEVycm9yKChqICYmIGouZXJyb3IpIHx8ICgiU2VydmVyIGVycm9yICIgKyByLnN0YXR1cykpOwogICAgICByZXR1",
        "cm4gajsKICAgIH0gY2F0Y2ggKGUpIHsKICAgICAgaWYgKGUubmFtZSA9PT0gIkFib3J0RXJyb3IiKSB0aHJvdyBuZXcgRXJyb3Io",
        "IlRoZSBzZXJ2ZXIgZGlkIG5vdCBhbnN3ZXIgaW4gdGltZS4iKTsKICAgICAgaWYgKGUgaW5zdGFuY2VvZiBUeXBlRXJyb3IpIHRo",
        "cm93IG5ldyBFcnJvcigiQ2Fubm90IHJlYWNoIHRoZSBzZXJ2ZXIuIElzIENvbGFiIHJ1bm5pbmc/IENvbGFiIGdpdmVzIGEgbmV3",
        "IGxpbmsgYWZ0ZXIgYSByZXN0YXJ0LiIpOwogICAgICB0aHJvdyBlOwogICAgfSBmaW5hbGx5IHsgY2xlYXJUaW1lb3V0KHRvKTsg",
        "fQogIH0KCiAgYXN5bmMgZnVuY3Rpb24gY29ubmVjdCgpIHsKICAgIGlmICghUy51cmwpIHsgY2hpcCgiIiwgIk5vdCBjb25uZWN0",
        "ZWQiKTsgcmV0dXJuIGZhbHNlOyB9CiAgICBjaGlwKCJidXN5IiwgIkNvbm5lY3RpbmfigKYiKTsgY21zZygiIik7CiAgICB0cnkg",
        "ewogICAgICBTLmluZm8gPSBhd2FpdCBhcGkoIi9hcGkvaW5mbyIpOwogICAgICBjb25zdCBncHUgPSAoUy5pbmZvLmdwdSB8fCAi",
        "R1BVIikucmVwbGFjZSgvXk5WSURJQVxzKy9pLCAiIik7CiAgICAgIGNoaXAoIm9rIiwgIkNvbm5lY3RlZCIpOwogICAgICBjbXNn",
        "KCJDb25uZWN0ZWQgdG8gIiArIGdwdSArICIuICIgKyBTLmluZm8ubG9yYXMubGVuZ3RoICsgIiBMb1JBIG1vZGVsIiArIChTLmlu",
        "Zm8ubG9yYXMubGVuZ3RoID09PSAxID8gIiIgOiAicyIpICsgIiBmb3VuZCBvbiBEcml2ZS4iKTsKICAgICAgZmlsbExvcmFzKCk7",
        "CiAgICAgIHJldHVybiB0cnVlOwogICAgfSBjYXRjaCAoZSkgewogICAgICBTLmluZm8gPSBudWxsOyBjaGlwKCJiYWQiLCAiT2Zm",
        "bGluZSIpOyBjbXNnKGUubWVzc2FnZSwgdHJ1ZSk7IHJldHVybiBmYWxzZTsKICAgIH0KICB9CgogIGZ1bmN0aW9uIGZpbGxMb3Jh",
        "cygpIHsKICAgIGNvbnN0IHNlbCA9ICQoImxvcmEiKSwga2VlcCA9IExTLmdldCgibG9yYSIsIG51bGwpOyAgIC8vIG51bGwgPSBu",
        "ZXZlciBjaG9zZW4sICIiID0gY2hvc2UgIk5vbmUiCiAgICBzZWwubGVuZ3RoID0gMTsKICAgIGNvbnN0IGxpc3QgPSAoUy5pbmZv",
        "ICYmIFMuaW5mby5sb3JhcykgfHwgW107CiAgICBsaXN0LmZvckVhY2gobCA9PiB7IGNvbnN0IG8gPSBkb2N1bWVudC5jcmVhdGVF",
        "bGVtZW50KCJvcHRpb24iKTsgby52YWx1ZSA9IGwubmFtZTsgby50ZXh0Q29udGVudCA9IGwubmFtZTsgc2VsLmFwcGVuZENoaWxk",
        "KG8pOyB9KTsKICAgIGNvbnN0IGhhcyA9IHYgPT4gWy4uLnNlbC5vcHRpb25zXS5zb21lKG8gPT4gby52YWx1ZSA9PT0gdik7CiAg",
        "ICBzZWwudmFsdWUgPSAoa2VlcCAhPT0gbnVsbCAmJiBoYXMoa2VlcCkpID8ga2VlcCA6IChsaXN0Lmxlbmd0aCA/IGxpc3RbbGlz",
        "dC5sZW5ndGggLSAxXS5uYW1lIDogIiIpOwogICAgb25Mb3JhKCk7CiAgfQogIGZ1bmN0aW9uIGN1clRyaWdnZXIoKSB7CiAgICBj",
        "b25zdCBuID0gJCgibG9yYSIpLnZhbHVlOyBjb25zdCBsID0gUy5pbmZvICYmIFMuaW5mby5sb3Jhcy5maW5kKHggPT4geC5uYW1l",
        "ID09PSBuKTsKICAgIHJldHVybiBsID8gKGwudHJpZ2dlciB8fCAiIikgOiAiIjsKICB9CiAgZnVuY3Rpb24gb25Mb3JhKCkgewog",
        "ICAgY29uc3QgbiA9ICQoImxvcmEiKS52YWx1ZSwgdCA9IGN1clRyaWdnZXIoKTsKICAgICQoInNjYWxlQm94IikuaGlkZGVuID0g",
        "IW47CiAgICAkKCJ0cmlnIikuaGlkZGVuID0gIShuICYmIHQpOwogICAgJCgidHJpZyIpLnRleHRDb250ZW50ID0gdCA/ICJUcmln",
        "Z2VyIHdvcmQ6ICIgKyB0ICsgIiAoYWRkZWQgYXV0b21hdGljYWxseSkiIDogIiI7CiAgICBpZiAoUy5pbmZvKSBMUy5zZXQoImxv",
        "cmEiLCBuKTsKICB9CiAgJCgibG9yYSIpLmFkZEV2ZW50TGlzdGVuZXIoImNoYW5nZSIsIG9uTG9yYSk7CgogICQoImNvbm5lY3Qi",
        "KS5hZGRFdmVudExpc3RlbmVyKCJjbGljayIsIGFzeW5jICgpID0+IHsKICAgIGNvbnN0IHAgPSBwYXJzZUxpbmsoJCgibGluayIp",
        "LnZhbHVlKTsKICAgIGlmIChwLnVybCkgUy51cmwgPSBwLnVybDsKICAgIFMua2V5ID0gcC5rZXkgfHwgJCgia2V5IikudmFsdWUu",
        "dHJpbSgpIHx8IFMua2V5OwogICAgaWYgKCFTLnVybCkgeyBjbXNnKCJQYXN0ZSB0aGUgbGluayBmcm9tIENvbGFiIGZpcnN0LiIs",
        "IHRydWUpOyByZXR1cm47IH0KICAgIExTLnNldCgidXJsIiwgUy51cmwpOyBMUy5zZXQoImtleSIsIFMua2V5KTsKICAgICQoImxp",
        "bmsiKS52YWx1ZSA9IFMudXJsOyAkKCJrZXkiKS52YWx1ZSA9IFMua2V5OwogICAgaWYgKGF3YWl0IGNvbm5lY3QoKSkgc2V0VGlt",
        "ZW91dCgoKSA9PiBzaG93KCJjcmVhdGUiKSwgNjAwKTsKICB9KTsKICAkKCJsaW5rIikuYWRkRXZlbnRMaXN0ZW5lcigiaW5wdXQi",
        "LCAoKSA9PiB7CiAgICBjb25zdCBwID0gcGFyc2VMaW5rKCQoImxpbmsiKS52YWx1ZSk7CiAgICBpZiAocC5rZXkpICQoImtleSIp",
        "LnZhbHVlID0gcC5rZXk7CiAgfSk7CiAgJCgiZm9yZ2V0IikuYWRkRXZlbnRMaXN0ZW5lcigiY2xpY2siLCAoKSA9PiB7CiAgICBM",
        "Uy5kZWwoInVybCIpOyBMUy5kZWwoImtleSIpOyBTLnVybCA9ICIiOyBTLmtleSA9ICIiOyBTLmluZm8gPSBudWxsOwogICAgJCgi",
        "bGluayIpLnZhbHVlID0gIiI7ICQoImtleSIpLnZhbHVlID0gIiI7IGNoaXAoIiIsICJOb3QgY29ubmVjdGVkIik7IGNtc2coIlNl",
        "cnZlciBmb3Jnb3R0ZW4uIik7IGZpbGxMb3JhcygpOwogIH0pOwoKICAvKiAtLS0tLS0tLS0tIHNsaWRlcnMgJiBzYXZlZCBpbnB1",
        "dHMgLS0tLS0tLS0tLSAqLwogIGZ1bmN0aW9uIGJpbmRSYW5nZShpZCwgdmlkLCBmbXQpIHsKICAgIGNvbnN0IGVsID0gJChpZCks",
        "IG91dCA9ICQodmlkKSwgc2F2ZWQgPSBMUy5nZXQoaWQsIG51bGwpOwogICAgaWYgKHNhdmVkICE9IG51bGwpIGVsLnZhbHVlID0g",
        "c2F2ZWQ7CiAgICBjb25zdCB1cGQgPSAoKSA9PiB7IG91dC50ZXh0Q29udGVudCA9IGZtdCgrZWwudmFsdWUpOyBMUy5zZXQoaWQs",
        "IGVsLnZhbHVlKTsgfTsKICAgIGVsLmFkZEV2ZW50TGlzdGVuZXIoImlucHV0IiwgdXBkKTsgdXBkKCk7CiAgfQogIGJpbmRSYW5n",
        "ZSgic2NhbGUiLCAic2NhbGVWIiwgdiA9PiB2LnRvRml4ZWQoMikpOwogIGJpbmRSYW5nZSgic3RlcHMiLCAic3RlcHNWIiwgdiA9",
        "PiBTdHJpbmcodikpOwogIGJpbmRSYW5nZSgiZ3VpZCIsICJndWlkViIsIHYgPT4gdi50b0ZpeGVkKDEpKTsKICBbInNtYXJ0Iiwg",
        "ImhpcmVzIl0uZm9yRWFjaChpZCA9PiB7IGNvbnN0IGVsID0gJChpZCksIHMgPSBMUy5nZXQoaWQsIG51bGwpOyBpZiAocyAhPSBu",
        "dWxsKSBlbC5jaGVja2VkID0gczsgZWwuYWRkRXZlbnRMaXN0ZW5lcigiY2hhbmdlIiwgKCkgPT4gTFMuc2V0KGlkLCBlbC5jaGVj",
        "a2VkKSk7IH0pOwogIGRvY3VtZW50LnF1ZXJ5U2VsZWN0b3JBbGwoImlucHV0W25hbWU9c2hhcGVdIikuZm9yRWFjaChyID0+IHsg",
        "aWYgKHIudmFsdWUgPT09IExTLmdldCgic2hhcGUiLCAicG9ydHJhaXQiKSkgci5jaGVja2VkID0gdHJ1ZTsgci5hZGRFdmVudExp",
        "c3RlbmVyKCJjaGFuZ2UiLCAoKSA9PiBMUy5zZXQoInNoYXBlIiwgci52YWx1ZSkpOyB9KTsKICAkKCJwcm9tcHQiKS52YWx1ZSA9",
        "IExTLmdldCgicHJvbXB0IiwgIiIpOwogICQoInByb21wdCIpLmFkZEV2ZW50TGlzdGVuZXIoImlucHV0IiwgKCkgPT4gTFMuc2V0",
        "KCJwcm9tcHQiLCAkKCJwcm9tcHQiKS52YWx1ZSkpOwogICQoIm5lZyIpLnZhbHVlID0gTFMuZ2V0KCJuZWciLCAiIik7CiAgJCgi",
        "bmVnIikuYWRkRXZlbnRMaXN0ZW5lcigiaW5wdXQiLCAoKSA9PiBMUy5zZXQoIm5lZyIsICQoIm5lZyIpLnZhbHVlKSk7CgogIC8q",
        "IC0tLS0tLS0tLS0gZ2FsbGVyeSBzdG9yYWdlIChJbmRleGVkREIsIGZhbGxzIGJhY2sgdG8gbWVtb3J5KSAtLS0tLS0tLS0tICov",
        "CiAgY29uc3QgbWVtID0gW107CiAgbGV0IGRicCA9IG51bGw7CiAgZnVuY3Rpb24gZGIoKSB7CiAgICBpZiAoZGJwKSByZXR1cm4g",
        "ZGJwOwogICAgZGJwID0gbmV3IFByb21pc2UocmVzID0+IHsKICAgICAgdHJ5IHsKICAgICAgICBjb25zdCByID0gaW5kZXhlZERC",
        "Lm9wZW4oImxvcmEtc3R1ZGlvIiwgMSk7CiAgICAgICAgci5vbnVwZ3JhZGVuZWVkZWQgPSAoKSA9PiByLnJlc3VsdC5jcmVhdGVP",
        "YmplY3RTdG9yZSgic2hvdHMiLCB7IGtleVBhdGg6ICJpZCIgfSk7CiAgICAgICAgci5vbnN1Y2Nlc3MgPSAoKSA9PiByZXMoci5y",
        "ZXN1bHQpOwogICAgICAgIHIub25lcnJvciA9ICgpID0+IHJlcyhudWxsKTsKICAgICAgICByLm9uYmxvY2tlZCA9ICgpID0+IHJl",
        "cyhudWxsKTsKICAgICAgfSBjYXRjaCAoZSkgeyByZXMobnVsbCk7IH0KICAgIH0pOwogICAgcmV0dXJuIGRicDsKICB9CiAgY29u",
        "c3QgdHggPSAoZCwgbW9kZSwgZm4pID0+IG5ldyBQcm9taXNlKHJlcyA9PiB7IHRyeSB7IGNvbnN0IHQgPSBkLnRyYW5zYWN0aW9u",
        "KCJzaG90cyIsIG1vZGUpOyBjb25zdCBycSA9IGZuKHQub2JqZWN0U3RvcmUoInNob3RzIikpOyB0Lm9uY29tcGxldGUgPSAoKSA9",
        "PiByZXMocnEgPyBycS5yZXN1bHQgOiB0cnVlKTsgdC5vbmVycm9yID0gKCkgPT4gcmVzKG51bGwpOyB9IGNhdGNoIChlKSB7IHJl",
        "cyhudWxsKTsgfSB9KTsKICBhc3luYyBmdW5jdGlvbiBhZGRTaG90KHMpIHsgY29uc3QgZCA9IGF3YWl0IGRiKCk7IGlmICghZCB8",
        "fCAoYXdhaXQgdHgoZCwgInJlYWR3cml0ZSIsIHN0ID0+IHN0LnB1dChzKSkpID09PSBudWxsKSBtZW0ucHVzaChzKTsgfQogIGFz",
        "eW5jIGZ1bmN0aW9uIGFsbFNob3RzKCkgeyBjb25zdCBkID0gYXdhaXQgZGIoKTsgbGV0IGEgPSBkID8gYXdhaXQgdHgoZCwgInJl",
        "YWRvbmx5Iiwgc3QgPT4gc3QuZ2V0QWxsKCkpIDogbnVsbDsgcmV0dXJuIChhIHx8IFtdKS5jb25jYXQobWVtKS5zb3J0KCh4LCB5",
        "KSA9PiB5LnQgLSB4LnQpOyB9CiAgYXN5bmMgZnVuY3Rpb24gZGVsU2hvdChpZCkgeyBjb25zdCBkID0gYXdhaXQgZGIoKTsgaWYg",
        "KGQpIGF3YWl0IHR4KGQsICJyZWFkd3JpdGUiLCBzdCA9PiBzdC5kZWxldGUoaWQpKTsgY29uc3QgaSA9IG1lbS5maW5kSW5kZXgo",
        "eCA9PiB4LmlkID09PSBpZCk7IGlmIChpID49IDApIG1lbS5zcGxpY2UoaSwgMSk7IH0KCiAgbGV0IHVybHMgPSBbXTsKICBhc3lu",
        "YyBmdW5jdGlvbiByZW5kZXJHYWxsZXJ5KCkgewogICAgdXJscy5mb3JFYWNoKHUgPT4gVVJMLnJldm9rZU9iamVjdFVSTCh1KSk7",
        "IHVybHMgPSBbXTsKICAgIGNvbnN0IGxpc3QgPSBhd2FpdCBhbGxTaG90cygpLCBib3ggPSAkKCJzaGVldCIpOwogICAgYm94LnRl",
        "eHRDb250ZW50ID0gIiI7CiAgICAkKCJlbXB0eSIpLmhpZGRlbiA9IGxpc3QubGVuZ3RoID4gMDsKICAgICQoImNvdW50IikudGV4",
        "dENvbnRlbnQgPSBsaXN0Lmxlbmd0aCA/IGxpc3QubGVuZ3RoICsgIiBwaG90byIgKyAobGlzdC5sZW5ndGggPT09IDEgPyAiIiA6",
        "ICJzIikgOiAiIjsKICAgIGxpc3QuZm9yRWFjaChzID0+IHsKICAgICAgY29uc3QgYiA9IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQo",
        "ImJ1dHRvbiIpOwogICAgICBiLnR5cGUgPSAiYnV0dG9uIjsgYi5jbGFzc05hbWUgPSAiZnJhbWUiOyBiLnNldEF0dHJpYnV0ZSgi",
        "YXJpYS1sYWJlbCIsICJPcGVuIHBob3RvICIgKyBzLm4pOwogICAgICBjb25zdCB1ID0gVVJMLmNyZWF0ZU9iamVjdFVSTChzLmJs",
        "b2IpOyB1cmxzLnB1c2godSk7CiAgICAgIGNvbnN0IGltID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudCgiaW1nIik7IGltLnNyYyA9",
        "IHU7IGltLmFsdCA9ICIiOyBpbS5sb2FkaW5nID0gImxhenkiOwogICAgICBjb25zdCBtID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVu",
        "dCgiZGl2Iik7IG0uY2xhc3NOYW1lID0gIm1ldGEiOwogICAgICBjb25zdCBhID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudCgic3Bh",
        "biIpOyBhLnRleHRDb250ZW50ID0gIk5vLiAiICsgU3RyaW5nKHMubikucGFkU3RhcnQoMiwgIjAiKTsKICAgICAgY29uc3QgYyA9",
        "IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoInNwYW4iKTsgYy50ZXh0Q29udGVudCA9ICJzZWVkICIgKyBzLnNlZWQ7CiAgICAgIG0u",
        "YXBwZW5kKGEsIGMpOyBiLmFwcGVuZChpbSwgbSk7CiAgICAgIGIuYWRkRXZlbnRMaXN0ZW5lcigiY2xpY2siLCAoKSA9PiBvcGVu",
        "Vmlld2VyKHMpKTsKICAgICAgYm94LmFwcGVuZENoaWxkKGIpOwogICAgfSk7CiAgfQoKICAvKiAtLS0tLS0tLS0tIHZpZXdlciAt",
        "LS0tLS0tLS0tICovCiAgbGV0IGN1ciA9IG51bGwsIGN1clVybCA9IG51bGw7CiAgZnVuY3Rpb24gb3BlblZpZXdlcihzKSB7CiAg",
        "ICBjdXIgPSBzOwogICAgaWYgKGN1clVybCkgVVJMLnJldm9rZU9iamVjdFVSTChjdXJVcmwpOwogICAgY3VyVXJsID0gVVJMLmNy",
        "ZWF0ZU9iamVjdFVSTChzLmJsb2IpOwogICAgJCgidmltZyIpLnNyYyA9IGN1clVybDsKICAgICQoInZpbmZvIikudGV4dENvbnRl",
        "bnQgPSAiTm8uICIgKyBTdHJpbmcocy5uKS5wYWRTdGFydCgyLCAiMCIpICsgIiDCtyBzZWVkICIgKyBzLnNlZWQgKyAocy5sb3Jh",
        "ID8gIiDCtyAiICsgcy5sb3JhIDogIiIpICsgIiDCtyAiICsgbmV3IERhdGUocy50KS50b0xvY2FsZVN0cmluZygpOwogICAgJCgi",
        "dnByb21wdCIpLnRleHRDb250ZW50ID0gcy5pZGVhIHx8ICIiOwogICAgY29uc3QgZCA9ICQoInZpZXdlciIpOyBpZiAoZC5zaG93",
        "TW9kYWwpIGQuc2hvd01vZGFsKCk7IGVsc2UgZC5zZXRBdHRyaWJ1dGUoIm9wZW4iLCAiIik7CiAgfQogIGZ1bmN0aW9uIGJsb2JU",
        "b0I2NChibG9iKSB7IHJldHVybiBuZXcgUHJvbWlzZSgocmVzLCByZWopID0+IHsgY29uc3QgciA9IG5ldyBGaWxlUmVhZGVyKCk7",
        "IHIub25sb2FkID0gKCkgPT4gcmVzKFN0cmluZyhyLnJlc3VsdCkuc3BsaXQoIiwiKVsxXSk7IHIub25lcnJvciA9IHJlajsgci5y",
        "ZWFkQXNEYXRhVVJMKGJsb2IpOyB9KTsgfQogIGFzeW5jIGZ1bmN0aW9uIG5hdGl2ZVNhdmUoYmxvYiwgbmFtZSwgc2hhcmUpIHsK",
        "ICAgIHRyeSB7CiAgICAgIGNvbnN0IGI2NCA9IGF3YWl0IGJsb2JUb0I2NChibG9iKTsKICAgICAgY29uc3Qgb3V0ID0gc2hhcmUg",
        "PyBOQVRJVkUuc2hhcmVJbWFnZShiNjQsIG5hbWUpIDogTkFUSVZFLnNhdmVJbWFnZShiNjQsIG5hbWUpOwogICAgICB0b2FzdChv",
        "dXQgPT09ICJvayIgPyAoc2hhcmUgPyAiT3BlbmluZyBzaGFyZeKApiIgOiAiU2F2ZWQgdG8gUGljdHVyZXMgLyBMb1JBIFN0dWRp",
        "byIpIDogU3RyaW5nKG91dCkpOwogICAgfSBjYXRjaCAoZSkgeyB0b2FzdCgiQ291bGQgbm90IHNhdmUgdGhlIHBob3RvLiIpOyB9",
        "CiAgfQogIGZ1bmN0aW9uIHNhdmVCbG9iKGJsb2IsIG5hbWUpIHsKICAgIGlmIChOQVRJVkUpIHsgbmF0aXZlU2F2ZShibG9iLCBu",
        "YW1lLCBmYWxzZSk7IHJldHVybjsgfQogICAgY29uc3QgdSA9IFVSTC5jcmVhdGVPYmplY3RVUkwoYmxvYiksIGEgPSBkb2N1bWVu",
        "dC5jcmVhdGVFbGVtZW50KCJhIik7CiAgICBhLmhyZWYgPSB1OyBhLmRvd25sb2FkID0gbmFtZTsgZG9jdW1lbnQuYm9keS5hcHBl",
        "bmRDaGlsZChhKTsgYS5jbGljaygpOyBhLnJlbW92ZSgpOwogICAgc2V0VGltZW91dCgoKSA9PiBVUkwucmV2b2tlT2JqZWN0VVJM",
        "KHUpLCA0MDAwKTsKICB9CiAgYXN5bmMgZnVuY3Rpb24gc2hhcmVCbG9iKGJsb2IsIG5hbWUpIHsKICAgIGlmIChOQVRJVkUpIHsg",
        "YXdhaXQgbmF0aXZlU2F2ZShibG9iLCBuYW1lLCB0cnVlKTsgcmV0dXJuOyB9CiAgICBjb25zdCBmID0gbmV3IEZpbGUoW2Jsb2Jd",
        "LCBuYW1lLCB7IHR5cGU6ICJpbWFnZS9qcGVnIiB9KTsKICAgIHRyeSB7IGlmIChuYXZpZ2F0b3IuY2FuU2hhcmUgJiYgbmF2aWdh",
        "dG9yLmNhblNoYXJlKHsgZmlsZXM6IFtmXSB9KSkgeyBhd2FpdCBuYXZpZ2F0b3Iuc2hhcmUoeyBmaWxlczogW2ZdIH0pOyByZXR1",
        "cm47IH0gfSBjYXRjaCAoZSkgeyBpZiAoZS5uYW1lID09PSAiQWJvcnRFcnJvciIpIHJldHVybjsgfQogICAgc2F2ZUJsb2IoYmxv",
        "YiwgbmFtZSk7CiAgfQogIGNvbnN0IGZuYW1lID0gcyA9PiAicGhvdG8tIiArIFN0cmluZyhzLm4pLnBhZFN0YXJ0KDIsICIwIikg",
        "KyAiLXNlZWQiICsgcy5zZWVkICsgIi5qcGciOwogICQoInZjbG9zZSIpLmFkZEV2ZW50TGlzdGVuZXIoImNsaWNrIiwgKCkgPT4g",
        "JCgidmlld2VyIikuY2xvc2UoKSk7CiAgJCgidnNhdmUiKS5hZGRFdmVudExpc3RlbmVyKCJjbGljayIsICgpID0+IGN1ciAmJiBz",
        "YXZlQmxvYihjdXIuYmxvYiwgZm5hbWUoY3VyKSkpOwogICQoInZzaGFyZSIpLmFkZEV2ZW50TGlzdGVuZXIoImNsaWNrIiwgKCkg",
        "PT4gY3VyICYmIHNoYXJlQmxvYihjdXIuYmxvYiwgZm5hbWUoY3VyKSkpOwogICQoInZkZWwiKS5hZGRFdmVudExpc3RlbmVyKCJj",
        "bGljayIsIGFzeW5jICgpID0+IHsgaWYgKCFjdXIpIHJldHVybjsgYXdhaXQgZGVsU2hvdChjdXIuaWQpOyAkKCJ2aWV3ZXIiKS5j",
        "bG9zZSgpOyByZW5kZXJHYWxsZXJ5KCk7IH0pOwogICQoInZ1c2UiKS5hZGRFdmVudExpc3RlbmVyKCJjbGljayIsICgpID0+IHsK",
        "ICAgIGlmICghY3VyKSByZXR1cm47CiAgICAkKCJwcm9tcHQiKS52YWx1ZSA9IGN1ci5pZGVhIHx8ICIiOyBMUy5zZXQoInByb21w",
        "dCIsICQoInByb21wdCIpLnZhbHVlKTsKICAgICQoInNlZWQiKS52YWx1ZSA9IGN1ci5zZWVkOwogICAgJCgidmlld2VyIikuY2xv",
        "c2UoKTsgc2hvdygiY3JlYXRlIik7CiAgfSk7CgogIC8qIC0tLS0tLS0tLS0gZ2VuZXJhdGUgLS0tLS0tLS0tLSAqLwogIGNvbnN0",
        "IHNsZWVwID0gbXMgPT4gbmV3IFByb21pc2UociA9PiBzZXRUaW1lb3V0KHIsIG1zKSk7CiAgbGV0IHRpY2sgPSBudWxsOwogIGZ1",
        "bmN0aW9uIHNldEJ1c3koYikgeyBTLmJ1c3kgPSBiOyAkKCJnbyIpLmRpc2FibGVkID0gYjsgJCgiZ28iKS50ZXh0Q29udGVudCA9",
        "IGIgPyAiV29ya2luZ+KApiIgOiAiR2VuZXJhdGUiOyB9CiAgZnVuY3Rpb24gc3RhdHVzKHRleHQsIGJhZCkgeyBjb25zdCBzID0g",
        "JCgic3RhdHVzIik7IHMudGV4dENvbnRlbnQgPSB0ZXh0OyBzLmNsYXNzTmFtZSA9ICJtc2cgZ3JvdyIgKyAoYmFkID8gIiBiYWQi",
        "IDogIiIpOyB9CgogIGFzeW5jIGZ1bmN0aW9uIGdlbmVyYXRlKGZvcmNlTmV3U2VlZCkgewogICAgaWYgKFMubW9kZSA9PT0gInBo",
        "b25lIikgcmV0dXJuIGdlbmVyYXRlUGhvbmUoZm9yY2VOZXdTZWVkKTsKICAgIGlmIChTLmJ1c3kpIHJldHVybjsKICAgIGNvbnN0",
        "IHByb21wdCA9ICQoInByb21wdCIpLnZhbHVlLnRyaW0oKTsKICAgIGlmICghcHJvbXB0KSB7ICQoInByb21wdCIpLmZvY3VzKCk7",
        "IHJldHVybjsgfQogICAgaWYgKCFTLmluZm8gJiYgIShhd2FpdCBjb25uZWN0KCkpKSB7IHNob3coImNvbm5lY3QiKTsgcmV0dXJu",
        "OyB9CiAgICBpZiAoZm9yY2VOZXdTZWVkKSAkKCJzZWVkIikudmFsdWUgPSAiIjsKICAgIGNvbnN0IHNlZWRWID0gJCgic2VlZCIp",
        "LnZhbHVlLnRyaW0oKTsKICAgIGNvbnN0IGJvZHkgPSB7CiAgICAgIHByb21wdCwgbmVnYXRpdmU6ICQoIm5lZyIpLnZhbHVlLnRy",
        "aW0oKSwgbG9yYTogJCgibG9yYSIpLnZhbHVlIHx8IG51bGwsIHRyaWdnZXI6IGN1clRyaWdnZXIoKSwKICAgICAgbG9yYV9zY2Fs",
        "ZTogKyQoInNjYWxlIikudmFsdWUsIHN0ZXBzOiArJCgic3RlcHMiKS52YWx1ZSwgZ3VpZGFuY2U6ICskKCJndWlkIikudmFsdWUs",
        "CiAgICAgIHNlZWQ6IHNlZWRWID09PSAiIiA/IC0xIDogTWF0aC5tYXgoMCwgcGFyc2VJbnQoc2VlZFYsIDEwKSB8fCAwKSwKICAg",
        "ICAgaGlyZXM6ICQoImhpcmVzIikuY2hlY2tlZCwgc21hcnQ6ICQoInNtYXJ0IikuY2hlY2tlZCwKICAgICAgc2hhcGU6IChkb2N1",
        "bWVudC5xdWVyeVNlbGVjdG9yKCJpbnB1dFtuYW1lPXNoYXBlXTpjaGVja2VkIikgfHwge30pLnZhbHVlIHx8ICJwb3J0cmFpdCIK",
        "ICAgIH07CiAgICBzZXRCdXN5KHRydWUpOwogICAgJCgicmVzdWx0IikuaGlkZGVuID0gZmFsc2U7ICQoInNob3QiKS5oaWRkZW4g",
        "PSB0cnVlOyAkKCJhZnRlciIpLmhpZGRlbiA9IHRydWU7ICQoImluZm8iKS5oaWRkZW4gPSB0cnVlOwogICAgJCgiYmFyIikuaGlk",
        "ZGVuID0gZmFsc2U7ICQoImNhbmNlbFJvdyIpLmhpZGRlbiA9IGZhbHNlOyAkKCJiYXIiKS5maXJzdEVsZW1lbnRDaGlsZC5zdHls",
        "ZS53aWR0aCA9ICIwJSI7CiAgICBzdGF0dXMoIlNlbmRpbmfigKYiKTsKICAgIGNvbnN0IHQwID0gRGF0ZS5ub3coKTsKICAgIGNs",
        "ZWFySW50ZXJ2YWwodGljayk7IHRpY2sgPSBzZXRJbnRlcnZhbCgoKSA9PiB7ICQoInRpbWVyIikudGV4dENvbnRlbnQgPSBNYXRo",
        "LmZsb29yKChEYXRlLm5vdygpIC0gdDApIC8gMTAwMCkgKyAicyI7IH0sIDUwMCk7CiAgICAkKCJyZXN1bHQiKS5zY3JvbGxJbnRv",
        "Vmlldyh7IGJlaGF2aW9yOiAic21vb3RoIiwgYmxvY2s6ICJuZWFyZXN0IiB9KTsKICAgIHRyeSB7CiAgICAgIGNvbnN0IHsgaWQg",
        "fSA9IGF3YWl0IGFwaSgiL2FwaS9qb2JzIiwgeyBtZXRob2Q6ICJQT1NUIiwgYm9keTogSlNPTi5zdHJpbmdpZnkoYm9keSkgfSk7",
        "CiAgICAgIFMuam9iSWQgPSBpZDsKICAgICAgbGV0IGZhaWxzID0gMDsKICAgICAgZm9yICg7OykgewogICAgICAgIGF3YWl0IHNs",
        "ZWVwKDEyMDApOwogICAgICAgIGxldCB2OwogICAgICAgIHRyeSB7IHYgPSBhd2FpdCBhcGkoIi9hcGkvam9icy8iICsgaWQpOyBm",
        "YWlscyA9IDA7IH0KICAgICAgICBjYXRjaCAoZSkgeyBpZiAoKytmYWlscyA+PSA2KSB0aHJvdyBlOyBjb250aW51ZTsgfQogICAg",
        "ICAgICQoImJhciIpLmZpcnN0RWxlbWVudENoaWxkLnN0eWxlLndpZHRoID0gTWF0aC5yb3VuZCh2LnByb2dyZXNzICogMTAwKSAr",
        "ICIlIjsKICAgICAgICBpZiAodi5zdGF0dXMgPT09ICJxdWV1ZWQiKSBzdGF0dXMoIldhaXRpbmcgaW4gbGluZSIgKyAodi5xdWV1",
        "ZSA+IDEgPyAiIMK3ICIgKyAodi5xdWV1ZSAtIDEpICsgIiBhaGVhZCIgOiAi4oCmIikpOwogICAgICAgIGVsc2UgaWYgKHYuc3Rh",
        "dHVzID09PSAicnVubmluZyIpIHN0YXR1cygiQ3JlYXRpbmcgeW91ciBwaG90b+KApiAiICsgTWF0aC5yb3VuZCh2LnByb2dyZXNz",
        "ICogMTAwKSArICIlIik7CiAgICAgICAgZWxzZSBpZiAodi5zdGF0dXMgPT09ICJkb25lIikgewogICAgICAgICAgY29uc3QgYmxv",
        "YiA9IGF3YWl0IGFwaSgiL2FwaS9qb2JzLyIgKyBpZCArICIvaW1hZ2UiLCB7IGJsb2I6IHRydWUgfSwgNjAwMDApOwogICAgICAg",
        "ICAgYXdhaXQgZmluaXNoKGJsb2IsIHYsIGJvZHksIHByb21wdCk7CiAgICAgICAgICBicmVhazsKICAgICAgICB9IGVsc2UgaWYg",
        "KHYuc3RhdHVzID09PSAiY2FuY2VsbGVkIikgeyBzdGF0dXMoIkNhbmNlbGxlZC4iKTsgJCgiYmFyIikuaGlkZGVuID0gdHJ1ZTsg",
        "YnJlYWs7IH0KICAgICAgICBlbHNlIHsgdGhyb3cgbmV3IEVycm9yKHYuZXJyb3IgfHwgIlNvbWV0aGluZyB3ZW50IHdyb25nLiIp",
        "OyB9CiAgICAgIH0KICAgIH0gY2F0Y2ggKGUpIHsKICAgICAgc3RhdHVzKGUubWVzc2FnZSwgdHJ1ZSk7ICQoImJhciIpLmhpZGRl",
        "biA9IHRydWU7CiAgICB9IGZpbmFsbHkgewogICAgICBjbGVhckludGVydmFsKHRpY2spOyBTLmpvYklkID0gbnVsbDsgJCgiY2Fu",
        "Y2VsUm93IikuaGlkZGVuID0gdHJ1ZTsgc2V0QnVzeShmYWxzZSk7CiAgICB9CiAgfQoKICBhc3luYyBmdW5jdGlvbiBmaW5pc2go",
        "YmxvYiwgdiwgYm9keSwgaWRlYSkgewogICAgY29uc3QgbiA9IExTLmdldCgibiIsIDApICsgMTsgTFMuc2V0KCJuIiwgbik7CiAg",
        "ICBjb25zdCBzID0geyBpZDogInMiICsgRGF0ZS5ub3coKSArIE1hdGgucmFuZG9tKCkudG9TdHJpbmcoMzYpLnNsaWNlKDIsIDYp",
        "LCB0OiBEYXRlLm5vdygpLCBuLCBibG9iLCBzZWVkOiB2LnNlZWQsIGlkZWEsIGxvcmE6IGJvZHkubG9yYSwgZmluYWw6IHYucHJv",
        "bXB0IH07CiAgICBTLmxhc3QgPSBzOwogICAgYXdhaXQgYWRkU2hvdChzKTsKICAgIGlmIChTLnNob3RVcmwpIFVSTC5yZXZva2VP",
        "YmplY3RVUkwoUy5zaG90VXJsKTsKICAgIFMuc2hvdFVybCA9IFVSTC5jcmVhdGVPYmplY3RVUkwoYmxvYik7CiAgICAkKCJzaG90",
        "Iikuc3JjID0gUy5zaG90VXJsOyAkKCJzaG90IikuaGlkZGVuID0gZmFsc2U7CiAgICAkKCJiYXIiKS5oaWRkZW4gPSB0cnVlOyAk",
        "KCJhZnRlciIpLmhpZGRlbiA9IGZhbHNlOwogICAgc3RhdHVzKCJEb25lLiBTYXZlZCB0byBHYWxsZXJ5LiIpOwogICAgJCgiaW5m",
        "byIpLmhpZGRlbiA9IGZhbHNlOyAkKCJpbmZvIikudGV4dENvbnRlbnQgPSAiTm8uICIgKyBTdHJpbmcobikucGFkU3RhcnQoMiwg",
        "IjAiKSArICIgwrcgc2VlZCAiICsgdi5zZWVkOwogICAgJCgic2VlZCIpLnBsYWNlaG9sZGVyID0gImxhc3Q6ICIgKyB2LnNlZWQ7",
        "CiAgfQoKICBhc3luYyBmdW5jdGlvbiBnZW5lcmF0ZVBob25lKGZvcmNlTmV3U2VlZCkgewogICAgaWYgKFMuYnVzeSkgcmV0dXJu",
        "OwogICAgY29uc3QgcHJvbXB0ID0gJCgicHJvbXB0IikudmFsdWUudHJpbSgpOwogICAgaWYgKCFwcm9tcHQpIHsgJCgicHJvbXB0",
        "IikuZm9jdXMoKTsgcmV0dXJuOyB9CiAgICByZWFkTmF0aXZlKCk7CiAgICBpZiAoIShOSSAmJiBOSS5tb2RlbFJlYWR5KSkgeyBh",
        "cHBseU1vZGUoKTsgc2hvdygiY29ubmVjdCIpOyAkKCJwbXNnIikudGV4dENvbnRlbnQgPSAiSW1wb3J0IHRoZSBtb2RlbCBmaXJz",
        "dC4iOyAkKCJwbXNnIikuY2xhc3NOYW1lID0gIm1zZyBiYWQiOyByZXR1cm47IH0KICAgIGlmIChmb3JjZU5ld1NlZWQpICQoInNl",
        "ZWQiKS52YWx1ZSA9ICIiOwogICAgY29uc3Qgc2VlZFYgPSAkKCJzZWVkIikudmFsdWUudHJpbSgpOwogICAgY29uc3QgYm9keSA9",
        "IHsgcHJvbXB0LCBzdGVwczogTWF0aC5tYXgoMTAsIE1hdGgubWluKDUwLCArJCgic3RlcHMiKS52YWx1ZSB8fCAyMCkpLCBzZWVk",
        "OiBzZWVkViA9PT0gIiIgPyAtMSA6IE1hdGgubWF4KDAsIHBhcnNlSW50KHNlZWRWLCAxMCkgfHwgMCkgfTsKICAgIHNldEJ1c3ko",
        "dHJ1ZSk7CiAgICAkKCJyZXN1bHQiKS5oaWRkZW4gPSBmYWxzZTsgJCgic2hvdCIpLmhpZGRlbiA9IHRydWU7ICQoImFmdGVyIiku",
        "aGlkZGVuID0gdHJ1ZTsgJCgiaW5mbyIpLmhpZGRlbiA9IHRydWU7CiAgICAkKCJiYXIiKS5oaWRkZW4gPSBmYWxzZTsgJCgiY2Fu",
        "Y2VsUm93IikuaGlkZGVuID0gZmFsc2U7ICQoImJhciIpLmZpcnN0RWxlbWVudENoaWxkLnN0eWxlLndpZHRoID0gIjAlIjsKICAg",
        "IHN0YXR1cygiU3RhcnRpbmfigKYiKTsKICAgIGNvbnN0IHQwID0gRGF0ZS5ub3coKTsKICAgIGNsZWFySW50ZXJ2YWwodGljayk7",
        "IHRpY2sgPSBzZXRJbnRlcnZhbCgoKSA9PiB7ICQoInRpbWVyIikudGV4dENvbnRlbnQgPSBNYXRoLmZsb29yKChEYXRlLm5vdygp",
        "IC0gdDApIC8gMTAwMCkgKyAicyI7IH0sIDUwMCk7CiAgICAkKCJyZXN1bHQiKS5zY3JvbGxJbnRvVmlldyh7IGJlaGF2aW9yOiAi",
        "c21vb3RoIiwgYmxvY2s6ICJuZWFyZXN0IiB9KTsKICAgIHRyeSB7CiAgICAgIGNvbnN0IG91dCA9IGF3YWl0IG5ldyBQcm9taXNl",
        "KChyZXNvbHZlLCByZWplY3QpID0+IHsKICAgICAgICBQSC5jdXIgPSB7IHJlc29sdmUsIHJlamVjdCB9OwogICAgICAgIGxldCBy",
        "OyB0cnkgeyByID0gSlNPTi5wYXJzZShOQVRJVkUuZ2VuZXJhdGUoSlNPTi5zdHJpbmdpZnkoYm9keSkpKTsgfSBjYXRjaCAoZSkg",
        "eyBQSC5jdXIgPSBudWxsOyByZWplY3QobmV3IEVycm9yKCJUaGUgYXBwIGNvdWxkIG5vdCBzdGFydCB0aGUgam9iLiIpKTsgcmV0",
        "dXJuOyB9CiAgICAgICAgaWYgKHIuZXJyb3IpIHsgUEguY3VyID0gbnVsbDsgcmVqZWN0KG5ldyBFcnJvcihyLmVycm9yKSk7IH0K",
        "ICAgICAgfSk7CiAgICAgIGNvbnN0IGJpbiA9IGF0b2Iob3V0LmltYWdlKSwgYXJyID0gbmV3IFVpbnQ4QXJyYXkoYmluLmxlbmd0",
        "aCk7CiAgICAgIGZvciAobGV0IGkgPSAwOyBpIDwgYmluLmxlbmd0aDsgaSsrKSBhcnJbaV0gPSBiaW4uY2hhckNvZGVBdChpKTsK",
        "ICAgICAgYXdhaXQgZmluaXNoKG5ldyBCbG9iKFthcnJdLCB7IHR5cGU6ICJpbWFnZS9qcGVnIiB9KSwgeyBzZWVkOiBvdXQuc2Vl",
        "ZCwgcHJvbXB0OiBvdXQucHJvbXB0IH0sIHsgbG9yYTogIm9uIHRoaXMgcGhvbmUiIH0sIHByb21wdCk7CiAgICB9IGNhdGNoIChl",
        "KSB7CiAgICAgIHN0YXR1cyhlLm1lc3NhZ2UsIGUubWVzc2FnZSAhPT0gIkNhbmNlbGxlZC4iKTsgJCgiYmFyIikuaGlkZGVuID0g",
        "dHJ1ZTsKICAgIH0gZmluYWxseSB7CiAgICAgIGNsZWFySW50ZXJ2YWwodGljayk7IFBILmN1ciA9IG51bGw7ICQoImNhbmNlbFJv",
        "dyIpLmhpZGRlbiA9IHRydWU7IHNldEJ1c3koZmFsc2UpOwogICAgfQogIH0KCiAgJCgiZ28iKS5hZGRFdmVudExpc3RlbmVyKCJj",
        "bGljayIsICgpID0+IGdlbmVyYXRlKGZhbHNlKSk7CiAgJCgiYWdhaW4iKS5hZGRFdmVudExpc3RlbmVyKCJjbGljayIsICgpID0+",
        "IGdlbmVyYXRlKHRydWUpKTsKICAkKCJjYW5jZWwiKS5hZGRFdmVudExpc3RlbmVyKCJjbGljayIsIGFzeW5jICgpID0+IHsgaWYg",
        "KFMubW9kZSA9PT0gInBob25lIikgeyB0cnkgeyBOQVRJVkUuY2FuY2VsKCk7IH0gY2F0Y2ggKGUpIHt9IHN0YXR1cygiQ2FuY2Vs",
        "bGluZ+KApiIpOyByZXR1cm47IH0gaWYgKFMuam9iSWQpIHsgdHJ5IHsgYXdhaXQgYXBpKCIvYXBpL2pvYnMvIiArIFMuam9iSWQs",
        "IHsgbWV0aG9kOiAiREVMRVRFIiB9KTsgfSBjYXRjaCAoZSkge30gc3RhdHVzKCJDYW5jZWxsaW5n4oCmIik7IH0gfSk7CiAgJCgi",
        "c2F2ZSIpLmFkZEV2ZW50TGlzdGVuZXIoImNsaWNrIiwgKCkgPT4gUy5sYXN0ICYmIHNhdmVCbG9iKFMubGFzdC5ibG9iLCBmbmFt",
        "ZShTLmxhc3QpKSk7CiAgJCgic2hhcmUiKS5hZGRFdmVudExpc3RlbmVyKCJjbGljayIsICgpID0+IFMubGFzdCAmJiBzaGFyZUJs",
        "b2IoUy5sYXN0LmJsb2IsIGZuYW1lKFMubGFzdCkpKTsKCiAgLyogLS0tLS0tLS0tLSBvbi1waG9uZSBtb2RlbCArIGV2ZW50cyBm",
        "cm9tIHRoZSBBbmRyb2lkIGFwcCAtLS0tLS0tLS0tICovCiAgY29uc3QgUEggPSB7IGN1cjogbnVsbCB9OwogIHdpbmRvdy5fX3N0",
        "dWRpb05hdGl2ZSA9IGZ1bmN0aW9uIChqc29uKSB7CiAgICBsZXQgZTsgdHJ5IHsgZSA9IEpTT04ucGFyc2UoanNvbik7IH0gY2F0",
        "Y2ggKHgpIHsgcmV0dXJuOyB9CiAgICBpZiAoZS50eXBlID09PSAicHJvZ3Jlc3MiKSB7CiAgICAgIGlmICghUEguY3VyKSByZXR1",
        "cm47CiAgICAgICQoImJhciIpLmZpcnN0RWxlbWVudENoaWxkLnN0eWxlLndpZHRoID0gTWF0aC5yb3VuZChlLnAgKiAxMDApICsg",
        "IiUiOwogICAgICBzdGF0dXMoIkNyZWF0aW5nIG9uIHRoaXMgcGhvbmXigKYgIiArIE1hdGgucm91bmQoZS5wICogMTAwKSArICIl",
        "Iik7CiAgICB9IGVsc2UgaWYgKGUudHlwZSA9PT0gImxvYWRpbmciKSB7CiAgICAgIGlmIChQSC5jdXIpIHN0YXR1cygiTG9hZGlu",
        "ZyB0aGUgbW9kZWwuIFRoZSBmaXJzdCBwaG90byB0YWtlcyBsb25nZXIuIik7CiAgICB9IGVsc2UgaWYgKGUudHlwZSA9PT0gImRv",
        "bmUiKSB7CiAgICAgIGlmIChQSC5jdXIpIHsgY29uc3QgYyA9IFBILmN1cjsgUEguY3VyID0gbnVsbDsgYy5yZXNvbHZlKGUpOyB9",
        "CiAgICB9IGVsc2UgaWYgKGUudHlwZSA9PT0gImVycm9yIiB8fCBlLnR5cGUgPT09ICJjYW5jZWxsZWQiKSB7CiAgICAgIGlmIChQ",
        "SC5jdXIpIHsgY29uc3QgYyA9IFBILmN1cjsgUEguY3VyID0gbnVsbDsgYy5yZWplY3QobmV3IEVycm9yKGUudHlwZSA9PT0gImNh",
        "bmNlbGxlZCIgPyAiQ2FuY2VsbGVkLiIgOiAoZS5tZXNzYWdlIHx8ICJTb21ldGhpbmcgd2VudCB3cm9uZy4iKSkpOyB9CiAgICB9",
        "IGVsc2UgaWYgKGUudHlwZSA9PT0gIm1vZGVsIikgewogICAgICAkKCJwYmFyIikuaGlkZGVuID0gZS5zdGF0ZSAhPT0gIndvcmtp",
        "bmciOwogICAgICAkKCJwYmFyIikuZmlyc3RFbGVtZW50Q2hpbGQuc3R5bGUud2lkdGggPSBNYXRoLnJvdW5kKChlLnAgfHwgMCkg",
        "KiAxMDApICsgIiUiOwogICAgICAkKCJwbXNnIikudGV4dENvbnRlbnQgPSBlLm1lc3NhZ2UgfHwgIiI7CiAgICAgICQoInBtc2ci",
        "KS5jbGFzc05hbWUgPSAibXNnIiArIChlLnN0YXRlID09PSAiZXJyb3IiID8gIiBiYWQiIDogIiIpOwogICAgICAkKCJwaW1wb3J0",
        "IikuZGlzYWJsZWQgPSAkKCJwZGwiKS5kaXNhYmxlZCA9IGUuc3RhdGUgPT09ICJ3b3JraW5nIjsKICAgICAgaWYgKGUuc3RhdGUg",
        "PT09ICJkb25lIiB8fCBlLnN0YXRlID09PSAiZXJyb3IiKSB7IHJlYWROYXRpdmUoKTsgYXBwbHlNb2RlKCk7IH0KICAgIH0KICB9",
        "OwogIGRvY3VtZW50LnF1ZXJ5U2VsZWN0b3JBbGwoImlucHV0W25hbWU9bW9kZV0iKS5mb3JFYWNoKHIgPT4gci5hZGRFdmVudExp",
        "c3RlbmVyKCJjaGFuZ2UiLCAoKSA9PiB7CiAgICBpZiAoIXIuY2hlY2tlZCkgcmV0dXJuOwogICAgUy5tb2RlID0gci52YWx1ZTsg",
        "TFMuc2V0KCJtb2RlIiwgUy5tb2RlKTsgYXBwbHlNb2RlKCk7CiAgfSkpOwogICQoInBpbXBvcnQiKS5hZGRFdmVudExpc3RlbmVy",
        "KCJjbGljayIsICgpID0+IHsgdHJ5IHsgTkFUSVZFLmltcG9ydE1vZGVsKCQoInB0cmlnIikudmFsdWUudHJpbSgpKTsgfSBjYXRj",
        "aCAoZSkgeyAkKCJwbXNnIikudGV4dENvbnRlbnQgPSAiQ291bGQgbm90IG9wZW4gdGhlIGZpbGUgcGlja2VyLiI7IH0gfSk7CiAg",
        "JCgicGRsIikuYWRkRXZlbnRMaXN0ZW5lcigiY2xpY2siLCAoKSA9PiB7CiAgICBjb25zdCB1ID0gJCgicHVybCIpLnZhbHVlLnRy",
        "aW0oKTsKICAgIGlmICghL15odHRwczpcL1wvL2kudGVzdCh1KSkgeyAkKCJwbXNnIikudGV4dENvbnRlbnQgPSAiUGFzdGUgYSBs",
        "aW5rIHRoYXQgc3RhcnRzIHdpdGggaHR0cHM6Ly8iOyAkKCJwbXNnIikuY2xhc3NOYW1lID0gIm1zZyBiYWQiOyByZXR1cm47IH0K",
        "ICAgIHRyeSB7IE5BVElWRS5kb3dubG9hZE1vZGVsKHUsICQoInB0cmlnIikudmFsdWUudHJpbSgpKTsgfSBjYXRjaCAoZSkgeyAk",
        "KCJwbXNnIikudGV4dENvbnRlbnQgPSAiQ291bGQgbm90IHN0YXJ0IHRoZSBkb3dubG9hZC4iOyB9CiAgfSk7CiAgJCgicGRlbCIp",
        "LmFkZEV2ZW50TGlzdGVuZXIoImNsaWNrIiwgKCkgPT4gewogICAgbGV0IHIgPSAib2siOyB0cnkgeyByID0gTkFUSVZFLmRlbGV0",
        "ZU1vZGVsKCk7IH0gY2F0Y2ggKGUpIHt9CiAgICBpZiAociA9PT0gImJ1c3kiKSB7ICQoInBtc2ciKS50ZXh0Q29udGVudCA9ICJX",
        "YWl0IHVudGlsIHRoZSBjdXJyZW50IGpvYiBmaW5pc2hlcy4iOyAkKCJwbXNnIikuY2xhc3NOYW1lID0gIm1zZyBiYWQiOyByZXR1",
        "cm47IH0KICAgIHJlYWROYXRpdmUoKTsgYXBwbHlNb2RlKCk7ICQoInBtc2ciKS50ZXh0Q29udGVudCA9ICJEZWxldGluZyB0aGUg",
        "bW9kZWzigKYiOyAkKCJwbXNnIikuY2xhc3NOYW1lID0gIm1zZyI7CiAgfSk7CgogIC8qIC0tLS0tLS0tLS0gaW5zdGFsbCAtLS0t",
        "LS0tLS0tICovCiAgbGV0IGRlZmVycmVkID0gbnVsbDsKICB3aW5kb3cuYWRkRXZlbnRMaXN0ZW5lcigiYmVmb3JlaW5zdGFsbHBy",
        "b21wdCIsIGUgPT4geyBlLnByZXZlbnREZWZhdWx0KCk7IGRlZmVycmVkID0gZTsgJCgiaW5zdGFsbCIpLmhpZGRlbiA9IGZhbHNl",
        "OyAkKCJpbnN0YWxsSGludCIpLmhpZGRlbiA9IHRydWU7IH0pOwogICQoImluc3RhbGwiKS5hZGRFdmVudExpc3RlbmVyKCJjbGlj",
        "ayIsIGFzeW5jICgpID0+IHsgaWYgKCFkZWZlcnJlZCkgcmV0dXJuOyBkZWZlcnJlZC5wcm9tcHQoKTsgdHJ5IHsgYXdhaXQgZGVm",
        "ZXJyZWQudXNlckNob2ljZTsgfSBjYXRjaCAoZSkge30gZGVmZXJyZWQgPSBudWxsOyAkKCJpbnN0YWxsIikuaGlkZGVuID0gdHJ1",
        "ZTsgfSk7CiAgd2luZG93LmFkZEV2ZW50TGlzdGVuZXIoImFwcGluc3RhbGxlZCIsICgpID0+IHsgJCgiaW5zdGFsbCIpLmhpZGRl",
        "biA9IHRydWU7ICQoImluc3RhbGxIaW50IikudGV4dENvbnRlbnQgPSAiSW5zdGFsbGVkLiI7ICQoImluc3RhbGxIaW50IikuaGlk",
        "ZGVuID0gZmFsc2U7IH0pOwoKICAvKiAtLS0tLS0tLS0tIHN0YXJ0IC0tLS0tLS0tLS0gKi8KICBhc3luYyBmdW5jdGlvbiBzdGFy",
        "dCgpIHsKICAgIGNvbnN0IGggPSBwYXJzZUxpbmsobG9jYXRpb24uaGFzaCA/IGxvY2F0aW9uLm9yaWdpbiArIGxvY2F0aW9uLmhh",
        "c2ggOiAiIik7CiAgICBpZiAoaC5rZXkpIHsKICAgICAgUy51cmwgPSBsb2NhdGlvbi5vcmlnaW47IFMua2V5ID0gaC5rZXk7IExT",
        "LnNldCgidXJsIiwgUy51cmwpOyBMUy5zZXQoImtleSIsIFMua2V5KTsKICAgICAgdHJ5IHsgaGlzdG9yeS5yZXBsYWNlU3RhdGUo",
        "bnVsbCwgIiIsIGxvY2F0aW9uLnBhdGhuYW1lICsgbG9jYXRpb24uc2VhcmNoKTsgfSBjYXRjaCAoZSkge30KICAgIH0gZWxzZSBp",
        "ZiAoIVMudXJsICYmICFOQVRJVkUpIHsKICAgICAgdHJ5IHsgY29uc3QgciA9IGF3YWl0IGZldGNoKCJhcGkvaGVhbHRoIik7IGNv",
        "bnN0IGogPSBhd2FpdCByLmpzb24oKTsgaWYgKGogJiYgai5uYW1lID09PSAibG9yYS1zdHVkaW8iKSB7IFMudXJsID0gbG9jYXRp",
        "b24ub3JpZ2luOyBMUy5zZXQoInVybCIsIFMudXJsKTsgfSB9IGNhdGNoIChlKSB7fQogICAgfQogICAgJCgibGluayIpLnZhbHVl",
        "ID0gUy51cmw7ICQoImtleSIpLnZhbHVlID0gUy5rZXk7CiAgICBpZiAoTkFUSVZFKSBhcHBseU1vZGUoKTsKICAgIGlmIChTLm1v",
        "ZGUgPT09ICJwaG9uZSIpIHsgaWYgKCEoTkkgJiYgTkkubW9kZWxSZWFkeSkpIHNob3coImNvbm5lY3QiKTsgcmV0dXJuOyB9CiAg",
        "ICBpZiAoUy51cmwpIHsgaWYgKCEoYXdhaXQgY29ubmVjdCgpKSkgc2hvdygiY29ubmVjdCIpOyB9IGVsc2UgeyBzaG93KCJjb25u",
        "ZWN0Iik7IGNtc2coIlBhc3RlIHRoZSBsaW5rIENvbGFiIGdhdmUgeW91IHRvIHN0YXJ0LiIpOyB9CiAgfQogIGlmICghTkFUSVZF",
        "ICYmICJzZXJ2aWNlV29ya2VyIiBpbiBuYXZpZ2F0b3IgJiYgL15odHRwcz86JC8udGVzdChsb2NhdGlvbi5wcm90b2NvbCkpIG5h",
        "dmlnYXRvci5zZXJ2aWNlV29ya2VyLnJlZ2lzdGVyKCJzdy5qcyIpLmNhdGNoKCgpID0+IHt9KTsKICBzdGFydCgpOwp9KSgpOwo8",
        "L3NjcmlwdD4KPC9ib2R5Pgo8L2h0bWw+Cg==",
    ]), "text/html; charset=utf-8"),
    "/manifest.webmanifest": ("".join([
        "ewogICJuYW1lIjogIkxvUkEgU3R1ZGlvIiwKICAic2hvcnRfbmFtZSI6ICJTdHVkaW8iLAogICJkZXNjcmlwdGlvbiI6ICJNYWtl",
        "IHBob3RvcyBmcm9tIHlvdXIgb3duIExvUkEgbW9kZWxzIHVzaW5nIHlvdXIgQ29sYWIgR1BVLiIsCiAgInN0YXJ0X3VybCI6ICIu",
        "LyIsCiAgInNjb3BlIjogIi4vIiwKICAiZGlzcGxheSI6ICJzdGFuZGFsb25lIiwKICAib3JpZW50YXRpb24iOiAicG9ydHJhaXQi",
        "LAogICJiYWNrZ3JvdW5kX2NvbG9yIjogIiMwYzExMTciLAogICJ0aGVtZV9jb2xvciI6ICIjMGMxMTE3IiwKICAiaWNvbnMiOiBb",
        "CiAgICB7ICJzcmMiOiAiaWNvbi0xOTIucG5nIiwgInNpemVzIjogIjE5MngxOTIiLCAidHlwZSI6ICJpbWFnZS9wbmciLCAicHVy",
        "cG9zZSI6ICJhbnkiIH0sCiAgICB7ICJzcmMiOiAiaWNvbi01MTIucG5nIiwgInNpemVzIjogIjUxMng1MTIiLCAidHlwZSI6ICJp",
        "bWFnZS9wbmciLCAicHVycG9zZSI6ICJhbnkiIH0sCiAgICB7ICJzcmMiOiAiaWNvbi01MTIucG5nIiwgInNpemVzIjogIjUxMng1",
        "MTIiLCAidHlwZSI6ICJpbWFnZS9wbmciLCAicHVycG9zZSI6ICJtYXNrYWJsZSIgfQogIF0KfQo=",
    ]), "application/manifest+json"),
    "/sw.js": ("".join([
        "Ly8gQXBwIGthIHNoZWxsIGNhY2hlIGthcnRhIGhhaSB0YWFraSBhcHAgQ29sYWIgYmFuZCBob25lIHBhciBiaGkga2h1bCBqYXll",
        "Lgpjb25zdCBWID0gInN0dWRpby12MSI7CmNvbnN0IFNIRUxMID0gWyIuLyIsICJpbmRleC5odG1sIiwgIm1hbmlmZXN0LndlYm1h",
        "bmlmZXN0IiwgImljb24tMTkyLnBuZyIsICJpY29uLTUxMi5wbmciXTsKc2VsZi5hZGRFdmVudExpc3RlbmVyKCJpbnN0YWxsIiwg",
        "ZSA9PiB7CiAgZS53YWl0VW50aWwoY2FjaGVzLm9wZW4oVikudGhlbihjID0+IGMuYWRkQWxsKFNIRUxMKSkudGhlbigoKSA9PiBz",
        "ZWxmLnNraXBXYWl0aW5nKCkpKTsKfSk7CnNlbGYuYWRkRXZlbnRMaXN0ZW5lcigiYWN0aXZhdGUiLCBlID0+IHsKICBlLndhaXRV",
        "bnRpbChjYWNoZXMua2V5cygpLnRoZW4oa3MgPT4gUHJvbWlzZS5hbGwoa3MuZmlsdGVyKGsgPT4gayAhPT0gVikubWFwKGsgPT4g",
        "Y2FjaGVzLmRlbGV0ZShrKSkpKS50aGVuKCgpID0+IHNlbGYuY2xpZW50cy5jbGFpbSgpKSk7Cn0pOwpzZWxmLmFkZEV2ZW50TGlz",
        "dGVuZXIoImZldGNoIiwgZSA9PiB7CiAgY29uc3QgdSA9IG5ldyBVUkwoZS5yZXF1ZXN0LnVybCk7CiAgaWYgKGUucmVxdWVzdC5t",
        "ZXRob2QgIT09ICJHRVQiIHx8IHUub3JpZ2luICE9PSBsb2NhdGlvbi5vcmlnaW4gfHwgdS5wYXRobmFtZS5zdGFydHNXaXRoKCIv",
        "YXBpLyIpKSByZXR1cm47CiAgZS5yZXNwb25kV2l0aCgKICAgIGZldGNoKGUucmVxdWVzdCkKICAgICAgLnRoZW4ociA9PiB7IGlm",
        "IChyLm9rKSB7IGNvbnN0IGMgPSByLmNsb25lKCk7IGNhY2hlcy5vcGVuKFYpLnRoZW4oeCA9PiB4LnB1dChlLnJlcXVlc3QsIGMp",
        "KTsgfSByZXR1cm4gcjsgfSkKICAgICAgLmNhdGNoKCgpID0+IGNhY2hlcy5tYXRjaChlLnJlcXVlc3QpLnRoZW4obSA9PiBtIHx8",
        "IGNhY2hlcy5tYXRjaCgiLi8iKSkpCiAgKTsKfSk7Cg==",
    ]), "text/javascript"),
}

# ----------------------------------------------------------------------------
# Safety filters REMOVED completely (prompt filter + negative + safety checker)
# ----------------------------------------------------------------------------
QUALITY = ("RAW photo, 8k uhd, dslr, soft natural lighting, high quality, film grain, Fujifilm XT3, "
           "sharp focus, natural skin texture, visible skin pores, realistic, photorealistic")
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
        pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, vae=vae, torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False)  # safety OFF
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
        final_neg = ", ".join(x for x in [p["negative"], BASE_NEG] if x)

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
        image = out.images[0]

        if p["hires"]:
            up = image.resize((int(w * 1.5), int(h * 1.5)), Image.LANCZOS)
            hs = 0.3
            gen2 = torch.Generator("cuda").manual_seed(seed)
            out = self.img2img(prompt_embeds=cond, negative_prompt_embeds=neg, image=up, strength=hs,
                               num_inference_steps=int(steps / hs), guidance_scale=p["guidance"],
                               generator=gen2, cross_attention_kwargs=cross, callback_on_step_end=cb)
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
