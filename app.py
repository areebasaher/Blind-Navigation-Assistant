# -*- coding: utf-8 -*-
"""
app.py - Blind Navigation Assistant (Gradio)

Pipeline (safety is RULE-BASED; the VLM only describes the scene):
  1. YOLO11 COCO-pretrained detector -> objects + boxes (left / ahead / right)
  2. Depth-Anything-V2 METRIC model  -> real distance in meters per object
  3. Rules decide STOP / move left / move right / clear
  4. Optional VLM adds one scene-description sentence
  5. Optional scene classifier (best.pt from train_model.py) adds scene type
  6. TTS speaks the result

Pick hardware with:  PROFILE=cpu | gpu_small | gpu   (auto if unset)
Override any model: DET_MODEL, DEPTH_MODEL, VLM_MODEL ("none" disables the VLM)
"""
import os
import tempfile

import gradio as gr
import numpy as np
import torch
from PIL import Image
from huggingface_hub import login
from ultralytics import YOLO
from transformers import (
    pipeline, AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig,
)

# ── CONFIG ────────────────────────────────────────────────────────────────
HAS_CUDA = torch.cuda.is_available()
VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9 if HAS_CUDA else 0

DEPTH_SMALL = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
DEPTH_BASE = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf"
QWEN = "Qwen/Qwen2.5-VL-3B-Instruct"

PROFILES = {
    # 8 GB RAM laptop, no GPU: no VLM (rule-based guidance is already complete)
    "cpu":       dict(det="yolo11n.pt", depth=DEPTH_SMALL, vlm=None, quant=False),
    # 4-6 GB GPU
    "gpu_small": dict(det="yolo11s.pt", depth=DEPTH_SMALL, vlm=QWEN, quant=True),
    # T4 or better
    "gpu":       dict(det="yolo11m.pt", depth=DEPTH_BASE, vlm=QWEN, quant=False),
}
# Auto-pick by VRAM: <4 GB (e.g. Quadro M1000M, 2 GB) -> cpu, 4-11 GB -> gpu_small, 12+ GB -> gpu
_auto = "gpu" if VRAM_GB >= 12 else "gpu_small" if VRAM_GB >= 4 else "cpu"
PROFILE = os.environ.get("PROFILE", _auto)
cfg = PROFILES[PROFILE]
HAS_CUDA = HAS_CUDA and PROFILE != "cpu"     # cpu profile never touches the GPU
if PROFILE == "cpu":
    torch.set_num_threads(min(4, os.cpu_count() or 4))

DET_MODEL = os.environ.get("DET_MODEL", cfg["det"])
DEPTH_MODEL = os.environ.get("DEPTH_MODEL", cfg["depth"])
VLM_MODEL = os.environ.get("VLM_MODEL", cfg["vlm"] or "none")
VLM_MODEL = None if VLM_MODEL.lower() == "none" else VLM_MODEL
SCENE_MODEL_PATH = os.environ.get("SCENE_MODEL_PATH", "models/scene_best.pt")  # optional

DET_CONF = 0.35
MAX_SIDE = 1024                      # downscale big phone photos for speed
VEHICLES = {"car", "bus", "truck", "motorcycle", "bicycle", "train"}
VEHICLE_ALERT_M = 6.0                # warn about a vehicle closer than this
OBSTACLE_ALERT_M = 2.0               # warn about any other object ahead closer than this
PATH_BLOCKED_M = 1.5                 # walking zone considered blocked below this
# NOTE: tune the three thresholds above on YOUR phone photos.


def authenticate_huggingface():
    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)


authenticate_huggingface()


# ── MODEL LOADING ─────────────────────────────────────────────────────────
def load_all():
    det = YOLO(DET_MODEL)
    depth = pipeline("depth-estimation", model=DEPTH_MODEL, device=0 if HAS_CUDA else -1)

    scene = YOLO(SCENE_MODEL_PATH) if os.path.exists(SCENE_MODEL_PATH) else None

    vlm = proc = None
    if VLM_MODEL:
        try:
            kw = {}
            if cfg["quant"] and HAS_CUDA:
                kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            proc = AutoProcessor.from_pretrained(VLM_MODEL)
            vlm = AutoModelForImageTextToText.from_pretrained(
                VLM_MODEL,
                torch_dtype=torch.float16 if HAS_CUDA else torch.float32,
                device_map="auto" if HAS_CUDA else None,
                **kw,
            ).eval()
        except Exception as e:  # app still works without the VLM
            print(f"[warn] VLM disabled: {e}")
            vlm = proc = None
    return det, depth, scene, vlm, proc


print(f"Profile: {PROFILE} | detector={DET_MODEL} | depth={DEPTH_MODEL} | vlm={VLM_MODEL}")
det_model, depth_pipe, scene_model, vlm_model, vlm_proc = load_all()
print("Models ready.")


# ── PERCEPTION ────────────────────────────────────────────────────────────
def depth_meters(image: Image.Image) -> np.ndarray:
    """Per-pixel distance in meters (metric depth model), same size as image."""
    d = depth_pipe(image)["predicted_depth"].squeeze().float()
    d = torch.nn.functional.interpolate(
        d[None, None], size=(image.height, image.width),
        mode="bilinear", align_corners=False)[0, 0]
    return d.cpu().numpy()


def detect(image: Image.Image, depth: np.ndarray):
    W = image.width
    r = det_model(image, conf=DET_CONF, verbose=False)[0]
    objs = []
    for b in r.boxes:
        x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
        x1, y1 = max(x1, 0), max(y1, 0)
        region = depth[y1:y2, x1:x2]
        if region.size == 0:
            continue
        cx = (x1 + x2) / 2
        objs.append(dict(
            label=det_model.names[int(b.cls)],
            conf=float(b.conf),
            dist=float(np.percentile(region, 20)),   # nearest part of the object
            side="on your left" if cx < W / 3 else "on your right" if cx > 2 * W / 3 else "ahead",
        ))
    return sorted(objs, key=lambda o: o["dist"])


def zone_clearance(depth: np.ndarray):
    """Nearest distance (m) in the walking band for left / center / right thirds.
    Band skips the very bottom rows so the floor right at the camera's feet
    isn't mistaken for an obstacle."""
    H, W = depth.shape
    rows = slice(int(H * 0.35), int(H * 0.75))
    cols = [slice(0, W // 3), slice(W // 3, 2 * W // 3), slice(2 * W // 3, W)]
    return [float(np.percentile(depth[rows, c], 15)) for c in cols]  # left, center, right


def scene_label(image: Image.Image) -> str:
    if scene_model is None:
        return ""
    probs = scene_model(image, verbose=False)[0].probs
    conf = float(probs.top1conf)
    return f"{scene_model.names[probs.top1]} ({conf * 100:.0f}%)" if conf >= 0.5 else ""


# ── RULE-BASED SAFETY + GUIDANCE ──────────────────────────────────────────
def build_guidance(objs, zones):
    left, center, right = zones
    hazards = []
    for o in objs:
        if o["label"] in VEHICLES and o["dist"] < VEHICLE_ALERT_M:
            hazards.append(o)
        elif o["side"] == "ahead" and o["dist"] < OBSTACLE_ALERT_M:
            hazards.append(o)

    blocked = center < PATH_BLOCKED_M
    msgs = [f"Warning: {h['label']} {h['side']}, about {h['dist']:.1f} meters."
            for h in hazards[:3]]

    if blocked or any(h["side"] == "ahead" for h in hazards):
        best = max((left, "left"), (right, "right"))
        if best[0] > PATH_BLOCKED_M and best[0] > center:
            msgs.append(f"Stop. Path ahead is blocked. Move slightly {best[1]}.")
        else:
            msgs.append("Stop. Path ahead is blocked. Wait or ask for help.")
        level = "STOP"
    elif hazards:
        msgs.append("Proceed slowly and stay alert.")
        level = "CAUTION"
    else:
        msgs.append("Path ahead looks clear.")
        level = "CLEAR"
    return level, " ".join(msgs)


def describe_scene(image: Image.Image, objs) -> str:
    """Optional VLM: ONE descriptive sentence. It never decides safety."""
    if vlm_model is None:
        return ""
    facts = "; ".join(f"{o['label']} {o['side']} {o['dist']:.1f} m" for o in objs[:6]) or "none"
    prompt = ("Describe this scene in one short sentence for a blind pedestrian. "
              f"Known objects: {facts}. Mention only what is clearly visible.")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = vlm_proc.apply_chat_template(messages, add_generation_prompt=True)
    inputs = vlm_proc(text=[text], images=[image], return_tensors="pt").to(vlm_model.device)
    with torch.no_grad():
        out = vlm_model.generate(**inputs, max_new_tokens=50, do_sample=False)
    return vlm_proc.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                 skip_special_tokens=True)[0].strip()


# ── TEXT TO SPEECH ────────────────────────────────────────────────────────
def text_to_speech(text: str):
    """TTS=offline uses pyttsx3 (no internet); default is gTTS. Each falls back to the other."""
    offline_first = os.environ.get("TTS") == "offline" or (os.name == "nt" and os.environ.get("TTS") != "gtts")
    order = ["offline", "gtts"] if offline_first else ["gtts", "offline"]  # Windows: built-in voices by default
    for backend in order:
        try:
            if backend == "gtts":
                from gtts import gTTS
                path = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
                gTTS(text=text[:900], lang="en").save(path)
            else:
                import pyttsx3
                path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
                eng = pyttsx3.init()
                eng.save_to_file(text[:900], path)
                eng.runAndWait()
            return path
        except Exception as e:
            print(f"[tts:{backend}] {e}")
    return None


# ── UI PIPELINE ───────────────────────────────────────────────────────────
def run_pipeline(image: Image.Image):
    if image is None:
        return "Please upload an image.", "", "", None
    image = image.convert("RGB")
    image.thumbnail((MAX_SIDE, MAX_SIDE))

    try:
        depth = depth_meters(image)
        objs = detect(image, depth)
        zones = zone_clearance(depth)
    except Exception as e:
        return f"Analysis failed: {e}", "", "", None

    objects_text = "\n".join(
        f"- {o['label']} {o['side']}, {o['dist']:.1f} m ({o['conf'] * 100:.0f}%)"
        for o in objs[:8]) or "No objects detected."
    scene = scene_label(image)
    if scene:
        objects_text = f"Scene: {scene}\n" + objects_text

    level, advice = build_guidance(objs, zones)
    safety_text = (f"{level}\nClearance (m) - left: {zones[0]:.1f} | "
                   f"center: {zones[1]:.1f} | right: {zones[2]:.1f}")

    try:
        extra = describe_scene(image, objs)
    except Exception as e:
        print(f"[vlm] {e}")
        extra = ""
    guidance = f"{advice} {extra}".strip()   # safety advice always comes first
    return objects_text, safety_text, guidance, text_to_speech(guidance)


with gr.Blocks(title="Blind Navigation Assistant") as demo:
    gr.Markdown("# 🦯 Blind Navigation Assistant\nUpload a photo of the path ahead.")
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Image", sources=["upload", "webcam"])
            btn = gr.Button("Analyze", variant="primary")
        with gr.Column():
            out_objs = gr.Textbox(label="Detected Objects (with distance)", lines=6)
            out_safety = gr.Textbox(label="Safety Status", lines=3)
            out_guid = gr.Textbox(label="Navigation Guidance", lines=4)
            out_audio = gr.Audio(label="Voice Guidance", autoplay=True)
    btn.click(run_pipeline, image_input, [out_objs, out_safety, out_guid, out_audio])

if __name__ == "__main__":
    demo.queue().launch()
