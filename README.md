# Blind Navigation Assistant

Upload a photo -> detected obstacles with **real distances in meters**,
STOP / CAUTION / CLEAR status, spoken guidance.

## What changed (and why)
| Problem | Fix |
|---|---|
| YOLO trained from random weights (`.yaml`) | Obstacle detection uses COCO-pretrained `yolo11n/s/m.pt` (no training). Scene classifier now starts from `yolo11s-cls.pt`, 50 epochs, early stopping, augmentation |
| Scene classifier can't detect cars/people | Real object detector with left / ahead / right position |
| Depth direction was inverted, global mean | Metric Depth-Anything-V2 -> meters, per-object and per-walking-zone |
| LLM decided safety | Safety is rule-based; VLM only adds one description sentence |
| Data leakage (val copy hack, random frame split) | Stratified / group split, rare classes dropped, no copying across splits |
| Heavy models | Hardware profiles, 4-bit Qwen option, offline TTS option |

## Run the app
```bash
pip install -r requirements.txt
python app.py                     # auto profile
PROFILE=cpu python app.py         # laptop, no GPU (no VLM)
PROFILE=gpu_small python app.py   # 4-6 GB GPU (needs bitsandbytes)
PROFILE=gpu python app.py         # T4 or better
TTS=offline python app.py         # pyttsx3 instead of gTTS
```
Override models: `DET_MODEL`, `DEPTH_MODEL`, `VLM_MODEL` (`none` disables the VLM).
`HF_TOKEN` is read from the environment / Space secret (never hard-coded).

## Train the optional scene classifier (Colab GPU)
```bash
python train_model.py
```
Weights are saved to `models/scene_best.pt`; `app.py` uses them automatically
if present. If the dataset has a video/scene id column, run
`GROUP_COL=that_column python train_model.py` to avoid leakage.
Check the printed class counts: many classes with few images = low accuracy
for any model; raise `MIN_IMAGES` or merge similar labels.

## Tune on your own photos
Edit in `app.py`: `VEHICLE_ALERT_M`, `OBSTACLE_ALERT_M`, `PATH_BLOCKED_M`.
Test with phone-camera photos held at chest height, not only dataset images.

## Hugging Face Spaces
Gradio SDK, GPU hardware, add `HF_TOKEN` secret, set variable `PROFILE=gpu`,
upload `app.py`, `requirements.txt`, and optionally `models/scene_best.pt`.

## Limits
Not a certified safety device. Use as an assistive aid only, never as the sole
means of navigation.

## Windows laptop setup (i7-6820HQ, 8 GB RAM, 2 GB Quadro M1000M)
The 2 GB GPU is too small, so the app runs the `cpu` profile (auto-selected).
Python 3.10-3.12, then in PowerShell:
```powershell
python -m venv venv; .\venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-laptop.txt
$env:PROFILE="cpu"; python app.py
```
- Speech uses Windows' built-in voices (offline). `$env:TTS="gtts"` switches to Google TTS.
- First run downloads ~150 MB of models. Expect roughly 2-5 s per image.
- Indoors, use the indoor depth model:
  `$env:DEPTH_MODEL="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"`
- Do NOT train on this laptop. Run `train_model.py` on Google Colab (free GPU),
  then download `models/scene_best.pt` into this project's `models` folder.
- Disk: you have ~17 GB free, enough for this setup but not for large VLMs.
