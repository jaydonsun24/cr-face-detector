# Live Facial Emotion → Clash Royale Emote Classifier

Point a webcam at your face, and the matching Clash Royale emote appears on screen.

**The chain:** `webcam frame → find the face → classify the emotion → look up the emote → draw both`

The model itself knows nothing about Clash Royale. It classifies a plain facial
emotion (happy, angry, sad, …); a lookup table in [config.py](config.py) maps each
emotion to an emote. The ML problem is ordinary emotion classification — the
"emote" part is a dictionary at the very end.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

In VS Code, select the interpreter inside `.venv` — Command Palette →
"Python: Select Interpreter".

## Layout

```
├── config.py           # paths, hyperparameters, emotion→emote map
├── data/               # dataset (gitignored)
├── models/             # trained weights .pth (gitignored)
├── assets/emotes/      # Clash Royale emote images
├── notebooks/          # exploration, and the Colab training mirror
└── src/
      ├── dataset.py    # Dataset/DataLoader + transforms
      ├── model.py      # pretrained backbone + swapped head
      ├── train.py      # training loop
      ├── evaluate.py   # confusion matrix, per-class accuracy
      ├── face_detect.py# webcam capture + face detection
      └── inference.py  # the live pipeline
```

## Build order

1. **Label space** — pick the emotions, map each to an emote in `config.py`
2. **Data flowing** — load into a DataLoader, look at a batch before modeling
3. **Face detection alone** — `python src/face_detect.py`, just boxes, no model
4. **Train** — transfer learning: freeze backbone, swap head, train the head
5. **Evaluate** — confusion matrix, not just overall accuracy
6. **Connect the pipeline** — `python src/inference.py`
7. **Close the real-world gap** — your live accuracy will be worse than
   validation. That gap is the lesson.

## Colab ↔ VS Code

They never connect. The handoff is one file:

1. Train in Colab (notebook mirrors `src/`)
2. `torch.save(model.state_dict(), ...)` → produces a `.pth`
3. Download it, drop it in `models/`
4. `src/inference.py` loads it and runs the webcam

A `.pth` stores **weights, not architecture** — so `model.py` must define the
same shape on both sides.

Note: this Mac has Apple Silicon, and `config.get_device()` returns `mps`, so
local training may be fast enough that Colab is optional.

## The two bugs that get everyone

**Preprocessing mismatch** — the resize and normalization at webcam time must
exactly match training, or predictions turn to nonsense. This is why both sides
import their transforms from `dataset.py` instead of redefining them.

**Flickering predictions** — a label that changes every frame feels broken even
at good accuracy. Average over the last N frames (`config.SMOOTHING_WINDOW`).

## Dataset

[FER2013](https://www.kaggle.com/datasets/msambare/fer2013) — 7 classes, small
grayscale images. Expected layout:

```
data/train/angry/*.jpg
data/train/happy/*.jpg
...
```

`ImageFolder` assigns class indices **alphabetically**, which is why
`config.EMOTIONS` is sorted that way. Verify against `dataset.classes` before
trusting any prediction.
