# Virtual Dress AI — Step-by-step Guide & Flowchart (MVP → API)

**Audience:** Beginner (I will explain like teaching a child).
**Goal:** Build an MVP that: user uploads a selfie + text (occasion/prefs) → AI analyzes skin tone, face shape, gender → recommends outfits → provides a simple virtual try-on (overlay) → exposes three APIs: `/analyze_face`, `/recommend_dress`, `/try_on`.

---

## Quick overview (one-line)

User uploads selfie + types what they want → Backend analyzes the face (skin tone, shape, gender) → Recommendation module filters the product catalog by occasion, gender, skin tone, and user text → Try-on module overlays the chosen dress on the user photo → Results returned to frontend.

---

## Flowchart (text version you can follow step-by-step)

```
[User] --upload selfie + text--> [API /analyze_face] --> [Face Analysis]
                                         |                      |
                                         |                      v
                                         |                 {skin_tone, face_shape, gender}
                                         v
                                  [API /recommend_dress] ---> [Recommendation Module]
                                         |                      |
                                         |                      v
                                         |                [Filtered product list]
                                         v
                                  [API /try_on] ------------> [Try-On Module]
                                                                    |
                                                                    v
                                                             [Try-on image returned]
```

---

# Step-by-step guide (very small steps, do exactly in order)

> **Before you start:** Use a desktop/laptop. We'll use **Python 3.10**, **VS Code**, and a virtual environment. Keep an example selfie image handy (clear face, plain background).

---

## Step 0 — Install software (Do this first)

1. Install **Python 3.10** from python.org. During install check **Add to PATH**.
2. Install **VS Code**. Open it after install.
3. In VS Code extensions install the **Python** extension (Microsoft).

---

## Step 1 — Create project & virtual env

Open a terminal (in VS Code: `Terminal → New Terminal`) and run:

```bash
mkdir virtualdress-ai
cd virtualdress-ai
python -m venv venv
# activate the venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

Your prompt should show `(venv)`.

---

## Step 2 — Install Python libraries (baseline)

Run this while venv is active:

```bash
pip install opencv-python mediapipe deepface fastapi uvicorn pillow numpy python-multipart
```

**Notes:**

* `deepface` will pull TensorFlow; if you don't want TF now, skip `deepface` and we'll use a fallback (but gender detection will be less accurate).
* Later, when you add advanced try-on (VITON-HD or Stable Diffusion) you'll install more packages (PyTorch with correct CUDA, etc.).

---

## Step 3 — Create project file structure (exact)

In your `virtualdress-ai` folder create this tree (in VS Code create new folders/files):

```
virtualdress-ai/
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI app
│   ├── face_analysis.py   # skin tone, face shape, gender
│   ├── recommend.py       # recommendation logic
│   ├── tryon.py           # simple try-on overlay (MVP)
│   └── data/
│       └── catalog.json   # small sample product catalog
└── venv/
```

Create empty `__init__.py` in `app` (makes it a package).

---

## Step 4 — `face_analysis.py` (copy this file exactly)

Create `app/face_analysis.py` and paste below. This file:

* loads the image
* uses Mediapipe FaceMesh to get face landmarks
* computes average skin RGB and converts to `light|medium|dark`
* computes a simple face-shape heuristic from bbox ratio
* uses DeepFace (if installed) to detect gender (fallback if not installed)

```python
# app/face_analysis.py
import cv2
import mediapipe as mp
import numpy as np

# DeepFace usage is optional; if not installed, we fallback to 'unknown'
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except Exception:
    DEEPFACE_AVAILABLE = False

mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)


def _rgb_to_luminance(rgb):
    # rgb: (B,G,R) from OpenCV
    b, g, r = rgb
    # convert to standard luminance Y (0..255)
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return y


def categorize_skin_tone(rgb):
    # rgb from cv2.mean -> (B,G,R)
    y = _rgb_to_luminance(rgb)
    # thresholds are approximate; tune later
    if y >= 180:
        return "light"
    elif y >= 100:
        return "medium"
    else:
        return "dark"


def classify_face_shape(landmarks, img_w, img_h):
    # landmarks: list of (x,y) in pixel coordinates
    xs = [p[0] for p in landmarks]
    ys = [p[1] for p in landmarks]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    w = maxx - minx
    h = maxy - miny
    if w == 0:
        return "unknown"
    ratio = h / w
    # Heuristic mapping (approximate)
    if ratio < 0.9:
        return "round"
    elif 0.9 <= ratio <= 1.15:
        return "square"
    elif 1.15 < ratio <= 1.4:
        return "oval"
    else:
        return "heart"


def analyze_face(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return {"error": "Could not read image"}

    h, w, _ = img.shape
    results = mp_face.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return {"error": "No face detected"}

    # Collect first face landmarks
    lm = results.multi_face_landmarks[0]
    pts = []
    for p in lm.landmark:
        pts.append((int(p.x * w), int(p.y * h)))

    # Skin tone: sample small mask in central face area
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    # sample first 100 landmarks to get central region
    for x, y in pts[:100]:
        cv2.circle(mask, (x, y), 2, 255, -1)
    mean_bgr = cv2.mean(img, mask=mask)[:3]
    mean_bgr = tuple(map(int, mean_bgr))
    skin_category = categorize_skin_tone(mean_bgr)

    # face shape
    face_shape = classify_face_shape(pts, w, h)

    # gender detection using DeepFace (best-effort)
    gender = "unknown"
    try:
        if DEEPFACE_AVAILABLE:
            df = DeepFace.analyze(img_path=image_path, actions=["gender"], enforce_detection=False)
            # DeepFace returns {'gender': 'Woman'} etc.
            gender_str = df.get('gender')
            if isinstance(gender_str, str):
                gender = gender_str.lower()
            else:
                gender = str(gender_str)
    except Exception:
        # ignore deepface errors; keep 'unknown'
        gender = "unknown"

    return {
        "skin_rgb": {"b": mean_bgr[0], "g": mean_bgr[1], "r": mean_bgr[2]},
        "skin_tone": skin_category,
        "face_shape": face_shape,
        "gender": gender
    }


if __name__ == '__main__':
    # quick local test
    print(analyze_face('sample_selfie.jpg'))
```

**How this works (in simple words):**

* We read the picture and use Mediapipe to find many points on the face.
* From those points we take colors in central face area and compute a brightness value (luminance). Based on brightness we say `light|medium|dark`.
* We also compute simple width/height ratio of the face area to guess shape (round/square/oval/heart).
* For gender we try DeepFace (optional). If DeepFace isn't available we return `unknown`.

**Common errors right here:**

* `No face detected` → use a clearer photo (face centered, not tilted); increase image size.
* `DeepFace` errors on install → install TensorFlow or skip DeepFace now.

---

## Step 5 — `recommend.py` (copy this file)

This module maps `occasion`, `gender`, `skin_tone`, and user text to candidate products. For MVP we'll use a small `catalog.json` to filter from.

Create `app/data/catalog.json` with sample products (example below). Then create `app/recommend.py`:

```json
// app/data/catalog.json  (example - use real JSON without comments)
[
  {"id":"p1","title":"Red Party Dress","gender":"female","occasion":["party"],"colors":["red","maroon"]},
  {"id":"p2","title":"Blue Casual Shirt","gender":"male","occasion":["casual"],"colors":["blue"]},
  {"id":"p3","title":"Simple Kurta","gender":"female","occasion":["temple","family"],"colors":["white","beige"]}
]
```

```python
# app/recommend.py
import json
import os

CATALOG_PATH = os.path.join(os.path.dirname(__file__), 'data', 'catalog.json')

# small list of color names to search in user text
COLOR_KEYWORDS = ['red','blue','black','white','green','yellow','pink','beige','maroon','brown']


def load_catalog():
    with open(CATALOG_PATH, 'r') as f:
        return json.load(f)


def extract_color_from_text(text):
    text = (text or '').lower()
    for c in COLOR_KEYWORDS:
        if c in text:
            return c
    return None


def recommend(occasion, gender, skin_tone, user_text, top_k=4):
    occasion = (occasion or '').lower()
    gender = (gender or '').lower()
    catalog = load_catalog()

    pref_color = extract_color_from_text(user_text)

    # filter by gender if known
    candidates = []
    for p in catalog:
        if gender and p.get('gender') and p.get('gender') != gender:
            continue
        if occasion and occasion not in [o.lower() for o in p.get('occasion', [])]:
            continue
        # color match bonus
        color_score = 1
        if pref_color and pref_color in [c.lower() for c in p.get('colors', [])]:
            color_score += 2
        if skin_tone == 'dark' and 'white' in [c.lower() for c in p.get('colors', [])]:
            color_score += 0.5  # example small boost for contrast
        candidates.append((p, color_score))

    # sort by score and return top_k
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [c[0] for c in candidates[:top_k]]


if __name__ == '__main__':
    print(recommend('party','female','medium','I want a red dress'))
```

**Explanation for beginners:**

* We read a small JSON catalog and filter by occasion and gender.
* If user text contains a color, we prefer products in that color.
* This is rule-based and easy to extend later.

---

## Step 6 — `tryon.py` (MVP simple overlay)

**Goal:** Give a working demo try-on **without** heavy deep models. This will not be perfect but will show the feature to the client immediately.

Create `app/tryon.py` and paste:

```python
# app/tryon.py
import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose.Pose(static_image_mode=True)


def overlay_transparent(background_rgb, overlay_png, x, y, overlay_size=None):
    # overlay_png read with alpha channel (BGRA)
    bg = background_rgb
    ol = overlay_png
    if overlay_size:
        ol = cv2.resize(ol, overlay_size, interpolation=cv2.INTER_AREA)
    h, w = ol.shape[:2]
    if x + w > bg.shape[1] or y + h > bg.shape[0]:
        # clip
        w = min(w, bg.shape[1] - x)
        h = min(h, bg.shape[0] - y)
        ol = ol[0:h, 0:w]
    # Separate channels
    if ol.shape[2] < 4:
        # not RGBA - just place
        bg[y:y+h, x:x+w] = ol
        return bg
    alpha = ol[:, :, 3] / 255.0
    for c in range(0, 3):
        bg[y:y+h, x:x+w, c] = (alpha * ol[:, :, c] + (1 - alpha) * bg[y:y+h, x:x+w, c])
    return bg


def simple_tryon(person_image_path, dress_image_path, out_path='out_tryon.png'):
    person = cv2.imread(person_image_path)
    if person is None:
        return {"error": "bad person image"}
    dress = cv2.imread(dress_image_path, cv2.IMREAD_UNCHANGED)  # need alpha
    if dress is None:
        return {"error": "bad dress image"}

    h, w, _ = person.shape
    results = mp_pose.process(cv2.cvtColor(person, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks:
        return {"error": "No pose landmarks found"}

    lm = results.pose_landmarks.landmark
    # get shoulders and hips
    left_sh = lm[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
    right_sh = lm[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_hip = lm[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]
    right_hip = lm[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]

    # convert normalized to pixel
    lx, ly = int(left_sh.x * w), int(left_sh.y * h)
    rx, ry = int(right_sh.x * w), int(right_sh.y * h)
    lhx, lhy = int(left_hip.x * w), int(left_hip.y * h)
    rhx, rhy = int(right_hip.x * w), int(right_hip.y * h)

    # compute bounding box for torso
    box_x = min(lx, rx) - 20
    box_w = abs(rx - lx) + 40
    box_y = min(ly, ry) - 10
    box_h = abs(((lhy + rhy) // 2) - box_y) + 30

    # bounds safe
    box_x = max(0, box_x)
    box_y = max(0, box_y)
    box_w = min(box_w, w - box_x)
    box_h = min(box_h, h - box_y)

    # resize dress to box
    try:
        person_out = overlay_transparent(person, dress, box_x, box_y, overlay_size=(box_w, box_h))
        cv2.imwrite(out_path, person_out)
        return {"out_path": out_path}
    except Exception as e:
        return {"error": str(e)}


if __name__ == '__main__':
    print(simple_tryon('sample_selfie.jpg', 'dress_example.png'))
```

**Notes for MVP try-on:**

* Use a dress PNG with transparent background (alpha channel).
* This method only resizes and places the dress at the torso box — no warping or cloth physics.
* It proves the concept and lets the client see a try-on quickly.

---

## Step 7 — `main.py` (FastAPI endpoints)

Create `app/main.py` and paste below. This exposes three endpoints required by your spec:

```python
# app/main.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import shutil
import base64

from app.face_analysis import analyze_face
from app.recommend import recommend
from app.tryon import simple_tryon

app = FastAPI()

@app.post('/analyze_face')
async def analyze_face_endpoint(file: UploadFile = File(...)):
    tmp = 'temp_selfie.jpg'
    with open(tmp, 'wb') as f:
        shutil.copyfileobj(file.file, f)
    result = analyze_face(tmp)
    return JSONResponse(result)

@app.post('/recommend_dress')
async def recommend_endpoint(
    file: UploadFile = File(...),
    occasion: str = Form(...),
    user_text: str = Form('')
):
    tmp = 'temp_reco.jpg'
    with open(tmp, 'wb') as f:
        shutil.copyfileobj(file.file, f)
    analysis = analyze_face(tmp)
    gender = analysis.get('gender', 'unknown')
    skin = analysis.get('skin_tone', 'medium')
    suggestions = recommend(occasion, gender, skin, user_text)
    return {'analysis': analysis, 'suggestions': suggestions}

@app.post('/try_on')
async def tryon_endpoint(
    file: UploadFile = File(...),
    dress_image_path: str = Form(...)
):
    # dress_image_path is a path to a PNG dress on server (for demo)
    tmp = 'temp_tryon.jpg'
    with open(tmp, 'wb') as f:
        shutil.copyfileobj(file.file, f)
    out = simple_tryon(tmp, dress_image_path, out_path='temp_out_tryon.png')
    if out.get('out_path'):
        with open(out['out_path'], 'rb') as f:
            b = f.read()
            encoded = base64.b64encode(b).decode('utf-8')
        return {'status': 'ok', 'image_base64': 'data:image/png;base64,' + encoded}
    return out

```

**How to run the server:**

Activate venv then run:

```bash
uvicorn app.main:app --reload
```

Open API docs in browser: `http://127.0.0.1:8000/docs` — this UI will let you upload files and call endpoints.

---

## Step 8 — `catalog.json` example (small)

Create `app/data/catalog.json` as a simple list (no comments) like:

```json
[
  {"id":"p1","title":"Red Party Dress","gender":"female","occasion":["party"],"colors":["red","maroon"]},
  {"id":"p2","title":"Blue Casual Shirt","gender":"male","occasion":["casual"],"colors":["blue"]},
  {"id":"p3","title":"Simple Kurta","gender":"female","occasion":["temple","family"],"colors":["white","beige"]}
]
```

This allows you to test recommendations.

---

## Step 9 — Quick demo test plan (do these tests in order)

1. **Start server**: `uvicorn app.main:app --reload`

2. **Test analyze_face** (use curl or the /docs UI):

```bash
curl -X POST "http://127.0.0.1:8000/analyze_face" -F "file=@sample_selfie.jpg"
```

**Expected JSON** (example):

```json
{
  "skin_rgb": {"b": 143, "g": 99, "r": 92},
  "skin_tone": "medium",
  "face_shape": "oval",
  "gender": "woman"
}
```

3. **Test recommend_dress** (send selfie, occasion, optional text):

```bash
curl -X POST "http://127.0.0.1:8000/recommend_dress" -F "file=@sample_selfie.jpg" -F "occasion=party" -F "user_text=I want red dress"
```

**Expected:** JSON with `analysis` and `suggestions` (list of products from catalog).

4. **Test try_on** (requires you have a local dress PNG path; for demo make `dress_example.png`):

```bash
curl -X POST "http://127.0.0.1:8000/try_on" -F "file=@sample_selfie.jpg" -F "dress_image_path=app/data/dress_example.png"
```

**Expected:** JSON with `image_base64` string. Paste that string into a browser address bar to view image.

---

## Step 10 — Troubleshooting (common problems & fixes)

* **No face detected**: Try clearer photo, plain background, no sunglasses, face straight to camera.
* **Mediapipe import error**: Reinstall with `pip install mediapipe` inside venv. Check Python version (3.10 recommended).
* **DeepFace errors on install**: It needs TensorFlow; either `pip install tensorflow` or skip DeepFace and use `gender='unknown'` for now.
* **Try-on overlay looks misaligned**: Use a dress PNG specifically designed for the demo (try small shifts in box calculations in `tryon.py`).
* **Large uploads/timeouts**: Keep incoming images <= 2–3 MB. Add checks in FastAPI to reject large files.
* **Base64 too long for response**: For production, save generated image to server or S3 and return URL instead.

---

## Step 11 — What comes next (recommended improvements)

1. Replace MVP try-on with **VITON-HD or Diffusion-based try-on** for realism.
2. Fine-tune a small classifier for face-shape detection if you need high accuracy.
3. Build a small UI (React) that calls these APIs and shows the results nicely.
4. Add a product-matching ML model (learn preferences from clicks/purchases).
5. Move images to S3 and serve via CDN; deploy models in Docker on GPU machines.

---

## Final checklist for you (tick as you complete)

* [ ] Python 3.10 installed
* [ ] VS Code installed + Python extension
* [ ] venv created & activated
* [ ] Libraries installed
* [ ] `app/face_analysis.py` created & working
* [ ] `app/recommend.py` created & working
* [ ] `app/tryon.py` created & working (MVP overlay)
* [ ] `app/main.py` created & server runs
* [ ] Test images and sample catalog present
* [ ] Run demo tests listed above

---

If you want, I can now:

* Walk you **live** through Step 0→Step 4 (tell me when you have VS Code open and I will give exact commands one-by-one).
* Or, I can produce the same guide as a downloadable single `README.md` file.

Tell me which option: **(A)** I walk you step-by-step now (I will give the exact terminal commands to run), or **(B)** produce README file for you to download.
