# Indian Sign Language Detector

This project detects Indian Sign Language hand gestures in real time using your webcam. You show a hand sign to the camera and it tells you which letter (A–Z) you're signing. It's built on top of two AI models working together — YOLO11 handles spotting the hand in the frame, and a Bidirectional LSTM takes over to classify the exact gesture based on hand landmark positions.

The whole thing runs as a desktop app with a dark-themed GUI window. Click one button, your webcam opens, and the detected letter shows up on screen instantly.

---

## What's inside

```
SignLanguageDetector/
├── SignDetection.ipynb       ← main notebook, run this to launch the app
├── Yolo_Training.ipynb       ← only needed if you want to retrain YOLO (not required)
├── Data/
│   ├── dataset.yaml          ← class config for YOLO training
│   └── images/train/         ← training images (A–Z hand signs)
├── model/
│   ├── best.pt               ← trained YOLO11 weights (ready to use)
│   ├── lstm_weights.h5       ← trained BiLSTM weights (generated on first run)
│   ├── X_train.npy           ← extracted hand landmark features
│   └── y_train.npy           ← labels for the above
└── README.md
```

The `model/best.pt` file is already trained and included. You don't need a GPU or Google Colab to run this — it works fully on CPU.

---

## How detection works

When you click Start, two things happen in sequence on every frame:

1. **YOLO11** scans the frame and draws a bounding box around the detected hand sign with a confidence score above 50%.
2. **MediaPipe** finds the exact positions of all 21 hand joints and passes those 42 normalized coordinates into the **BiLSTM** model which outputs the final letter prediction.

The result gets overlaid on the OpenCV window as `Gesture = <letter>`.

---

## Getting it running

There are a few version-specific things to get right here. Python 3.10 is required — TensorFlow 2.13 doesn't work on Python 3.11 or 3.14. The steps below are exactly what worked.

### 1. Install Python 3.10

Download from python.org or use the bundled installer if it's in the folder. During install, check **"Add Python to PATH"**.

Verify it's available:

```bash
py -3.10 --version
```

### 2. Create a virtual environment

Open a terminal inside the project folder and run:

```bash
py -3.10 -m venv venv
venv\Scripts\activate
```

Your terminal prompt should show `(venv)` — this means you're isolated from your system Python.

### 3. Install dependencies

These exact versions are what work together without conflicts:

```bash
pip install tensorflow==2.13.0 mediapipe==0.10.9 ultralytics opencv-python scikit-learn seaborn matplotlib notebook ipykernel
```

> `mediapipe==0.10.9` is pinned because newer versions removed the `solutions` API that this project uses.
> `tensorflow==2.13.0` is pinned because it ships with Keras 2.x which matches the saved model format.

### 4. Register the kernel with Jupyter

This step is easy to miss. Without it, the notebook opens but shows "No kernel found":

```bash
python -m ipykernel install --user --name=signlang --display-name "Python 3.10 (SignLang)"
```

### 5. Launch Jupyter

```bash
jupyter notebook
```

Open **SignDetection.ipynb** in the browser that opens.

Go to **Kernel > Change Kernel** and select **Python 3.10 (SignLang)**.

Then run **Cell > Run All**.

### 6. Use the app

The last cell opens a dark-themed GUI window. Click **START DETECTION**, then hold a hand sign in front of your webcam. The detected letter will appear on screen. Press `Q` inside the camera window to stop.

---

## First run note

The first time you run Cell 10, the LSTM model trains from scratch on your machine. This takes around 30–60 seconds since the dataset is small (66 samples). After that, it saves to `model/lstm_weights.h5` and loads instantly on every run after.

---

## If you want to retrain YOLO

You don't need to, since `model/best.pt` is already included. But if you want to train on new data, open `Yolo_Training.ipynb` in **Google Colab** (it requires a GPU — Colab's free T4 works fine):

```python
from ultralytics import YOLO
model = YOLO("yolo11n.pt")
model.train(data="Data/dataset.yaml", epochs=40, imgsz=640)
```

After training, copy `runs/detect/train/weights/best.pt` into your local `model/` folder.

---

## Tech stack

| What | Why |
|------|-----|
| YOLO11n | Fast hand detection and localization in each frame |
| MediaPipe Hands | Extracts 21 hand joint positions reliably |
| Bidirectional LSTM | Classifies the gesture from normalized landmark data |
| TensorFlow 2.13 | Runs the LSTM model locally on CPU |
| OpenCV | Webcam capture and drawing overlays |
| Tkinter | Desktop GUI launcher |

---

## Common issues

**"No kernel" in Jupyter** — run step 4 above (register ipykernel), then Kernel > Change Kernel inside the notebook.

**`mediapipe has no attribute solutions`** — wrong mediapipe version installed. Run `pip install mediapipe==0.10.9`.

**`tensorflow` install fails** — you're not in the Python 3.10 venv. Run `python --version` and make sure it says 3.10.x before installing.

**Model loading error on cell 10** — this happens if the old `.keras` file from Colab is still present. The notebook handles this automatically now and retrains instead.
