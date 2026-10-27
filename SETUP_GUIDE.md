# 🖐️ Gesture Genix — Setup Guide for Team Members
### (Windows Laptop — Full Setup from Scratch)

---

## ✅ Summary Checklist

```
□ Step 1  — Copy gesture-genix folder to Desktop
□ Step 2  — Install Python 3.11 (check "Add to PATH"!)
□ Step 3  — Open cmd inside gesture-genix folder
□ Step 4  — python -m venv venv
□ Step 5  — venv\Scripts\activate
□ Step 6  — pip install -r requirements.txt
□ Step 7  — cd src
□ Step 8  — python capture.py        (test camera)
□ Step 9  — python data_collector.py (collect data)
□ Step 10 — python train_classifier.py (train model)
□ Step 11 — python inference.py      (run it!) 🚀
```

---

## 🔷 STEP 1 — Copy the Project to This Laptop

1. Get the `gesture-genix.zip` file from your team leader (USB / Google Drive / WhatsApp)
2. Right-click the `.zip` file → **Extract All**
3. Move the extracted `gesture-genix` folder to your **Desktop**
4. Final path should look like:
   ```
   C:\Users\YourName\Desktop\gesture-genix\
   ```

The folder should contain:
```
gesture-genix/
├── requirements.txt
├── README.md
├── src/
│   ├── config.py
│   ├── landmark_utils.py
│   ├── capture.py
│   ├── data_collector.py
│   ├── train_classifier.py
│   ├── nlp_layer.py
│   └── inference.py
└── data/
    └── models/
        ├── gesture_classifier.pkl
        ├── label_map.json
        └── hand_landmarker.task
```

---

## 🔷 STEP 2 — Install Python on This Laptop

1. Open your browser → go to: **https://www.python.org/downloads/**
2. Click **"Download Python 3.11.x"**
3. Open the downloaded `.exe` file
4. ⚠️ **VERY IMPORTANT** — Check this box BEFORE clicking install:
   ```
   ✅  Add python.exe to PATH
   ```
   > If you miss this, nothing will work later!
5. Click **"Install Now"** → wait → click **Close**

**Verify it worked** — open **Command Prompt** (search `cmd` in Start Menu):
```
python --version
```
You should see:
```
Python 3.11.x
```
✅ Python is ready!

---

## 🔷 STEP 3 — Open Command Prompt in the Project Folder

1. Open **File Explorer** → go to Desktop → open the `gesture-genix` folder
2. Click on the **address bar** at the top
3. Type `cmd` and press **Enter**

A black Command Prompt opens already inside the folder.

Confirm by typing:
```
dir
```
You should see `requirements.txt` and `README.md` listed. ✅

---

## 🔷 STEP 4 — Create a Virtual Environment

```
python -m venv venv
```

Wait 5–10 seconds (no output = success).

Verify:
```
dir venv
```
You should see a `Scripts` folder. ✅

---

## 🔷 STEP 5 — Activate the Virtual Environment

```
venv\Scripts\activate
```

Your prompt changes to show `(venv)` at the start:
```
(venv) C:\Users\YourName\Desktop\gesture-genix>
```
✅ You're now inside the virtual environment.

> ⚠️ Every time you close and reopen Command Prompt, you must run this activate command again before doing anything.

---

## 🔷 STEP 6 — Install All Libraries

```
pip install -r requirements.txt
```

This downloads and installs 7 libraries. Takes **3–10 minutes** depending on internet speed.
You'll see lots of download progress bars — that's completely normal.

When done you'll see:
```
Successfully installed mediapipe-x.x opencv-python-x.x numpy-x.x ...
```

**Verify everything installed correctly:**
```
python -c "import mediapipe, cv2, numpy, pandas, sklearn, joblib, pyttsx3; print('All OK!')"
```
Expected output:
```
All OK!
```
✅

---

## 🔷 STEP 7 — Go Into the `src` Folder

⚠️ **All scripts must be run from inside `src\`. This is the most common mistake!**

```
cd src
```

Your prompt now shows:
```
(venv) C:\Users\YourName\Desktop\gesture-genix\src>
```
✅

---

## 🔷 STEP 8 — Test Camera + Hand Tracking (Phase 1)

```
python capture.py
```

A window opens showing your webcam. Hold your hand up in front of the camera.

| What you see | Meaning |
|---|---|
| Green dots + lines on hand | ✅ Working perfectly |
| "No hand detected" (grey text) | Move hand closer / improve lighting |
| Window doesn't open at all | Camera issue — see troubleshooting below |

> 📌 **Note:** On the very first run, it may auto-download the hand tracking model (~30 MB).
> Keep internet on for first run. After that it works fully offline.

Press **`q`** to close the window. ✅

---

## 🔷 STEP 9 — Collect Training Data (Phase 2)

> ⏭️ **Skip this step if** you received `gesture_classifier.pkl` and `label_map.json` files
> from your team leader and they're already inside `data\models\`. Jump to Step 11.

```
python data_collector.py
```

A window opens with a HUD. Collect data for all 10 gestures:

| Key | Gesture |
|---|---|
| `0` | hello |
| `1` | yes |
| `2` | no |
| `3` | please |
| `4` | thank_you |
| `5` | i_love_you |
| `6` | peace |
| `7` | thumbs_up |
| `8` | thumbs_down |
| `9` | stop |

**For EACH gesture, do this:**
1. Press the number key (e.g. `1` for yes)
2. Make that hand gesture in front of the camera
3. Press **`r`** → red **"● REC"** appears on screen
4. Hold the gesture — watch the green bar fill up — collect **150–200 samples**
5. Press **`r`** again to stop recording
6. Repeat for the next gesture

**Other controls:**
| Key | Action |
|---|---|
| `s` | Show sample count summary |
| `q` | Quit and save |

Press `q` when all 10 gestures are done. ✅

---

## 🔷 STEP 10 — Train the Gesture Classifier (Phase 3)

> ⏭️ **Skip this step if** you received `gesture_classifier.pkl` and `label_map.json`
> from your team leader. Jump to Step 11.

```
python train_classifier.py
```

Trains in 30–60 seconds. You'll see:
```
[INFO] Loaded xxxx samples ...
[INFO] Training ...
[INFO] Training complete.

Test accuracy : 97.xx%

✓ Done. Run  inference.py  to test live recognition.
```

✅ Aim for **90%+ accuracy**. If below 85%, go back to Step 9 and collect more data, then retrain.

---

## 🔷 STEP 11 — Run Live Gesture Recognition 🚀 (Phase 4)

```
python inference.py
```

The system is now fully live!

**What you'll see on screen:**
- **Line 1:** Current gesture being recognized + confidence % → e.g. `hello (94%)`
- **Line 2:** Sentence being built → e.g. `Buffer: hello yes please`
- **Line 3:** Last spoken sentence → e.g. `Spoken: hello yes please`

**How it works:**
1. Make a gesture → it's recognized and added to the sentence buffer
2. Lower your hand and wait **1.2 seconds** → sentence is **spoken aloud automatically**
3. Buffer clears and is ready for the next sentence

**Keyboard controls:**
| Key | Action |
|---|---|
| `q` | Quit |
| `c` | Clear the buffer silently |
| `s` | Speak the current sentence right now |

---

## ❌ Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'config'` | You forgot `cd src` — always run scripts from inside `src\` |
| `python is not recognized` | Reinstall Python and check ✅ "Add to PATH" during setup |
| Camera not opening / black screen | Close Zoom, Teams, or any app using the camera. Try again. |
| No green hand skeleton | Improve lighting. Keep hand 30–70 cm from camera. |
| Accuracy is very low (below 80%) | Collect 200+ samples per gesture in Step 9, then retrain in Step 10 |
| No audio / no speech | Check Windows volume. Check speaker output device. |
| PowerShell venv error | Run: `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` |

---

## 🔁 Every Time You Come Back to Work

```bash
# 1. Open cmd inside gesture-genix folder
# 2. Activate virtual environment:
venv\Scripts\activate

# 3. Go into src:
cd src

# 4. Run whichever phase you need:
python capture.py            ← Phase 1: test camera
python data_collector.py     ← Phase 2: collect data
python train_classifier.py   ← Phase 3: train model
python inference.py          ← Phase 4: run live 🚀
```

---

*Gesture Genix | MCA Capstone Project | Python 3.11 | MediaPipe + scikit-learn + pyttsx3*
