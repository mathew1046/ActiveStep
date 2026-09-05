# ActiveStep on Google Colab

This subfolder contains everything needed to continue the model-assessment
and improvement work on Google Colab (free GPU/TPU runtime).

## What's here

| File | Purpose |
|---|---|
| `requirements_colab.txt` | Trimmed pip requirements (Colab already has TF/numpy/sklearn) |
| `ActiveStep_Colab.ipynb` | Ready-to-run Colab notebook: clone → install → audit → cache → smoke → full assessment → report |

## Steps to continue in Colab

### 1. Upload the notebook

1. Open [colab.research.google.com](https://colab.research.google.com).
2. **File → Upload notebook** and select `colab/ActiveStep_Colab.ipynb`
   from this repository (or drag-and-drop the `.ipynb` file into Colab).
3. **Runtime → Change runtime type → T4 GPU** (recommended for CNN training;
   CPU works but is slower).

### 2. Run the setup cells

The first two cells handle everything:

- **Cell 1** clones your GitHub repo into `/content/ActiveStep`.
  If the repo is private, the notebook includes a commented-out alternative
  that mounts Google Drive instead — put a `.zip` or clone there first.
- **Cell 2** installs `tensorflow-model-optimization` (the only dep Colab
  doesn't already have).

### 3. Run the assessment pipeline

The remaining cells mirror `scripts/run_assessment.sh` but run directly in
Python (no conda needed):

| Cell | What it does | Expected time (T4) |
|---|---|---|
| Audit | Dataset inventory → `models/dataset_audit.json` | < 30 s |
| Cache | Build streaming-safe window cache (`.cache/windows`, ~250 MB) | 1–2 min |
| Smoke | 1-epoch mechanics check on subjects 1–3 | ~1 min |
| Full | Complete nested LOPO across all 10 subjects, 3 candidates | 15–40 min |
| Report | Render `assessment_report.md` from `summary.json` | < 5 s |
| Results | Display key metrics tables inline | instant |

### 4. Iterate on model improvements

After the baseline reproduces, create new cells below the baseline to
experiment. The suggested next experiments (from the assessment report)
are:

1. **FI activity gate** — add an absolute locomotor-band power floor to
   `src/features.py` so the FI ratio doesn't explode during standing.
2. **Endpoint-focused labels** — change the window target in `src/data.py`
   from "any freeze in window" to "freeze at window end" to reduce delay.
3. **Class-weight tuning** — the corrected data makes the current weight
   ~12×; try 3–5× and observe cross-subject score distributions.
4. **Temporal post-processing** — add score smoothing, consecutive-positive
   trigger, refractory period, and recovery hysteresis in `src/detector.py`,
   tuned inside the inner folds only.

Each experiment should use a **new `--tag`** (e.g. `fi_gate_v1`) so results
don't overwrite `baseline_v1`.

### 5. Save results back

Colab notebooks auto-save to your Drive. To persist model artifacts:

```python
# Zip the assessment output and download it
!cd /content/ActiveStep && zip -r /content/baseline_v1.zip models/nested/baseline_v1/
from google.colab import files
files.download('/content/baseline_v1.zip')
```

Or mount Drive and copy directly:

```python
from google.colab import drive
drive.mount('/content/drive')
!cp -r /content/ActiveStep/models/nested/baseline_v1 /content/drive/MyDrive/ActiveStep/
```

## Notes

- **Colab session timeouts**: Free Colab sessions disconnect after ~12 h of
  inactivity and ~90 min of idle. The full assessment fits well within one
  session. If it disconnects mid-run, re-run with `--force` removed —
  completed folds are cached and skipped.
- **Disk**: The repo is ~100 MB (dataset 87 MB + models 12 MB). The window
  cache adds ~250 MB. Colab's `/content` has ~100 GB free, so this is fine.
- **TensorFlow version**: Colab ships TF 2.15+. The code is tested against
  2.18.1. If you hit API issues, pin with `!pip install tensorflow==2.18.1`.
- **No hardware deps needed**: The Colab notebook only touches the model
  pipeline (`src/`), not the firmware/runtime/hardware code. Dependencies
  like `gpiozero`, `smbus2`, `sounddevice` are not installed and not needed.
