# Radiology AI Workstation — Pipeline Overview

---

## What happens when you upload an X-ray?

```
X-ray image
     │
     ▼
┌─────────────────────────────────────────────────────┐
│  STEP 1 — DenseNet Classifier                       │
│                                                     │
│  Model: TorchXRayVision DenseNet-121                │
│  Input: 224 × 224 grayscale                         │
│  Output: probability score for 18 pathologies       │
│                                                     │
│  Answers: "Is there Pneumonia? Effusion? Opacity?"  │
└─────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────┐
│  STEP 2 — Grad-CAM (Vision Heatmap)                 │
│                                                     │
│  Asks the model: "WHERE did you look to decide?"    │
│  Output: grayscale attention map (224 × 224)        │
│                                                     │
│  Problem: the model sometimes looks outside lungs   │
│  (bones, edges, background) → not clinically useful │
└─────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────┐
│  STEP 3 — Lung Segmentation (Threshold Mask)        │
│                                                     │
│  Lungs are dark on X-ray (filled with air).         │
│  We invert-threshold the image to isolate them.     │
│  Morphological clean-up removes noise.              │
│  Output: binary mask — white = lung, black = other  │
│                                                     │
│  This is image-specific: every patient is different │
└─────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────┐
│  STEP 4 — Medical Heatmap                           │
│                                                     │
│  medical_cam = Grad-CAM × lung_mask                 │
│                                                     │
│  Anything the model "saw" outside the lungs         │
│  gets zeroed out. Only lung-region attention        │
│  survives. This is the key medical safety step.     │
│                                                     │
│  Then:                                              │
│    • Values < 0.45 → set to zero  (noise removal)  │
│    • Gaussian blur → smooth the map                 │
│    • COLORMAP_JET → blue=low, red=high attention    │
│    • Blend 65% image + 35% heatmap                  │
└─────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────┐
│  STEP 5 — Pathology Localization                    │
│                                                     │
│  Threshold high-activation regions (> 0.55)        │
│  Find contours → draw yellow outlines               │
│  Label with top finding name + confidence score     │
│                                                     │
│  These are the areas MOST likely to contain         │
│  the detected pathology                             │
└─────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────┐
│  STEP 6 — Anatomy Overlay (Medically Derived)       │
│                                                     │
│  Using the lung mask, we derive:                    │
│    • Lung field outlines  → from real contours      │
│    • Diaphragm curves     → bottom boundary of lung │
│    • Cardiac silhouette   → gap between the lungs   │
│    • Trachea / Carina     → geometric approximation │
│    • Rib markers          → geometric approximation │
│                                                     │
│  Layers are toggleable in the sidebar               │
└─────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────┐
│  STEP 7 — Structured Report (Rule-Based)            │
│                                                     │
│  Filters predictions above confidence threshold     │
│  Groups findings into clinical patterns:            │
│    • Airspace disease (Pneumonia, Consolidation)    │
│    • Pleural abnormality (Effusion, Thickening)     │
│    • Mass-like lesion (Mass, Nodule)                │
│    • Possible collapse (Atelectasis)                │
│  Generates FINDINGS + IMPRESSION + NOTE             │
└─────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────┐
│  STEP 8 — LLM Narrative Report (On Demand)          │
│                                                     │
│  Sends findings to Ollama (llama3.1)                │
│  Prompt instructs: be realistic, don't over-diagnose│
│  Output: FINDINGS / IMPRESSION / RECOMMENDATION     │
│          written in clinical language               │
└─────────────────────────────────────────────────────┘
```

---

## What each image panel shows

| Panel | What you're looking at |
|---|---|
| **Anatomy Viewer** | The X-ray with anatomy layers drawn on top. Lung outlines, diaphragm, and heart are derived from this specific image, not fixed shapes. |
| **Lung Segmentation** | Just the threshold mask — shows exactly which pixels the system identified as lung tissue. |
| **Medical Heatmap** | Where the AI sees pathology, constrained to the lungs. Red = high attention. Yellow outlines = localised abnormality regions. |
| **Vision Heatmap** | Raw Grad-CAM with no medical constraints. Useful for debugging the model, not for clinical interpretation. |

---

## What is medically accurate vs. approximate

| Feature | Accuracy | Why |
|---|---|---|
| Lung field outlines | **Image-derived** | Traced from actual pixel intensities |
| Diaphragm curves | **Image-derived** | Bottom boundary of the lung mask |
| Cardiac silhouette | **Image-derived** | Sized from mediastinal gap between lungs |
| Trachea / Carina | Approximate | Cannot be reliably extracted from threshold mask |
| Rib markers | Approximate | Requires dedicated rib segmentation model |
| Heatmap location | Approximate | Grad-CAM is indirect — classifier-based, not segmentation |

---

## What this is NOT

- Not a diagnostic tool
- Not a replacement for a radiologist
- Grad-CAM localization is **weak localization**, not true lesion segmentation
- Threshold lung mask works on most clean X-rays but may fail on very dense lungs

---

## Roadmap to true medical accuracy

```
Current: threshold lung mask
    ↓
Phase 2: U-Net lung segmentation (MONAI / nnU-Net)
    ↓
Phase 3: True lesion segmentation (nnU-Net pathology models)
    ↓
Phase 4: DICOM support + window/level + measurements
    ↓
Phase 5: Full PACS-style workstation
```

Recommended training datasets when ready:
- **JSRT** — Japanese Society of Radiology Technology
- **Montgomery County CXR Set** — lung segmentation masks included
- **Shenzhen CXR Set** — TB screening with masks
