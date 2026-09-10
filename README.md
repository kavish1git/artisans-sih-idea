---
title: Artisan Smart Cataloging CV
emoji: 🏺
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 8000
---

# AI-Driven Market Linkage & Smart Cataloging for Marginalized Artisans (SIH26090)

## Production-Ready Computer Vision System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-Headless-5C3EE8.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 1. Project Overview

Targeting **Smart India Hackathon (SIH) Problem Statement SIH26090**, this Computer Vision module empowers rural, marginalized Indian artisans, weavers, potters, and sculptors with an automated, production-grade image enhancement and cataloging engine.

In rural workshop settings, artisans photograph products using budget smartphones under challenging real-world conditions:
- Cluttered backgrounds (floors, beds, outdoor workshops)
- Poor, uneven indoor lighting & harsh shadows
- Camera tilt and suboptimal framing
- Low digital literacy

This system transforms unedited raw smartphone photos into pristine, standardized, marketplace-compliant e-commerce catalog listings (1080×1080 1:1 format) with soft grounding shadows and instant spoken-language feedback.

---

## 2. End-to-End Pipeline Architecture

```text
               Smartphone Photo
                      ↓
      [Module 1] Image Quality Analysis (Resolution, Blur, Lighting, Framing)
                      ↓
      [Module 2] Product Segmentation (Deep Matting & Alpha Mask)
                      ↓
      [Module 3] Background Removal (Edge-Preserving Transparent PNG)
                      ↓
      [Module 4] Grounding Shadow Synthesis (Professional vs. Natural)
                      ↓
      [Module 5] Lighting & Dynamic Range Normalization
                      ↓
      [Module 6] Adaptive Color Constancy & White Balance
                      ↓
      [Module 7] Smart Centering & Aspect-Ratio Cropping (1:1, 4:5, 3:4, 16:9)
                      ↓
      [Module 8] Clean Neutral Backdrop Generation (White / Off-White / Studio)
                      ↓
      [Module 9] Conservative Fine-Detail Enhancement
                      ↓
      [Module 10] Marketplace Formatting (1080x1080 JPEG/WebP, 90 Quality)
                      ↓
      [Module 11] Before vs. After Quality Verification
                      ↓
           Final Marketplace Image & Voice Prompt
```

---

## 3. Project Directory Structure

```text
artisan-computer-vision/
├── api/
│   ├── __init__.py
│   └── main.py              # FastAPI server & REST endpoints
├── src/
│   ├── __init__.py
│   ├── utils.py             # Module 22: Safe I/O, security checks, color conversions
│   ├── quality.py           # Module 1 & 21: Quality analysis & artisan voice prompts
│   ├── segmentation.py      # Module 2: Pretrained U2-Net / matting model
│   ├── background.py        # Module 3: Alpha matting & transparent PNG
│   ├── shadow.py            # Module 4: Grounding shadow generation
│   ├── lighting.py          # Module 5: Adaptive illumination & dynamic range
│   ├── white_balance.py     # Module 6: Shades-of-gray / gray-world constancy
│   ├── crop.py              # Module 7: Bounding box padding & aspect ratio
│   ├── marketplace.py       # Module 8 & 9: Clean backdrop & conservative enhancement
│   ├── formatter.py         # Module 10: 1080x1080 JPEG/WebP export
│   └── pipeline.py          # Module 11 & 12: Unified pipeline orchestration
├── static/
│   └── index.html           # Module 14: Interactive Web UI for SIH live jury demo
├── tests/
│   ├── images/              # Synthetic & real handicraft test fixtures
│   ├── generate_test_images.py
│   ├── test_quality.py      # Unit tests for image loading & quality analysis
│   ├── test_segmentation.py # Tests for segmentation and background removal
│   ├── test_processing_modules.py # Tests for lighting, shadow, crop, format
│   ├── test_pipeline.py     # End-to-end integration tests
│   └── test_api.py          # FastAPI endpoint tests
├── scripts/
│   ├── benchmark.py         # Module 16: Latency, memory, P50/P95 benchmarking
│   └── evaluate.py          # Module 17 & 18: Quantitative IoU, Dice, Precision, Recall
├── experiments/
│   └── model_comparison.csv # Module 17: U2-Net vs GrabCut benchmark data
├── reports/
│   └── cv_evaluation.md     # Module 18: Comprehensive evaluation report
├── requirements.txt
├── .env.example
├── Dockerfile
└── README.md
```

---

## 4. Setup & Local Installation

### Prerequisites
- Python 3.11 or 3.12
- pip package manager

### Installation Steps

```bash
# 1. Navigate to project folder
cd artisan-computer-vision

# 2. Create and activate virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate

# Linux / macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 5. Running the Application & Live Demo Web UI

Start the FastAPI application with Uvicorn:

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Once running:
- **Interactive SIH Demo UI**: Open [http://localhost:8000](http://localhost:8000) in your web browser.
- **Swagger Interactive API Documentation**: Open [http://localhost:8000/docs](http://localhost:8000/docs).
- **Health Check**: Open [http://localhost:8000/api/v1/health](http://localhost:8000/api/v1/health).

---

## 6. Running Automated Tests

Run the comprehensive 30-test automated test suite:

```bash
python -m pytest tests/ -v
```

All 30 unit and integration tests pass with a 100% success rate across:
- Magic byte & MIME validation, decompression bomb protection
- Blur, brightness, exposure, noise, composition analysis
- U²-Net segmentation and transparent PNG background removal
- Aspect-ratio centering (1:1, 4:5, 3:4, 16:9)
- Natural lighting and Shades-of-Gray white balance correction
- Grounding shadow generation (professional vs. natural)
- Marketplace formatting (1080x1080 JPEG/WebP)
- Full end-to-end pipeline execution
- FastAPI REST endpoints

---

## 7. Performance Benchmarking & Evaluation

### Run Benchmark
```bash
python scripts/benchmark.py
```
*Output on standard CPU:*
- **Average Processing Time**: ~1.1 seconds
- **P50 (Median) Latency**: ~818 ms (Sub-second inference!)
- **Success Rate**: 100.0%
- **Memory Footprint**: ~670 MB

### Run Quantitative Evaluation
```bash
python scripts/evaluate.py
```
*Generates:*
- `experiments/model_comparison.csv`
- `reports/cv_evaluation.md`

| Model Architecture | IoU (Jaccard) | Dice (F1) | Precision | Recall | Avg Latency (CPU) | Role |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **U²-Net Portable (u2netp)** | **99.7%** | **99.8%** | **99.7%** | **100.0%** | **~797 ms** | **Primary Engine** |
| **Adaptive GrabCut Saliency** | **92.9%** | **96.3%** | **92.9%** | **100.0%** | **~2534 ms** | **Offline Fallback** |

---

## 8. API Documentation & Examples

### Endpoint: `POST /api/v1/process-image`

Accepts `multipart/form-data`:
- `image`: File (JPEG, PNG, WebP)
- `background`: `white` (default), `off-white`, `light-gray`, `transparent`
- `aspect_ratio`: `1:1` (default), `4:5`, `3:4`, `16:9`
- `shadow_mode`: `professional` (default), `natural`, `none`
- `enhancement`: `auto` (default), `low`, `medium`, `none`

#### Example using `cURL`:
```bash
curl -X POST "http://localhost:8000/api/v1/process-image" \
  -F "image=@tests/images/pottery_normal.jpg" \
  -F "background=white" \
  -F "aspect_ratio=1:1" \
  -F "shadow_mode=professional"
```

#### Example using Python `requests`:
```python
import requests

url = "http://localhost:8000/api/v1/process-image"
files = {"image": open("pottery.jpg", "rb")}
data = {
    "background": "white",
    "aspect_ratio": "1:1",
    "shadow_mode": "professional"
}

response = requests.post(url, files=files, data=data)
result = response.json()

print(f"Catalog URL: {result['image_url']}")
print(f"Quality Before: {result['before_score']}/100")
print(f"Quality After: {result['after_score']}/100")
print(f"Dominant Colors: {result['dominant_colors']}")
print(f"Artisan Voice Prompt: {result['recommendation']}")
```

#### Example JSON Response:
```json
{
  "status": "success",
  "image_url": "http://localhost:8000/output/catalog_1a2b3c4d5e6f.jpg",
  "output_image": "C:\\Users\\Kavish\\...\\output\\catalog_1a2b3c4d5e6f.jpg",
  "transparent_url": "http://localhost:8000/output/transparent_1a2b3c4d5e6f.png",
  "transparent_image": "C:\\Users\\Kavish\\...\\output\\transparent_1a2b3c4d5e6f.png",
  "before_score": 54,
  "after_score": 91,
  "improvement": 37,
  "bounding_box": { "x": 164, "y": 73, "width": 750, "height": 935 },
  "dominant_colors": ["#b25227", "#f3eae1", "#cda84b", "#49392e"],
  "recommendation": "Product detected clearly! Photo quality is great for cataloging.",
  "processing_time_ms": 1135.2,
  "aspect_ratio": "1:1",
  "background": "white",
  "warnings": []
}
```

---

## 9. Accessibility & Artisan Voice Feedback (Module 21)

Unlike developer-centric tools that return cryptic metrics (`confidence = 0.42`), this module returns plain spoken feedback designed for audio synthesis in mobile apps:

| Image Defect | Artisan Voice Recommendation |
| :--- | :--- |
| **Blurry photo** | *"Photo is blurry. Please hold your phone steady and tap to focus."* |
| **Dark lighting** | *"Photo is too dark. Please move closer to a window or turn on a light."* |
| **Harsh glare** | *"Photo has harsh glare. Please shield direct sunlight from the craft."* |
| **Small subject** | *"Product is too small in the frame. Please move the camera closer."* |
| **Subject cut off** | *"Product is too close to the edges. Please move the camera back slightly."* |
| **Good quality** | *"Product detected clearly! Photo quality is great for cataloging."* |

The included Web UI features a **"Read Aloud"** button leveraging browser Text-to-Speech (`window.speechSynthesis`) to demonstrate the voice prompt during SIH judging.

---

## 10. Security & Robustness (Module 22)

- **File Header & Magic Byte Validation**: Rejects spoofed extensions and corrupted payloads.
- **Decompression Bomb Protection**: 15 MB file size limit and strict dimension caps ($64 \times 64$ to $8192 \times 8192$).
- **Safe Temporary Files**: Random UUID identifiers (`catalog_{uuid}.jpg`) to prevent collisions and directory traversal attacks.
- **Zero-Crash Design**: Corrupt files or extreme lighting defects produce helpful status warnings and guidance rather than server 500 crashes.
