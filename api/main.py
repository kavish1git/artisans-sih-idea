"""FastAPI Application Backend (Module 13 & 22)
Exposes REST endpoints for the Artisan Computer Vision system:
- POST /api/v1/process-image: Multipart image upload and catalog pipeline execution
- POST /api/v1/quality: Lightweight pre-flight quality check and voice recommendation
- GET  /api/v1/health: Service health check
- Serves interactive demo web UI and output image assets.
"""

import os
import io
import time
import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.utils import validate_image_bytes, ImageValidationError, MAX_FILE_SIZE_BYTES
from src.pipeline import get_pipeline
from src.quality import analyze_image_quality

# Initialize FastAPI
app = FastAPI(
    title="Artisan Smart Cataloging CV API (SIH26090)",
    description="Production-ready computer vision pipeline transforming smartphone artisan photos into marketplace catalog listings.",
    version="1.0.0",
)

# CORS middleware for mobile and frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "output"
STATIC_DIR = BASE_DIR / "static"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# Mount static and output directories
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=FileResponse)
async def serve_demo_ui():
    """Serves the interactive SIH demonstration web dashboard."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return JSONResponse({"message": "Artisan Smart Cataloging CV API is operational."})


@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint matching Module 13 specification."""
    return {
        "status": "healthy",
        "service": "artisan-computer-vision",
        "version": "1.0.0",
        "timestamp": time.time(),
    }


@app.post("/api/v1/quality")
async def check_quality_only(image: UploadFile = File(...)):
    """
    Fast pre-flight image quality check endpoint.
    Returns quality score and actionable artisan voice feedback before full processing.
    """
    try:
        content = await image.read()
        validate_image_bytes(content, filename=image.filename)
        quality_report = analyze_image_quality(content)
        return {
            "status": "success",
            "overall_score": quality_report["overall_score"],
            "recommendation": quality_report["recommendation"],
            "metrics": quality_report,
        }
    except ImageValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Quality analysis error: {str(e)}",
        )


@app.post("/api/v1/process-image")
async def process_image_endpoint(
    request: Request,
    image: UploadFile = File(..., description="Raw smartphone handicraft photo"),
    background: str = Form("white", description="Background type: white, off-white, light-gray, transparent"),
    aspect_ratio: str = Form("1:1", description="Target aspect ratio: 1:1, 4:5, 3:4, 16:9"),
    enhancement: str = Form("auto", description="Enhancement mode: auto, low, medium, none"),
    shadow_mode: str = Form("professional", description="Shadow style: professional, natural, none"),
    model_name: str = Form("isnet-general-use", description="Model: isnet-general-use, silueta, u2netp, u2net"),
    alpha_matting: bool = Form(True, description="Enable edge and thread matting"),
    detection_mode: str = Form("auto", description="Detection mode: auto, focused, full_set"),
):
    """
    Main cataloging transformation pipeline endpoint.
    Accepts multipart form-data and returns comprehensive product metadata and processed asset URLs.
    """
    # 1. Enforce payload size limit
    content = await image.read()
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of 15 MB.",
        )

    # 2. Validate format and magic bytes
    try:
        validate_image_bytes(content, filename=image.filename)
    except ImageValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    # 3. Execute Computer Vision Pipeline
    try:
        pipeline = get_pipeline()
        result = pipeline.process(
            source=content,
            background=background,
            aspect_ratio=aspect_ratio,
            enhancement=enhancement,
            shadow_mode=shadow_mode,
            target_dim=1080,
            output_format="JPEG" if background != "transparent" else "PNG",
            model_name=model_name,
            alpha_matting=alpha_matting,
            detection_mode=detection_mode,
            save_files=True,
        )

        base_url = str(request.base_url).rstrip("/")

        # Convert local disk paths to accessible API URLs
        output_url = None
        transparent_url = None

        if result.get("output_image"):
            out_name = Path(result["output_image"]).name
            output_url = f"{base_url}/output/{out_name}"

        if result.get("transparent_image"):
            trans_name = Path(result["transparent_image"]).name
            transparent_url = f"{base_url}/output/{trans_name}"

        response_payload = {
            "status": result["status"],
            "image_url": output_url,
            "output_image": result.get("output_image"),
            "transparent_url": transparent_url,
            "transparent_image": result.get("transparent_image"),
            "before_score": result.get("before_score", 0),
            "after_score": result.get("after_score", 0),
            "improvement": result.get("improvement", 0),
            "quality_before": result.get("quality_before", {}),
            "quality_after": result.get("quality_after", {}),
            "bounding_box": result.get("bounding_box", {}),
            "dominant_colors": result.get("dominant_colors", []),
            "recommendation": result.get("recommendation", ""),
            "processing_time_ms": result.get("processing_time_ms", 0),
            "aspect_ratio": result.get("aspect_ratio", aspect_ratio),
            "background": result.get("background", background),
            "warnings": result.get("warnings", []),
        }

        return JSONResponse(content=response_payload)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline processing failure: {str(e)}",
        )
