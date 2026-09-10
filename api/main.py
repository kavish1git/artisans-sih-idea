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
    background: str = Form("auto", description="Automatic contrast-aware background selection"),
    aspect_ratio: str = Form("auto", description="Automatic e-commerce aspect ratio alignment"),
    enhancement: str = Form("auto", description="Conservative photo-realistic enhancement"),
    shadow_mode: str = Form("auto", description="Intelligent grounding shadow decision"),
    model_name: str = Form("auto", description="Automatic segmentation model selection"),
):
    """
    Main fully automatic cataloging transformation pipeline endpoint.
    Accepts ONLY the raw artisan image and automatically determines:
    - Handicraft category & craft identity
    - Visual attributes (color names, shape, orientation, pattern, texture)
    - Optimal contrast background, grounding shadow, and framing.
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
            output_format="JPEG",
            model_name=model_name,
            save_files=True,
        )

        base_url = str(request.base_url).rstrip("/")

        # Handle retake recommendations gracefully
        if result.get("status") == "needs_retake":
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "status": "needs_retake",
                    "reason": result.get("reason", "The product could not be isolated clearly."),
                    "voice_prompt": result.get("voice_prompt", "Please place the craft on a contrasting surface and retake."),
                    "quality_score": result.get("quality_score", 0),
                    "processing": {"time_ms": result.get("processing_time_ms", 0)},
                },
            )

        # Convert local disk paths to accessible API URLs
        output_url = None
        transparent_url = None
        orig_url = None

        img_dict = result.get("image", {})
        if img_dict.get("processed"):
            out_name = Path(img_dict["processed"]).name
            output_url = f"{base_url}/output/{out_name}"

        if img_dict.get("transparent"):
            trans_name = Path(img_dict["transparent"]).name
            transparent_url = f"{base_url}/output/{trans_name}"

        if img_dict.get("original"):
            orig_name = Path(img_dict["original"]).name
            orig_url = f"{base_url}/output/{orig_name}"

        response_payload = {
            "status": "success",
            "product": result.get("product", {}),
            "visual_attributes": result.get("visual_attributes", {}),
            "geometry": result.get("geometry", {}),
            "image": {
                "original": orig_url,
                "processed": output_url,
                "transparent": transparent_url,
                "width": 1080,
                "height": 1080,
            },
            "quality": result.get("quality", {}),
            "processing": result.get("processing", {}),
            "voice_prompt": result.get("voice_prompt", ""),
            "warnings": result.get("warnings", []),
            # Compatibility helpers for existing UI / client scripts
            "image_url": output_url,
            "transparent_url": transparent_url,
            "before_score": result.get("quality", {}).get("before", 0),
            "after_score": result.get("quality", {}).get("after", 0),
            "improvement": result.get("quality", {}).get("improvement", 0),
            "dominant_colors": result.get("visual_attributes", {}).get("palette_hex", []),
            "recommendation": result.get("voice_prompt", ""),
            "processing_time_ms": result.get("processing", {}).get("time_ms", 0),
        }

        return JSONResponse(content=response_payload)


    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline processing failure: {str(e)}",
        )

