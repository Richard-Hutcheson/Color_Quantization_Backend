from typing import Annotated

from fastapi import APIRouter, Form, HTTPException, Response, UploadFile
from fastapi.exceptions import RequestValidationError

from src.services.images import build_filled_paint_by_numbers_image, build_paint_by_numbers_image, build_palette_image, extract_dominant_colors

router = APIRouter(prefix="/api")

# Allowed MIME types for uploaded images
_ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}

# 10 MB file size limit
_MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024


@router.post("/images", status_code=200)
async def post_images(
    image: UploadFile,
    # color_count must be between 1 and 64 (inclusive)
    color_count: Annotated[int, Form(ge=1, le=64)],
) -> Response:
    # Validate the uploaded file's content type
    if image.content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Invalid file type. Only JPG and PNG images are accepted.",
        )

    image_bytes = await image.read()

    # Validate the file size
    if len(image_bytes) > _MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="File too large for maximum allowed size.",
        )

    try:
        result = extract_dominant_colors(image_bytes, color_count)
        build_palette_image(result.response.colors)
        build_paint_by_numbers_image(result.full_image, result.label_map, result.response.colors)
        build_filled_paint_by_numbers_image(result.full_image, result.label_map, result.response.colors)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process image: {exc}",
        )

    return Response(status_code=200)
