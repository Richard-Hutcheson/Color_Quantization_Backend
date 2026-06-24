import json
import time
from typing import Annotated

from fastapi import APIRouter, Form, HTTPException, UploadFile
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from src.models.images import ImagesResponse, PaintColor
from src.services.images import build_color_palette_data, build_filled_paint_by_numbers_image, build_paint_by_numbers_image, build_palette_image, compute_blob_centers, compute_mix_recipe, extract_dominant_colors

router = APIRouter(prefix="/api")

# Allowed MIME types for uploaded images
_ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}

# 10 MB file size limit
_MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024

_DEFAULT_PALETTE: list[PaintColor] = [
    PaintColor(name="Titanium White", r=0xF3, g=0xF4, b=0xF7),
    PaintColor(name="Ivory Black",    r=0x23, g=0x1F, b=0x20),
    PaintColor(name="Cadmium Red",    r=0xE3, g=0x00, b=0x22),
    PaintColor(name="Cadmium Yellow", r=0xFD, g=0xDA, b=0x0D),
    PaintColor(name="Phthalo Blue",   r=0x00, g=0x0F, b=0x89),
]


@router.post("/images", status_code=200)
async def post_images(
    image: UploadFile,
    # color_count must be between 1 and 64 (inclusive)
    color_count: Annotated[int, Form(ge=1, le=64)],
    palette_json: Annotated[str | None, Form()] = None,
) -> ImagesResponse:
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

    if palette_json is not None:
        try:
            raw = json.loads(palette_json)
            palette = [PaintColor.model_validate(entry) for entry in raw]
            if len(palette) == 0:
                raise ValueError("Palette must contain at least one paint.")
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=f"Invalid palette_json: {exc}")
    else:
        palette = _DEFAULT_PALETTE

    try:
        print(f"[images] Request received — filename={image.filename!r}, color_count={color_count}")

        print("[images] Extracting dominant colors...")
        t0 = time.perf_counter()
        result = extract_dominant_colors(image_bytes, color_count)
        print(f"[images] Dominant colors extracted in {time.perf_counter() - t0:.2f}s")

        print("[images] Computing mix recipes...")
        t0 = time.perf_counter()
        result.response.colors = [
            color.model_copy(update={"recipe": compute_mix_recipe(color, palette)})
            for color in result.response.colors
        ]
        print(f"[images] Mix recipes computed in {time.perf_counter() - t0:.2f}s")

        print("[images] Computing blob centers...")
        t0 = time.perf_counter()
        blob_centers = compute_blob_centers(result.label_map, result.response.colors)
        print(f"[images] Blob centers computed in {time.perf_counter() - t0:.2f}s")

        print("[images] Building palette image...")
        t0 = time.perf_counter()
        build_palette_image(result.response.colors)
        print(f"[images] Palette image built in {time.perf_counter() - t0:.2f}s")

        print("[images] Building paint-by-numbers image...")
        t0 = time.perf_counter()
        pbn_image = build_paint_by_numbers_image(result.full_image, result.label_map, result.response.colors, blob_centers)
        print(f"[images] Paint-by-numbers image built in {time.perf_counter() - t0:.2f}s")

        print("[images] Building filled paint-by-numbers image...")
        t0 = time.perf_counter()
        pbn_filled_image = build_filled_paint_by_numbers_image(result.full_image, result.label_map, result.response.colors, blob_centers)
        print(f"[images] Filled paint-by-numbers image built in {time.perf_counter() - t0:.2f}s")

        color_palette = build_color_palette_data(result.response.colors)

        print("[images] Done.")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process image: {exc}",
        )

    return result.response.model_copy(update={
        "paint_by_numbers_image": pbn_image,
        "paint_by_numbers_filled_image": pbn_filled_image,
        "color_palette": color_palette,
    })
