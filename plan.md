# Plan: POST /api/images — Dominant Color Extraction

## Overview
Add a `POST /api/images` endpoint that accepts a JPG/PNG image and a `color_count` integer, runs KMeans clustering (scikit-learn) to extract the X most dominant colors, and returns them in the response.

Follows the existing pattern: `routers/` → `services/` → `models/`.

---

## Files to Create / Modify

| File | Action | Purpose |
|---|---|---|
| `src/models/images.py` | Create | Pydantic response model for dominant colors |
| `src/services/images.py` | Create | `extract_dominant_colors()` — KMeans color extraction logic |
| `src/routers/images.py` | Create | `POST /api/images` — validation + calls service |
| `src/main.py` | Modify | Register the new images router |

---

## Implementation Details

### Router — `src/routers/images.py`
- Accept `multipart/form-data` with:
  - `image`: `UploadFile` (validated to be `image/jpeg` or `image/png`)
  - `color_count`: `int` (validated via Form field + Annotated constraints)
- Validation failures → `422 Unprocessable Entity` (FastAPI default) with descriptive messages
- Any processing error → `500 Internal Server Error` with a descriptive message
- Calls `extract_dominant_colors(image_bytes, color_count)` from the service layer

### Service — `src/services/images.py`
1. Load image bytes with **Pillow** → convert to RGB
2. Resize a copy to max 1024px (longest side) for clustering only — full-resolution image preserved for future use
3. Flatten pixel array with **numpy** into shape `(N, 3)`
4. Run **KMeans** (scikit-learn) with `n_clusters=color_count`
5. Return cluster centers as dominant colors
6. Raise a descriptive exception on any failure so the router can catch it

### Model — `src/models/images.py`
- `DominantColor`: `r`, `g`, `b` (int 0–255) + `hex` (str, computed)
- `ImagesResponse`: `color_count: int`, `colors: list[DominantColor]`

---

## Validation Rules
| Field | Rule |
|---|---|
| `image` | Content-type must be `image/jpeg` or `image/png`; file must be ≤ 10 MB; must be readable by Pillow |
| `color_count` | Integer, 1–64 (inclusive) |

---

## Decisions
1. **Response format** — RGB + hex, but endpoint returns HTTP 200 with no body for Part 1
2. **`color_count` bounds** — min 1, max 64
3. **File size limit** — reject if > 10 MB (checked on raw bytes before processing)
4. **Image resize** — downsample to max 1024px longest side **for clustering only**; full-resolution image is preserved for future use (remapping, paint-by-numbers overlay, etc.)
