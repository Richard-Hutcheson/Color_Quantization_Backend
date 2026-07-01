import base64
import io
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt
from scipy.optimize import nnls
from skimage.color import lab2rgb, rgb2lab
from sklearn.cluster import KMeans

from src.models.images import AchievedColor, DominantColor, ImagesResponse, MixRecipe, PaintColor, PaletteEntry

# Images are downsampled to this size (longest side) before clustering
# to keep KMeans fast regardless of the original resolution.
_CLUSTER_MAX_PX = 1024

# Directory where palette images are saved
_PALETTE_OUTPUT_DIR = Path(__file__).parents[2] / "output" / "dominant_color_palette"

# Directory where paint-by-numbers images are saved
_PBN_OUTPUT_DIR = Path(__file__).parents[2] / "output" / "paint_by_numbers"

# Directory where filled paint-by-numbers images are saved
_PBN_FILLED_OUTPUT_DIR = Path(__file__).parents[2] / "output" / "paint_by_numbers_filled"

# Gaussian blur applied to the cluster image before KMeans fitting.
# Smooths out texture/detail so KMeans groups broad color regions instead of
# fine surface variation. Kernel must be odd.
_CLUSTER_BLUR_KERNEL = 15
_CLUSTER_BLUR_SIGMA = 3.0

# Blobs smaller than this fraction of total image pixels are skipped for number placement
_MIN_BLOB_AREA_RATIO = 0.001

# cv2.putText parameters for region numbers
_PBN_FONT = cv2.FONT_HERSHEY_SIMPLEX
_PBN_FONT_SCALE = 1.4
_PBN_FONT_THICKNESS = 2
_PBN_TEXT_MARGIN = 12
_PBN_EDGE_INWARD_MIN_DIST_RATIO = 0.65


@dataclass
class ExtractionResult:
    """Internal result type carrying both the API response model and the
    full-resolution image data needed for downstream processing steps."""
    response: ImagesResponse
    full_image: Image.Image
    label_map: np.ndarray  # shape (H, W), int cluster index per pixel


def extract_dominant_colors(image_bytes: bytes, color_count: int) -> ExtractionResult:
    """
    Determine the `color_count` most dominant colors in the image using KMeans clustering.

    The full-resolution image is preserved in memory for future processing steps
    (pixel remapping, paint-by-numbers overlay, etc.). Only a downsampled copy is
    passed to KMeans to keep clustering fast.
    """
    # Load the full-resolution image and normalise to RGB
    full_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Build a smaller copy for clustering only, then blur it before fitting.
    # Blurring removes texture/fine detail so KMeans assigns pixels to broad
    # color regions rather than surface noise.
    cluster_image = _resize_for_clustering(full_image)
    cluster_array = np.array(cluster_image, dtype=np.uint8)
    cluster_array = cv2.GaussianBlur(
        cluster_array,
        (_CLUSTER_BLUR_KERNEL, _CLUSTER_BLUR_KERNEL),
        _CLUSTER_BLUR_SIGMA,
    )

    # Flatten pixels into a 2-D array of shape (num_pixels, 3)
    pixel_array = cluster_array.reshape(-1, 3).astype(np.float32)

    # Run KMeans to find the dominant color clusters
    kmeans = KMeans(n_clusters=color_count, random_state=42, n_init="auto")
    kmeans.fit(pixel_array)

    # Cluster centers are the dominant colors; round to nearest integer (0-255)
    centers = np.round(kmeans.cluster_centers_).astype(int)

    colors = [DominantColor(r=int(r), g=int(g), b=int(b)) for r, g, b in centers]

    print(f"Dominant colors ({color_count}):")
    for color in colors:
        print(f"  RGB({color.r}, {color.g}, {color.b}), HEX: {color.hex}")

    # Reuse the labels KMeans already computed on the downsampled cluster image.
    # Reshape into a 2-D label map and scale up to full resolution with nearest-neighbor
    # interpolation, then absorb small isolated islands into their nearest large neighbor.
    cluster_h, cluster_w = cluster_array.shape[:2]
    cluster_label_map = kmeans.labels_.reshape(cluster_h, cluster_w).astype(np.uint8)
    img_w, img_h = full_image.size
    label_map = cv2.resize(cluster_label_map, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
    label_map = label_map.astype(np.int32)
    min_blob_px = max(1, int(img_h * img_w * _MIN_BLOB_AREA_RATIO))
    label_map = _absorb_small_blobs(label_map, min_blob_px)

    return ExtractionResult(
        response=ImagesResponse(color_count=color_count, colors=colors),
        full_image=full_image,
        label_map=label_map,
    )


def build_palette_image(colors: list[DominantColor]) -> None:
    """
    Render a JPG palette grid showing each dominant color as a labeled swatch.
    Each cell contains a color rectangle with the RGB value and hex code below it.
    Colors are arranged in rows of up to 8 swatches.
    The image is saved to the output/dominant_color_palette directory.
    """
    SWATCH_W = 150
    SWATCH_H = 150
    LABEL_H = 55       # vertical space reserved for two lines of text below the swatch
    PADDING = 12       # spacing around each swatch
    MAX_COLS = 8
    BACKGROUND = (245, 245, 245)
    TEXT_COLOR = (50, 50, 50)
    FONT_SIZE = 14

    cols = min(len(colors), MAX_COLS)
    rows = math.ceil(len(colors) / cols)

    cell_w = SWATCH_W + PADDING * 2
    cell_h = SWATCH_H + LABEL_H + PADDING * 2

    img = Image.new("RGB", (cell_w * cols, cell_h * rows), color=BACKGROUND)
    draw = ImageDraw.Draw(img)

    # load_default(size=N) returns a FreeType font in Pillow >= 10; fall back to
    # the basic bitmap font on older versions so the function never hard-crashes.
    try:
        font = ImageFont.load_default(size=FONT_SIZE)
    except TypeError:
        font = ImageFont.load_default()

    # Draw each color as a swatch with its RGB and hex label, placed in grid order
    for i, color in enumerate(colors):
        col = i % cols
        row = i // cols

        x = col * cell_w + PADDING
        y = row * cell_h + PADDING

        # Color swatch rectangle
        draw.rectangle([x, y, x + SWATCH_W, y + SWATCH_H], fill=(color.r, color.g, color.b))

        # Center the text under the swatch
        text_x = x + SWATCH_W // 2
        rgb_y = y + SWATCH_H + 10
        hex_y = rgb_y + FONT_SIZE + 6

        draw.text((text_x, rgb_y), f"RGB({color.r}, {color.g}, {color.b})", fill=TEXT_COLOR, font=font, anchor="mt")
        draw.text((text_x, hex_y), color.hex, fill=TEXT_COLOR, font=font, anchor="mt")

    _PALETTE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = _PALETTE_OUTPUT_DIR / "palette.jpg"
    img.save(output_path, format="JPEG", quality=95)
    print(f"Palette image saved to {output_path}")


def build_color_palette_data(colors: list[DominantColor]) -> dict[str, PaletteEntry]:
    """
    Return the dominant color palette as a dict keyed by the 1-based number
    that is stamped in the paint-by-numbers image.
    """
    return {
        str(idx + 1): PaletteEntry(r=color.r, g=color.g, b=color.b)
        for idx, color in enumerate(colors)
    }


def build_paint_by_numbers_image(
    full_image: Image.Image,
    label_map: np.ndarray,
    colors: list[DominantColor],
    blob_centers: list[list[tuple[int, int]]],
) -> str:
    """
    Generate a paint-by-numbers overlay at the original image resolution.

    The output is a white-background image with:
    - Black borders wherever two adjacent pixels belong to different color clusters.
    - A number placed inside every blob, using pre-computed interior points from
      blob_centers (see compute_blob_centers).

    The number-to-RGB key is printed to the console.
    The image is saved to output/paint_by_numbers/pbn.jpg.
    """
    img_h, img_w = label_map.shape

    # White background
    overlay = np.full((img_h, img_w, 3), 255, dtype=np.uint8)

    # Build a boolean mask of border pixels: any pixel whose 4-neighbor has a
    # different cluster label is considered a border and drawn black.
    h_edge = label_map[:-1, :] != label_map[1:, :]  # shape (H-1, W)
    v_edge = label_map[:, :-1] != label_map[:, 1:]  # shape (H, W-1)

    border = np.zeros((img_h, img_w), dtype=bool)
    # Expand each edge onto both neighboring pixels so the border appears as a
    # 2-pixel-wide line straddling the true region boundary rather than a 1-pixel
    # line sitting entirely on one side of it.
    border[:-1, :] |= h_edge
    border[1:, :] |= h_edge
    border[:, :-1] |= v_edge
    border[:, 1:] |= v_edge

    overlay[border] = [0, 0, 0]

    # For each color, stamp its 1-based number at the interior point of every blob
    print("Paint-by-numbers key:")
    for idx, color in enumerate(colors):
        number = idx + 1
        print(f"  {number}: RGB({color.r}, {color.g}, {color.b})")
        # Each entry in blob_centers[idx] is the most interior (cx, cy) of one connected blob
        for cx, cy in blob_centers[idx]:
            text_origin = _compute_pbn_text_origin(
                cx,
                cy,
                str(number),
                img_w,
                img_h,
            )
            cv2.putText(
                overlay,
                str(number),
                text_origin,
                _PBN_FONT,
                _PBN_FONT_SCALE,
                (0, 0, 0),
                _PBN_FONT_THICKNESS,
                cv2.LINE_AA,
            )

    _PBN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = _PBN_OUTPUT_DIR / "pbn.jpg"
    pil_image = Image.fromarray(overlay)
    pil_image.save(output_path, format="JPEG", quality=95)
    print(f"Paint-by-numbers image saved to {output_path}")

    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=95)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def build_filled_paint_by_numbers_image(
    full_image: Image.Image,
    label_map: np.ndarray,
    colors: list[DominantColor],
    blob_centers: list[list[tuple[int, int]]],
) -> str:
    """
    Generate a color-filled version of the paint-by-numbers image.

    Each region is filled with its cluster color via a vectorized LUT lookup.
    Black borders are drawn at region boundaries. Numbers are placed using
    pre-computed interior points from blob_centers (see compute_blob_centers),
    with a luminance-based contrasting text color.

    The image is saved to output/paint_by_numbers_filled/pbn_filled.jpg.
    """
    img_h, img_w = label_map.shape

    # Vectorized color fill: build a (color_count, 3) LUT and index it with
    # the label map in one operation instead of N per-label mask writes.
    color_lut = np.array([[c.r, c.g, c.b] for c in colors], dtype=np.uint8)
    overlay = color_lut[label_map]

    # Draw black borders at region boundaries
    h_edge = label_map[:-1, :] != label_map[1:, :]
    v_edge = label_map[:, :-1] != label_map[:, 1:]

    border = np.zeros((img_h, img_w), dtype=bool)
    # Expand each edge onto both neighboring pixels (same logic as build_paint_by_numbers_image)
    border[:-1, :] |= h_edge
    border[1:, :] |= h_edge
    border[:, :-1] |= v_edge
    border[:, 1:] |= v_edge

    overlay[border] = [0, 0, 0]

    # For each color, stamp its number at the interior point of every blob using a contrasting color
    for idx, color in enumerate(colors):
        number = idx + 1
        # Perceived luminance — pick white text on dark fills, black on light fills
        luminance = 0.299 * color.r + 0.587 * color.g + 0.114 * color.b
        text_color = (255, 255, 255) if luminance < 128 else (0, 0, 0)
        # Each entry in blob_centers[idx] is the most interior (cx, cy) of one connected blob
        for cx, cy in blob_centers[idx]:
            text_origin = _compute_pbn_text_origin(
                cx,
                cy,
                str(number),
                img_w,
                img_h,
            )
            cv2.putText(
                overlay,
                str(number),
                text_origin,
                _PBN_FONT,
                _PBN_FONT_SCALE,
                text_color,
                _PBN_FONT_THICKNESS,
                cv2.LINE_AA,
            )

    _PBN_FILLED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = _PBN_FILLED_OUTPUT_DIR / "pbn_filled.jpg"
    pil_image = Image.fromarray(overlay)
    pil_image.save(output_path, format="JPEG", quality=95)
    print(f"Filled paint-by-numbers image saved to {output_path}")

    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=95)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def compute_blob_centers(
    label_map: np.ndarray,
    colors: list[DominantColor],
) -> list[list[tuple[int, int]]]:
    """
    For each color index, choose a label anchor point for every blob in the label map.

    Returns a list (one entry per color) of lists of (cx, cy) positions — one per
    connected blob. Interior blobs use the distance-transform peak (furthest from
    boundaries). Blobs touching the image border use an inward-biased anchor that
    stays interior while shifting closer to the image center.

    Computed once and shared between build_paint_by_numbers_image and
    build_filled_paint_by_numbers_image to avoid redundant connected-component
    analysis and distance transforms.
    """
    img_h, img_w = label_map.shape
    y_coords, x_coords = np.indices((img_h, img_w))
    center_x = (img_w - 1) / 2.0
    center_y = (img_h - 1) / 2.0
    center_dist_map = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)

    centers: list[list[tuple[int, int]]] = []
    # Iterate over each color label to find all disconnected blobs of that color
    for idx in range(len(colors)):
        mask = (label_map == idx).astype(np.uint8)
        # connectedComponentsWithStats labels each disconnected blob; component 0 is always
        # the background (pixels where mask == 0), so real blobs start at index 1.
        num_labels, label_img, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        blob_centers: list[tuple[int, int]] = []
        # Choose an anchor per blob. Edge-touching blobs are biased inward.
        for comp in range(1, num_labels):
            comp_mask = (label_img == comp).astype(np.uint8)
            # distanceTransform assigns each foreground pixel its Euclidean distance to the
            # nearest background pixel; the maximum is the point furthest from any edge —
            # always safely inside the region even for concave shapes.
            dist = cv2.distanceTransform(comp_mask, cv2.DIST_L2, 5)
            left = int(stats[comp, cv2.CC_STAT_LEFT])
            top = int(stats[comp, cv2.CC_STAT_TOP])
            width = int(stats[comp, cv2.CC_STAT_WIDTH])
            height = int(stats[comp, cv2.CC_STAT_HEIGHT])
            edge_touching = (
                left == 0
                or top == 0
                or (left + width) == img_w
                or (top + height) == img_h
            )
            blob_centers.append(_select_blob_anchor(comp_mask, dist, edge_touching, center_dist_map))
        centers.append(blob_centers)
    return centers


def compute_mix_recipe(target: DominantColor, palette: list[PaintColor]) -> MixRecipe:
    """
    Compute the paint mixing recipe that best approximates `target` using the
    colors in `palette`.

    The solver works in LAB color space so that the least-squares minimisation
    is perceptually uniform.  Non-Negative Least Squares (NNLS) is used so that
    every weight is ≥ 0 — you cannot use a negative amount of paint.

    Returns a MixRecipe with:
    - percentages: integer 0–100 for each paint, always summing to 100
    - achieved_color: the color you actually get by mixing those percentages
    """
    # skimage expects float64 RGB in [0, 1] with shape (1, 1, 3)
    def _to_lab(r: int, g: int, b: int) -> np.ndarray:
        rgb = np.array([[[r / 255.0, g / 255.0, b / 255.0]]], dtype=np.float64)
        return rgb2lab(rgb).reshape(3)

    target_lab = _to_lab(target.r, target.g, target.b)
    palette_lab = np.array([_to_lab(p.r, p.g, p.b) for p in palette])  # (num_paints, 3)

    # Solve: palette_lab.T @ weights ≈ target_lab, weights ≥ 0
    weights, _ = nnls(palette_lab.T, target_lab)

    total = weights.sum()
    if total == 0:
        weights = np.ones(len(palette), dtype=np.float64)
        total = float(len(palette))

    normalized = weights / total

    # Largest-remainder rounding so percentages always sum to exactly 100
    raw = normalized * 100.0
    floors = np.floor(raw).astype(int)
    remainders = raw - floors
    deficit = 100 - floors.sum()
    top_indices = np.argsort(remainders)[::-1][:deficit]
    floors[top_indices] += 1
    percentages = {p.name: int(floors[i]) for i, p in enumerate(palette)}

    # Reconstruct the achieved color from the normalized weights
    achieved_lab = palette_lab.T @ normalized  # shape (3,)
    achieved_rgb_float = lab2rgb(achieved_lab.reshape(1, 1, 3)).reshape(3)
    achieved_rgb = np.clip(np.round(achieved_rgb_float * 255), 0, 255).astype(int)
    achieved_color = AchievedColor(r=int(achieved_rgb[0]), g=int(achieved_rgb[1]), b=int(achieved_rgb[2]))

    return MixRecipe(percentages=percentages, achieved_color=achieved_color)


def _absorb_small_blobs(label_map: np.ndarray, min_blob_px: int) -> np.ndarray:
    """
    Remove blobs smaller than `min_blob_px` by overwriting their pixels with the
    label of the nearest large-enough neighbor.

    Uses scipy's distance_transform_edt to efficiently find the nearest valid pixel
    for every invalid (small-blob) pixel in one vectorised pass.
    """
    label_map = label_map.copy()
    num_colors = int(label_map.max()) + 1

    # Mark every pixel that belongs to a blob large enough to receive a number
    valid = np.zeros(label_map.shape, dtype=bool)
    # Scan every blob of every color; flag pixels in blobs that meet the size threshold
    for idx in range(num_colors):
        mask = (label_map == idx).astype(np.uint8)
        n, label_img, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        # Component 0 is background; skip it. For each real blob, check its pixel area.
        for comp in range(1, n):
            if stats[comp, cv2.CC_STAT_AREA] >= min_blob_px:
                valid |= label_img == comp

    if valid.all():
        return label_map

    # distance_transform_edt with return_indices=True returns the (row, col) coordinates
    # of the nearest True (valid) pixel for every False (invalid) pixel.
    _, nearest = distance_transform_edt(~valid, return_indices=True)
    # nearest[0] and nearest[1] are the row and column index arrays; use them to copy
    # the label of the nearest valid pixel into every invalid pixel in one vectorized step.
    label_map[~valid] = label_map[nearest[0][~valid], nearest[1][~valid]]
    return label_map


def _compute_pbn_text_origin(
    cx: int,
    cy: int,
    text: str,
    img_w: int,
    img_h: int,
) -> tuple[int, int]:
    """Return a bottom-left text origin centered on (cx, cy) and clamped inward."""
    (text_w, text_h), baseline = cv2.getTextSize(
        text,
        _PBN_FONT,
        _PBN_FONT_SCALE,
        _PBN_FONT_THICKNESS,
    )

    # cv2.putText expects a bottom-left baseline origin.
    x = int(round(cx - text_w / 2))
    y = int(round(cy + text_h / 2))

    min_x = _PBN_TEXT_MARGIN
    max_x = img_w - text_w - _PBN_TEXT_MARGIN
    min_y = text_h + _PBN_TEXT_MARGIN
    max_y = img_h - baseline - _PBN_TEXT_MARGIN

    if max_x < min_x:
        x = max(0, img_w - text_w)
    else:
        x = max(min_x, min(x, max_x))

    if max_y < min_y:
        y = min(img_h - baseline, max(text_h, y))
    else:
        y = max(min_y, min(y, max_y))

    return (x, y)


def _select_blob_anchor(
    comp_mask: np.ndarray,
    dist: np.ndarray,
    edge_touching: bool,
    center_dist_map: np.ndarray,
) -> tuple[int, int]:
    """Choose a blob anchor; edge-touching blobs are biased inward."""
    _, _, _, max_loc = cv2.minMaxLoc(dist)
    if not edge_touching:
        return max_loc

    max_dist = float(dist.max())
    min_allowed_dist = max_dist * _PBN_EDGE_INWARD_MIN_DIST_RATIO
    eligible = (comp_mask == 1) & (dist >= min_allowed_dist)
    if not np.any(eligible):
        eligible = comp_mask == 1

    ys, xs = np.where(eligible)
    if ys.size == 0:
        return max_loc

    best_idx = int(np.argmin(center_dist_map[ys, xs]))
    return (int(xs[best_idx]), int(ys[best_idx]))


def _resize_for_clustering(image: Image.Image) -> Image.Image:
    """Return a copy of `image` scaled so its longest side is at most `_CLUSTER_MAX_PX`.
    If the image is already small enough, it is returned unchanged."""
    width, height = image.size
    longest_side = max(width, height)

    if longest_side <= _CLUSTER_MAX_PX:
        return image

    scale = _CLUSTER_MAX_PX / longest_side
    new_size = (int(width * scale), int(height * scale))
    return image.resize(new_size, Image.LANCZOS)
