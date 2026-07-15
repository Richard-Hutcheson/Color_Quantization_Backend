import base64
import io
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt
from scipy.optimize import minimize
from skimage.color import rgb2lab
from sklearn.cluster import KMeans

from src.models.images import AchievedColor, DominantColor, ImagesResponse, MixRecipe, PaintColor, PaletteEntry

# Images are downsampled to this size (longest side) before fitting KMeans
# to keep clustering fast regardless of the original resolution. Only the
# cluster *centers* are learned at this resolution; per-pixel labels are then
# assigned at full resolution (see extract_dominant_colors) so region
# boundaries stay crisp rather than blocky.
_CLUSTER_MAX_PX = 1280

# Directory where palette images are saved
_PALETTE_OUTPUT_DIR = Path(__file__).parents[2] / "output" / "dominant_color_palette"

# Directory where paint-by-numbers images are saved
_PBN_OUTPUT_DIR = Path(__file__).parents[2] / "output" / "paint_by_numbers"

# Directory where filled paint-by-numbers images are saved
_PBN_FILLED_OUTPUT_DIR = Path(__file__).parents[2] / "output" / "paint_by_numbers_filled"

# Light, edge-preserving denoising applied before clustering.
#
# The previous pipeline used a heavy Gaussian blur (kernel 15, sigma 3.0) which
# smeared fine texture — distinct water ripples, food pieces and highlights all
# merged into large flat blobs. We instead use a bilateral filter, which smooths
# flat areas (reducing noise so KMeans forms clean regions) while preserving the
# edges that carry fine detail.
#
# _CLUSTER_BILATERAL_DIAMETER: neighborhood diameter in pixels.
# _CLUSTER_BILATERAL_SIGMA_COLOR: how different colors may be and still be mixed.
# _CLUSTER_BILATERAL_SIGMA_SPACE: spatial reach of the filter.
_CLUSTER_BILATERAL_DIAMETER = 9
_CLUSTER_BILATERAL_SIGMA_COLOR = 40.0
_CLUSTER_BILATERAL_SIGMA_SPACE = 10.0

# Blobs smaller than this fraction of total image pixels are skipped for number placement
_MIN_BLOB_AREA_RATIO = 0.00020
# Blobs must also meet a minimum short-side size (as a fraction of the
# shorter image side) to avoid preserving long, thin slivers.
# Example: on a 1000px short side, 0.002 means the blob's bounding-box short side
# must be at least 2px to be preserved.
_MIN_BLOB_SHORT_SIDE_RATIO = 0.0015
# Long-thin components with a very high bounding-box aspect ratio are treated as
# narrow artifacts and absorbed.
# Example: 24.0 means a 240x10 or 120x5 bounding box (ratio 24:1) is still allowed,
# while anything more elongated is merged into nearby robust regions.
_MAX_BLOB_ASPECT_RATIO = 24.0

# After boundary smoothing, run one stricter tiny-blob cleanup pass. Smoothing can
# create tiny isolated islands at label junctions; this pass absorbs those artifacts.
_POST_SMOOTH_BLOB_AREA_MULTIPLIER = 1.25

# Paintability smoothing applied to the label map after blob absorption.
# We combine:
# 1) multiscale mask regularization (downsample + blur + upsample + relabel)
# 2) median filtering in label space
# This removes persistent squiggly boundaries while retaining medium-level detail.
_LABEL_BOUNDARY_REGULARIZE_SCALE = 3
_LABEL_BOUNDARY_REGULARIZE_SIGMA = 1.2
_LABEL_BOUNDARY_SMOOTHING_KERNEL = 7
_LABEL_BOUNDARY_SMOOTHING_PASSES = 2

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

    # Build a smaller copy for fitting KMeans only, then apply a light,
    # edge-preserving bilateral filter. This denoises flat areas so KMeans forms
    # clean regions while preserving the edges that carry fine detail (ripples,
    # food pieces, highlights).
    cluster_image = _resize_for_clustering(full_image)
    cluster_array = np.array(cluster_image, dtype=np.uint8)
    cluster_array = cv2.bilateralFilter(
        cluster_array,
        _CLUSTER_BILATERAL_DIAMETER,
        _CLUSTER_BILATERAL_SIGMA_COLOR,
        _CLUSTER_BILATERAL_SIGMA_SPACE,
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

    # Assign labels at full resolution for crisp boundaries.
    #
    # Rather than nearest-neighbor upscaling the coarse cluster-resolution label
    # grid (which produces blocky, jagged region edges), we assign every
    # full-resolution pixel to its nearest cluster center in color space. The
    # full image is lightly bilateral-filtered first so the assignment follows
    # broad regions instead of per-pixel sensor noise, then tiny isolated islands
    # are absorbed into their nearest large neighbor.
    img_w, img_h = full_image.size
    full_array = np.array(full_image, dtype=np.uint8)
    full_array = cv2.bilateralFilter(
        full_array,
        _CLUSTER_BILATERAL_DIAMETER,
        _CLUSTER_BILATERAL_SIGMA_COLOR,
        _CLUSTER_BILATERAL_SIGMA_SPACE,
    )
    label_map = _assign_nearest_center(full_array, kmeans.cluster_centers_)
    min_blob_px = max(1, int(img_h * img_w * _MIN_BLOB_AREA_RATIO))
    label_map = _absorb_small_blobs(label_map, min_blob_px)
    label_map = _smooth_label_boundaries(label_map)
    label_map = _absorb_small_blobs(label_map, int(max(1, round(min_blob_px * _POST_SMOOTH_BLOB_AREA_MULTIPLIER))))

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
) -> str:
    """
    Generate a color-filled preview of the finished painting.

    Each region is filled with its cluster color via a vectorized LUT lookup.
    No borders or numbers are drawn — the preview is a clean, flat-color
    posterization (matching the competitor reference) so fine detail reads
    clearly. The numbered outline for painting lives in the separate
    paint-by-numbers image (see build_paint_by_numbers_image).

    The image is saved to output/paint_by_numbers_filled/pbn_filled.jpg.
    """
    # Vectorized color fill: build a (color_count, 3) LUT and index it with
    # the label map in one operation instead of N per-label mask writes.
    color_lut = np.array([[c.r, c.g, c.b] for c in colors], dtype=np.uint8)
    overlay = color_lut[label_map]

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
    center_x = (img_w - 1) / 2.0
    center_y = (img_h - 1) / 2.0

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
            left = int(stats[comp, cv2.CC_STAT_LEFT])
            top = int(stats[comp, cv2.CC_STAT_TOP])
            width = int(stats[comp, cv2.CC_STAT_WIDTH])
            height = int(stats[comp, cv2.CC_STAT_HEIGHT])

            # Work in the component's tight bounding box instead of full-frame masks.
            # This avoids an O(H*W) distance transform per blob and dramatically cuts
            # runtime when there are many small components.
            comp_roi = (label_img[top:top + height, left:left + width] == comp).astype(np.uint8)
            padded_roi = cv2.copyMakeBorder(
                comp_roi,
                1,
                1,
                1,
                1,
                cv2.BORDER_CONSTANT,
                value=0,
            )
            # distanceTransform assigns each foreground pixel its Euclidean distance to the
            # nearest background pixel; the maximum is the point furthest from any edge —
            # always safely inside the region even for concave shapes.
            dist = cv2.distanceTransform(padded_roi, cv2.DIST_L2, 5)[1:-1, 1:-1]
            edge_touching = (
                left == 0
                or top == 0
                or (left + width) == img_w
                or (top + height) == img_h
            )
            blob_centers.append(
                _select_blob_anchor(
                    comp_roi,
                    dist,
                    edge_touching,
                    center_x,
                    center_y,
                    left,
                    top,
                )
            )
        centers.append(blob_centers)
    return centers


def compute_mix_recipe(target: DominantColor, palette: list[PaintColor]) -> MixRecipe:
    """
    Compute the paint mixing recipe that best approximates `target` using the
    colors in `palette`.

    The solver uses a simple subtractive paint model:
    - convert paint colors to linear-light reflectance
    - blend in absorbance space (Beer-Lambert style)
    - compare the mixed result to the target in LAB

    This better matches real paint behaviour than linear LAB mixing.

    Returns a MixRecipe with:
    - percentages: integer 0–100 for each paint, always summing to 100
    - achieved_color: the color you actually get by mixing those percentages
    """
    # skimage expects float64 RGB in [0, 1] with shape (1, 1, 3)
    def _rgb01_to_lab(rgb01: np.ndarray) -> np.ndarray:
        return rgb2lab(rgb01.reshape(1, 1, 3)).reshape(3)

    def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
        return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)

    def _linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
        return np.where(
            rgb <= 0.0031308,
            12.92 * rgb,
            1.055 * (np.clip(rgb, 0.0, 1.0) ** (1.0 / 2.4)) - 0.055,
        )

    palette_rgb = np.array([[p.r, p.g, p.b] for p in palette], dtype=np.float64) / 255.0
    target_rgb = np.array([target.r, target.g, target.b], dtype=np.float64) / 255.0
    target_lab = _rgb01_to_lab(target_rgb)

    # Blend in absorbance space to approximate subtractive paint mixing.
    palette_linear = _srgb_to_linear(palette_rgb)
    absorbance = -np.log(np.clip(palette_linear, 1e-6, 1.0))

    def _mix_rgb(weights: np.ndarray) -> np.ndarray:
        mixed_absorbance = absorbance.T @ weights
        mixed_linear = np.exp(-mixed_absorbance)
        return np.clip(_linear_to_srgb(mixed_linear), 0.0, 1.0)

    # Constrain weights to be a real mix: non-negative and summing to 1.
    def _objective(weights: np.ndarray) -> float:
        achieved_lab = _rgb01_to_lab(_mix_rgb(weights))
        diff = achieved_lab - target_lab
        return float(diff @ diff)

    num_paints = len(palette)
    initial = np.full(num_paints, 1.0 / num_paints, dtype=np.float64)
    result = minimize(
        _objective,
        initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * num_paints,
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
    )
    if not result.success:
        raise ValueError(f"Mix optimisation failed: {result.message}")

    weights = np.clip(result.x.astype(np.float64), 0.0, 1.0)
    total = weights.sum()
    if total == 0:
        raise ValueError("Mix optimisation produced zero total weight.")
    normalized = weights / total

    # Largest-remainder rounding so percentages always sum to exactly 100
    raw = normalized * 100.0
    floors = np.floor(raw).astype(int)
    remainders = raw - floors
    deficit = 100 - floors.sum()
    top_indices = np.argsort(remainders)[::-1][:deficit]
    floors[top_indices] += 1
    percentages = {p.name: int(floors[i]) for i, p in enumerate(palette)}

    achieved_rgb = np.clip(np.round(_mix_rgb(normalized) * 255.0), 0, 255).astype(int)
    achieved_color = AchievedColor(r=int(achieved_rgb[0]), g=int(achieved_rgb[1]), b=int(achieved_rgb[2]))

    return MixRecipe(percentages=percentages, achieved_color=achieved_color)


def _absorb_small_blobs(label_map: np.ndarray, min_blob_px: int) -> np.ndarray:
    """
    Remove tiny/thin blobs by overwriting their pixels with the label of the
    nearest robust neighbor.

    Robustness is decided per component using area plus simple shape constraints
    (minimum short side and maximum aspect ratio). This keeps cleanup fast while
    still absorbing long narrow strips.
    """
    label_map = label_map.copy()
    num_colors = int(label_map.max()) + 1
    img_h, img_w = label_map.shape
    min_short_side_px = max(2, int(min(img_h, img_w) * _MIN_BLOB_SHORT_SIDE_RATIO))

    valid = np.zeros(label_map.shape, dtype=bool)
    for idx in range(num_colors):
        mask = (label_map == idx).astype(np.uint8)
        n, label_img, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for comp in range(1, n):
            comp_area = int(stats[comp, cv2.CC_STAT_AREA])
            comp_w = int(stats[comp, cv2.CC_STAT_WIDTH])
            comp_h = int(stats[comp, cv2.CC_STAT_HEIGHT])
            short_side = max(1, min(comp_w, comp_h))
            long_side = max(comp_w, comp_h)
            aspect_ratio = long_side / short_side

            if (
                comp_area >= min_blob_px
                and short_side >= min_short_side_px
                and aspect_ratio <= _MAX_BLOB_ASPECT_RATIO
            ):
                valid |= label_img == comp

    if not valid.any():
        return label_map

    if valid.all():
        return label_map

    # Assign every invalid pixel to its nearest valid component in one vectorized pass.
    _, nearest = distance_transform_edt(~valid, return_indices=True)
    label_map[~valid] = label_map[nearest[0][~valid], nearest[1][~valid]]

    return label_map


def _smooth_label_boundaries(label_map: np.ndarray) -> np.ndarray:
    """
    Smooth squiggly boundaries in the discrete label map.

    First, each label mask is regularized at a coarser scale and projected back
    to full resolution, then a median filter performs local majority cleanup.
    This yields cleaner, more paintable outlines than median filtering alone.
    """
    if _LABEL_BOUNDARY_SMOOTHING_PASSES <= 0 and _LABEL_BOUNDARY_REGULARIZE_SCALE <= 1:
        return label_map

    smoothed = label_map.astype(np.int32)
    img_h, img_w = smoothed.shape
    num_labels = int(smoothed.max()) + 1

    if _LABEL_BOUNDARY_REGULARIZE_SCALE > 1 and num_labels > 1:
        small_w = max(1, img_w // _LABEL_BOUNDARY_REGULARIZE_SCALE)
        small_h = max(1, img_h // _LABEL_BOUNDARY_REGULARIZE_SCALE)
        regularized_layers: list[np.ndarray] = []
        for idx in range(num_labels):
            mask = (smoothed == idx).astype(np.float32)
            small = cv2.resize(mask, (small_w, small_h), interpolation=cv2.INTER_AREA)
            small = cv2.GaussianBlur(
                small,
                (0, 0),
                _LABEL_BOUNDARY_REGULARIZE_SIGMA,
                _LABEL_BOUNDARY_REGULARIZE_SIGMA,
            )
            up = cv2.resize(small, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
            regularized_layers.append(up)
        smoothed = np.argmax(np.stack(regularized_layers, axis=2), axis=2).astype(np.int32)

    for _ in range(_LABEL_BOUNDARY_SMOOTHING_PASSES):
        smoothed = cv2.medianBlur(smoothed.astype(np.uint8), _LABEL_BOUNDARY_SMOOTHING_KERNEL).astype(np.int32)
    return smoothed


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
    center_x: float,
    center_y: float,
    offset_x: int,
    offset_y: int,
) -> tuple[int, int]:
    """Choose a blob anchor; edge-touching blobs are biased inward."""
    _, _, _, max_loc = cv2.minMaxLoc(dist)
    if not edge_touching:
        return (offset_x + int(max_loc[0]), offset_y + int(max_loc[1]))

    max_dist = float(dist.max())
    min_allowed_dist = max_dist * _PBN_EDGE_INWARD_MIN_DIST_RATIO
    eligible = (comp_mask == 1) & (dist >= min_allowed_dist)
    if not np.any(eligible):
        eligible = comp_mask == 1

    ys, xs = np.where(eligible)
    if ys.size == 0:
        return (offset_x + int(max_loc[0]), offset_y + int(max_loc[1]))

    global_x = xs.astype(np.float32) + float(offset_x)
    global_y = ys.astype(np.float32) + float(offset_y)
    best_idx = int(np.argmin((global_x - center_x) ** 2 + (global_y - center_y) ** 2))
    return (int(global_x[best_idx]), int(global_y[best_idx]))


def _assign_nearest_center(image_array: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    Assign every pixel in `image_array` (H, W, 3) to the index of its nearest
    color in `centers` (num_centers, 3), using squared Euclidean distance in RGB.

    Returns an (H, W) int32 label map. The distance computation is fully
    vectorized and evaluated in row-chunks so peak memory stays bounded on large
    images (avoids allocating an H*W*num_centers float array all at once).
    """
    img_h, img_w = image_array.shape[:2]
    pixels = image_array.reshape(-1, 3).astype(np.float32)
    centers_f = centers.astype(np.float32)

    labels = np.empty(pixels.shape[0], dtype=np.int32)
    # Process in chunks to cap the size of the intermediate distance matrix.
    chunk = 1_000_000
    for start in range(0, pixels.shape[0], chunk):
        end = start + chunk
        block = pixels[start:end]
        # (chunk, num_centers) squared distances via broadcasting
        dists = np.sum((block[:, None, :] - centers_f[None, :, :]) ** 2, axis=2)
        labels[start:end] = np.argmin(dists, axis=1)

    return labels.reshape(img_h, img_w).astype(np.int32)


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
