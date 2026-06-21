import io
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt
from sklearn.cluster import KMeans

from src.models.images import DominantColor, ImagesResponse

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

# Median blur applied to the label map after upscaling to full resolution.
# Each pixel takes the most common label among its neighbors, which dissolves
# small isolated islands without shifting the main region boundaries.
# Applied this many times in succession; each pass removes finer islands.
_LABEL_MEDIAN_KERNEL = 21  # must be odd
_LABEL_MEDIAN_PASSES = 2

# Blobs smaller than this fraction of total image pixels are skipped for number placement
_MIN_BLOB_AREA_RATIO = 0.001

# cv2.putText parameters for region numbers
_PBN_FONT = cv2.FONT_HERSHEY_SIMPLEX
_PBN_FONT_SCALE = 0.8
_PBN_FONT_THICKNESS = 1


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
    # Reshape into a 2-D label map, scale up to full resolution with nearest-neighbor
    # interpolation, then apply repeated median blur passes to dissolve small islands.
    cluster_h, cluster_w = cluster_array.shape[:2]
    cluster_label_map = kmeans.labels_.reshape(cluster_h, cluster_w).astype(np.uint8)
    img_w, img_h = full_image.size
    label_map = cv2.resize(cluster_label_map, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
    for _ in range(_LABEL_MEDIAN_PASSES):
        label_map = cv2.medianBlur(label_map, _LABEL_MEDIAN_KERNEL)
    label_map = label_map.astype(np.int32)

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


def build_paint_by_numbers_image(
    full_image: Image.Image,
    label_map: np.ndarray,
    colors: list[DominantColor],
) -> None:
    """
    Generate a paint-by-numbers overlay at the original image resolution.

    The output is a white-background image with:
    - Black borders wherever two adjacent pixels belong to different color clusters.
    - A number placed inside every region. Blobs too small to receive a number are
      absorbed into the nearest large neighbor so no unlabeled islands remain.

    The number-to-RGB key is printed to the console.
    The image is saved to output/paint_by_numbers/pbn.jpg.
    """
    img_h, img_w = label_map.shape
    total_pixels = img_h * img_w
    min_blob_px = max(1, int(total_pixels * _MIN_BLOB_AREA_RATIO))

    # Remove small islands: any blob below min_blob_px is absorbed into the
    # nearest large-enough neighboring region so every visible area gets a number.
    label_map = _absorb_small_blobs(label_map, min_blob_px)

    # White background
    overlay = np.full((img_h, img_w, 3), 255, dtype=np.uint8)

    # Build a boolean mask of border pixels: any pixel whose 4-neighbor has a
    # different cluster label is considered a border and drawn black.
    h_edge = label_map[:-1, :] != label_map[1:, :]  # shape (H-1, W)
    v_edge = label_map[:, :-1] != label_map[:, 1:]  # shape (H, W-1)

    border = np.zeros((img_h, img_w), dtype=bool)
    border[:-1, :] |= h_edge
    border[1:, :] |= h_edge
    border[:, :-1] |= v_edge
    border[:, 1:] |= v_edge

    overlay[border] = [0, 0, 0]

    # Print the key and place numbers inside each qualifying blob
    print("Paint-by-numbers key:")
    for idx, color in enumerate(colors):
        number = idx + 1
        print(f"  {number}: RGB({color.r}, {color.g}, {color.b})")

        mask = (label_map == idx).astype(np.uint8)
        num_labels, label_img, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        # Component label 0 is the background (pixels where mask == 0); skip it.
        # All remaining blobs are guaranteed large enough after _absorb_small_blobs.
        for comp in range(1, num_labels):
            # Use distanceTransform to find the most interior point of this blob.
            # The centroid of a concave shape can fall outside the region entirely;
            # the distance-transform peak is always safely inside it.
            comp_mask = (label_img == comp).astype(np.uint8)
            dist = cv2.distanceTransform(comp_mask, cv2.DIST_L2, 5)
            _, _, _, max_loc = cv2.minMaxLoc(dist)
            cx, cy = max_loc
            cv2.putText(
                overlay,
                str(number),
                (cx, cy),
                _PBN_FONT,
                _PBN_FONT_SCALE,
                (0, 0, 0),
                _PBN_FONT_THICKNESS,
                cv2.LINE_AA,
            )

    _PBN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = _PBN_OUTPUT_DIR / "pbn.jpg"
    Image.fromarray(overlay).save(output_path, format="JPEG", quality=95)
    print(f"Paint-by-numbers image saved to {output_path}")


def build_filled_paint_by_numbers_image(
    full_image: Image.Image,
    label_map: np.ndarray,
    colors: list[DominantColor],
) -> None:
    """
    Generate a color-filled version of the paint-by-numbers image.

    Each region is flood-filled with its cluster color. Black borders are drawn
    at region boundaries. Numbers are placed at the most interior point of each
    blob using a contrasting color (white on dark regions, black on light ones)
    for readability.

    The image is saved to output/paint_by_numbers_filled/pbn_filled.jpg.
    """
    img_h, img_w = label_map.shape
    total_pixels = img_h * img_w
    min_blob_px = max(1, int(total_pixels * _MIN_BLOB_AREA_RATIO))

    label_map = _absorb_small_blobs(label_map, min_blob_px)

    # Fill each pixel with its cluster color
    overlay = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    for idx, color in enumerate(colors):
        mask = label_map == idx
        overlay[mask] = [color.r, color.g, color.b]

    # Draw black borders at region boundaries (same logic as PBN)
    h_edge = label_map[:-1, :] != label_map[1:, :]
    v_edge = label_map[:, :-1] != label_map[:, 1:]

    border = np.zeros((img_h, img_w), dtype=bool)
    border[:-1, :] |= h_edge
    border[1:, :] |= h_edge
    border[:, :-1] |= v_edge
    border[:, 1:] |= v_edge

    overlay[border] = [0, 0, 0]

    # Place numbers inside each blob using a contrasting text color
    for idx, color in enumerate(colors):
        number = idx + 1
        # Perceived luminance — pick white text on dark fills, black on light fills
        luminance = 0.299 * color.r + 0.587 * color.g + 0.114 * color.b
        text_color = (255, 255, 255) if luminance < 128 else (0, 0, 0)

        mask = (label_map == idx).astype(np.uint8)
        num_labels, label_img, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        for comp in range(1, num_labels):
            comp_mask = (label_img == comp).astype(np.uint8)
            dist = cv2.distanceTransform(comp_mask, cv2.DIST_L2, 5)
            _, _, _, max_loc = cv2.minMaxLoc(dist)
            cx, cy = max_loc
            cv2.putText(
                overlay,
                str(number),
                (cx, cy),
                _PBN_FONT,
                _PBN_FONT_SCALE,
                text_color,
                _PBN_FONT_THICKNESS,
                cv2.LINE_AA,
            )

    _PBN_FILLED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = _PBN_FILLED_OUTPUT_DIR / "pbn_filled.jpg"
    Image.fromarray(overlay).save(output_path, format="JPEG", quality=95)
    print(f"Filled paint-by-numbers image saved to {output_path}")


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
    for idx in range(num_colors):
        mask = (label_map == idx).astype(np.uint8)
        n, label_img, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for comp in range(1, n):
            if stats[comp, cv2.CC_STAT_AREA] >= min_blob_px:
                valid |= label_img == comp

    if valid.all():
        return label_map

    # For every invalid pixel, copy the label of the nearest valid pixel
    _, nearest = distance_transform_edt(~valid, return_indices=True)
    label_map[~valid] = label_map[nearest[0][~valid], nearest[1][~valid]]
    return label_map


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
