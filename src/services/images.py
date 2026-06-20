import io
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans

from src.models.images import DominantColor, ImagesResponse

# Images are downsampled to this size (longest side) before clustering
# to keep KMeans fast regardless of the original resolution.
_CLUSTER_MAX_PX = 1024

# Directory where palette images are saved
_PALETTE_OUTPUT_DIR = Path(__file__).parents[2] / "output" / "dominant_color_palette"


def extract_dominant_colors(image_bytes: bytes, color_count: int) -> ImagesResponse:
    """
    Determine the `color_count` most dominant colors in the image using KMeans clustering.

    The full-resolution image is preserved in memory for future processing steps
    (pixel remapping, paint-by-numbers overlay, etc.). Only a downsampled copy is
    passed to KMeans to keep clustering fast.
    """
    # Load the full-resolution image and normalise to RGB
    full_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Build a smaller copy for clustering only
    cluster_image = _resize_for_clustering(full_image)

    # Flatten pixels into a 2-D array of shape (num_pixels, 3)
    pixel_array = np.array(cluster_image, dtype=np.float32).reshape(-1, 3)

    # Run KMeans to find the dominant color clusters
    kmeans = KMeans(n_clusters=color_count, random_state=42, n_init="auto")
    kmeans.fit(pixel_array)

    # Cluster centers are the dominant colors; round to nearest integer (0-255)
    centers = np.round(kmeans.cluster_centers_).astype(int)

    colors = [DominantColor(r=int(r), g=int(g), b=int(b)) for r, g, b in centers]

    print(f"Dominant colors ({color_count}):")
    for color in colors:
        print(f"  RGB({color.r}, {color.g}, {color.b}), HEX: {color.hex}")

    return ImagesResponse(color_count=color_count, colors=colors)


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
