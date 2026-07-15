# Color_Quantization_Backend

## Overview
This repo is the frontend application of a paint by numbers web application. The purpose of the application is to allow users to upload a photo and select X amount of colors from the photo they want. The backend will determine what are the X most dominant colors, how to create each one of the colors from a base palette of Red, Green, Blue, White, and Black, return the recreated image with those X amount of colors, and return a paint by numbers overlay. The frontend will display this returned information to the user.

## Tech Stack
* Python 3.14
* FastAPI
    * uvicorn — runs the FastAPI server
    * python-multipart — required for file uploads
* Image handling & processing
    * Pillow — load, resize, and save images
    * numpy — pixel array manipulation
    * opencv-python — blur, edge detection, and drawing borders/numbers for the paint-by-numbers overlay
* Color extraction & remapping
    * scikit-learn — KMeans clustering (extracts palette AND gives you per-pixel labels for remapping)
* Color mixing recipes
    * scipy — constrained optimization for subtractive paint mixing weights
    * scikit-image — RGB → LAB conversion for perceptual color-distance matching

## Running the API

**First time only:**
```bash
make setup
```

**Start the server:**
```bash
make dev
```

The API will be available at `http://localhost:8000`.  
Interactive docs are at `http://localhost:8000/docs`.

## How It Works
The backend exposes one main endpoint:

`POST /api/images`

The request contains:
* `image` (JPG or PNG)
* `color_count` (1–64)
* optional `palette_json` (custom paint palette)

### Processing pipeline
1. Validate upload type and file size.
2. Load the image in RGB.
3. Downsample for clustering speed, apply bilateral denoising, and run KMeans to find `color_count` dominant colors.
4. Reassign every full-resolution pixel to the nearest cluster center for crisp region boundaries.
5. Clean the label map (small-blob absorption + boundary smoothing) so paint regions are more usable.
6. For each dominant color, compute a paint recipe from the palette using constrained subtractive mixing optimization.
7. Build output assets:
   * dominant color palette image
   * paint-by-numbers line image (white background + black region edges + region numbers)
   * filled paint-by-numbers image (regions filled with their assigned dominant colors)

### Response payload
The response includes:
* `color_count`
* `colors`: dominant colors with:
  * RGB + hex
  * `recipe`:
    * `percentages` per paint (always sums to 100)
    * `achieved_color` (the model-predicted mixed result)
* `paint_by_numbers_image` (base64 JPG)
* `paint_by_numbers_filled_image` (base64 JPG)
* `color_palette` (indexed dominant colors for UI display)