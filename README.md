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
    * scipy — NNLS solver for calculating how to mix each palette color from R, G, B, White, Black
    * scikit-image — RGB → LAB color conversion (makes mixing math perceptually accurate)

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

### Finding the Dominant Colors

1. **Shrink and blur the photo.** The image is scaled down so its longest side is at most 1024 pixels, then a blur is applied. The blur smooths out fine texture and surface detail (think fabric grain or wood patterns) so the algorithm focuses on broad color regions rather than noise.

2. **Group pixels by color similarity (KMeans clustering).** Every pixel is treated as a point in 3D space based on its red, green, and blue values. KMeans clustering partitions all pixels into *N* groups (where *N* is the number of colors you requested), where each group contains pixels that are closest in color to one another. The center of each group is the dominant color for that group.

3. **Scale the result back up.** Each pixel in the full-resolution photo gets assigned the label of whichever dominant color it belongs to. Tiny isolated patches that are too small to label are merged into the nearest surrounding color region.

### Building the Paint-by-Numbers Overlay

1. **Find the borders between color regions.** Any pixel that sits next to a pixel belonging to a different color group is considered a border. These borders are drawn black on a white canvas, producing the outline drawing you'd see in a paint-by-numbers kit.

2. **Find a good spot to place each number.** For every disconnected blob of the same color, the algorithm finds the point that is furthest from the blob's edges — the most "interior" spot. This ensures numbers are placed well inside their region rather than right on a border where they'd be hard to read.

3. **Stamp the numbers.** Each color is assigned a number (1, 2, 3, …). That number is printed at the interior point of every blob belonging to that color. The `color_palette` in the API response maps each number to its RGB and hex values so you know which paint to use for each region.

4. **The filled version** follows the same process but fills each region with its actual color before drawing the borders and numbers, giving a preview of what the finished painting will look like.
