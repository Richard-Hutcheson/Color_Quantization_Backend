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

**1. Create and activate the virtual environment:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**2. Install dependencies (first time only):**
```bash
pip install -r requirements.txt
```

**3. Start the server:**
```bash
uvicorn src.main:app --reload
```

The API will be available at `http://localhost:8000`.  
Interactive docs are at `http://localhost:8000/docs`.

