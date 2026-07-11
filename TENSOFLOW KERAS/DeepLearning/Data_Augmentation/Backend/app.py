from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import shutil
import os

from augment import generate_images

app = FastAPI()

# ----------------------------
# Enable CORS
# ----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")


@app.get("/")
def home():
    return {
        "message": "AI Image Augmentation Backend Running 🚀"
    }


@app.post("/augment")
async def augment(file: UploadFile = File(...)):

    filepath = os.path.join(UPLOAD_DIR, file.filename)

    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    images = generate_images(
        image_path=filepath,
        output_dir=OUTPUT_DIR,
        count=10
    )

    return {
        "status": "success",
        "generated_images": images
    }