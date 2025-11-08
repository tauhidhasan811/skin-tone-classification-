from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
from asset.agegen import get_age_gen 

app = FastAPI(title="Age & Gender Prediction API")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        temp_path = "temp_image.jpg"
        pil_image.save(temp_path)

        result = get_age_gen(temp_path) 

        return JSONResponse(content=result)  

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

