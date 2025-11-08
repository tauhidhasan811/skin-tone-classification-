

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from PIL import Image
import io
import numpy as np
from deepface import DeepFace
from transformers import AutoImageProcessor
from asset.model import ViTTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 10
trainer = ViTTrainer(num_classes=num_classes)
trainer.load("vit_skin_model_1.pth")
model = trainer.model
model.eval()
model.to(device)
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")


skin_tones = {
    0: "Light White",
    1: "White",
    2: "Light Cream",
    3: "Cream / Pale Beige",
    4: "Light Brown",
    5: "Medium Brown",
    6: "Brown",
    7: "Dark Brown",
    8: "Very Dark Brown",
    9: "Deep Dark"
}

app = FastAPI(title="Age, Gender & Skin Tone API")

@app.get("/")
def home():
    return {"message": "API running! Use POST /predict to get predictions."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith("image/"):
            return JSONResponse(content={"error": "Please upload an image file."}, status_code=400)

        image_bytes = await file.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(pil_image)[:, :, ::-1]  

        result = DeepFace.analyze(img_path=img_np, actions=['age', 'gender'], enforce_detection=False)
        if isinstance(result, list):
            gender = result[0]['dominant_gender']
            age = result[0]['age']
        else:
            gender = result['dominant_gender']
            age = result['age']

        inputs = processor(images=pil_image, return_tensors="pt")
        img_tensor = inputs["pixel_values"].to(device)
        with torch.no_grad():
            outputs = model(img_tensor)
            logits = outputs.logits
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            predicted_class = int(probs.argmax())
            skin_tone_name = skin_tones[predicted_class]


        return JSONResponse(content={
            "age": age,
            "gender": gender,
            "skin_tone": skin_tone_name
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
